/* cmgemm.c - standalone correctness harness for the cooperative-matrix
 * Q8_0 prefill GEMM (vulkan/matmul_q8_0_mm_f16_cm.spv).
 *
 * Runs the real kernel against a CPU reference over several shapes,
 * including deliberately non-multiple dimensions, which is where the
 * earlier hand-assembled SPIR-V prototype silently returned zero rows.
 *
 * Build: cc -O2 -o tools/cmgemm tools/cmgemm.c -lvulkan -lm
 * Run:   ./tools/cmgemm [path-to-spv]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <time.h>

#define VKC(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "FAIL %s: %d\n", #expr, (int)_r); exit(1); } } while (0)

#define BM 64u
#define BN 64u
#define BK 32u
#define Q8_0_U16 17u

static uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (m ? 0x200 : 0));
    if (e >= 31) return (uint16_t)(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000;
        uint32_t half = m >> (14 - e);
        if (m & (0x1fffffu >> (14 - e))) half += 1;
        return (uint16_t)(sign | (half >> 13));
    }
    uint32_t half = (m + 0x1000) >> 13;
    if (half & 0x400) { half >>= 1; e++; }
    return (uint16_t)(sign | ((uint32_t)(e & 0x1f) << 10) | (half & 0x3ff));
}
static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
    if (e == 0) {
        if (m == 0) f = sign;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; }
               m &= 0x3ff; f = sign | (e << 23) | (m << 13); }
    } else if (e == 31) f = sign | 0x7f800000 | (m << 13);
    else f = sign | ((e + (127 - 15)) << 23) | (m << 13);
    float out; memcpy(&out, &f, sizeof(out)); return out;
}

static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dv;
static VkQueue queue;
static uint32_t qfam;
static VkPhysicalDeviceMemoryProperties memprops;
static VkShaderModule shader;

static uint32_t pick_mem(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t t = 0; t < memprops.memoryTypeCount; t++)
        if ((bits & (1u << t)) &&
            (memprops.memoryTypes[t].propertyFlags & want) == want) return t;
    fprintf(stderr, "no memory type\n"); exit(1);
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; } Buf;

static Buf mkbuf(VkDeviceSize size) {
    Buf b; b.size = size;
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = size,
                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
    VKC(vkCreateBuffer(dv, &bci, NULL, &b.buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dv, b.buf, &req);
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = pick_mem(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
    VKC(vkAllocateMemory(dv, &mai, NULL, &b.mem));
    VKC(vkBindBufferMemory(dv, b.buf, b.mem, 0));
    return b;
}
static void *map(Buf b) { void *p; VKC(vkMapMemory(dv, b.mem, 0, b.size, 0, &p)); return p; }

/* Deterministic pseudo-random in [-1,1). */
static float rnd(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return (float)((int32_t)(*s >> 8) % 2000 - 1000) / 1000.0f;
}

static int run_shape_on(VkShaderModule mod, uint32_t bm,
                        uint32_t out_dim, uint32_t n_tok, uint32_t blocks,
                        float scale, double *out_maxrel) {
    uint32_t in_dim = blocks * BK;
    size_t wu16 = (size_t)out_dim * blocks * Q8_0_U16;
    size_t xn = (size_t)n_tok * in_dim;
    size_t yn = (size_t)n_tok * out_dim;

    Buf wb = mkbuf(wu16 * 2), xb = mkbuf(xn * 4), yb = mkbuf(yn * 4);

    /* Build Q8_0 weights: per block a f16 scale then 32 int8. */
    uint16_t *w = map(wb);
    uint32_t seed = 12345u ^ (out_dim * 7919u) ^ (n_tok * 104729u);
    float *wref = malloc(sizeof(float) * (size_t)out_dim * in_dim);
    for (uint32_t r = 0; r < out_dim; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint16_t *blk = w + ((size_t)r * blocks + b) * Q8_0_U16;
            float d = 0.002f + 0.03f * fabsf(rnd(&seed));
            blk[0] = f2h(d);
            d = h2f(blk[0]);
            signed char *qs = (signed char *)(blk + 1);
            for (uint32_t e = 0; e < BK; e++) {
                int q = (int)(rnd(&seed) * 127.0f);
                if (q > 127) q = 127; if (q < -127) q = -127;
                qs[e] = (signed char)q;
                wref[(size_t)r * in_dim + b * BK + e] = (float)q * d;
            }
        }
    }
    vkUnmapMemory(dv, wb.mem);

    float *x = map(xb);
    float *xref = malloc(sizeof(float) * xn);
    for (size_t i = 0; i < xn; i++) { x[i] = rnd(&seed); xref[i] = x[i]; }
    vkUnmapMemory(dv, xb.mem);

    float *yp = map(yb);
    memset(yp, 0xCD, yn * 4);          /* poison: catches untouched outputs */
    vkUnmapMemory(dv, yb.mem);

    /* descriptors */
    VkDescriptorSetLayoutBinding bnd[3];
    for (int i = 0; i < 3; i++)
        bnd[i] = (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dsci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bnd };
    VkDescriptorSetLayout dsl; VKC(vkCreateDescriptorSetLayout(dv, &dsci, NULL, &dsl));
    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 20 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout pl; VKC(vkCreatePipelineLayout(dv, &plci, NULL, &pl));
    VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main" },
        .layout = pl };
    VkPipeline pipe; VKC(vkCreateComputePipelines(dv, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps };
    VkDescriptorPool dp; VKC(vkCreateDescriptorPool(dv, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    VkDescriptorSet ds; VKC(vkAllocateDescriptorSets(dv, &dsai, &ds));
    VkDescriptorBufferInfo dbi[3] = { { wb.buf, 0, VK_WHOLE_SIZE },
                                      { xb.buf, 0, VK_WHOLE_SIZE },
                                      { yb.buf, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++)
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i] };
    vkUpdateDescriptorSets(dv, 3, wds, 0, NULL);

    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = qfam };
    VkCommandPool cp; VKC(vkCreateCommandPool(dv, &cpi, NULL, &cp));
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cb; VKC(vkAllocateCommandBuffers(dv, &cbai, &cb));
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VKC(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    struct { uint32_t out_dim, n_tok, blocks, row_bytes; float scale; }
        push = { out_dim, n_tok, blocks, blocks * 34u, scale };
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cb, (out_dim + bm - 1) / bm, (n_tok + BN - 1) / BN, 1);
    VKC(vkEndCommandBuffer(cb));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &cb };
    VKC(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VKC(vkDeviceWaitIdle(dv));

    yp = map(yb);
    double maxrel = 0; int bad = 0, poisoned = 0;
    uint32_t poison; memset(&poison, 0xCD, 4);
    float poisonf; memcpy(&poisonf, &poison, 4);
    for (uint32_t c = 0; c < n_tok; c++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            double ref = 0;
            for (uint32_t k = 0; k < in_dim; k++)
                ref += (double)wref[(size_t)r * in_dim + k] * (double)xref[(size_t)c * in_dim + k];
            ref *= scale;
            float got = yp[(size_t)c * out_dim + r];
            if (memcmp(&got, &poisonf, 4) == 0) { poisoned++; continue; }
            double den = fabs(ref) > 1.0 ? fabs(ref) : 1.0;
            double rel = fabs(got - ref) / den;
            if (rel > maxrel) maxrel = rel;
            /* Only gross breakage counts as bad: random-sign dot products
             * cancel heavily, so a few percent relative is normal f16 input
             * rounding, not a defect. Kernel-vs-kernel maxrel is the real
             * signal and is reported by the caller. */
            if (rel > 0.10) {
                if (bad < 4) printf("    bad[tok %u][row %u]: got %.5f want %.5f\n", c, r, got, ref);
                bad++;
            }
        }
    }
    vkUnmapMemory(dv, yb.mem);
    int ok = (bad == 0 && poisoned == 0);
    if (out_maxrel) *out_maxrel = maxrel;

    free(wref); free(xref);
    vkDestroyCommandPool(dv, cp, NULL); vkDestroyDescriptorPool(dv, dp, NULL);
    vkDestroyPipeline(dv, pipe, NULL); vkDestroyPipelineLayout(dv, pl, NULL);
    vkDestroyDescriptorSetLayout(dv, dsl, NULL);
    vkDestroyBuffer(dv, wb.buf, NULL); vkFreeMemory(dv, wb.mem, NULL);
    vkDestroyBuffer(dv, xb.buf, NULL); vkFreeMemory(dv, xb.mem, NULL);
    vkDestroyBuffer(dv, yb.buf, NULL); vkFreeMemory(dv, yb.mem, NULL);
    return ok;
}

static VkShaderModule load_mod(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    uint32_t *code = malloc(sz);
    if (fread(code, 1, sz, fp) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(fp);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sz, .pCode = code };
    VkShaderModule m; VKC(vkCreateShaderModule(dv, &smci, NULL, &m));
    return m;
}

static VkShaderModule mod_cm, mod_base;
/* BM of the coopmat kernel under test; must match its tile. */
static uint32_t cm_bm = 128u;

/* Run one shape on the coopmat kernel and, when available, on the shipping
 * packed-f16 kernel, so the numerics delta is measured against what the
 * engine already produces rather than only against an exact reference. */
static int run_shape(uint32_t out_dim, uint32_t n_tok, uint32_t blocks, float scale) {
    double rel_cm = 0, rel_base = -1.0;
    int ok = run_shape_on(mod_cm, cm_bm, out_dim, n_tok, blocks, scale, &rel_cm);
    if (mod_base) ok &= run_shape_on(mod_base, 128u, out_dim, n_tok, blocks, scale, &rel_base);
    /* The coopmat kernel must not be less accurate than the kernel it
     * replaces; it accumulates in f32 so it should normally be better. */
    if (rel_base >= 0.0) {
        int regressed = rel_cm > rel_base * 1.5 + 1e-4;
        ok &= !regressed;
        printf("  out_dim=%-5u n_tok=%-5u blocks=%-3u  maxrel cm=%.5f  f16=%.5f  %s%s\n",
               out_dim, n_tok, blocks, rel_cm, rel_base, ok ? "OK" : "FAIL",
               regressed ? " (cm less accurate)" : "");
    }
    else
        printf("  out_dim=%-5u n_tok=%-5u blocks=%-3u  maxrel cm=%.5f  %s\n",
               out_dim, n_tok, blocks, rel_cm, ok ? "OK" : "FAIL");
    return ok;
}


/* Timed A/B: same shape, same buffers, both kernels. Weight/activation
 * contents do not affect timing, so this reuses the correctness setup at
 * benchmark sizes. Buffers stay a few MB, so this never approaches the
 * memory pressure of a full model run. */
static double bench_one(VkShaderModule mod, uint32_t bm, uint32_t out_dim,
                        uint32_t n_tok, uint32_t blocks, int iters) {
    uint32_t in_dim = blocks * BK;
    Buf wb = mkbuf((size_t)out_dim * blocks * Q8_0_U16 * 2);
    Buf xb = mkbuf((size_t)n_tok * in_dim * 4);
    Buf yb = mkbuf((size_t)n_tok * out_dim * 4);
    uint16_t *w = map(wb);
    for (size_t i = 0; i < (size_t)out_dim * blocks; i++) w[i * Q8_0_U16] = f2h(0.01f);
    vkUnmapMemory(dv, wb.mem);

    VkDescriptorSetLayoutBinding bnd[3];
    for (int i = 0; i < 3; i++)
        bnd[i] = (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dsci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bnd };
    VkDescriptorSetLayout dsl; VKC(vkCreateDescriptorSetLayout(dv, &dsci, NULL, &dsl));
    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 20 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout pl; VKC(vkCreatePipelineLayout(dv, &plci, NULL, &pl));
    VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main" },
        .layout = pl };
    VkPipeline pipe; VKC(vkCreateComputePipelines(dv, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps };
    VkDescriptorPool dp; VKC(vkCreateDescriptorPool(dv, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    VkDescriptorSet ds; VKC(vkAllocateDescriptorSets(dv, &dsai, &ds));
    VkDescriptorBufferInfo dbi[3] = { { wb.buf, 0, VK_WHOLE_SIZE },
                                      { xb.buf, 0, VK_WHOLE_SIZE },
                                      { yb.buf, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++)
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i] };
    vkUpdateDescriptorSets(dv, 3, wds, 0, NULL);

    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = qfam };
    VkCommandPool cp; VKC(vkCreateCommandPool(dv, &cpi, NULL, &cp));
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cb; VKC(vkAllocateCommandBuffers(dv, &cbai, &cb));
    struct { uint32_t out_dim, n_tok, blocks, row_bytes; float scale; }
        push = { out_dim, n_tok, blocks, blocks * 34u, 1.0f };

    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        VKC(vkBeginCommandBuffer(cb, &cbbi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        for (int i = 0; i < iters; i++) {
            vkCmdDispatch(cb, (out_dim + bm - 1) / bm, (n_tok + BN - 1) / BN, 1);
            VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT };
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        }
        VKC(vkEndCommandBuffer(cb));
        struct timespec t0, t1;
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .commandBufferCount = 1, .pCommandBuffers = &cb };
        clock_gettime(CLOCK_MONOTONIC, &t0);
        VKC(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        VKC(vkDeviceWaitIdle(dv));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        if (el < best) best = el;
        vkResetCommandPool(dv, cp, 0);
    }
    vkDestroyCommandPool(dv, cp, NULL); vkDestroyDescriptorPool(dv, dp, NULL);
    vkDestroyPipeline(dv, pipe, NULL); vkDestroyPipelineLayout(dv, pl, NULL);
    vkDestroyDescriptorSetLayout(dv, dsl, NULL);
    vkDestroyBuffer(dv, wb.buf, NULL); vkFreeMemory(dv, wb.mem, NULL);
    vkDestroyBuffer(dv, xb.buf, NULL); vkFreeMemory(dv, xb.mem, NULL);
    vkDestroyBuffer(dv, yb.buf, NULL); vkFreeMemory(dv, yb.mem, NULL);
    return best / iters;
}

static void bench_shape(uint32_t out_dim, uint32_t n_tok, uint32_t blocks) {
    int iters = 50;
    double tc = bench_one(mod_cm, cm_bm, out_dim, n_tok, blocks, iters);
    double flop = 2.0 * out_dim * n_tok * (blocks * BK);
    if (mod_base) {
        double tb = bench_one(mod_base, 128u, out_dim, n_tok, blocks, iters);
        printf("  %5ux%-5u k=%-5u   f16 %7.3f ms (%6.1f GF/s)   cm %7.3f ms (%6.1f GF/s)   %.2fx\n",
               out_dim, n_tok, blocks * BK, tb * 1e3, flop / tb / 1e9,
               tc * 1e3, flop / tc / 1e9, tb / tc);
    } else {
        printf("  %5ux%-5u k=%-5u   cm %7.3f ms (%6.1f GF/s)\n",
               out_dim, n_tok, blocks * BK, tc * 1e3, flop / tc / 1e9);
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "vulkan/matmul_q8_0_mm_f16_cm.spv";
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ii = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &ai };
    VKC(vkCreateInstance(&ii, NULL, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice *devs = malloc(sizeof(*devs) * n);
    vkEnumeratePhysicalDevices(inst, &n, devs);
    phys = devs[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    vkGetPhysicalDeviceMemoryProperties(phys, &memprops);

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties *qp = malloc(sizeof(*qp) * nq);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
    qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &prio };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .cooperativeMatrix = VK_TRUE };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &cmf, .shaderFloat16 = VK_TRUE, .storageBuffer8BitAccess = VK_TRUE,
        .uniformAndStorageBuffer8BitAccess = VK_TRUE };
    VkPhysicalDeviceVulkan11Features f11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &f12, .storageBuffer16BitAccess = VK_TRUE,
        .uniformAndStorageBuffer16BitAccess = VK_TRUE };
    VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                     .pNext = &f11 };
    const char *exts[] = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts };
    VKC(vkCreateDevice(phys, &dci, NULL, &dv));
    vkGetDeviceQueue(dv, qfam, 0, &queue);

    VkPhysicalDeviceSubgroupProperties sg = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
    VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                       .pNext = &sg };
    vkGetPhysicalDeviceProperties2(phys, &p2);
    printf("device: %s (subgroup %u)\nshader: %s\n", props.deviceName, sg.subgroupSize, path);

    { const char *e = getenv("CM_BM"); if (e) cm_bm = (uint32_t)atoi(e); }
    mod_cm = load_mod(path);
    const char *basep = argc > 2 ? argv[2] : "vulkan/matmul_q8_0_mm_f16.spv";
    FILE *bt = fopen(basep, "rb");
    if (bt) { fclose(bt); mod_base = load_mod(basep); printf("baseline: %s\n", basep); }
    else printf("baseline: (not found, cm-only)\n");

    int all = 1;
    /* exact tile multiples */
    all &= run_shape(64, 64, 2, 1.0f);
    all &= run_shape(128, 128, 4, 1.0f);
    all &= run_shape(256, 64, 8, 0.5f);
    /* the shape that broke the hand-assembled prototype */
    all &= run_shape(130, 67, 2, 1.0f);
    /* assorted ragged edges */
    all &= run_shape(1, 1, 1, 1.0f);
    all &= run_shape(65, 3, 3, 1.0f);
    all &= run_shape(31, 129, 5, 2.0f);
    all &= run_shape(1536, 97, 8, 1.0f);
    printf("%s\n", all ? "CM-GEMM-OK" : "CM-GEMM-FAIL");

    if (getenv("CMGEMM_BENCH")) {
        printf("\nthroughput (best of 3, 50 dispatches each):\n");
        bench_shape(2048, 1024, 64);
        bench_shape(4096, 1024, 64);
        bench_shape(2048, 512,  64);
        bench_shape(6144, 1024, 64);
        bench_shape(1024, 1024, 128);
    }
    return all ? 0 : 1;
}
