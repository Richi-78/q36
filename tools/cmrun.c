/* cmrun.c - run cm_kernel.spv (16x16x16 F16 x F16 -> F32 coop matrix) on the
 * first device advertising that shape, and verify against a CPU reference.
 * Build: cc -O2 -o cmrun cmrun.c -lvulkan
 * Shader: /tmp/cm_kernel.spv      (C), A f16[256]; B f16[256]; OUT f32[256])
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>

#define CKM_ERR(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "FAIL %s: %d\n", #expr, (int)_r); exit(1); } } while (0)

static uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) {          /* inf / nan */
        return (uint16_t)(sign | 0x7c00 | (m ? 0x200 : 0));
    }
    if (e >= 31) return (uint16_t)(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000;
        uint32_t half = m >> (14 - e);
        if (m & (0x1fffffu >> (14 - e))) half += 1;   /* round to nearest */
        return (uint16_t)(sign | (half >> 13));
    }
    uint32_t half = (m + 0x1000) >> 13;        /* round to nearest */
    if (half & 0x400) { half >>= 1; e++; }
    return (uint16_t)(sign | ((uint32_t)(e & 0x1f) << 10) | (half & 0x3ff));
}
static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = sign;
        else {
            e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3ff; f = sign | (e << 23) | (m << 13);
        }
    } else if (e == 31) f = sign | 0x7f800000 | (m << 13);
    else f = sign | ((e + (127 - 15)) << 23) | (m << 13);
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

int main(void) {
    VkInstance inst;
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ii = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &ai };
    CKM_ERR(vkCreateInstance(&ii, NULL, &inst));

    uint32_t nd = 0;
    CKM_ERR(vkEnumeratePhysicalDevices(inst, &nd, NULL));
    VkPhysicalDevice dev = VK_NULL_HANDLE;
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getCm;
    getCm = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!getCm) { fprintf(stderr, "no entrypoint\n"); return 1; }
    VkPhysicalDevice *devs = calloc(nd, sizeof(*devs));
    CKM_ERR(vkEnumeratePhysicalDevices(inst, &nd, devs));
    for (uint32_t d = 0; d < nd && !dev; d++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[d], &p);
        uint32_t pn = 0; getCm(devs[d], &pn, NULL);
        VkCooperativeMatrixPropertiesKHR *pr = calloc(pn ? pn : 1, sizeof(*pr));
        for (uint32_t i = 0; i < pn; i++) pr[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        getCm(devs[d], &pn, pr);
        for (uint32_t i = 0; i < pn; i++) {
            if (pr[i].MSize == 16 && pr[i].NSize == 16 && pr[i].KSize == 16 &&
                pr[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                pr[i].BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                pr[i].CType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                dev = devs[d];
                VkPhysicalDeviceSubgroupProperties subgroup = {
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                };
                VkPhysicalDeviceProperties2 props2 = {
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                    .pNext = &subgroup,
                };
                vkGetPhysicalDeviceProperties2(dev, &props2);
                printf("device: %s (subgroup %u)\n", p.deviceName, subgroup.subgroupSize);
                break;
            }
        }
        free(pr);
    }
    if (!dev) { fprintf(stderr, "no 16x16x16 f16f32 device\n"); return 1; }

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .cooperativeMatrix = VK_TRUE,
    };
    uint32_t qfi = UINT32_MAX, qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qn, NULL);
    VkQueueFamilyProperties *qprops = calloc(qn ? qn : 1, sizeof(*qprops));
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qn, qprops);
    for (uint32_t i = 0; i < qn; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qfi = i;
            break;
        }
    }
    free(qprops);
    if (qfi == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 1; }
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qfi, .queueCount = 1,
    };
    const char *exts[] = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &cmf,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts,
    };
    VkDevice dv; CKM_ERR(vkCreateDevice(dev, &dci, NULL, &dv));
    VkQueue q; vkGetDeviceQueue(dv, qfi, 0, &q);

    /* shader */
    FILE *fp = fopen("/tmp/cm_kernel.spv", "rb");
    if (!fp) { fprintf(stderr, "no spv\n"); return 1; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); rewind(fp);
    uint32_t *code = malloc(sz); if (fread(code, 1, sz, fp) != (size_t)sz) return 1;
    fclose(fp);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = (size_t)sz, .pCode = code };
    CKM_ERR(vkCreateShaderModule(dv, &smci, NULL, &sm));

    VkDescriptorSetLayoutBinding bnd[3] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo dsli = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bnd,
    };
    VkDescriptorSetLayout dsl; CKM_ERR(vkCreateDescriptorSetLayout(dv, &dsli, NULL, &dsl));
    VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .setLayoutCount = 1, .pSetLayouts = &dsl };
    VkPipelineLayout pl; CKM_ERR(vkCreatePipelineLayout(dv, &pli, NULL, &pl));
    VkPipelineShaderStageCreateInfo st = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                          .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm,
                                          .pName = "main" };
    VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                        .stage = st, .layout = pl };
    VkPipeline pp; CKM_ERR(vkCreateComputePipelines(dv, VK_NULL_HANDLE, 1, &cpi, NULL, &pp));

    /* buffers: A f16[256], B f16[256], out f32[256] */
    VkBuffer buf[3]; VkDeviceMemory mem[3];
    VkDeviceSize sizes[3] = { 512, 512, 1024 };
    VkMemoryRequirements req[3];
    for (int i = 0; i < 3; i++) {
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .size = sizes[i],
                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
        CKM_ERR(vkCreateBuffer(dv, &bci, NULL, &buf[i]));
        vkGetBufferMemoryRequirements(dv, buf[i], &req[i]);
        VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize = req[i].size };
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(dev, &mp);
        for (uint32_t t = 0; t < mp.memoryTypeCount; t++) {
            if ((req[i].memoryTypeBits & (1u << t)) &&
                (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                mai.memoryTypeIndex = t; break;
            }
        }
        CKM_ERR(vkAllocateMemory(dv, &mai, NULL, &mem[i]));
        CKM_ERR(vkBindBufferMemory(dv, buf[i], mem[i], 0));
    }
    /* data */
    uint16_t *a = malloc(512);
    uint16_t *b = malloc(512);
    /* build simple exact pattern instead */
    for (int m = 0; m < 16; m++)
        for (int k = 0; k < 16; k++)
            a[m * 16 + k] = f2h((float)((m * 7 + k * 3) % 9) / 4.0f - 0.5f);
    for (int k = 0; k < 16; k++)
        for (int n = 0; n < 16; n++)
            b[k * 16 + n] = f2h((float)((k * 5 + n * 11) % 7) / 6.0f - 0.5f);
    void *ptr;
    vkMapMemory(dv, mem[0], 0, 512, 0, &ptr); memcpy(ptr, a, 512); vkUnmapMemory(dv, mem[0]);
    vkMapMemory(dv, mem[1], 0, 512, 0, &ptr); memcpy(ptr, b, 512); vkUnmapMemory(dv, mem[1]);

    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps };
    VkDescriptorPool dp; CKM_ERR(vkCreateDescriptorPool(dv, &dpci, NULL, &dp));
    VkDescriptorSet ds; VkDescriptorSetLayout dsl_single = dsl;
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                    .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl_single };
    CKM_ERR(vkAllocateDescriptorSets(dv, &dai, &ds));
    VkDescriptorBufferInfo dbi[3] = {
        { buf[0], 0, VK_WHOLE_SIZE }, { buf[1], 0, VK_WHOLE_SIZE }, { buf[2], 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++) {
        w[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                       .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .pBufferInfo = &dbi[i] };
    }
    vkUpdateDescriptorSets(dv, 3, w, 0, NULL);

    VkCommandPool cp;
    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .queueFamilyIndex = qfi };
    CKM_ERR(vkCreateCommandPool(dv, &cpci, NULL, &cp));
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1 };
    CKM_ERR(vkAllocateCommandBuffers(dv, &cbai, &cb));
    VkCommandBufferBeginInfo cbb = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    CKM_ERR(vkBeginCommandBuffer(cb, &cbb));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, 1, 1, 1);
    CKM_ERR(vkEndCommandBuffer(cb));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                        .pCommandBuffers = &cb };
    CKM_ERR(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CKM_ERR(vkDeviceWaitIdle(dv));

    float *out = malloc(1024);
    vkMapMemory(dv, mem[2], 0, 1024, 0, &ptr); memcpy(out, ptr, 1024); vkUnmapMemory(dv, mem[2]);

    float maxerr = 0; int bad = 0;
    for (int m = 0; m < 16; m++) for (int n = 0; n < 16; n++) {
        float ref = 0;
        for (int k = 0; k < 16; k++) ref += h2f(a[m * 16 + k]) * h2f(b[k * 16 + n]);
        float e = fabsf(out[m * 16 + n] - ref);
        if (e > maxerr) maxerr = e;
        ;
        if (e > 0.05f) { bad++; if (bad < 5) printf("  bad[%d][%d]: got %.4f want %.4f\n", m, n, out[m*16+n], ref); }
    }
    float checksum = 0; for (int i = 0; i < 256; i++) checksum += out[i];
    printf("C checksum=%.3f maxerr=%.5f bad=%d\n", checksum, maxerr, bad);
    printf("%s\n", (bad == 0 && maxerr < 0.02f) ? "CM-PROBE-OK" : "CM-PROBE-FAIL");
    return (bad == 0 && maxerr < 0.02f) ? 0 : 1;
}