/* cmprobe.c - standalone probe of VK_KHR_cooperative_matrix on this device.
 * Build: cc -o cmprobe cmprobe.c -lvulkan */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static const char *ctn(VkComponentTypeKHR t) {
    switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "F16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "F32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR: return "F64";
    case VK_COMPONENT_TYPE_SINT8_KHR: return "S8";
    case VK_COMPONENT_TYPE_UINT8_KHR: return "U8";
    case VK_COMPONENT_TYPE_SINT16_KHR: return "S16";
    case VK_COMPONENT_TYPE_UINT16_KHR: return "U16";
    case VK_COMPONENT_TYPE_SINT32_KHR: return "S32";
    case VK_COMPONENT_TYPE_UINT32_KHR: return "U32";
    default: return "?";
    }
}
static void chk(VkResult r, const char *what) {
    if (r != VK_SUCCESS) { fprintf(stderr, "FAIL %s: %d\n", what, (int)r); exit(1); }
}

int main(void) {
    VkInstance inst;
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ii = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &ai };
    chk(vkCreateInstance(&ii, NULL, &inst), "vkCreateInstance");

    uint32_t n = 0;
    chk(vkEnumeratePhysicalDevices(inst, &n, NULL), "enum count");
    VkPhysicalDevice *devs = calloc(n, sizeof(*devs));
    chk(vkEnumeratePhysicalDevices(inst, &n, devs), "enum");

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getCmProps =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!getCmProps) { printf("loader has no vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\n"); return 1; }

    for (uint32_t d = 0; d < n; d++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[d], &p);
        VkPhysicalDeviceSubgroupProperties subgroup = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
        };
        VkPhysicalDeviceProperties2 props2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &subgroup,
        };
        vkGetPhysicalDeviceProperties2(devs[d], &props2);
        printf("== device %u: %s (api 0x%x, subgroupSize %u)\n",
               d, p.deviceName, p.apiVersion, subgroup.subgroupSize);

        uint32_t en = 0;
        vkEnumerateDeviceExtensionProperties(devs[d], NULL, &en, NULL);
        VkExtensionProperties *exts = calloc(en, sizeof(*exts));
        vkEnumerateDeviceExtensionProperties(devs[d], NULL, &en, exts);
        int cm = 0;
        for (uint32_t e = 0; e < en; e++) {
            if (!strcmp(exts[e].extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
                cm = 1;
                printf("  VK_KHR_cooperative_matrix present (spec %u)\n", exts[e].specVersion);
            }
        }
        free(exts);
        if (!cm) { printf("  no coop matrix ext\n"); continue; }

        VkPhysicalDeviceCooperativeMatrixFeaturesKHR feat = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
        VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                         .pNext = &feat };
        vkGetPhysicalDeviceFeatures2(devs[d], &f2);
        printf("  feature cooperativeMatrix = %u\n", feat.cooperativeMatrix);

        uint32_t pn = 0;
        getCmProps(devs[d], &pn, NULL);
        VkCooperativeMatrixPropertiesKHR *props =
            calloc(pn, sizeof(*props));
        for (uint32_t i = 0; i < pn; i++) props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        getCmProps(devs[d], &pn, props);
        printf("  %u cooperative matrix properties:\n", pn);
        for (uint32_t i = 0; i < pn; i++) {
            VkCooperativeMatrixPropertiesKHR *pr = &props[i];
            printf("    %ux%ux%d A=%s B=%s C=%s R=%s scope=%s sat=%u\n",
                   pr->MSize, pr->NSize, pr->KSize,
                   ctn(pr->AType), ctn(pr->BType), ctn(pr->CType), ctn(pr->ResultType),
                   pr->scope == VK_SCOPE_SUBGROUP_KHR ? "SUBGROUP" : "other",
                   pr->saturatingAccumulation);
        }
        free(props);
    }
    return 0;
}
