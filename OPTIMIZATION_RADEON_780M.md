# Radeon 780M Vulkan Optimization Recap

## Hardware and runtime

- GPU: AMD Radeon 780M Graphics, RADV Phoenix
- Vulkan device ID: `1002:1900`
- Mesa/RADV: 26.1.5
- Vulkan subgroup size: 64
- Device-local memory reported by Vulkan: approximately 15.81 GiB
- Q36 model: Qwen3.6-35B-A3B IQ2_XXS/Q2_K mixed quantization
- Backend: generic Vulkan

## Work completed

Q36 was initially using generic Vulkan defaults that were largely developed and
benchmarked around the AMD BC-250. Profiling on the Radeon 780M showed that the
`delta_net_cols` prefill kernel was substantially slower on Phoenix APUs.

The following device-specific behavior was added to `q36_vulkan.c`:

- Detect AMD Phoenix devices with vendor/device ID `1002:1900`.
- Report `Phoenix tuning` during Vulkan startup.
- Disable `delta_net_cols` by default on Phoenix APUs.
- Preserve the existing behavior on the BC-250 and other generic devices.
- Preserve the environment override:

```sh
Q36_VK_DELTA_COL_PREFILL=1   # force-enable
Q36_VK_DELTA_COL_PREFILL=0   # force-disable
```

The change is intentionally narrow so other Vulkan hardware is not silently
forced onto an unvalidated tuning profile.

All Vulkan binaries were rebuilt, and the Vulkan kernel test passed:

```text
vulkan-kernels: OK
q36 tests: ok
```

## Measurements

Initial measurements at the default 800 MHz GPU clock showed approximately:

```text
Before Phoenix tuning: 167 t/s prefill at 512 tokens
After disabling delta_net_cols: 227 t/s prefill at 512 tokens
```

This was approximately a 36% improvement for that short prefill case.

With the GPU forced to high-performance mode and reaching 2700 MHz, the later
measurements were:

| Context | Prefill | Decode |
|---:|---:|---:|
| 512 | 231.92 t/s | 24.68 t/s |
| 1024 | 251.75 t/s | 24.73 t/s |
| 3072 | 199.81 t/s | 24.08 t/s |
| 4096 | 158.58 t/s | 23.97 t/s |

The GPU frequency was verified through:

```sh
cat /sys/class/drm/card1/device/hwmon/hwmon3/freq1_input
```

`2700000000` means 2700 MHz. The `pp_dpm_sclk` file lists available clock
states, but does not necessarily identify the current clock.

## Profiling findings

Using `Q36_VK_PROF_KERNEL=1` identified the largest costs on the 780M:

- Long-context attention prefill: `attn_prefill_qtile2`
- Routed MoE prefill GEMMs:
  - `moe_gate_up_gemm`
  - `moe_down_gemm`
- Recurrent decode processing: `delta_net_decode`
- Dense Q8 matmul kernels

Tests showed that:

- Keeping `attn_prefill_qtile2` enabled is faster than its fallback.
- Keeping the GPU recurrent and fused paths enabled is faster overall.
- Disabling MoE GEMM gives no useful improvement at longer contexts.
- Disabling `delta_net_cols` is the clear Phoenix-specific win.

## Why llama.cpp may report around 300 t/s

The comparison is useful, but may not be directly equivalent. Results depend on:

- Model architecture and quantization
- Prompt length and benchmark method
- Whether the measurement is prompt processing or generation
- Kernel scheduling and synchronization strategy
- KV-cache type and context length
- Whether the llama.cpp path uses a different Vulkan shader strategy

Q36 is a model-specific Qwen3.6 engine with recurrent layers, routed MoE
experts, custom IQ2 kernels, and a different execution graph. Its current
performance is therefore not expected to match a general llama.cpp Vulkan
backend automatically.

## Useful benchmark commands

Short benchmark:

```sh
./q36-bench --vulkan \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 1024 --ctx-max 1024 --gen-tokens 8
```

Long-context benchmark:

```sh
./q36-bench --vulkan \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 1024 --ctx-max 4096 \
  --step-incr 1024 --gen-tokens 8
```

Kernel profiling:

```sh
Q36_VK_PROF_KERNEL=1 ./q36-bench --vulkan \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 4096 --ctx-max 4096 --gen-tokens 8
```

Force the previous delta-column path for comparison:

```sh
Q36_VK_DELTA_COL_PREFILL=1 ./q36-bench --vulkan ...
```

## Possible future improvements

### 1. Phoenix-specific attention shader

`attn_prefill_qtile2` is the largest long-context prefill cost. A Phoenix-tuned
variant could investigate:

- Different token/head tiling
- Reduced shared-memory pressure
- More suitable workgroup shape for Radeon 780M
- Better use of 64-lane subgroups
- Reduced barriers between score and value phases
- Specialization for Q4_0 value-cache rows

This is the most promising prefill optimization, but it requires numerical
parity tests against the current kernel.

### 2. Phoenix-specific MoE GEMM tuning

The routed MoE GEMMs consume a large portion of prefill time. Possible work:

- Test alternative expert/pair tile sizes
- Tune workgroup dimensions for the 780M's available compute units
- Reduce redundant weight reads
- Compare expert-major and token-major layouts
- Add a device-specific threshold for GEMM versus matvec dispatch

### 3. Decode optimization

Decode throughput is approximately 24 tokens/s and remains fairly stable as
context grows. Potential targets are:

- `delta_net_decode`
- Dense Q8 decode matvecs
- Host/GPU submission synchronization
- Small-dispatch batching
- Avoiding unnecessary readbacks between layers

The profiler showed significant synchronization overhead, so reducing command
submission or flush frequency may help as much as changing shader arithmetic.

### 4. Runtime device capability profile

The current implementation identifies Phoenix using the device ID. A more
extensible design could maintain a small Vulkan tuning profile containing:

- Preferred prefill chunk size
- Delta-net kernel selection
- Attention variant
- MoE GEMM threshold
- Workgroup specialization
- KV-cache preferences

Explicit environment variables should remain available for diagnostics.

### 5. Benchmark methodology comparison

To understand the gap with llama.cpp, run both engines with exactly the same:

- GGUF model
- Prompt token sequence
- Context lengths
- KV-cache precision
- Number of generated tokens
- Warm/cold model state
- GPU performance mode

The comparison should separately record prefill and decode throughput.

## Current recommendation

Use the normal Vulkan build with the Phoenix-specific default enabled:

```sh
make
./q36 --vulkan -p "Hello"
```

For sustained testing, performance mode can be enabled temporarily:

```sh
sudo sh -c 'echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level'
```

Restore automatic power management afterward:

```sh
sudo sh -c 'echo auto > /sys/class/drm/card1/device/power_dpm_force_performance_level'
```

The current code change is safe, validated by the Vulkan kernel tests, and
provides a meaningful improvement over the original generic defaults on the
Radeon 780M.
