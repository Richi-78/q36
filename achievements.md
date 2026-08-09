# Phoenix / Radeon 780M optimization results

## 1. Prefill chunk sweep — completed

Command used:

```sh
./q36-bench --vulkan --prefill-chunk N \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 --ctx-max 2048 --gen-tokens 8
```

| chunk | prefill tok/s | generation tok/s |
|---:|---:|---:|
| 256 | 160.04 | 24.99 |
| 512 | 191.68 | 24.67 |
| 1024 | 218.97 | 24.78 |
| 1536 | 118.90 | 25.19 |
| 2048 | 104.06 | 25.32 |

The existing `Q36_GPU_PREFILL_CHUNK_DEFAULT` of **1024** is confirmed as the
best prompt-throughput choice. No source change was needed; the effective
setting is already persisted in `q36.c`.

## 2. Cooperative matrix — runtime enumeration corrected

The Phoenix profile/device exposes `VK_KHR_cooperative_matrix` and enables
`cooperativeMatrix`. The captured `VP_VULKANINFO` JSON does **not** enumerate
per-shape/type properties; that absence is a limitation of the static capture,
not evidence that RADV exposes no usable shapes. Vulkan requires querying
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` at runtime. `q36` now logs
the returned M/N/K dimensions, component types, scope, and saturation support
when initializing Vulkan.

A hand-written SPIR-V prototype was subsequently added for the suitable FP16
shape. The minimal cooperative-matrix shader assembles, validates, and passes
its Phoenix runtime CPU-reference check. The full Q8 GEMM prototype is still
being tested separately and has not been connected to the production graph.

## 3. Integer dot products — investigated

The device reports accelerated signed/unsigned/mixed 8-bit integer dot
products, including packed 4x8-bit forms. The current Q8 kernels consume F32
activations and perform per-block floating-point scaling; using integer dots
would require a new activation quantization/scaling path, not a drop-in shader
replacement. No unvalidated path was enabled.

## 4. Q4 comparison — skipped

A comparable Q4 model is not available, per user direction. No Q4 benchmark or
implementation change was attempted.

## 5. MoE tuning — partial benchmark

Using the existing `Q36_VK_MOE_GEMM_MIN` dispatch threshold with a 1024-token
prefill:

| threshold | prefill tok/s |
|---:|---:|
| 64 | 222.79 |
| 128 (default) | **225.09** |
| 256 | 224.35 |
| 512 | 224.12 |
| 4096 (GEMM effectively disabled) | 200.61 |

The existing threshold of 128 remains best. The active GEMM tile shaders are
already 256-thread, 64-row x 32-slot kernels; no unvalidated tile variant was
persisted.

## 6–7. Synchronization and GPU-resident routing — reviewed

`q36_vulkan.c` already uses GPU-built MoE tiles when the resident expert-bank
cache is available (`q36_vk_moe_tiles_gpu`), and falls back to labeled host
readbacks only when residency/streaming constraints require it. The resident
path is therefore retained; changing the fallback orchestration would require
model/runtime validation and was not safe to guess.

## 8–9. Layout and 64-lane delta decode — reviewed

The existing Phoenix path uses subgroup size 64 where applicable. Delta decode
still intentionally uses 32 threads because each workgroup owns a 32-column
state block and preserves the established update ordering. No layout or
64-thread variant was persisted without a correctness/performance result.

## 10. Clock behavior — not run

No continuous clock/busy sampling was performed; benchmark execution was kept
short and sequential as required for the shared-memory Phoenix system.

## Q4_NL MTP model test — attempted

Tested `/run/media/wikirichi/1ACCDA82CCDA5819/Users/Wikirichi/publicmodels/unsloth/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` (18.5 GB) with the Vulkan benchmark at context 2048, prefill chunk 1024, and 8 generated tokens. Loading reached the Phoenix device, then stopped because the model contains tensor type **20 (`IQ4_NL`)** for `blk.0.ffn_down_exps.weight`, which the current Vulkan backend does not implement.

This is an unsupported-format failure, not a performance result. The MTP container itself was not tested with a separate draft model because the base model cannot currently load.

### Net result

The validated optimization decisions are to retain the current defaults:
**1024-token prefill chunks** and **MoE GEMM threshold 128**. Cooperative-matrix
support remains an open optimization opportunity pending the runtime shape
query and a measured prototype; it is not a negative result. The two retained
defaults were already production defaults, so no tuning change was warranted.

## 11. GEMM tile-shape sweep — completed (all negative, shapes confirmed optimal)

A full A/B round was run on the current default kernels, each variant built and
measured on-device with the standard 2048-token benchmark
(`--ctx-start 2048 --ctx-max 2048 --gen-tokens 8`). Every alternative was
implemented as a separate kernel with identical per-output accumulation order
(numerically safe, bit-identical per output where the 32-slot/64-col parent
claim applies), then removed again after the measurement:

| variant | prefill tok/s | gen tok/s | vs default |
|---:|---:|---:|---|
| **default** (f16 Q8 128x64, MoE 32-slot, delta 32-col) | **208.90** | 24.23 | — |
| dense Q8 GEMM 128x128 tile (8x8 regs/thread) | 188.82 | 25.44 | −10% |
| dense Q8 GEMM 64x64 f32 tile (`Q36_VK_Q8_MM_F16=0`) | 176.73 | 25.22 | −15% |
| MoE gate/up+down 64-slot 512-thread tiles | 183.19 | 25.31 | −12% |
| delta-net decode 64-col (64-lane wave) | 162.58 | 18.49 | −22% |
| delta-net fast for prefill (`Q36_VK_DELTA_DECODE=0`) | 150.22 | 24.29 | −28% |
| delta_net_cols forced on at 2048 (`Q36_VK_DELTA_COL_PREFILL=1`) | 165.50 | 24.80 | −21% |

Takeaways:

- Both directions from the shipped tiles (wider tiles, smaller tiles) lose on the
  780M. The packed-f16 128x64 dense GEMM and 64x32-slot MoE GEMMs sit at a sharp
  local optimum for this device; LDS/register/occupancy geometry is not the
  remaining lever.
- Wider register tiles (8x8) lose ~10% to register pressure; wider MoE slot
  tiles lose ~12%; a 64-lane delta decode loses ~22% (the 32KB state block
  halves workgroups per CU). The 32-column delta-decode design is the right
  one.
- The Phoenix default (delta_net_cols disabled) is confirmed correct at 2048
  tokens too, not only at the 512-token probe that originally motivated it.
- All four experimental kernel sources were removed after measurement; the
  working tree keeps the shipped kernels only. (Experiment diff is preserved in
  a git stash: `stash@{0}`.)

## 12. Cooperative-matrix runtime evidence — concrete

The runtime query (already logged at Vulkan init) returns **14 shapes** on this
device. The directly usable one for the GEMMs is:

```text
16x16x16 A=FLOAT16 B=FLOAT16 C=FLOAT32 R=FLOAT32 scope=SUBGROUP saturating=0
```

i.e. the classic RDNA3 MFMA tile with f32 accumulation, plus several saturating
and non-saturating INT8/INT16 variants. This is the strongest remaining
candidate for the dense Q8 + MoE GEMM family (~65% of profiled GPU time).

One tooling blocker was verified: **glslangValidator 11.16.2 does not support
`GL_KHR_cooperative_matrix`** (a `coorMat` shader fails to compile). The
prototype therefore uses hand-assembled SPIR-V
(`OpCooperativeMatrixMulAddKHR` + `OpTypeCooperativeMatrixKHR`).

### Net result

All measured geometry experiments confirm the shipped defaults; no tuning change
was warranted. Cooperative matrices remain the open, highest-upside item, now
with a confirmed runtime shape list, a working minimal probe, and an
unvalidated Q8 GEMM prototype requiring further tile/correctness work.

## 13. Cooperative-matrix SPIR-V prototype — compiled, GEMM still failing

The following prototype/tooling work was added after the runtime shape query:

- `vulkan/cm_kernel.spvasm` assembles and passes `spirv-val`; `tools/cmrun`
  executes the 16x16x16 F16×F16→F32 operation on the Radeon 780M with
  `CM-PROBE-OK`.
- `vulkan/matmul_q8_0_mm_f16_cm.spvasm` now assembles and passes validation.
  The fixes include correct 16-bit integer weight loads, push-constant
  interface declaration, both 16-value halves of each Q8_0 block, a barrier
  between K tiles, correct `d*128` scaling, output bounds checks, and corrected
  cooperative-output indexing.
- A standalone non-multiple-dimension runtime check (`out_dim=130`,
  `n_tok=67`, `blocks=2`) still fails: the first 64 output rows are valid but
  rows 64 and above become zero (`maxerr=4.343750`). The 128-row/four-subgroup
  GEMM layout is therefore not production-safe and has not been integrated.
- `tools/cmprobe.c` now reports the actual subgroup size (64 on Phoenix), and
  `tools/cmrun.c` selects a compute-capable queue family.

## 14. Latest production benchmark — current baseline

Three sequential runs used:

```sh
./q36-bench --vulkan --prefill-chunk 1024 \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 --ctx-max 2048 --gen-tokens 8
```

| run | prefill tok/s | generation tok/s |
|---:|---:|---:|
| 1 | 223.77 | 25.22 |
| 2 | 226.68 | 25.42 |
| 3 | 225.63 | 25.44 |
| **average** | **225.36** | **25.36** |

This is approximately 7.9% above the previous 208.90 t/s prefill baseline.
The cooperative-matrix prototype was not used for these production results.
