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

> **Superseded — see section 15. This blocker does not exist.** The version
> string `11:16.2.0` is Fedora's *epoch 11* plus version *16.2.0*; the local
> glslang is 16.2.0, not 11.16.2, and it compiles `GL_KHR_cooperative_matrix`
> fine. The hand-assembled SPIR-V (and its 64-row bug below) was unnecessary.

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

## 15. Cooperative-matrix Q8 GEMM — shipped as GLSL, measured, opt-in

### The tooling blocker was a misread version string

Sections 12 and 13 record `GL_KHR_cooperative_matrix` as unsupported by the
local toolchain. That conclusion was wrong. `glslangValidator --version`
prints:

```text
Glslang Version: 11:16.2.0
ESSL Version: OpenGL ES GLSL 3.20 glslang Khronos. 16.2.0
```

`11:` is the Fedora packaging **epoch**, not a major version: this is glslang
**16.2.0**, which has supported `GL_KHR_cooperative_matrix` since 13.x. The
machine additionally has `/usr/bin/glslc` = shaderc **v2026.1**, which the
repo's `./glslc` wrapper never reaches because it execs `glslangValidator`
directly.

Both compilers emit real cooperative-matrix SPIR-V. Parsing the binaries shows
capability `4433` (`CooperativeMatrixKHR`), `SPV_KHR_cooperative_matrix`, and
both `OpTypeCooperativeMatrixKHR` and `OpCooperativeMatrixMulAddKHR`. A plain
GLSL probe run through `tools/cmrun` on the 780M returns:

```text
device: AMD Radeon 780M Graphics (RADV PHOENIX) (subgroup 64)
C checksum=-2.167 maxerr=0.00000 bad=0
CM-PROBE-OK
```

Hand-assembled SPIR-V was therefore never required, and the 64-row zeroing bug
in section 13 was an artifact of hand-assembling rather than a hardware or
layout constraint.

### The kernel

`vulkan/matmul_q8_0_mm_f16_cm.comp` replaces the `.spvasm` prototype. One
256-thread workgroup owns a **128-row x 64-token** tile; four subgroups sit in
a 2x2 grid, each holding 4x2 accumulators of the 16x16x16 F16xF16->F32 shape.
Weights dequantize to f16 with the block scale folded in, activations round
once to f16, and all accumulation is f32 inside the cooperative matrix.

The shader guards on `gl_SubgroupID < 4`, so it is correct on both wave64 (4
full subgroups) and wave32 (subgroups 4-7 idle) without any
required-subgroup-size pipeline state.

Build is opt-in and does not disturb the other shaders:

```sh
make q8-cm          # uses GLSLC_CM (default: system glslc), not ./glslc
```

### Why the tile is 128x64: this GEMM is DRAM-bound, not issue-bound

The first version used a 64x64 tile and measured **exactly tied** with the
shipping packed-f16 kernel (~2500 GF/s each). Traffic analysis explains it — a
tiled GEMM re-reads X `out_dim/BM` times and W `n_tok/BN` times, and X is f32
against ~1.06 B/element Q8_0 weights, so the activation stream dominates:

| tile | X | W | total | achieved |
|---|---|---|---|---|
| f16 128x64 | 134.2 MB | 71.3 MB | 205.5 MB | 59.8 GB/s |
| cm 64x64 | 268.4 MB | 71.3 MB | 339.7 MB | **99.2 GB/s** |

The 64x64 tile was saturating DDR5 while the WMMA units idled; the two kernels
tied by coincidence. Doubling BM halved the X stream. Going further to 256x64
*lost* the gain again (16 accumulators/subgroup plus 25 KB LDS costs more
occupancy than the traffic saving buys), so 128x64 is the measured optimum:

| tile | 2048x1024 k=2048 | vs f16 |
|---|---|---|
| cm 64x64 | 3.426 ms | 1.00x |
| **cm 128x64** | **2.388 ms** | **1.44x** |
| cm 256x64 | 2.665 ms | 1.29x |

Isolated kernel throughput, `CMGEMM_BENCH=1 ./tools/cmgemm` (best of 3, 50
dispatches):

| shape | f16 | cm | speedup |
|---|---|---|---|
| 2048x1024 k=2048 | 2501 GF/s | 3597 GF/s | 1.44x |
| 4096x1024 k=2048 | 2561 | 3609 | 1.41x |
| 2048x512 k=2048 | 2449 | 3543 | 1.45x |
| 6144x1024 k=2048 | 2584 | 3597 | 1.39x |
| 1024x1024 k=4096 | 2489 | 3822 | 1.54x |

### Correctness

`tools/cmgemm` runs the real kernel against an f64 CPU reference across eight
shapes, poisons the output buffer first to catch untouched elements, and also
runs the shipping kernel for comparison. The shape that broke the hand-written
prototype (`out_dim=130, n_tok=67, blocks=2`) passes. The cooperative kernel is
**more** accurate than the kernel it replaces on every shape, because it
accumulates in f32 rather than in 8-term f16 chains:

```text
  out_dim=130   n_tok=67    blocks=2    maxrel cm=0.00703  f16=0.01949  OK
  out_dim=31    n_tok=129   blocks=5    maxrel cm=0.01283  f16=0.04686  OK
  out_dim=1536  n_tok=97    blocks=8    maxrel cm=0.01681  f16=0.03475  OK
  CM-GEMM-OK
```

`./q36_test --vulkan-kernels` and `make test` pass with and without the path
enabled.

The batch-size invariance gate — the one this kernel could plausibly have
broken, since it changes the accumulation order — passes exactly with the path
enabled:

```text
q36-test: session-sync-resume warm-vs-cold step 0 top1 ref=674 cand=674
          top5_overlap=5/5 top15_overlap=15/15 top20_overlap=20/20
          top64_overlap=64/64 rms=0 max_abs=0
session-sync-resume: OK
```

**Not yet run with the path enabled:** `--gpu-cpu-parity` and
`--vulkan-fusion-parity` (both were interrupted mid-capture). Those should pass
before this is considered for default-on; until then `Q36_VK_Q8_MM_CM` stays
opt-in and the default route is untouched.

### End-to-end

`--ctx-start 2048 --ctx-max 2048 --prefill-chunk 1024 --gen-tokens 8`, three
warm runs each. (Discard first-run numbers: a cold page cache costs ~8%.)

| | run 1 | run 2 | run 3 | avg |
|---|---:|---:|---:|---:|
| baseline | 226.59 | 218.64 | 221.73 | **222.32** |
| `Q36_VK_Q8_MM_CM=1` | 242.35 | 238.37 | 238.11 | **239.61** |

**+7.8% prefill**, generation unchanged (~25 t/s). Per-kernel GPU time:

| shape | baseline | coopmat | speedup |
|---|---:|---:|---:|
| dense_q8_0_p_8192x2048 | 2293.1 ms | 1686.3 ms | 1.36x |
| dense_q8_0_p_4096x2048 | 850.1 | 617.8 | 1.38x |
| dense_q8_0_p_2048x4096 | 1134.4 | 776.8 | 1.46x |
| dense_q8_0_p_512x2048 | 391.4 | 252.7 | 1.55x |
| dense_q8_0_p_2048x512 | 140.3 | 87.3 | 1.61x |
| dense_q8_0_p_32x2048 | 47.8 | 48.1 | 1.00x (falls back) |
| **total dense Q8 prefill** | **4857** | **3469** | **1.40x** |

1.40x on a 26% share predicts +8.5% end-to-end; +7.8% measured.

`out_dim < 128` keeps the specialized kernels: the 128-row tile would leave
most of a narrow projection idle, which is why `dense_q8_0_p_32x2048` is
deliberately unchanged.

### Note on benchmark methodology

Two traps cost real time here and are worth recording:

1. **Grid/tile mismatch is silent.** The dispatch grid in `q36_vulkan.c` must
   match the shader's `BM`. An oversized grid still produces *correct* output
   (out-of-range rows are guarded) but runs the full k-loop doing nothing —
   the first end-to-end measurement showed a 5% *loss* from exactly this. The
   `groups=` field in `Q36_VK_PROF_KERNEL=1` output is the check: it must not
   change when only the kernel changes.
2. **First run after a build is cold.** Baseline measured 205.68 cold and
   222.32 warm. Always discard run 1.

### Next

The same treatment applies to `moe_gate_up_gemm` (4874 ms, 26%) and
`moe_down_gemm` (2680 ms, 14%) — together twice the dense Q8 share, and both
have the identical structure (dequantize to a staged f16 tile, then a
hand-rolled packed-f16 inner loop). The dequant staging is reusable as-is; only
the inner loop changes. Note those kernels stage from IQ2_XXS and Q2_K rather
than Q8_0, and their B operand is already a routed token tile, so the traffic
arithmetic that picked 128x64 here has to be redone for them.

## 16. Cooperative-matrix MoE GEMMs — measured, +45% prefill, opt-in

The uncommitted MoE CM work (`vulkan/moe_gate_up_gemm_cm.comp`,
`vulkan/moe_down_gemm_cm.comp`, dispatch via `Q36_VK_MOE_MM_CM` in
`q36_vulkan.c`, `make q8-cm` builds all three) was benchmarked as-is.
Same 2048-token bench as §15, three warm runs each (run 1 discarded):

| | run 1 | run 2 | run 3 | run 4 | avg (2-4) |
|---|---:|---:|---:|---:|---:|
| baseline | 225.72 | 225.14 | 224.41 | 224.12 | **224.56** |
| `Q36_VK_Q8_MM_CM=1 Q36_VK_MOE_MM_CM=1` | 326.90 | 326.48 | 326.65 | 326.83 | **326.65** |

**+45.5% prefill**, generation unchanged (~24.5 t/s). Per-kernel GPU time
(2048 ctx, `Q36_VK_PROF_KERNEL=1`):

| op | before (§15) | with CM | speedup |
|---|---:|---:|---:|
| `moe_iq2_gate_up_gemm` | 4874 ms | 1870 ms | 2.6x |
| `moe_q2k_down_gemm` | 2680 ms | 1108 ms | 2.4x |
| dense Q8 total | 4857 ms | ~3456 ms | 1.4x |

MoE CM beats the dense-CM ratio because the baseline MoE kernels were
further from the traffic roof (IQ2_XXS/Q2_K dequant + f16 accumulation
chains); f32 WMMA accumulation plus the halved 128-row dispatch grid
removed both. The 128-row tile with `(dim+127)/128` grid is confirmed in
the profile (`moe_iq2_gate_up_gemm_cm`, `moe_q2k_down_gemm_cm` dispatch
under those names).

### Correctness (all with both CM switches on)

- `./q36_test --vulkan-kernels`: OK; routed `gemm_max_rel` 0.00895 → 0.00507
  (CM more accurate, same direction as dense in §15).
- `tools/cmgemm` dense shapes: OK (unchanged).
- `--session-sync-resume`: OK, exact (`rms=0 max_abs=0`, top64 64/64).
- `--vulkan-fusion-parity`: OK (top1 match, rms ~1.2e-5).
- `--ssd-streaming-parity`: OK, exact on cold-pressure and full-layer.
- Greedy CLI (`--temp 0 -n 20`) on `long_memory_archive.txt` (~4k-token
  prefill): byte-identical output vs baseline; prefill 203.5 → 282.7 t/s
  on that prompt too.
- `--gpu-cpu-parity`: still blocked on the CPU side (30-min timeout in
  `long_*` CPU capture, same as §15). Pre-existing harness slowness, not a
  CM signal; stays the gate before default-on.

### Retuning with CM on — no change

- Prefill chunk (2048 ctx): 512 → 300, 1024 → 327, 2048 → ~106 (same
  collapse as §1). **1024 stays.**
- `Q36_VK_MOE_GEMM_MIN` 64/128/256: interleaved runs show no separation
  (327.2 vs 327.5 after cooldown); an apparent +0.4% for 64 did not
  reproduce. **128 stays.**
- Sustained back-to-back benching throttles this APU (316 → 286 → 327
  across four consecutive runs, gen tok/s dipping with it). Interleave
  configs and cool down before trusting <1% deltas.

### Where prefill goes next

8k-ctx profile with CM on (`--ctx-start 8192 --ctx-max 8192`, 200.6 t/s):
`attn_prefill_qtile2` is 40.7 s = **50.0%** of GPU time (2.2 s at 2k → 40.7 s
at 8k, ~quadratic as expected); MoE CM total ~11.8 s, dense Q8 ~9.2 s.
Attention is now the prefill bottleneck at length. The GEMM family is done;
`attn_prefill_qtile2`/`attn_combine` is the next kernel project.

## 17. Prefill attention — subgroup-max tile reduction, +4% @2k / +21% @8k

Roofline from the §16 8k profile: 10 full-attention layers (40 / interval 4),
n_kv=2, K=Q8_0 (272 B/row/head), V=Q4_0 (144 B) → ~285 GB KV+Q+partials
traffic in 40.68 s ≈ **7 GB/s** vs ~100 GB/s roof. Latency/occupancy-bound,
not bandwidth-bound — headroom without changing a single byte.

`vulkan/attn_prefill_qtile2.comp` (128 threads = 2 tokens × 8 GQA heads,
64 lanes own 4 dims each, 64-key tiles, 4 barriers/tile) had every lane
redundantly scan all 64 LDS scores per row for the online-softmax tile max:
64 lanes × 64 reads for 64 values. Replaced with one element per lane +
`subgroupMax` (new `GL_KHR_shader_subgroup_arithmetic` require; the 64 lanes
of a row group are one wave64 subgroup; other subgroup sizes keep the serial
scan). `max()` on finite values is order-independent → bit-identical.

| bench (CM on) | before | after | delta |
|---|---:|---:|---:|
| 2048 ctx prefill | 326.65 | ~341 (337.6/342.2/344.7/344.3) | **+4.4%** |
| 8192 ctx prefill | 200.58 | 243.4/244.3 | **+21%** |
| qtile2 @2k | 2166 ms | 1605 ms | 1.35x |
| qtile2 @8k | 40681 ms | 26202 ms | 1.55x |

Generation unchanged (~24.4 t/s; decode doesn't use this kernel).

### Correctness

- Greedy CLI (`--temp 0 -n 20`, `long_memory_archive.txt`, CM on):
  byte-identical vs pre-change shader.
- `./q36_test --vulkan-kernels`: OK. `--session-sync-resume`: OK exact.
- `--vulkan-fusion-parity`: OK (top1 match, rms ~1.1e-5).

### Negative probes (reverted, not kept)

- **Exp-fusion** (fold `exp(w-nm)` into AV loop, drop 1 barrier + 1 LDS
  pass): prefill 326.7 → 321.8, kernel 2166 → 2328 ms. The 64x extra
  transcendentals (512 vs 8 exp/lane) cost more than the barrier saved.
  Kernel is ALU-heavy already (Q8/Q4 dequant) — don't add exp.
- **TILE 64→128** (halve barriers/tile, LDS 20→24 KB): 2k and 8k both
  inside noise (kernel 25.6 s vs 26.2 s @8k). Barrier count is not the
  binding constraint; kept 64 for the smaller LDS footprint.

### Note

Mid-experiment the tree briefly lost `l[rr] += w;` through a bad hunk
(fused-test binary was invalid, timing only). Restored and verified via
`git diff` clean + rebuild + greedy-diff before measuring. Lesson: attention
edits get a greedy-diff before any number is trusted.
