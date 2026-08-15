# Agent Notes

`q36.c` is a Qwen3.6-35B-A3B specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase.
new branch "q36/tree/q36-phoenix is psecific for Radeon 780M GFX (codename gfx1103)

## Goals

- Keep the production path as whole-model Vulkan graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on Linux; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Vulkan smoke tests for build verification.
- On the 780M (shared memory), run benchmarks inside a memory-capped scope.
  GPU buffers come out of system RAM, and when TTM cannot satisfy an
  allocation it swaps buffer objects to shmem; with zram as the only swap
  that spirals into a global OOM which takes out the desktop session, not
  just the benchmark. Observed once: `q36-bench invoked oom-killer`,
  `Free swap = 24kB`, GNOME and the browser killed. Use:

  ```sh
  systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 -- ./q36-bench ...
  ```

  `MemorySwapMax=0` is the important half: it prevents the zram spiral.

## Layout

- `q36.c`: model loading, tokenizer, CPU reference code, Vulkan graph scheduling,
  sessions, disk-cache payload serialization.
- `q36_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `q36_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `q36_vulkan.c`: Vulkan runtime and kernel wrappers.
- `vulkan/*.comp`: compute kernels.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## Hardware Targets

### AMD Radeon 780M (RDNA 3, 12 CUs / 768 shaders, shared memory, tipically DDR5)
    Specific branch at https://github.com/Richi-78/q36/tree/q36-phoenix
    Via Vulkan/RADV on Linux
    - Options for Q2 and possibly Q4 for systemns with 32GB RAM (24GB assigned via 
      amdgpu.gttsize and ttm.pages_limit GRUB options 
      (as suggested in https://github.com/kyuz0/amd-strix-halo-toolboxes#host-configuration)

### AMD BC-250 (RDNA 2, 24 CUs / 1536 shaders, 16 GB unified GDDR6) via
    Vulkan/RADV on Linux. Codename "Cyan Skillfish", cut-down PS5 APU. 
    - Unified memory, ~10-14 GB usable for the model after OS and KV cache.
    - Weight buffers map directly from GGUF with no copy via
      VK_EXT_external_memory_host. No staging-buffer path.
    - Q2 routed-expert quant only. The memory budget is too tight for Q4.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Vulkan are available. Use live server tests only when intentionally
testing the API surface.

Benchmark hygiene: the first run after a build is cold and reads ~8% low
(205.68 cold vs 222.32 warm on the same binary). Discard run 1 and average
three. When changing a kernel's tile, check the `groups=` column of
`Q36_VK_PROF_KERNEL=1` — it must not move unless the dispatch grid was
supposed to change. A grid that no longer matches the shader's `BM` still
produces correct output (out-of-range rows are guarded) while silently
running workgroups that do nothing.

### Cooperative-matrix kernels

`vulkan/*_cm.comp` need a glslang with `GL_KHR_cooperative_matrix` and are
built by an opt-in target, so a stock build never requires one:

```sh
make q8-cm                    # compiles with $(GLSLC_CM), default: system glslc
./tools/cmgemm                # correctness vs f64 reference + shipping kernel
CMGEMM_BENCH=1 ./tools/cmgemm # throughput A/B, no model needed
```

`GLSLC_CM` is deliberately separate from `GLSLC` so building these does not
change the compiler used for the other shaders. Do not judge glslang support
from the version string alone — a Fedora epoch (`11:16.2.0` is epoch 11,
version 16.2.0) has already been misread here as an ancient version and cost
a round of unnecessary hand-written SPIR-V.

## Notes
- Do not edit local `ds4/` or `llama.cpp/` checkouts. They are ignored and are
  optional references, not build dependencies.
