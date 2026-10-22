# FlashRT overlay

The engine is [FlashRT](https://github.com/flashrt-project/FlashRT) (Apache-2.0, arXiv 2606.20537), an open-source small-batch latency-focused inference engine. This repository does **not** redistribute FlashRT: consumers clone the pinned upstream revision, copy our files on top (the overlay), and build. The kernel `.so` and every generated artifact are regenerable from this directory.

## Pin

| | |
|---|---|
| Upstream | https://github.com/flashrt-project/FlashRT |
| Revision | `f2f04db8` (2026-09-29, "AMD MI300X (CDNA3) support — Pi0.5 at 35.1 ms BF16, GROOT N1.7 at 19.6 ms", #214) |
| CUTLASS | NVIDIA/cutlass `v4.4.2` (`da5e086dab31d63815acafdac9a9c5893b1c69e2`) — upstream CMake expects `third_party/cutlass`; it prints the exact clone command at configure time |
| Build target | Jetson Orin sm_87 |

## Why an overlay, not a fork

Fourteen modified files + four new kernels, all additive and env-gated: with every `FVK_*`/`FLASHRT_*` variable at 0/absent, the engine behaves like upstream at this revision. A fork would entangle our operator work with upstream HEAD (which added Thor/SM120/MPS work after our base) and force consumers to track upstream churn they do not need. The overlay keeps our diff reviewable, the pin reproducible, and upstreaming straightforward.

## File list (18 files)

### New kernels (4)

| File | What it adds |
|---|---|
| `csrc/kernels/gateup_side_geglu.cu` | GEG8: fused gate/up INT8 epilogue kernel — computes the gate/up split + GeGLU after the mixed INT8 GEMM in one pass (−3.2 ms vs separate kernels) |
| `csrc/kernels/int8_fast.cu` | FAST_QUANT: fast activation-quantization kernels for the INT8 path |
| `csrc/kernels/norm_masked.cu` | masked-norm kernel: RMSNorm applied over a token mask (prompt prefix masking) |
| `csrc/gemm/cutlass_sm80_bf16_geglu.cu` | CUTLASS bf16 gated-GeGLU GEMM for the encoder gate/up path |

### Modified (14)

| File | What changed |
|---|---|
| `CMakeLists.txt` | registers the 4 new translation units; output .so lands in `flash_rt/` (upstream default) |
| `csrc/bindings.cpp` | Python bindings for the new kernels |
| `csrc/gemm/gemm_runner.{cu,h}` | CUTLASS INT8 GEMM runner additions |
| `csrc/kernels/activation.{cu,cuh}` | gated-GeGLU device functions shared by the fused epilogue |
| `csrc/kernels/elementwise.{cu,cuh}` | vectorized elementwise ops (`FVK_PI05_RTX_VEC_ELEMENTWISE=1`) |
| `csrc/kernels/norm.cuh` | AdaRMS apply path |
| `csrc/kernels/quantize.{cu,cuh}` | INT8 quant kernels incl. the fast-quant dispatch |
| `csrc/kernels/rope.cu` | vectorized QKV split kernel (8 bf16/thread uint4) |
| `examples/orin/bench_pi05.py` | extends upstream bench to 3 views / 3 cache-frames |
| `flash_rt/frontends/torch/pi05_rtx.py` | encoder INT8 (GU_MIX) wiring: `_enc_gateup_mixed`, K16 mix-set loading, env gating |
| `flash_rt/models/pi05/pipeline_rtx.py` | CUDA Graph capture, GU_MIX mixed gate/up dispatch, GEG8 epilogue, masked-norm apply |
