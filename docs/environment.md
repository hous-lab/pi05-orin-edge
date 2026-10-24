# Environment

Two machines, deliberately kept as simple as possible:

| Role | Hardware | OS/stack | Runs |
|---|---|---|---|
| deploy target | Jetson AGX Orin 64GB (SM87) | L4T 36.5.0 (JetPack 6.x), CUDA 12.6, python 3.10.12, torch 2.8.0+cu126 (system stack) | FlashRT + overlay, server, CUDA Graph capture, INT8 kernels |
| eval host | x86_64 (RTX 4090 Laptop) | Ubuntu 22.04, python venv | LIBERO-10 sim, eval harness, weight download + conversion |

## Orin runtime stack

- torch comes from the NVIDIA JetPack stack (2.8.0+cu126). Do not pip-install torch on Orin.
- Everything else installs offline from a wheelhouse: build it on x86 with
  [environment/make_orin_wheelhouse.sh](../environment/make_orin_wheelhouse.sh), copy it over, then run [environment/setup_orin_venv.sh](../environment/setup_orin_venv.sh) on Orin.
- FlashRT is cloned at the pinned rev by [flashrt_overlay/apply_overlay.sh](../flashrt_overlay/apply_overlay.sh); CUTLASS v4.4.2 is
  cloned into `third_party/cutlass` for the INT8 GEMM path (sm87 gets it automatically).
- cuDNN 9.3.0 system; TensorRT present (trtexec 10.3) but NOT used by this stack - the runtime
  is torch + custom CUDA kernels + CUDA Graph.

## x86 eval stack

- Eval venv needs: openpi-client, imageio (+ffmpeg), tyro, numpy 1.26.x, and the LIBERO sim
  (mujoco stack). We install LIBERO from the openpi repo checkout: PYTHONPATH points at
  `<openpi>/third_party/libero`.
- No GPU needed on x86 for eval (sim + websocket client only).

## Platform traps we hit (see [docs/lessons.md](../docs/lessons.md) for the full ledger)

- **Clock skew after reboot**: Orin wall clock can lag; future-mtime object files make make
  silently skip recompiles. Touch sources or delete the stale .o before rebuilding; watch for
  Clock skew warnings.
- **GPU health windows**: after long sessions the GPU can enter degraded windows where latencies
  inflate 2x+ and numerics can corrupt. Judge every latency/parity claim inside a verified
  healthy window (probe + latency anchor + numeric anchor) - see [docs/recipe.md](../docs/recipe.md).
- **A/B latency discipline**: lock clocks (jetson_clocks, MAXN) before any A/B; unlocked runs
  mix DVFS states and the delta is meaningless.
