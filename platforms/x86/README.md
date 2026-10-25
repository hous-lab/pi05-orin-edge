# platforms/x86 - RTX 4090 Laptop (SM89) campaign

Control-variable rerun of the Orin deployment on an RTX 4090 Laptop 16GB: same
weights (byte-identical bf16 bundle), same eval protocol (LIBERO-10, 10 tasks x
50 episodes, seed 7, replan 5, H=10), two quantization configs:

- **config b** - the shipped Orin INT8 recipe (`configs/gumix_geg8.env`), INT8
  CUTLASS kernels rebuilt for sm_89.
- **config a** - FP8 W8A16-fp16 (FP8 E4M3 weights x FP16 activations, cuBLASLt
  descale), the RTX-native scheme; impossible on Orin (SM87 has no FP8).

The only intended variable is the platform. Fingerprints verify which path is
live at startup: config a prints `FP8 quantized 253 GEMM weights (layout=nk)`
and zero INT8 lines; config b prints the five INT8 recipe lines (90 decoder
GEMM, K16 mix 18 layers, 36 ATTN, 36 GATEUP, 18 DOWN).

## Files

- [serve_with_stats.py](serve_with_stats.py) - passive wrapper around the unmodified production
  server: captures the pipeline instance, dumps model-side latency records
  (json + npz) on SIGINT/exit.
- `launch_server.sh a|b [port]` - server launcher for the two configs.
- `run_eval.sh <outdir> [host] [port] [episodes]` - closed-loop eval driver.
- `bench_pi05_rtx.py` - standalone latency bench (single process, no server):
  `FLASHRT_ROOT=... PI05_CKPT_DIR=... PI05_CALIB_NPZ=... python3 bench_pi05_rtx.py --fp8 1 --out out.json`

## Usage

The FlashRT sm_89 tree uses the same overlay ([flashrt_overlay/](../../flashrt_overlay/README.md)) applied to the pinned upstream and built with the local x86 CUDA toolkit (arch-native), then pointed to via `FLASHRT_ROOT`.

```bash
# server (config b), then eval against it
FLASHRT_ROOT=<FlashRT sm_89 tree> PI05_CKPT_DIR=<bf16 bundle> \
PI05_MIX_SETS=<tools/mix_sets_k16.npz> platforms/x86/launch_server.sh b 8021
LIBERO_PATH=<openpi>/third_party/libero \
PI05_PY_EVAL=<python with imageio/tyro/libero/openpi_client> \
platforms/x86/run_eval.sh <outdir> 127.0.0.1 8021 50

# standalone bench
FLASHRT_ROOT=... PI05_CKPT_DIR=... PI05_CALIB_NPZ=... \
python3 platforms/x86/bench_pi05_rtx.py --fp8 1 --out bench_fp8.json
```

## Results (4090 Laptop, 2026-10-25)

Accuracy (500 episodes per arm):

| Arm | Successes/500 | Rate | 95% CI |
|---|---|---|---|
| b: Orin INT8 recipe on sm_89 | 450/500 | 90.0% | ±2.6pp |
| a: FP8 W8A16-fp16 | 445/500 | 89.0% | ±2.7pp |
| Orin reference (same protocol) | 461/500 | 92.2% | ±2.3pp |

All pairwise deltas are within binomial noise (B vs Orin p=0.23, A vs B p=0.60,
A vs Orin p=0.082). Losses concentrate on cliff tasks that shift per config
(b loses 8 episodes on task2; a loses 15 on task8 but gains 8 back on task9) -
marginal-task lottery, not a systematic accuracy floor drop.

Latency, model caliber, over the full closed loop (per-inference wall time
recorded server-side, one sample per policy call):

| Arm | calls | p50 | p95 | p99 | max | jitter (p95-p50) |
|---|---|---|---|---|---|---|
| b: INT8 recipe | 26,924 | 24.78 | 25.18 | 28.96 | 71.9 | 0.40 ms |
| a: FP8 | 28,644 | 25.56 | 28.46 | 43.38 | 87.7 | 2.90 ms |

FP8 ≈ INT8 at p50 on x86: the workload is compute/attention-bound, not
bandwidth-bound, so the weight format barely moves model latency (this is the
opposite of Orin, where the INT8 mix buys 3.5x by bandwidth). FP8 does hold a
fatter tail: its path allocates per-inference staging buffers and coexists with
the LIBERO simulator render context on the same 16GB GPU - at task boundaries
the two peaks can collide into a transient `cudaMalloc` OOM. If a task fails at
its start with `CUDA error 2`, restart the server (clears fragmentation) and
rerun the driver: the harness resumes by skipping completed task JSONs.

Full records: [benchmark/results/x86_4090l_b_500ep](../../benchmark/results/x86_4090l_b_500ep/),
[benchmark/results/x86_4090l_a_500ep](../../benchmark/results/x86_4090l_a_500ep/)
(task JSONs, run logs, server-side latency json + raw records npz, standalone
bench json).
