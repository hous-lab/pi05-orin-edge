# platforms/arm64 - Jetson AGX Orin 64GB (SM87)

Production platform this repo targets. The serve/eval pipeline lives at the
repo root (runtime/, benchmark/, configs/); this folder holds the Orin-specific
launcher with passive latency instrumentation and the stats-rerun record.

## Files

- `launch_server.sh [port]` - production server via the passive stats wrapper
  (byte-identical serving path; dumps model-side per-inference records on
  SIGINT/exit).
- [serve_with_stats.py](serve_with_stats.py) - the wrapper (same as [platforms/x86](../x86/README.md)).
- `../../benchmark/results/orin_gumix_geg8_stats_500ep/` - the full-percentile
  stats rerun record (500 episodes, same protocol as the accuracy anchors).

## Usage

```bash
FLASHRT_ROOT=<FlashRT sm_87 tree> \
PI05_CKPT_DIR=<bf16 bundle> \
PI05_MIX_SETS=<tools/mix_sets_k16.npz> \
STATS_OUT=<persistent path>/stats_server.json \
platforms/arm64/launch_server.sh 8020
```

Startup fingerprints (see docs/recipe.md): five INT8 recipe lines + two CUDA
Graph lines + `serving on 0.0.0.0:8020`. Lock clocks before latency work
(MAXN + jetson_clocks, docs/recipe.md "GPU health windows").

## Latency rerun (2026-10-25)

Model-caliber closed-loop percentiles over 500 episodes (LIBERO-10, MAXN +
jetson_clocks, server-side records, one sample per policy call):

| calls | p50 | p95 | p99 | max | jitter (p95-p50) |
|---|---|---|---|---|---|
| 26,650 | 93.30 | 93.80 | 93.94 | 125.1 | 0.50 ms |
