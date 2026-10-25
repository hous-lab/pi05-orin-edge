#!/bin/bash
# Orin (SM87) production server launcher (shipped INT8 recipe) with passive
# latency-stats instrumentation.
# Usage: platforms/arm64/launch_server.sh [port]  (default 8020)
#
# Env:
#   FLASHRT_ROOT    FlashRT tree (sm_87 build)              [required]
#   PI05_CKPT_DIR   bf16 bundle + norm_stats + tokenizer    [required]
#   PI05_MIX_SETS   K16 mix-set npz                         [required]
#   STATS_OUT       stats dump path (default ./stats_server.json)
#   PI05_PY         python to use (default python3)
set -euo pipefail
PORT=${1:-8020}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
: "${FLASHRT_ROOT:?set FLASHRT_ROOT}"
: "${PI05_CKPT_DIR:?set PI05_CKPT_DIR}"
: "${PI05_MIX_SETS:?set PI05_MIX_SETS (tools/mix_sets_k16.npz)}"
export PI05_MIX_SETS
# shellcheck disable=SC1091
source "$REPO/configs/gumix_geg8.env"
export STATS_OUT=${STATS_OUT:-$PWD/stats_server.json}
exec ${PI05_PY:=python3} "$HERE/serve_with_stats.py" --views 2 --pool 1 --layers 27 --steps 1 --port "$PORT"
