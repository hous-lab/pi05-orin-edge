#!/bin/bash
# Closed-loop eval driver (any platform): 10 tasks x N episodes.
# Usage: platforms/x86/run_eval.sh <outdir> [host] [port] [episodes]
set -euo pipefail
OUT=${1:?usage: run_eval.sh <outdir> [host] [port] [episodes]}
HOST=${2:-127.0.0.1}
PORT=${3:-8021}
EPS=${4:-50}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
: "${LIBERO_PATH:?set LIBERO_PATH (path to libero checkout, e.g. <openpi>/third_party/libero)}"
: "${PI05_PY_EVAL:?set PI05_PY_EVAL (python with imageio/tyro/libero/openpi_client)}"
cd "$REPO"
PYTHONPATH="$LIBERO_PATH" exec "$PI05_PY_EVAL" benchmark/libero_eval_multitask.py \
  --episodes "$EPS" --outdir "$OUT" --host "$HOST" --port "$PORT"
