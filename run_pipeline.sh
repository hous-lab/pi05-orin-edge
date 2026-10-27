#!/usr/bin/env bash
# Top-level pipeline driver (x86 side). Stages:
#   1 guard  - fail fast with pointers when upstream inputs are missing
#   2 convert  - fp32 source -> bf16 deploy bundle (skipped if bundle already exists)
#   3 sync   - rsync repo to Orin (skipped unless ORIN_SSH_TARGET is set)
#   4 serve  - build + launch on Orin (printed, or executed via ssh when set)
#   5 eval   - LIBERO-10 closed-loop from x86
set -euo pipefail
cd "$(dirname "$0")"

CKPT_SRC=${PI05_CKPT_SRC:-ckpts/pi05_libero_base}
CKPT_DIR=${PI05_CKPT_DIR:-ckpts/pi05_libero_official}
FLASHRT_ROOT=${FLASHRT_ROOT:-$(dirname "$PWD")/FlashRT}
SKIP_CONVERT=${SKIP_CONVERT:-0}

# ---- 1. guard
MISSING=""
[ -f "$CKPT_SRC/model.safetensors" ] || MISSING="$MISSING
  - fp32 source weights: $CKPT_SRC/model.safetensors (bash scripts/setup/download_lerobot_base.sh)"
[ -f "$CKPT_DIR/norm_stats.json" ] || MISSING="$MISSING
  - norm_stats.json: $CKPT_DIR/norm_stats.json (bash scripts/setup/download_norm_stats.sh)"
[ -f "$CKPT_DIR/assets/tokenizer/tokenizer.model" ] || MISSING="$MISSING
  - tokenizer: $CKPT_DIR/assets/tokenizer/tokenizer.model (bash scripts/setup/download_paligemma_tokenizer.sh)"
if [ -n "$MISSING" ]; then
  echo "Missing upstream inputs:" >&2
  echo "$MISSING" >&2
  exit 1
fi
echo "[1/5] guard OK"

# ---- 2. convert (skipped when the bundle already exists)
if [ "$SKIP_CONVERT" = 1 ] || [ -f "$CKPT_DIR/model.safetensors" ]; then
  echo "[2/5] convert skipped (bundle exists; set SKIP_CONVERT=0 to force)"
else
  echo "[2/5] converting fp32 -> bf16 bundle (several minutes, streams 14.5 GB)"
  mkdir -p "$CKPT_DIR"
  python3 scripts/export/convert_official_weights.py \
    --src "$CKPT_SRC/model.safetensors" --out "$CKPT_DIR/model.safetensors"
fi
echo "[3/5] bundle ready: $CKPT_DIR"

# ---- 3. sync to Orin
if [ -n "${ORIN_SSH_TARGET:-}" ]; then
  bash scripts/ops/sync_to_orin.sh "$ORIN_SSH_TARGET"
  echo "[4/5] synced; weights are NOT synced by default - copy $CKPT_DIR manually (8.3 GB)"
else
  echo "[4/5] ORIN_SSH_TARGET not set - skip sync"
fi

# ---- 4/5. serve + eval (two machines, so these stay explicit)
cat <<EOF
Next steps (see docs/pipeline_guide.md for full commands):
  Orin: source configs/gumix_geg8.env && python scripts/runtime/server_flashrt_official_ws.py \
        --views 2 --pool 1 --layers 27 --steps 1 --port 8020
  x86:  PYTHONPATH=<openpi>/third_party/libero python benchmark/libero_eval_multitask.py \
        --host <orin-ip> --port 8020
EOF
