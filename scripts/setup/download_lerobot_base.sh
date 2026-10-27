#!/usr/bin/env bash
# fp32 source weights: lerobot/pi05_libero_base model.safetensors (14.47 GB, F32).
# This is the INPUT to scripts/export/convert_official_weights.py, not the deploy bundle.
# Tries the official endpoint first, then the hf-mirror endpoint (CN networks).
set -euo pipefail
DEST=${1:-${PI05_CKPT_SRC:-./ckpts/pi05_libero_base}}
REPO_ID="lerobot/pi05_libero_base"
SHA256="21b8711787c4a75861b02cff6aa81675a3a943d32b435a68262ac4461e476ba4"
mkdir -p "$DEST"
for EP in https://huggingface.co https://hf-mirror.com; do
  echo "== trying $EP"
  if HF_ENDPOINT=$EP huggingface-cli download "$REPO_ID" model.safetensors \
       --local-dir "$DEST" --local-dir-use-symlinks False; then
    break
  fi
  echo "== $EP failed"
done
echo "$SHA256  $DEST/model.safetensors" | sha256sum -c -
echo "OK: $DEST/model.safetensors"
