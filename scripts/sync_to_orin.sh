#!/usr/bin/env bash
# rsync the repo to the Orin box (code only; weights/data/wheelhouse are copied separately).
# Usage: bash scripts/sync_to_orin.sh [user@host] [dest]
set -euo pipefail
TARGET=${1:-${ORIN_SSH_TARGET:-orin}}
DEST=${2:-pi05-orin-edge}
rsync -az --delete \
  --exclude .git \
  --exclude .omc \
  --exclude .venv-orin \
  --exclude ckpts \
  --exclude data \
  --exclude wheelhouse \
  --exclude benchmark/results/videos* \
  ./ "$TARGET:$DEST/"
echo "synced to $TARGET:$DEST"
echo "weights: scp ckpts/pi05_libero_official/{model.safetensors,norm_stats.json} + assets/tokenizer/tokenizer.model"
