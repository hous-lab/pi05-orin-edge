#!/usr/bin/env bash
# norm_stats.json: per-task normalization quantiles from the openpi-assets bucket.
# This is the exact file behind our 500-episode anchor (sha256 below).
set -euo pipefail
DEST=${1:-${PI05_CKPT_DIR:-./ckpts/pi05_libero_official}}
URL="https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
SHA256="b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"

mkdir -p "$DEST"
OUT="$DEST/norm_stats.json"
curl -fSL --retry 5 --max-time 120 "$URL" -o "$OUT"
echo "$SHA256  $OUT" | sha256sum -c -
echo "OK: $OUT"