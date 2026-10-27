#!/usr/bin/env bash
# PaliGemma SentencePiece tokenizer (Google big_vision public bucket, no auth).
# Placed where the runtime expects it: {PI05_CKPT_DIR}/assets/tokenizer/tokenizer.model
# (FlashRT's own locator also accepts $FLASH_RT_PALIGEMMA_TOKENIZER directly; its
# upstream script scripts/download_paligemma_tokenizer.sh fetches the same file.)
set -euo pipefail
DEST=${1:-${PI05_CKPT_DIR:-./ckpts/pi05_libero_official}}
URL="https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
MD5="1420adc9856720a559e8a87284b195e2"
OUT="$DEST/assets/tokenizer/tokenizer.model"

mkdir -p "$(dirname "$OUT")"
curl -fSL --retry 5 --max-time 300 "$URL" -o "$OUT"
echo "$MD5  $OUT" | md5sum -c -
echo "OK: $OUT"