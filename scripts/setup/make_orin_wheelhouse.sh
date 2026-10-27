#!/usr/bin/env bash
# aarch64/py3.10 wheelhouse on x86 for Orin offline install (M1).
set -euo pipefail
WH=${1:-./wheelhouse_orin}
mkdir -p "$WH"

cat > /tmp/req-orin.in <<'EOF'
transformers==4.53.2
sentencepiece
protobuf
numpy==1.26.4
pillow
einops
safetensors
websockets
openpi-client
dm-tree
msgpack
tyro
typing-extensions
imageio
imageio-ffmpeg
EOF

cd "$WH"
uv pip compile /tmp/req-orin.in \
  --python-version 3.10 \
  --python-platform aarch64-manylinux2014 \
  -o req-orin.txt

pip download -r req-orin.txt --index-url https://pypi.org/simple/ --retries 10 --timeout 30 -d . \
  --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 --platform any \
  --python-version 3.10 --implementation cp --abi cp310 --abi abi3 --abi none \
  --only-binary :all:

echo "wheelhouse ready: $WH"
ls "$WH" | head -30
