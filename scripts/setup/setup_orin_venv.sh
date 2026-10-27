#!/usr/bin/env bash
# Orin offline venv: system torch stack visible + python-only deps from wheelhouse.
# Run on Orin from anywhere: bash scripts/setup/setup_orin_venv.sh
# The runtime stack is FlashRT + our overlay; the openpi repo is NOT needed on Orin
# (it is only required on x86 for the parity reference, see docs/pipeline_guide.md).
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJ"

# 1) venv: python3.10 + system site-packages (torch 2.8.0+cu126 comes from the
#    NVIDIA JetPack/L4T stack, not from pip)
if [ ! -e .venv-orin ]; then python3 -m venv --system-site-packages .venv-orin; fi

V=$PROJ/.venv-orin/bin

# 2) offline install of python-only deps (no network, does not touch NVIDIA stack)
$V/pip install --no-index --find-links "$PROJ/wheelhouse" \
  transformers==4.53.2 sentencepiece protobuf numpy==1.26.4 pillow einops safetensors \
  websockets openpi-client dm-tree msgpack tyro

# 3) import gate: FlashRT + server deps
FLASHRT_ROOT=${FLASHRT_ROOT:-$PROJ/../FlashRT} \
  $V/python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
import websockets, msgpack, tyro
from openpi_client import msgpack_numpy
import flash_rt.frontends.torch.pi05_rtx as pr
print("flash_rt import OK:", pr is not None)
PY
echo SETUP_ORIN_DONE