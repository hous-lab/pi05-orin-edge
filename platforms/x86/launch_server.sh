#!/bin/bash
# x86 RTX (SM89+) server launcher - two configs:
#   a = FP8 W8A16-fp16 (RTX-native: FP8 E4M3 weights x FP16 activations)
#   b = the Orin INT8 recipe (configs/gumix_geg8.env) rebuilt for sm_89
# Usage: platforms/x86/launch_server.sh <a|b> [port]
#
# Env:
#   FLASHRT_ROOT    FlashRT tree (sm_89 build, flash_rt importable)  [required]
#   PI05_CKPT_DIR   bf16 bundle + norm_stats + tokenizer dir          [required]
#   PI05_MIX_SETS   K16 mix-set npz (config b only)                   [config b]
#   STATS_OUT       server-side stats dump path (default ./stats_server.json)
#   PI05_PY         python to use (default python3)
set -euo pipefail
CFG=${1:?usage: launch_server.sh <a|b> [port]}
PORT=${2:-8021}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
: "${FLASHRT_ROOT:?set FLASHRT_ROOT}"
: "${PI05_CKPT_DIR:?set PI05_CKPT_DIR}"

if [ "$CFG" = "b" ]; then
  : "${PI05_MIX_SETS:?config b needs PI05_MIX_SETS (tools/mix_sets_k16.npz)}"
  export PI05_MIX_SETS
  # shellcheck disable=SC1091
  source "$REPO/configs/gumix_geg8.env"
else
  # config a: FP8 is the frontend default; every INT8 gate explicitly off so the
  # server's FORCE_INT8 setdefault cannot re-arm the INT8 path.
  export FVK_PI05_RTX_FORCE_INT8=0
  export FVK_PI05_RTX_INT8_DECODER_ONLY=0
  export FVK_PI05_RTX_ENC8_ATTN=0
  export FVK_PI05_RTX_ENC8_MLP=0
  export FVK_PI05_RTX_ENC8_DOWN=0
  export FVK_PI05_RTX_ENC8_MIX=0
  export FVK_PI05_RTX_ENC8_MIX_GEG8=0
fi

export STATS_OUT=${STATS_OUT:-$PWD/stats_server.json}
exec ${PI05_PY:-python3} "$HERE/serve_with_stats.py" --views 2 --pool 1 --steps 1 --port "$PORT"
