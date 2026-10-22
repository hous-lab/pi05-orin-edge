#!/usr/bin/env bash
# Create a FlashRT working tree at the pinned revision and apply this repo's overlay.
# Usage: ./flashrt_overlay/apply_overlay.sh [TARGET_DIR]
#   TARGET_DIR defaults to ./FlashRT (next to this repo checkout).
# Build instructions are in docs/pipeline_guide.md / run_pipeline.sh, not here.
set -euo pipefail

REV="f2f04db8"
CUTLASS_TAG="v4.4.2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$REPO_ROOT/../FlashRT}"

if [ -d "$TARGET/flash_rt" ]; then
  echo "error: $TARGET already contains a flash_rt/ tree - refusing to touch it" >&2
  exit 1
fi

echo "== cloning FlashRT @ $REV into $TARGET"
git clone https://github.com/flashrt-project/FlashRT.git "$TARGET"
git -C "$TARGET" checkout "$REV"

echo "== cloning CUTLASS $CUTLASS_TAG (upstream CMake expects third_party/cutlass)"
git clone --depth 1 --branch "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass.git "$TARGET/third_party/cutlass"

echo "== applying overlay ($(find "$REPO_ROOT/flashrt_overlay/tree" -type f | wc -l) files)"
cp -r "$REPO_ROOT/flashrt_overlay/tree/." "$TARGET/"
echo "done. next: build per docs/pipeline_guide.md (cmake -B build ...), then run the parity gate."
