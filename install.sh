#!/usr/bin/env bash
# Copy this project's scripts and captions into a MotionCLIP checkout.
# Usage: bash install.sh /path/to/MotionCLIP
set -euo pipefail

TARGET="${1:?usage: bash install.sh /path/to/MotionCLIP}"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -d "$TARGET/src" ] || { echo "error: $TARGET does not look like a MotionCLIP checkout"; exit 1; }

for d in patches data training evaluation visualization; do
  cp "$HERE/$d"/*.py "$HERE/$d"/*.sh "$TARGET"/ 2>/dev/null || true
done
cp "$HERE"/data/captions/captions_*.json "$TARGET"/

echo "Copied into $TARGET. Next:"
echo "  cd $TARGET && python patch_mclip.py && python patch_ml.py && python patch_vae.py"
