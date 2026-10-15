#!/usr/bin/env bash
# Booster's sim-trained YOLOv8 detector from the public robocup_demo repo (Git LFS object).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor/models"
FILE="$DEST/sim_data_det_0126.onnx"
URL="https://media.githubusercontent.com/media/BoosterRobotics/robocup_demo/main/src/vision/model/sim_data_det_0126.onnx"
SHA256="4410f04175bb3a7bc79ed7cfce2a5ab0e9c695be8cb488fbfd9d5619dc492005"

# check a line of "<hash>  <file>" the same way on macOS (shasum) and Linux (sha256sum)
if command -v shasum >/dev/null 2>&1; then
  SHA_CMD=(shasum -a 256)
else
  SHA_CMD=(sha256sum)
fi

mkdir -p "$DEST"
if [ -f "$FILE" ] && echo "$SHA256  $FILE" | "${SHA_CMD[@]}" -c --status; then
  echo "model present: $FILE"
  exit 0
fi
curl -fsSL -o "$FILE.tmp" "$URL"
echo "$SHA256  $FILE.tmp" | "${SHA_CMD[@]}" -c -
mv "$FILE.tmp" "$FILE"
echo "fetched: $FILE"
