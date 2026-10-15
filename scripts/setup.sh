#!/usr/bin/env bash
# One command from a fresh clone to a runnable simulator (Linux x86-64 incl. WSL2, or Apple Silicon macOS):
#   bash scripts/setup.sh
# Installs pixi if missing, pulls the Booster submodules, installs the ROS 2 / MuJoCo / PyTorch environment,
# downloads the YOLO ball model, the Booster SDK headers and the rerun C++ SDK, builds brain and the bridge
# from the Robotedge repo, and runs the unit tests. Safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64|Darwin-arm64) ;;
  *) echo "Unsupported platform $(uname -s)-$(uname -m): the pixi environment covers linux-64 and osx-arm64." >&2; exit 1 ;;
esac

missing=()
for tool in git curl rsync patch unzip; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing system tools: ${missing[*]}" >&2
  echo "Install them with your package manager, e.g. Debian/Ubuntu: sudo apt install ${missing[*]}" >&2
  echo "                                              Fedora:        sudo dnf install ${missing[*]}" >&2
  echo "                                              Arch:          sudo pacman -S ${missing[*]}" >&2
  exit 1
fi

if ! command -v pixi >/dev/null 2>&1; then
  echo "== installing pixi"
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi

echo "== Booster submodules"
git submodule update --init --recursive

echo "== environment (ROS 2, MuJoCo, PyTorch; a few GB the first time)"
pixi install

# The Robotedge repo is found by scripts/activate.sh: $ROBOTEDGE_ROOT, a sibling folder, or ~/.
ROBOTEDGE_ROOT="$(pixi run bash -c 'echo "$ROBOTEDGE_ROOT"' | tail -n 1)"
if [ ! -d "$ROBOTEDGE_ROOT/src/brain" ]; then
  echo "Robotedge repo not found at $ROBOTEDGE_ROOT." >&2
  echo "Clone Robotedge-5v5-RCAP2026 next to this folder, or export ROBOTEDGE_ROOT=/path/to/it, then re-run." >&2
  exit 1
fi
echo "== using Robotedge repo at $ROBOTEDGE_ROOT"

echo "== YOLO ball model"
pixi run fetch-models

echo "== build brain + bridge (first build also compiles the rerun C++ SDK; several minutes)"
pixi run build-sim

echo "== tests"
pixi run test

cat <<MSG

Ready. Open the viewer with:
  pixi run keeper-quick      # goalkeeper, robot starts playing
  pixi run keeper            # goalkeeper, full match procedure
  pixi run striker           # striker
then the controller tab at http://127.0.0.1:8765/
MSG
