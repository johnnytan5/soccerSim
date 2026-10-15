#!/usr/bin/env bash
# Usage: pixi run bash scripts/run_sim.sh [--viewer] [--quick] [tree:=game.xml] [role:=striker]
#   default : real match procedure — robot boots in prepare mode (press "Stand up (LT+Up)" in the
#             controller tab), stands at the sideline, referee holds INITIAL until you press READY.
#   --quick : robot already in soccer mode, scripted referee INITIAL 3 s -> READY 20 s -> SET 3 s -> PLAYING.
# Extra bridge parameters: BRIDGE_ARGS='-p ball_detector:=ground_truth' pixi run bash scripts/run_sim.sh --viewer
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# colcon's setup.bash reads unset variables (COLCON_TRACE), so relax -u while sourcing it.
set +u
# shellcheck disable=SC1091
source "$ROOT/ros_ws/install/setup.bash"
set -u

PY=python
# MuJoCo's passive viewer needs its mjpython launcher on macOS; plain python works on Linux.
VIEWER_PY=python
if [ "$(uname -s)" = "Darwin" ]; then
  VIEWER_PY=mjpython
fi
VIEWER=""
MATCH_ARGS="-p boot_sequence:=true -p referee_schedule:=0:0"
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --viewer) PY="$VIEWER_PY"; VIEWER="--viewer" ;;
    --quick) MATCH_ARGS="" ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

# shellcheck disable=SC2086
"$PY" -m soccer_sim_bridge.bridge_node $VIEWER --ros-args $MATCH_ARGS ${BRIDGE_ARGS:-} &
BRIDGE_PID=$!
trap 'kill $BRIDGE_PID 2>/dev/null || true' EXIT
echo "controller tab: http://127.0.0.1:8765/"
ros2 launch soccer_sim_bridge brain.launch.py "$@"
