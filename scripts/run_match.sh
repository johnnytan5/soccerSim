#!/usr/bin/env bash
# Usage: pixi run bash scripts/run_match.sh [--viewer] ROSTER_JSON [tree:=game.xml] [domain_base:=70] [enable_com:=true]
#   The match bridge (all robots + GameController) runs in the background, one brain_node per robot in the foreground.
#   Extra bridge options: MATCH_ARGS='--start full_auto --half-minutes 10 --ball-detector ground_truth'
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set +u
# shellcheck disable=SC1091
source "$ROOT/ros_ws/install/setup.bash"
set -u

PY=python
VIEWER_PY=python
if [ "$(uname -s)" = "Darwin" ]; then
  VIEWER_PY=mjpython
fi
VIEWER=""
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --viewer) PY="$VIEWER_PY"; VIEWER="--viewer" ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
ROSTER="$1"; shift
DOMAIN_BASE=70
for a in "$@"; do
  case "$a" in domain_base:=*) DOMAIN_BASE="${a#domain_base:=}" ;; esac
done

# shellcheck disable=SC2086
"$PY" -m soccer_sim_bridge.match_node $VIEWER --roster "$ROSTER" --domain-base "$DOMAIN_BASE" ${MATCH_ARGS:-} &
BRIDGE_PID=$!
echo "controller tab: http://127.0.0.1:8765/"
ros2 launch soccer_sim_bridge match.launch.py roster:="$ROSTER" "$@" &
LAUNCH_PID=$!
trap 'kill $BRIDGE_PID $LAUNCH_PID 2>/dev/null || true' EXIT
# The match is over when either side is: brains without a bridge (or a bridge without brains) are useless.
wait -n $BRIDGE_PID $LAUNCH_PID || true
