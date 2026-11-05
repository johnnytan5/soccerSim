#!/usr/bin/env bash
# Usage: pixi run [-e gpu] bash scripts/run_match.sh [--viewer|--env-check] ROSTER_JSON [tree:=game.xml] [domain_base:=70] [enable_com:=true]
#   The match bridge (all robots + GameController) runs in the background, one brain_node per robot in the foreground.
#   Extra bridge options: MATCH_ARGS='--start full_auto --half-minutes 10 --ball-detector ground_truth'
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set +u
# shellcheck disable=SC1091
source "$ROOT/scripts/ros_env.sh"
# SS_WS_DIR: colon-separated extra workspaces overlaid on ros_ws (the per-team brain builds, ros_ws_brains/<name>).
WS_DIRS=("$ROOT/ros_ws")
if [ -n "${SS_WS_DIR:-}" ]; then
  IFS=: read -ra EXTRA_WS <<< "$SS_WS_DIR"
  for ws in "${EXTRA_WS[@]}"; do
    [ -n "$ws" ] && [ "$(cd "$ws" && pwd)" != "$ROOT/ros_ws" ] && WS_DIRS+=("$ws")
  done
fi
# Sources the overlays, then keeps the active pixi env (e.g. `-e gpu` for onnxruntime-gpu) ahead of the one they were built in.
ss_source_overlay "${WS_DIRS[@]}"
set -u
export SOCCER_SIM_ROOT="$ROOT"  # match.launch.py finds scripts/ns_exec.sh through it

PY=python
VIEWER_PY=python
if [ "$(uname -s)" = "Darwin" ]; then
  VIEWER_PY=mjpython
fi
VIEWER=""
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --viewer) PY="$VIEWER_PY"; VIEWER="--viewer" ;;
    --env-check)  # print the interpreter and onnxruntime providers a match would get, then exit (no match runs)
      "$PY" -c 'import sys, onnxruntime as o; print("python", sys.executable); print("onnxruntime", o.__file__, o.get_available_providers())'
      exit 0 ;;
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
# Polled rather than `wait -n`, which macOS's bash 3.2 does not have.
while kill -0 "$BRIDGE_PID" 2>/dev/null && kill -0 "$LAUNCH_PID" 2>/dev/null; do
  sleep 1
done
