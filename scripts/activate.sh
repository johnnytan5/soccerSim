# Sourced by pixi on every `pixi run` / `pixi shell` (see [activation] in pixi.toml). POSIX sh.

# Robotedge repo: $ROBOTEDGE_ROOT if set, else a sibling folder of this repo, else ~/Robotedge-5v5-RCAP2026.
if [ -z "${ROBOTEDGE_ROOT:-}" ]; then
  sibling="$(cd "$PIXI_PROJECT_ROOT/.." && pwd)/Robotedge-5v5-RCAP2026"
  if [ -d "$sibling/src/brain" ]; then
    ROBOTEDGE_ROOT="$sibling"
  else
    ROBOTEDGE_ROOT="$HOME/Robotedge-5v5-RCAP2026"
  fi
fi
export ROBOTEDGE_ROOT

if [ "$(uname -s)" = "Linux" ]; then
  # MuJoCo has no default GL backend on Linux. The head camera renders offscreen, so use EGL
  # (works without a display). Export MUJOCO_GL yourself to override.
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  # WSLg runs an X server on :0, but only interactive WSL shells have DISPLAY set.
  if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && grep -qi microsoft /proc/version 2>/dev/null && [ -S /tmp/.X11-unix/X0 ]; then
    export DISPLAY=:0
  fi
fi
