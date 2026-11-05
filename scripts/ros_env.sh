# Sourced by run_match.sh / run_sim.sh (bash). ss_source_overlay WS_DIR...: source each colcon workspace's
# install/setup.bash, then put the active pixi environment ($CONDA_PREFIX) back in front.
#
# colcon bakes the environment a workspace was built in into install/setup.bash (COLCON_CURRENT_PREFIX=
# ".../.pixi/envs/default") and chain-sources it, which prepends that env's bin/ to PATH and its site-packages to
# PYTHONPATH. Under `pixi run -e gpu` (onnxruntime-gpu) that silently swapped python, and with it onnxruntime, back to
# the default env, so YOLO ran on the CPU. All pixi envs carry the same ROS packages from one lock file, so the build
# env's paths are rewritten onto the active one instead of rebuilding the workspace per env.

ss_underlays() {  # ss_underlays WS_DIR: the pixi env prefixes WS_DIR/install/setup.bash chain-sources
  sed -n 's/^COLCON_CURRENT_PREFIX="\(.*\)"$/\1/p' "$1/install/setup.bash" 2>/dev/null | grep '/\.pixi/envs/[^/]*$' || true
}

ss_rebase_env() {  # ss_rebase_env OLD NEW: rewrite OLD (and OLD/...) to NEW in the path-list variables, drop duplicates
  local old="${1%/}" new="${2%/}" var p out
  local -a parts
  [ -n "$old" ] && [ -n "$new" ] && [ "$old" != "$new" ] || return 0
  for var in PATH PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PKG_CONFIG_PATH; do
    [ -n "${!var-}" ] || continue
    IFS=: read -ra parts <<< "${!var}"
    out=""
    for p in "${parts[@]}"; do  # no associative arrays: macOS ships bash 3.2
      case "$p" in "$old" | "$old"/*) p="$new${p#"$old"}" ;; esac
      [ -n "$p" ] || continue
      case ":$out:" in *":$p:"*) continue ;; esac
      out="${out:+$out:}$p"
    done
    export "$var=$out"
  done
}

ss_source_overlay() {  # ss_source_overlay WS_DIR...: source the workspaces in order, then rebase onto $CONDA_PREFIX
  local ws u prefix="${CONDA_PREFIX:-}"  # captured before the setup scripts run
  for ws in "$@"; do
    # shellcheck disable=SC1091
    source "$ws/install/setup.bash"
  done
  [ -n "$prefix" ] || return 0
  for ws in "$@"; do
    for u in $(ss_underlays "$ws"); do
      ss_rebase_env "$u" "$prefix"
    done
  done
}
