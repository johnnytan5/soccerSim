#!/usr/bin/env bash
# Usage: ns_exec.sh ss-rN CMD [ARGS...]
# Run CMD inside network namespace ss-rN as the *calling user* with the *whole* environment (pixi, ROS, LD_LIBRARY_PATH,
# ROS_DOMAIN_ID, ...). sudo's env_reset would drop most of it, so the environment is handed to `env -i` (as root, in the
# namespace) and setpriv then drops back to this user. Used as the launch prefix of each brain by match.launch.py.
set -euo pipefail
ns="$1"; shift
mapfile -d '' vars < <(env -0)
exec sudo -n ip netns exec "$ns" env -i "${vars[@]}" \
  setpriv --reuid="$(id -u)" --regid="$(id -g)" --init-groups --inh-caps=-all -- "$@"
