#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGES="${PACKAGES:-brain}"
SRC="$ROOT/ros_ws/src"
# ROBOTEDGE_ROOT comes from scripts/activate.sh (run through `pixi run`).
: "${ROBOTEDGE_ROOT:?run through pixi: pixi run build-sim}"
if [ ! -d "$ROBOTEDGE_ROOT/src/brain" ]; then
  echo "Robotedge repo not found at $ROBOTEDGE_ROOT (no src/brain)." >&2
  echo "Clone Robotedge-5v5-RCAP2026 next to this repo, or export ROBOTEDGE_ROOT=/path/to/it." >&2
  exit 1
fi

# Message packages are used straight from the Robotedge working tree (read-only).
mkdir -p "$SRC"
[ -L "$SRC/robotedge" ] && rm "$SRC/robotedge"
mkdir -p "$SRC/robotedge"
for pkg in booster_msgs booster_ros2_interface robocup_ros2_interface; do
  ln -sfn "$ROBOTEDGE_ROOT/src/$pkg" "$SRC/robotedge/$pkg"
done

# brain is copied from the Robotedge working tree and patched for clang (behaviour-identical;
# see patches/brain-clang.patch). Re-copied on every build so local brain edits are always what runs.
rsync -a --delete --exclude build --exclude install --exclude log --exclude __pycache__ \
  "$ROBOTEDGE_ROOT/src/brain/" "$SRC/brain/"
# apply_patch NAME: apply patches/NAME.patch to the copy; "already in the tree" (an RCAP branch that adopted the
# change) counts as applied, because patch(1) exits 1 for it just as it does for a hunk that no longer fits.
apply_patch() {
  local patch_file="$ROOT/patches/$1.patch"
  if patch -p1 -d "$SRC/brain" --reverse --dry-run --silent < "$patch_file" >/dev/null 2>&1; then
    echo "patches/$1.patch is already part of $ROBOTEDGE_ROOT/src/brain; skipping it."
    return 0
  fi
  patch -p1 -d "$SRC/brain" --forward --silent < "$patch_file"
}

if ! apply_patch brain-clang; then
  echo "patches/brain-clang.patch no longer applies to $ROBOTEDGE_ROOT/src/brain; refresh it." >&2
  exit 1
fi

# Optional: publish brain's decision on /brain/decision so the viewer overlay never misses a change
# (brain's own /speak announcements are rate-limited). The overlay falls back to /speak without it.
if ! apply_patch brain-decision-topic; then
  echo "warning: patches/brain-decision-topic.patch no longer applies; the viewer overlay will use /speak instead." >&2
  rm -f "$SRC"/brain/src/brain.cpp.rej "$SRC"/brain/src/brain.cpp.orig
fi

# The build only needs the Booster SDK headers. Use $SDK_ZIP (a sdk_release.zip) if you set one,
# otherwise take them from Booster's public SDK repo.
if [ ! -d "$ROOT/vendor/booster_sdk/include/booster" ]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  if [ -n "${SDK_ZIP:-}" ] && [ -f "$SDK_ZIP" ]; then
    echo "Booster SDK headers from $SDK_ZIP"
    unzip -q "$SDK_ZIP" -d "$tmp"
    sdk_include="$tmp/sdk_release/include"
  else
    echo "Downloading Booster SDK headers (github.com/BoosterRobotics/booster_robotics_sdk)"
    git clone --depth 1 --quiet https://github.com/BoosterRobotics/booster_robotics_sdk.git "$tmp/sdk"
    sdk_include="$tmp/sdk/include"
  fi
  [ -d "$sdk_include/booster" ] || { echo "no include/booster in the Booster SDK" >&2; exit 1; }
  mkdir -p "$ROOT/vendor/booster_sdk"
  cp -R "$sdk_include" "$ROOT/vendor/booster_sdk/include"
fi

bash "$ROOT/scripts/build_rerun.sh"

# ament links every message package's *__rosidl_generator_py dylib into brain_node. Linux never
# loads their Python symbols, but macOS binds them at startup and aborts, so ask the Mach-O linker
# to drop them. The flag is only meaningful there.
EXE_LINKER_FLAGS=""
if [ "$(uname -s)" = "Darwin" ]; then
  EXE_LINKER_FLAGS="-DCMAKE_EXE_LINKER_FLAGS=-Wl,-dead_strip_dylibs"
fi

cd "$ROOT/ros_ws"
# shellcheck disable=SC2086
colcon build --symlink-install \
  --packages-up-to $PACKAGES \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  "-DCMAKE_CXX_FLAGS=-I$ROOT/vendor/booster_sdk/include" \
  "-Drerun_sdk_DIR=$ROOT/vendor/rerun_sdk/lib/cmake/rerun_sdk" \
  ${EXE_LINKER_FLAGS:+"$EXE_LINKER_FLAGS"}
