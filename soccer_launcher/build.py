"""Staleness check for ros_ws against the rcap tree. The build itself is run by runner.py (streamed)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import rcap
from .store import project_root

BUILD_ENV = {"PACKAGES": "brain soccer_sim_bridge"}  # identical to the `build-sim` pixi task
BUILD_ARGV = ["bash", "scripts/setup_ws.sh"]


@dataclass(frozen=True)
class BuildStatus:
    stale: bool
    reason: str  # "" when up to date
    fingerprint: str


def _stamp_path(launcher_dir: Path | None) -> Path:
    return (launcher_dir or project_root() / ".launcher") / "build_stamp.json"


def check(rcap_root: Path, launcher_dir: Path | None = None, local_root: Path | None = None) -> BuildStatus:
    local_root = local_root or project_root()
    fp = rcap.fingerprint(rcap_root, local_root)
    if not (local_root / "ros_ws" / "install" / "setup.bash").is_file():
        return BuildStatus(True, "not built yet", fp)
    try:
        stamp = json.loads(_stamp_path(launcher_dir).read_text())
    except (OSError, ValueError):
        return BuildStatus(True, "no record of the last build", fp)
    if stamp.get("rcap_root") != str(rcap_root):
        return BuildStatus(True, "built from a different rcap clone", fp)
    if stamp.get("fingerprint") != fp:
        return BuildStatus(True, "rcap code changed since the last build", fp)
    return BuildStatus(False, "", fp)


def write_stamp(rcap_root: Path, fingerprint: str, launcher_dir: Path | None = None) -> None:
    path = _stamp_path(launcher_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"rcap_root": str(rcap_root), "fingerprint": fingerprint}))
    os.replace(tmp, path)
