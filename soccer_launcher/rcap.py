"""Locate and inspect the rcap (Robotedge-5v5-RCAP2026) clone. Read-only: never fetches or checks out."""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .store import project_root

REPO_NAME = "Robotedge-5v5-RCAP2026"
# What scripts/setup_ws.sh consumes from the rcap tree.
_RCAP_INPUTS = ("src/brain", "src/booster_msgs", "src/booster_ros2_interface", "src/robocup_ros2_interface")
# Build-affecting files that live in soccerSim itself (relative to its root).
_LOCAL_INPUTS = ("patches", "scripts/setup_ws.sh", "scripts/build_rerun.sh")
_SKIP_DIRS = {"build", "install", "log", "__pycache__", ".git"}


def locate(override: str = "", env: dict[str, str] | None = None) -> Path:
    """Same order as scripts/activate.sh: explicit setting, $ROBOTEDGE_ROOT, sibling clone, ~/<repo>."""
    env = os.environ if env is None else env
    if override.strip():
        return Path(override).expanduser()
    if env.get("ROBOTEDGE_ROOT"):
        return Path(env["ROBOTEDGE_ROOT"]).expanduser()
    sibling = project_root().parent / REPO_NAME
    if (sibling / "src" / "brain").is_dir():
        return sibling
    return Path.home() / REPO_NAME


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


@dataclass(frozen=True)
class RcapInfo:
    root: Path
    valid: bool  # has src/brain
    branch: str = ""
    sha: str = ""
    subject: str = ""
    dirty: bool = False
    detached: bool = False


def inspect(root: Path) -> RcapInfo:
    if not (root / "src" / "brain").is_dir():
        return RcapInfo(root, False)
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") or ""
    return RcapInfo(
        root=root,
        valid=True,
        branch=branch if branch != "HEAD" else "",
        detached=branch == "HEAD",
        sha=_git(root, "rev-parse", "--short", "HEAD") or "",
        subject=_git(root, "log", "-1", "--format=%s") or "",
        dirty=bool(_git(root, "status", "--porcelain", "--", "src")),
    )


def _walk(base: Path):
    if base.is_file():
        yield base
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def fingerprint(root: Path, local_root: Path | None = None) -> str:
    """sha256 over the paths + contents of everything the build reads (rcap inputs and local patches/scripts)."""
    local_root = local_root or project_root()
    h = hashlib.sha256()
    for base_root, rels in ((root, _RCAP_INPUTS), (local_root, _LOCAL_INPUTS)):
        for rel in rels:
            base = base_root / rel
            if not base.exists():
                continue
            for f in _walk(base):
                try:
                    data = f.read_bytes()
                except OSError:
                    continue
                h.update(f.relative_to(base_root).as_posix().encode())
                h.update(b"\0")
                h.update(hashlib.sha256(data).digest())
    return h.hexdigest()
