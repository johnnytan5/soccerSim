"""Per-team brain builds: one Robotedge folder per team, each built into its own workspace ros_ws_brains/<name>."""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

from . import build
from .config import LaunchConfig
from .store import Store, project_root

TEAMS = ("home", "away")
WS_DIR = "ros_ws_brains"


def sanitize(folder: str | Path) -> str:
    """Workspace directory name of a folder: its readable basename plus a short hash of the full path, so two
    folders called 'Robotedge' in different places never share a build."""
    path = os.path.abspath(os.path.expanduser(str(folder)))
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(path.rstrip("/"))).strip("._") or "brain"
    return f"{base}-{hashlib.sha1(path.encode()).hexdigest()[:8]}"


def workspace(folder: str | Path, local_root: Path | None = None) -> Path:
    return (local_root or project_root()) / WS_DIR / sanitize(folder)


def install_dir(folder: str | Path, local_root: Path | None = None) -> Path:
    return workspace(folder, local_root) / "install"


def team_folders(cfg: LaunchConfig) -> dict[str, str]:
    """{team: absolute folder} for the teams that have their own folder set. Only matches use them."""
    if not cfg.is_match:
        return {}
    out = {}
    for team in TEAMS:
        v = getattr(cfg, f"{team}_brain_root").strip()
        if v:
            out[team] = os.path.abspath(os.path.expanduser(v))
    return out


def build_env(folder: str | Path, local_root: Path | None = None) -> dict[str, str]:
    """Environment for scripts/setup_ws.sh to build that folder's brain into its own workspace."""
    return {"SS_WS_DIR": str(workspace(folder, local_root)), "ROBOTEDGE_ROOT": str(folder)}


def check(folder: str | Path, local_root: Path | None = None) -> build.BuildStatus:
    return build.check(Path(folder), None, local_root, ws=workspace(folder, local_root))


def main(argv: list[str] | None = None) -> int:
    """pixi run build-brains [FOLDER ...]: build each folder's brain if stale (no folders: those in the last saved
    launcher config)."""
    import subprocess

    folders = list(sys.argv[1:] if argv is None else argv)
    root = project_root()
    if not folders:
        store = Store()
        store.load()
        folders = list(dict.fromkeys(team_folders(store.last or LaunchConfig()).values()))
    if not folders:
        print("no team brain folders: pass them as arguments or set them in the launcher")
        return 0
    for folder in map(os.path.abspath, map(os.path.expanduser, folders)):
        status = check(folder, root)
        if not status.stale:
            print(f"{folder}: up to date ({workspace(folder, root)})")
            continue
        print(f"{folder}: building ({status.reason}) into {workspace(folder, root)}")
        code = subprocess.run(build.BUILD_ARGV, cwd=root, env={**os.environ, **build_env(folder, root)}).returncode
        if code != 0:
            return code
        build.write_stamp(Path(folder), status.fingerprint, None, workspace(folder, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
