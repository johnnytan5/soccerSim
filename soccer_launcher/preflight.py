"""Checks shown before launch. Each is (level, text); level is "ok", "warn" or "fail" (fail blocks launch)."""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path

from .config import LaunchConfig
from .rcap import RcapInfo
from .store import project_root

MODEL = Path("vendor/models/sim_data_det_0126.onnx")


@dataclass(frozen=True)
class Check:
    level: str
    text: str
    fix: str = ""  # Tools task that fixes it, e.g. "fetch-models"


def port_free(port: int) -> bool:
    if port <= 0:
        return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def run(cfg: LaunchConfig, info: RcapInfo, stale: bool, root: Path | None = None,
        env: dict[str, str] | None = None, platform: str | None = None) -> list[Check]:
    root = root or project_root()
    env = os.environ if env is None else env
    platform = platform or os.uname().sysname
    out: list[Check] = []
    for problem in cfg.validate():
        out.append(Check("fail", problem))
    if info.valid:
        where = info.branch or ("detached " + info.sha)
        out.append(Check("ok", f"rcap: {info.root} ({where}{' +local changes' if info.dirty else ''})"))
    else:
        out.append(Check("fail", f"rcap clone not found at {info.root} (no src/brain); set the path in Advanced"))
    if (root / MODEL).is_file():
        out.append(Check("ok", "YOLO ball model present"))
    elif cfg.ball_detector == "yolo":
        out.append(Check("fail", f"missing {MODEL}", fix="fetch-models"))
    else:
        out.append(Check("warn", f"missing {MODEL} (not needed for ground_truth)"))
    out.append(Check("warn", "build is stale: it will be rebuilt first") if stale else Check("ok", "build is up to date"))
    if cfg.viewer and platform == "Linux" and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")) \
            and not Path("/tmp/.X11-unix/X0").exists():
        out.append(Check("warn", "viewer is on but no display found (DISPLAY/WSLg)"))
    if cfg.ros_domain_id is None and not env.get("ROS_DOMAIN_ID"):
        out.append(Check("warn", "default ROS domain: two sims at once would cross-talk (set a ROS_DOMAIN_ID)"))
    return out


def blocking(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.level == "fail"]
