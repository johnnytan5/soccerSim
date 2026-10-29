"""Checks shown before launch. Each is (level, text); level is "ok", "warn" or "fail" (fail blocks launch)."""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path

from . import settings
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
    registry = settings.registry_for(info.root if info.valid else None)
    for problem in cfg.validate(registry):
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
    elif cfg.simulator != "2d":
        out.append(Check("warn", f"missing {MODEL} (not needed for ground_truth)"))
    out.append(Check("warn", "build is stale: it will be rebuilt first") if stale else Check("ok", "build is up to date"))
    mujoco_viewer = cfg.simulator == "mujoco" and cfg.viewer
    if cfg.simulator == "2d":
        out.append(Check("ok", "2D simulator: robots on the kinematic simulator, shown on the browser map "
                               "(no MuJoCo, no viewer window)"))
    if info.valid and not settings.load_schema(info.root):
        out.append(Check("ok", "this rcap checkout has no world_model schema, so no world-model settings are offered"))
    for key in settings.unknown_keys(cfg.settings, registry):
        out.append(Check("warn", f"setting {key} is not known to this rcap checkout and will be ignored"))
    if cfg.settings and not cfg.is_match:
        out.append(Check("warn", "settings only reach brains in a match; the single-robot launch ignores them"))
    if mujoco_viewer and platform == "Linux" and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")) \
            and not Path("/tmp/.X11-unix/X0").exists():
        out.append(Check("warn", "viewer is on but no display found (DISPLAY/WSLg)"))
    if mujoco_viewer and cfg.graphics == "cpu":
        if platform == "Darwin":
            out.append(Check("warn", "graphics: CPU only applies on Linux; the viewer on macOS always uses the system's graphics"))
        else:
            out.append(Check("warn", "the viewer is drawn by the CPU (software rendering) and will be slow with several robots; "
                                     "GPU is the default"))
    if cfg.is_match:
        out += _match_checks(cfg, platform)
    elif cfg.ros_domain_id is None and not env.get("ROS_DOMAIN_ID"):
        out.append(Check("warn", "default ROS domain: two sims at once would cross-talk (set a ROS_DOMAIN_ID)"))
    return out


def _has_default_route(route_file: str = "/proc/net/route") -> bool:
    try:
        with open(route_file) as f:
            return any(line.split()[1] == "00000000" for line in list(f)[1:] if len(line.split()) > 1)
    except OSError:
        return True  # cannot tell (not Linux): do not nag


def _match_checks(cfg: LaunchConfig, platform: str) -> list[Check]:
    n = len(cfg.players)
    out = [Check("ok", f"match: {sum(p.get('team') == 'home' for p in cfg.players)} home, "
                       f"{sum(p.get('team') == 'away' for p in cfg.players)} away ({n} robot{'s' if n != 1 else ''}, one ROS domain each)")]
    if cfg.ball_detector == "yolo" and n > 1:
        total = 25 if n <= 5 else 12.5  # camera frames the sim can render per second while holding 50 Hz
        out.append(Check("warn", f"YOLO with {n} robots: the sim can render about {total:g} camera frames a second in "
                                 f"total, so each robot's camera updates only {total / n:.1f} times a second. "
                                 "Ground truth ball gives every robot full-rate perception"))
    if cfg.detailed_bodies and cfg.viewer and cfg.simulator == "mujoco" and n > 2:
        out.append(Check("warn", f"detailed robot bodies: {n} robots are about {n} million triangles, so the viewer "
                                 "will trail the sim. The simple team-coloured bodies stay smooth"))
    if platform == "Linux" and not _has_default_route():
        out.append(Check("warn", "no network route: teammates find each other over UDP and may not connect "
                                 "(connect to any network, even without internet)"))
    return out


def blocking(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.level == "fail"]
