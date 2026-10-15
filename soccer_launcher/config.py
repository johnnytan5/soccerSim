"""Launch configuration: the one object the form edits, the store saves and command.py turns into a command."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields

ROLES = ("goal_keeper", "striker", "defender")
TREES = ("game.xml", "chase.xml", "demo.xml")
DETECTORS = ("yolo", "ground_truth")
CAMERA_GL = ("auto", "software", "gpu")
MUJOCO_GL = ("", "egl", "glfw", "osmesa")
MATCH_MODES = ("quick", "full")
GOALKEEPER_ID = 1  # brain forces this player id to be the keeper whatever `role` says

_SCHEDULE_RE = re.compile(r"^\d:\d+(\.\d+)?(,\d:\d+(\.\d+)?)*$")


@dataclass
class LaunchConfig:
    # Role & match
    role: str = "goal_keeper"
    player_id: int = 1
    tree: str = "game.xml"
    match_mode: str = "quick"  # quick: scripted referee, soccer mode; full: real match procedure
    viewer: bool = True
    # Perception
    ball_detector: str = "yolo"
    # Field & referee (empty/None = bridge default)
    spawn_x: float | None = None
    spawn_y: float | None = None
    spawn_yaw: float | None = None
    ball_x: float | None = None
    ball_y: float | None = None
    referee_schedule: str = ""
    # Advanced
    rcap_root: str = ""  # "" = auto-detect (same order as scripts/activate.sh)
    sim_vy_max: float | None = None
    camera_gl: str = "auto"
    mujoco_gl: str = ""
    ros_domain_id: int | None = None
    extra_bridge_args: str = ""
    extra_launch_args: str = ""

    def validate(self) -> list[str]:
        """Human-readable problems; an empty list means launchable."""
        errs: list[str] = []
        if self.role not in ROLES:
            errs.append(f"role must be one of {', '.join(ROLES)}")
        if self.tree not in TREES:
            errs.append(f"tree must be one of {', '.join(TREES)}")
        if self.match_mode not in MATCH_MODES:
            errs.append("match mode must be quick or full")
        if self.ball_detector not in DETECTORS:
            errs.append(f"ball detector must be one of {', '.join(DETECTORS)}")
        if self.camera_gl not in CAMERA_GL:
            errs.append(f"camera GL must be one of {', '.join(CAMERA_GL)}")
        if not 1 <= self.player_id <= 5:
            errs.append("player id must be 1-5")
        elif self.player_id == GOALKEEPER_ID and self.role != "goal_keeper":
            errs.append("player 1 is always the keeper; use player id 2-5 for a striker/defender")
        elif self.player_id != GOALKEEPER_ID and self.role == "goal_keeper":
            errs.append("a goal_keeper must be player id 1")
        if self.ros_domain_id is not None and not 0 <= self.ros_domain_id <= 232:
            errs.append("ROS_DOMAIN_ID must be 0-232")
        if self.sim_vy_max is not None and not self.sim_vy_max > 0:
            errs.append("SIM_VY_MAX must be positive")
        if self.referee_schedule and not _SCHEDULE_RE.match(self.referee_schedule):
            errs.append("referee schedule looks like 0:3,1:20,2:3,3:0 (state:seconds pairs)")
        for name in ("spawn_x", "spawn_y", "spawn_yaw", "ball_x", "ball_y", "sim_vy_max"):
            v = getattr(self, name)
            if v is not None and not math.isfinite(v):
                errs.append(f"{name} must be a finite number")
        return errs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LaunchConfig":
        """Tolerant: unknown keys are ignored, wrong-typed values fall back to the default."""
        base = cls()
        known = {f.name: f for f in fields(cls)}
        for key, value in (data or {}).items():
            f = known.get(key)
            if f is None:
                continue
            default = getattr(base, key)
            try:
                if isinstance(default, bool):
                    value = bool(value) if isinstance(value, bool) else default
                elif isinstance(default, int):
                    value = int(value)
                elif isinstance(default, str):
                    value = str(value)
                elif default is None:  # optional number
                    value = None if value in (None, "") else float(value)
                    if key == "ros_domain_id" and value is not None:
                        value = int(value)
            except (TypeError, ValueError):
                continue
            setattr(base, key, value)
        return base


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    config: LaunchConfig | None = None  # None => not launchable yet
    blocked_reason: str = ""

    @property
    def available(self) -> bool:
        return self.config is not None


_NEEDS_MULTI = "needs the multi-robot sim (today's sim has exactly one robot)"

PRESETS: tuple[Preset, ...] = (
    Preset("Lone keeper", "Goalkeeper, real match procedure + viewer (pixi run keeper)",
           LaunchConfig(role="goal_keeper", player_id=1, match_mode="full")),
    Preset("Lone keeper (quick)", "Goalkeeper, scripted referee, viewer (pixi run keeper-quick)",
           LaunchConfig(role="goal_keeper", player_id=1, match_mode="quick")),
    Preset("Lone striker (quick)", "Striker as player 2, scripted referee, viewer (pixi run striker)",
           LaunchConfig(role="striker", player_id=2, match_mode="quick")),
    Preset("Lone striker (full)", "Striker as player 2, real match procedure, viewer",
           LaunchConfig(role="striker", player_id=2, match_mode="full")),
    Preset("Keeper, ground-truth ball", "Keeper quick with perfect ball detection (debug the brain, not YOLO)",
           LaunchConfig(role="goal_keeper", player_id=1, match_mode="quick", ball_detector="ground_truth")),
    Preset("Team vs team (5v5)", "2 teams of 5", None, _NEEDS_MULTI),
    Preset("Goalie vs 1 striker", "1 goalie against 1 opponent striker", None, _NEEDS_MULTI),
    Preset("Goalie vs 2 strikers", "1 goalie against 2 opponent strikers", None, _NEEDS_MULTI),
    Preset("Goalies only", "Both teams' goalies only", None, _NEEDS_MULTI),
)
