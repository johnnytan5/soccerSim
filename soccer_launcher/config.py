"""Launch configuration: the one object the form edits, the store saves and command.py turns into a command."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields

from soccer_sim.roster import AWAY, HOME, PlayerSpec, standard_team, validate_roster

from . import settings as setting_registry

ROLES = ("goal_keeper", "striker", "defender")
TREES = ("game.xml", "chase.xml", "demo.xml")
DETECTORS = ("yolo", "ground_truth")
CAMERA_GL = ("auto", "software", "gpu")
GRAPHICS = ("gpu", "cpu")  # what draws the MuJoCo viewer window
MUJOCO_GL = ("", "egl", "glfw", "osmesa")
MATCH_MODES = ("quick", "full")
SIMULATORS = ("2d", "mujoco")  # 2d: browser map on the kinematic simulator; mujoco: MuJoCo physics (+ viewer window)
GOALKEEPER_ID = 1  # brain forces this player id to be the keeper whatever `role` says
DEFAULT_DOMAIN_BASE = 70  # match runs use one ROS domain per robot, counting up from here
MAX_ROBOTS = 10
MAX_PACKET_BUDGET = 65535  # the GameController's TeamInfo.message_budget is a uint16

_SCHEDULE_RE = re.compile(r"^\d:\d+(\.\d+)?(,\d:\d+(\.\d+)?)*$")


@dataclass
class LaunchConfig:
    # Role & match
    role: str = "goal_keeper"
    player_id: int = 1
    tree: str = "game.xml"
    match_mode: str = "quick"  # quick: scripted referee, soccer mode; full: real match procedure
    simulator: str = "mujoco"
    viewer: bool = True  # mujoco only: open the 3D viewer window
    graphics: str = "gpu"  # viewer window: your graphics card, or "cpu" = Mesa software rendering (Linux only)
    # Match: a roster of {team: home|away, player_id, role} plays a whole game (empty = the single robot above)
    players: list = field(default_factory=list)
    half_minutes: float = 10.0
    continuous: bool = False  # no halves or clock limit
    detailed_bodies: bool = False  # viewer: real T2 meshes instead of simple team-coloured bodies (slow with many robots)
    packet_budget: int = 60000  # UDP packets each team may send per match, as in RCAP (0 = no limit); over it the score is zeroed
    # Perception
    ball_detector: str = "yolo"
    # Field & referee (empty/None = bridge default)
    spawn_x: float | None = None
    spawn_y: float | None = None
    spawn_yaw: float | None = None
    ball_x: float | None = None
    ball_y: float | None = None
    referee_schedule: str = ""
    # Tunable settings from soccer_launcher.settings: registry key -> value (absent = keep config.yaml's value)
    settings: dict = field(default_factory=dict)
    # Advanced
    rcap_root: str = ""  # "" = auto-detect (same order as scripts/activate.sh)
    sim_vy_max: float | None = None
    camera_gl: str = "auto"
    mujoco_gl: str = ""
    ros_domain_id: int | None = None
    extra_bridge_args: str = ""
    extra_launch_args: str = ""

    @property
    def is_match(self) -> bool:
        return bool(self.players)

    def roster(self) -> list[PlayerSpec]:
        return [PlayerSpec.from_dict(p) for p in self.players if isinstance(p, dict)]

    def validate(self, registry=None) -> list[str]:
        """Human-readable problems; an empty list means launchable. `registry`: the settings of the RCAP checkout
        being launched (soccer_launcher.settings.registry_for); defaults to the static ones."""
        errs: list[str] = []
        if self.simulator not in SIMULATORS:
            errs.append("simulator must be 2d or mujoco")
        elif self.simulator == "2d":
            if not self.is_match:
                errs.append("the 2D simulator runs matches: add players (single-robot presets use MuJoCo)")
            if self.ball_detector != "ground_truth":
                errs.append("the 2D simulator has no camera: use the ground-truth ball detector")
        errs += setting_registry.validate(self.settings, registry if registry is not None else list(setting_registry.STATIC))
        if self.is_match:
            errs += self._validate_match()
        elif self.role not in ROLES:
            errs.append(f"role must be one of {', '.join(ROLES)}")
        if self.tree not in TREES:
            errs.append(f"tree must be one of {', '.join(TREES)}")
        if self.match_mode not in MATCH_MODES:
            errs.append("match mode must be quick or full")
        if self.ball_detector not in DETECTORS:
            errs.append(f"ball detector must be one of {', '.join(DETECTORS)}")
        if self.graphics not in GRAPHICS:
            errs.append("graphics must be gpu or cpu")
        if self.camera_gl not in CAMERA_GL:
            errs.append(f"camera GL must be one of {', '.join(CAMERA_GL)}")
        if self.is_match:
            pass
        elif not 1 <= self.player_id <= 5:
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

    def _validate_match(self) -> list[str]:
        errs: list[str] = []
        if not all(isinstance(p, dict) for p in self.players):
            return ["players must be a list of {team, player_id, role}"]
        errs += validate_roster(self.roster())
        if len(self.players) > MAX_ROBOTS:
            errs.append(f"at most {MAX_ROBOTS} robots")
        if not (self.continuous or (math.isfinite(self.half_minutes) and 0.1 <= self.half_minutes <= 60)):
            errs.append("half length must be 0.1-60 minutes")
        if not 0 <= self.packet_budget <= MAX_PACKET_BUDGET:
            errs.append(f"packet budget must be 0-{MAX_PACKET_BUDGET} (0 = no limit)")
        base = self.ros_domain_id if self.ros_domain_id is not None else DEFAULT_DOMAIN_BASE
        if base + len(self.players) - 1 > 232:
            errs.append(f"each robot needs its own ROS domain: {base} + {len(self.players)} robots goes past 232")
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
            if key == "players":
                if isinstance(value, list):
                    base.players = [_clean_player(p) for p in value if isinstance(p, dict)]
                continue
            if key == "settings":
                base.settings = setting_registry.clean(value)
                continue
            try:
                if isinstance(default, bool):
                    value = bool(value) if isinstance(value, bool) else default
                elif isinstance(default, int):
                    value = int(value)
                elif isinstance(default, float):
                    value = float(value)
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


def _clean_player(p: dict) -> dict:
    try:
        return {"team": str(p.get("team", HOME)), "player_id": int(p.get("player_id", 0)), "role": str(p.get("role", ""))}
    except (TypeError, ValueError):
        return {"team": HOME, "player_id": 0, "role": ""}


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    config: LaunchConfig | None = None  # None => not launchable yet
    blocked_reason: str = ""
    custom: bool = False  # the roster editor's starting point: picking it opens the editor

    @property
    def available(self) -> bool:
        return self.config is not None


def _players(home_ids=(), away_ids=(), away_roles=None) -> list[dict]:
    roster = standard_team(HOME, home_ids) + standard_team(AWAY, away_ids)
    if away_roles:  # {player_id: role} for the away side
        roster = [PlayerSpec(p.team, p.player_id, away_roles.get(p.player_id, p.role)) if p.team == AWAY else p
                  for p in roster]
    return [p.to_dict() for p in roster]


def _match(players: list[dict], **kw) -> LaunchConfig:
    # Several robots cannot each render a camera and run YOLO in real time: perfect ball detection by default.
    return LaunchConfig(players=players, ball_detector="ground_truth", match_mode="quick", **kw)


COMPETITION = Preset(
    "Competition", "5v5 as at the competition: real match procedure, two 10-minute halves, 60000-packet budget, "
    "team comms at 3 Hz, noisy perception. The default; change anything you like.",
    LaunchConfig(
        simulator="2d", players=_players((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)), ball_detector="ground_truth",
        match_mode="full", half_minutes=10.0, continuous=False, packet_budget=60000,
        settings={
            "enable_com": True,
            "communication.team_broadcast_rate_hz": 3.0,
            "communication.discovery_rate_hz": 1.0,
            # ponytail: placeholders until measured on the real robots; without noise fusion has nothing to fix
            "sim.detection_noise_std": 0.05,
            "sim.detection_noise_range_gain": 0.03,
            "sim.odom_drift_pct": 1.0,
            "sim.odom_yaw_drift_deg": 1.0,
            "sim.collisions": True,
            "sim.falls": True,
        }))

PRESETS: tuple[Preset, ...] = (
    COMPETITION,
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
    Preset("Team vs team (5v5)", "Full match: 2 teams of 5, referee, two halves",
           _match(_players((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)))),
    Preset("5 robots, no opponents", "One team of 5 on the pitch alone: practise attack and formations",
           _match(_players((1, 2, 3, 4, 5)))),
    Preset("Goalie vs 1 striker", "1 goalie against 1 opponent striker",
           _match(_players((1,), (2,)), continuous=True)),
    Preset("Goalie vs 2 strikers", "1 goalie against 2 opponent strikers",
           _match(_players((1,), (2, 3)), continuous=True)),
    Preset("Goalies only", "Both teams' goalies only", _match(_players((1,), (1,)), continuous=True)),
    Preset("Custom match", "Build your own: pick who is on the pitch, their team and role",
           _match(_players((1, 2), (2,)), continuous=True), custom=True),
)


DEFAULT_CONFIG = COMPETITION.config  # what a fresh launcher shows
