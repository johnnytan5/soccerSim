"""Tunable settings: one registry drives validation, the command line and the launcher form.

Each entry is either a brain parameter (becomes a ROS parameter override of every brain_node through the
`brain_params` launch argument) or a simulator parameter (becomes a `--sim-param NAME=VALUE` of the match
bridge). A blank / absent value means "don't override": the brain keeps config.yaml's competition value.

Entries come from two places:
  * the static lists below (existing brain comms / tracking knobs and the simulator's realism knobs), and
  * `world_model.schema.json` of the RCAP checkout being launched (generated from the C++ parameter table, so the
    world-model settings follow the code); a checkout without the file simply has none of them.
Adding a future group (swarm coordination, ...) is adding entries here or a schema in the RCAP code.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

SCHEMA_PATH = Path("src") / "brain" / "config" / "world_model.schema.json"
WM_PREFIX = "world_model."

BRAIN, SIM = "brain", "sim"
TYPES = ("float", "int", "bool", "str")


@dataclass(frozen=True)
class Setting:
    key: str  # stable id and the key in LaunchConfig.settings: the ROS parameter name, or "sim.<name>"
    label: str
    type: str  # float | int | bool | str
    group: str
    target: str  # brain | sim
    param: str  # ROS parameter name (brain) / SimConfig-KinematicParams field or "seed" (sim)
    default: float | int | bool | str | None = None  # the competition value, shown as the placeholder
    lo: float | None = None
    hi: float | None = None
    unit: str = ""
    help: str = ""
    choices: tuple = ()
    inert: bool = False  # exists in config.yaml but the brain never reads it
    only_sim: str = ""  # "2d": only meaningful for the kinematic simulator
    source: str = "static"  # static | rcap

    def to_public(self) -> dict:
        return {"key": self.key, "label": self.label, "type": self.type, "group": self.group, "target": self.target,
                "default": self.default, "min": self.lo, "max": self.hi, "unit": self.unit, "help": self.help,
                "choices": list(self.choices), "inert": self.inert, "only_sim": self.only_sim, "source": self.source}


def _brain(param: str, label: str, type_: str, default, lo=None, hi=None, unit="", help="", group="Team comms & ball",
           inert=False) -> Setting:
    return Setting(param, label, type_, group, BRAIN, param, default, lo, hi, unit, help, inert=inert)


def _sim(name: str, label: str, type_: str, default, lo=None, hi=None, unit="", help="", only_sim="") -> Setting:
    return Setting(f"sim.{name}", label, type_, "Simulation realism", SIM, name, default, lo, hi, unit, help,
                   only_sim=only_sim)


STATIC: tuple[Setting, ...] = (
    _brain("enable_com", "Team communication", "bool", True, help="Off: robots do not send or use teammate packets."),
    _brain("communication.team_broadcast_rate_hz", "State rate", "float", 3.0, 0.1, 20.0, "Hz",
           "Teammate state packets per second (the team budget counts every packet)."),
    _brain("communication.discovery_rate_hz", "Discovery rate", "float", 1.0, 0.1, 20.0, "Hz",
           "Peer-discovery packets per second; below 0.5 Hz one lost packet breaks restart detection."),
    _brain("strategy.cooperation.tactical_packet_timeout_ms", "Teammate stale timeout", "float", 600.0, 100.0, 5000.0,
           "ms", "The brain raises this to 3 x the state interval + 100 ms itself (1099 ms at 3 Hz)."),
    _brain("strategy.tm_ball_dist_threshold", "Teammate ball distance", "float", 1.5, 0.0, 10.0, "m",
           "A teammate's ball counts as 'reliable' when it is farther than this from this robot."),
    _brain("strategy.ball_memory_timeout", "Ball memory", "float", 1.0, 0.0, 10.0, "s",
           "How long a lost ball stays 'known' at its last position."),
    _brain("strategy.ball_confidence_threshold", "Ball confidence threshold", "float", 40.0, 0.0, 100.0, "",
           "Ball detections below this confidence are ignored."),
    _brain("strategy.ball_confidence_decay_rate", "Ball confidence decay", "float", 3.0, 0.0, 100.0, "",
           "In config.yaml but never read by the brain: changing it has no effect.", inert=True),
    _brain("obstacle_avoidance.robot_obstacle_memory_msecs", "Robot track memory", "float", 1000.0, 0.0, 10000.0, "ms",
           "How long an unseen robot stays an obstacle.", group="Robot tracking"),
    _brain("obstacle_avoidance.robot_tracking_alpha", "Robot tracking weight", "float", 0.65, 0.0, 1.0, "",
           "Weight of a new robot observation in the position low-pass filter.", group="Robot tracking"),
    _brain("obstacle_avoidance.robot_match_distance", "Robot match distance", "float", 0.7, 0.1, 5.0, "m",
           "Cross-frame association distance for robot detections.", group="Robot tracking"),
    _brain("obstacle_avoidance.robot_min_confidence", "Robot min confidence", "float", 40.0, 0.0, 100.0, "",
           "Robot detections below this confidence are ignored.", group="Robot tracking"),
    _brain("obstacle_avoidance.robot_confirm_frames", "Robot confirm frames", "int", 2, 1, 20, "frames",
           "Consecutive detections before a robot counts as an obstacle.", group="Robot tracking"),
    _sim("detection_noise_std", "Detection noise", "float", None, 0.0, 1.0, "m",
         "Gaussian noise on every detection's position (ball, robots, posts, markings)."),
    _sim("detection_noise_range_gain", "Noise growth with range", "float", None, 0.0, 0.5, "m/m",
         "Extra noise per metre of range, so far objects are noisier: sigma = noise + gain * range."),
    _sim("odom_drift_pct", "Odometry drift", "float", None, 0.0, 20.0, "% of distance",
         "Position error growing with distance walked (fixed bias plus noise per run).", only_sim="2d"),
    _sim("odom_yaw_drift_deg", "Heading drift", "float", None, 0.0, 20.0, "deg per 10 m",
         "Heading error growing with distance walked.", only_sim="2d"),
    _sim("seed", "Random seed", "int", None, 0, 2**31 - 1, "",
         "Fixes the noise / drift / fall streams (wall-clock brains still make runs non-identical)."),
    _sim("collisions", "Robot-robot collisions", "bool", None, help="Robots block and push each other.", only_sim="2d"),
    _sim("falls", "Falls", "bool", None, help="Robots can fall (hard contact) and need time to get up.", only_sim="2d"),
)


# ---- the RCAP checkout's world-model schema ---------------------------------------------------------
_schema_cache: dict[str, tuple[float, list[Setting]]] = {}


def _from_schema(doc: dict) -> list[Setting]:
    out: list[Setting] = []
    ns = str(doc.get("namespace", "world_model"))
    for p in doc.get("params", []):
        try:
            kind = {"bool": "bool", "int": "int", "double": "float", "string": "str"}[p["type"]]
            key = f"{ns}.{p['key']}"
            group = f"World model: {p.get('group', 'general')}"
            out.append(Setting(key, p["key"], kind, group, BRAIN, key, p.get("default"), p.get("min"), p.get("max"),
                               p.get("unit", ""), p.get("help", ""), tuple(p.get("choices", ())), source="rcap"))
        except (KeyError, TypeError):
            continue
    return out


def load_schema(rcap_root: str | Path | None) -> list[Setting]:
    """World-model settings declared by the RCAP checkout (empty when it has no schema)."""
    if not rcap_root:
        return []
    path = Path(rcap_root) / SCHEMA_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    cached = _schema_cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        settings = _from_schema(json.loads(path.read_text()))
    except (OSError, ValueError):
        settings = []
    _schema_cache[str(path)] = (mtime, settings)
    return settings


def registry_for(rcap_root: str | Path | None = None) -> list[Setting]:
    return list(STATIC) + load_schema(rcap_root)


def schema_hash(rcap_root: str | Path | None) -> str:
    """Changes whenever the checkout's schema does (the front end refetches on it)."""
    import hashlib
    reg = load_schema(rcap_root)
    return hashlib.sha1(json.dumps([s.to_public() for s in reg], sort_keys=True).encode()).hexdigest()[:12] if reg else ""


def public(rcap_root: str | Path | None = None) -> list[dict]:
    return [s.to_public() for s in registry_for(rcap_root)]


# ---- values -----------------------------------------------------------------------------------------
def _coerce(setting: Setting, raw):
    """Typed value, or None to mean 'not set'. Raises ValueError for junk."""
    if raw is None or raw == "":
        return None
    if setting.type == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        raise ValueError("must be on or off")
    if setting.type == "str":
        text = str(raw)
        if setting.choices and text not in setting.choices:
            raise ValueError(f"must be one of {', '.join(setting.choices)}")
        return text
    if isinstance(raw, bool):
        raise ValueError("must be a number")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("must be a finite number")
    if setting.type == "int":
        if value != int(value):
            raise ValueError("must be a whole number")
        return int(value)
    return value


def clean(raw, registry: list[Setting] | None = None) -> dict:
    """Tolerant load: keep scalar values, drop junk. Keys the registry does not know are kept (they may belong to a
    different RCAP checkout) but never reach a command line."""
    if not isinstance(raw, dict):
        return {}
    by_key = {s.key: s for s in (registry if registry is not None else STATIC)}
    out = {}
    for key, value in raw.items():
        if not isinstance(key, str) or isinstance(value, (dict, list)):
            continue
        setting = by_key.get(key)
        if setting is None:
            if isinstance(value, (bool, int, float, str)) and value != "":
                out[key] = value
            continue
        try:
            typed = _coerce(setting, value)
        except (TypeError, ValueError):
            continue
        if typed is not None:
            out[key] = typed
    return out


def validate(values: dict, registry: list[Setting]) -> list[str]:
    errs = []
    by_key = {s.key: s for s in registry}
    for key, raw in values.items():
        s = by_key.get(key)
        if s is None:
            continue  # reported by preflight as a warning (different RCAP checkout)
        try:
            v = _coerce(s, raw)
        except (TypeError, ValueError) as e:
            errs.append(f"{s.label}: {e}")
            continue
        if v is None or s.type in ("bool", "str"):
            continue
        if s.lo is not None and v < s.lo or s.hi is not None and v > s.hi:
            errs.append(f"{s.label} must be {_fmt(s.lo)} to {_fmt(s.hi)}{(' ' + s.unit) if s.unit else ''}")
    return errs


def _fmt(v) -> str:
    return "" if v is None else (str(int(v)) if float(v) == int(v) else f"{v:g}")


def unknown_keys(values: dict, registry: list[Setting]) -> list[str]:
    known = {s.key for s in registry}
    return sorted(k for k in values if k not in known)


def brain_overrides(values: dict, registry: list[Setting]) -> dict:
    """ROS parameter name -> typed value. ROS is strict about int vs double, so floats stay floats (40 -> 40.0)."""
    out = {}
    for s in registry:
        if s.target != BRAIN or s.inert or s.key not in values:
            continue
        try:
            v = _coerce(s, values[s.key])
        except (TypeError, ValueError):
            continue
        if v is None:
            continue
        out[s.param] = float(v) if s.type == "float" else v
    return out


def sim_params(values: dict, registry: list[Setting], simulator: str) -> list[tuple[str, object]]:
    out = []
    for s in registry:
        if s.target != SIM or s.key not in values or (s.only_sim and s.only_sim != simulator):
            continue
        try:
            v = _coerce(s, values[s.key])
        except (TypeError, ValueError):
            continue
        if v is not None:
            out.append((s.param, v))
    return out
