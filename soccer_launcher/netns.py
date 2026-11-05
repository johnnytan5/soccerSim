"""Network profile: run each brain in its own network namespace with the team channel impaired (scripts/netem.sh).

Pure helpers (parse / plan) plus the side-effecting `session`, which brings the namespaces up, falls back to profile
none with a warning when this machine cannot do it, and always tears them down again."""
from __future__ import annotations

import atexit
import contextlib
import dataclasses
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LaunchConfig

PROFILES = ("none", "good", "rcap_bad", "terrible", "blackout")
BROADCAST = "10.77.0.255"  # directed broadcast of br-sim (10.77.0.0/24); game_control_ip is the bridge itself
GATEWAY = "10.77.0.1"
SLOTS = "0123456789"  # SS_SLOT: parallel copies of the whole setup (see scripts/netem.sh)
_PROFILE_RE = re.compile(r"^(none|good|rcap_bad|terrible|blackout(:\d+(\.\d+)?)?|lat\d+_loss\d+(\.\d+)?)$")
_OVERRIDE_RE = re.compile(r"^(home|away)(\d+)$")


def slot_names(slot: str = "", robot: int = 1) -> dict[str, str]:
    """Names netem.sh derives for a slot ('' = the default set). Interface names are <= 15 characters (checked by a test)."""
    if slot not in ("", *SLOTS):
        raise ValueError(f"slot must be empty or one digit 0-9, not '{slot}'")
    if slot == "":
        net, ns, br, sep = "10.77.0", "ss-r", "br-sim", ""
    else:
        net, ns, br, sep = f"10.77.{int(slot) + 1}", f"ss{slot}-r", f"br-sim{slot}", f"{slot}_"
    return {"bridge": br, "subnet": f"{net}.0/24", "gateway": f"{net}.1", "broadcast": f"{net}.255",
            "namespace": f"{ns}{robot}", "host": f"ssh{sep}{robot}", "peer": f"ssv{sep}{robot}", "ifb": f"ssi{sep}{robot}",
            "ns_prefix": ns, "robot_ip": f"{net}.{10 + robot}"}


def slot_of(cfg: LaunchConfig, env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return cfg.netns_slot or env.get("SS_SLOT", "")


def valid_profile(text: str) -> bool:
    return bool(_PROFILE_RE.match(text))


def parse_overrides(text: str) -> dict[tuple[str, int], str]:
    """'home3=terrible, away2=blackout:10' -> {('home', 3): 'terrible', ('away', 2): 'blackout:10'}. Raises ValueError."""
    out: dict[tuple[str, int], str] = {}
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        key, sep, prof = part.partition("=")
        m = _OVERRIDE_RE.match(key)
        if not sep or not m or not valid_profile(prof):
            raise ValueError(f"'{part}': use home3=terrible (home|away + player id = none|good|rcap_bad|terrible|blackout[:T])")
        out[(m.group(1), int(m.group(2)))] = prof
    return out


def validate(cfg: LaunchConfig) -> list[str]:
    errs = []
    if not valid_profile(cfg.network_profile):
        errs.append("network profile must be none, good, rcap_bad, terrible, blackout[:seconds] or lat<ms>_loss<pct>")
    if cfg.netns_slot not in ("", *SLOTS):
        errs.append("netns slot must be empty or one digit 0-9")
    try:
        parse_overrides(cfg.network_overrides)
    except ValueError as e:
        errs.append(f"network overrides {e}")
    if cfg.network_profile != "none" or cfg.network_overrides.strip():
        if not cfg.is_match:
            errs.append("a network profile only applies to matches (the single-robot launch runs in the root namespace)")
    return errs


def active(cfg: LaunchConfig) -> bool:
    """True when any robot gets a profile other than none (the default: today's behaviour, no namespaces)."""
    return cfg.is_match and any(p != "none" for p in robot_profiles(cfg))


def robot_profiles(cfg: LaunchConfig) -> list[str]:
    """Profile of each robot in roster order (robot i lives in namespace ss-r{i+1}, or ss<slot>-r{i+1}). Bad overrides are ignored here
    (validate reports them)."""
    try:
        over = parse_overrides(cfg.network_overrides)
    except ValueError:
        over = {}
    base = cfg.network_profile if valid_profile(cfg.network_profile) else "none"
    return [over.get((p.get("team"), int(p.get("player_id", 0))), base) for p in cfg.players]


def ingress_enabled(cfg: LaunchConfig, env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(cfg.network_ingress) or env.get("SS_NETEM_INGRESS") == "1"


def brain_extras(cfg: LaunchConfig) -> tuple[dict[str, str], dict]:
    """(extra `ros2 launch` arguments as {name: value}, brain parameter overrides) that put the brains on the veth."""
    slot = slot_of(cfg)
    if slot == "":  # the default set: the command line stays exactly as before
        return ({"netns": "true", "game_control_ip": GATEWAY}, {"communication.discovery_address": BROADCAST})
    n = slot_names(slot)
    return ({"netns": "true", "game_control_ip": n["gateway"], "ns_prefix": n["ns_prefix"]},
            {"communication.discovery_address": n["broadcast"]})


def netem_env(cfg: LaunchConfig, env: dict[str, str] | None = None) -> dict[str, str]:
    return {"SS_ROBOTS": str(len(cfg.players)), "SS_NETEM_INGRESS": "1" if ingress_enabled(cfg, env) else "0",
            "SS_SLOT": slot_of(cfg, env)}


# ---- preflight --------------------------------------------------------------------------------------
def _run_ok(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _module_ok(name: str, probe: Callable[[list[str]], bool]) -> bool:
    return probe(["sudo", "-n", "modprobe", "-n", name]) or Path("/sys/module", name).is_dir()


def missing(ingress: bool = False, probe: Callable[[list[str]], bool] = _run_ok, which=shutil.which) -> list[str]:
    """What this machine lacks for a network profile (empty list = ready)."""
    out = []
    if not which("tc") or not which("ip"):
        out.append("iproute2 (tc / ip) not installed")
        return out
    if not probe(["sudo", "-n", "tc", "-V"]):
        out.append("passwordless `sudo -n tc` not available")
    if not probe(["sudo", "-n", "ip", "netns", "list"]):
        out.append("passwordless `sudo -n ip netns` not available")
    if not _module_ok("sch_netem", probe):
        out.append("kernel module sch_netem (netem) missing")
    if ingress and not _module_ok("ifb", probe):
        out.append("kernel module ifb missing (`sudo modprobe ifb`): per-receiver loss unavailable")
    return out


# ---- bring-up / tear-down -----------------------------------------------------------------------------
def _script() -> str:
    from .store import project_root  # late: store imports config, which imports this module
    return str(project_root() / "scripts" / "netem.sh")


def _netem(args: list[str], cfg: LaunchConfig, log: Callable[[str], None], env: dict[str, str] | None = None) -> bool:
    try:
        p = subprocess.run(["bash", _script(), *args], capture_output=True, text=True, timeout=120,
                           env={**os.environ, **netem_env(cfg, env)})
    except (OSError, subprocess.SubprocessError) as e:
        log(f"netem.sh {' '.join(args)} failed: {e}")
        return False
    for line in (p.stdout + p.stderr).splitlines():
        log(f"[netem] {line}")
    return p.returncode == 0


def bring_up(cfg: LaunchConfig, log: Callable[[str], None]) -> bool:
    ok = _netem(["up"], cfg, log)
    for i, prof in enumerate(robot_profiles(cfg), start=1):
        if ok and prof not in ("good", "none"):  # `up` already left every namespace on good
            ok = _netem(["apply", prof, str(i)], cfg, log) and ok
    return ok


def tear_down(cfg: LaunchConfig, log: Callable[[str], None] = lambda _l: None) -> None:
    _netem(["down"], cfg, log)


@contextlib.contextmanager
def session(cfg: LaunchConfig, log: Callable[[str], None]) -> Iterator[LaunchConfig]:
    """Yields the config to launch: `cfg` with namespaces up, or a copy with profile none when they cannot be had."""
    if not active(cfg):
        yield cfg
        return
    lacks = missing(ingress_enabled(cfg))
    fallback = dataclasses.replace(cfg, network_profile="none", network_overrides="", network_ingress=False)
    if lacks:
        log("WARNING network profile ignored, running without namespaces: " + "; ".join(lacks))
        yield fallback
        return
    cleanup = lambda: tear_down(cfg)  # noqa: E731
    atexit.register(cleanup)
    try:
        if not bring_up(cfg, log):
            log("WARNING could not set up the network namespaces, running without them")
            yield fallback
            return
        log("network profile: " + ", ".join(f"{p.get('team')}{p.get('player_id')}={prof}"
                                            for p, prof in zip(cfg.players, robot_profiles(cfg)))
            + (" (per-receiver ingress)" if ingress_enabled(cfg) else ""))
        yield cfg
    finally:
        tear_down(cfg, log)
        atexit.unregister(cleanup)
