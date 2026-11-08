"""LaunchConfig -> the exact command + environment that scripts/run_sim.sh needs. Pure, no I/O."""
from __future__ import annotations

import json
import shlex

from .config import DEFAULT_DOMAIN_BASE, LaunchConfig


def bridge_args(cfg: LaunchConfig, web_port: int | None = None) -> list[str]:
    """ROS parameter overrides for the bridge, as `-p name:=value` tokens (run_sim.sh word-splits them)."""
    params: list[tuple[str, object]] = [("ball_detector", cfg.ball_detector)]
    for name in ("spawn_x", "spawn_y", "spawn_yaw", "ball_x", "ball_y"):
        v = getattr(cfg, name)
        if v is not None:
            params.append((name, v))
    if cfg.referee_schedule:
        params.append(("referee_schedule", cfg.referee_schedule))
    if web_port is not None:  # the launcher proxies the bridge's controller server on this internal port
        params.append(("web_port", web_port))
    out: list[str] = []
    for k, v in params:
        out += ["-p", f"{k}:={v}"]
    out += shlex.split(cfg.extra_bridge_args)
    return out


def match_args(cfg: LaunchConfig, web_port: int | None = None) -> list[str]:
    """Options for the match bridge (soccer_sim_bridge.match_node); run_match.sh word-splits them."""
    out = ["--ball-detector", cfg.ball_detector, "--start", "full_auto" if cfg.match_mode == "full" else "quick",
           "--half-minutes", f"{cfg.half_minutes:g}",
           "--packet-budget", str(cfg.packet_budget)]
    if cfg.continuous:
        out.append("--continuous")
    if cfg.detailed_bodies:
        out.append("--detailed-bodies")
    for flag, v in (("--ball-x", cfg.ball_x), ("--ball-y", cfg.ball_y)):
        if v is not None:
            out += [flag, f"{v:g}"]
    if web_port is not None:
        out += ["--web-port", str(web_port)]
    return out + shlex.split(cfg.extra_bridge_args)


def _match_command(cfg: LaunchConfig, web_port: int | None) -> tuple[list[str], dict[str, str]]:
    roster = json.dumps([{k: p[k] for k in ("team", "player_id", "role")} for p in cfg.players], separators=(",", ":"))
    base = cfg.ros_domain_id if cfg.ros_domain_id is not None else DEFAULT_DOMAIN_BASE
    argv = ["bash", "scripts/run_match.sh"]
    if cfg.viewer:
        argv.append("--viewer")
    argv += [roster, f"tree:={cfg.tree}", f"domain_base:={base}"]
    argv += shlex.split(cfg.extra_launch_args)
    return argv, {"MATCH_ARGS": shlex.join(match_args(cfg, web_port))}


def _graphics_env(cfg: LaunchConfig, env: dict[str, str]) -> None:
    """The viewer uses the GPU unless asked otherwise; only the CPU choice needs saying."""
    if cfg.graphics == "cpu":
        env["SIM_VIEWER_GL"] = "cpu"


def build_command(cfg: LaunchConfig, rcap_root: str | None = None, web_port: int | None = None) -> tuple[list[str], dict[str, str]]:
    """Returns (argv, extra_env). argv is relative to the soccerSim root; env is merged over os.environ."""
    if cfg.is_match:
        argv, env = _match_command(cfg, web_port)
        if cfg.sim_vy_max is not None:
            env["SIM_VY_MAX"] = str(cfg.sim_vy_max)
        _graphics_env(cfg, env)
        if cfg.camera_gl != "auto":
            env["SIM_CAMERA_GL"] = cfg.camera_gl
        if cfg.mujoco_gl:
            env["MUJOCO_GL"] = cfg.mujoco_gl
        if rcap_root:
            env["ROBOTEDGE_ROOT"] = rcap_root
        return argv, env
    argv = ["bash", "scripts/run_sim.sh"]
    if cfg.viewer:
        argv.append("--viewer")
    if cfg.match_mode == "quick":
        argv.append("--quick")
    argv += [f"tree:={cfg.tree}", f"role:={cfg.role}", f"player_id:={cfg.player_id}"]
    argv += shlex.split(cfg.extra_launch_args)

    env: dict[str, str] = {"BRIDGE_ARGS": shlex.join(bridge_args(cfg, web_port))}
    if cfg.sim_vy_max is not None:
        env["SIM_VY_MAX"] = str(cfg.sim_vy_max)
    _graphics_env(cfg, env)
    if cfg.camera_gl != "auto":
        env["SIM_CAMERA_GL"] = cfg.camera_gl
    if cfg.mujoco_gl:
        env["MUJOCO_GL"] = cfg.mujoco_gl
    if cfg.ros_domain_id is not None:
        env["ROS_DOMAIN_ID"] = str(cfg.ros_domain_id)
    if rcap_root:
        env["ROBOTEDGE_ROOT"] = rcap_root
    return argv, env


def render(cfg: LaunchConfig, rcap_root: str | None = None, web_port: int | None = None) -> str:
    """The equivalent one-liner for the preview pane / copy-paste."""
    argv, env = build_command(cfg, rcap_root, web_port)
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return f"{prefix} pixi run {shlex.join(argv)}"

