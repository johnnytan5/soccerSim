"""Which simulator the match bridge runs on: MuJoCo physics ("mujoco") or the 2D kinematic simulator ("kinematic").

Both expose the same World / SoccerController surface (see kinematic.py), so match_node only talks to what this module
hands back. Importing it loads neither backend: "kinematic" never touches mujoco, torch or booster_deploy.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np

from .config import SimConfig
from .kinematic_params import KinematicParams

BACKENDS = ("mujoco", "kinematic")


def _parse(value: str, current):
    if isinstance(current, bool):
        if value.lower() not in ("true", "false", "1", "0"):
            raise ValueError("expected true or false")
        return value.lower() in ("true", "1")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:  # Optional[int], i.e. seed
        return int(value)
    raise ValueError("not a settable value")


def resolve_params(cfg: SimConfig, pairs: list[str]) -> tuple[SimConfig, KinematicParams]:
    """Apply `NAME=VALUE` overrides to the simulator settings. NAME is a SimConfig field (noise, drift, seed, ...) or a
    KinematicParams field (collisions, falls, ...); anything else is an error rather than being ignored."""
    sim_fields = {f.name for f in dataclasses.fields(SimConfig)}
    kin_fields = {f.name for f in dataclasses.fields(KinematicParams)}
    kin = KinematicParams()
    for pair in pairs:
        name, sep, text = pair.partition("=")
        if not sep:
            raise ValueError(f"--sim-param expects NAME=VALUE, got {pair!r}")
        try:
            if name in kin_fields:
                kin = dataclasses.replace(kin, **{name: _parse(text, getattr(kin, name))})
            elif name in sim_fields and name not in ("field", "camera"):
                cfg = dataclasses.replace(cfg, **{name: _parse(text, getattr(cfg, name))})
            else:
                raise ValueError("unknown setting")
        except ValueError as e:
            raise ValueError(f"--sim-param {pair}: {e}") from None
    return cfg, kin


def robot_rngs(seed: int | None, n: int) -> list[np.random.Generator]:
    """One independent random stream per robot; the same seed always gives the same streams."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def load_backend(name: str) -> SimpleNamespace:
    """(make_world(cfg, n, params), make_robot(cfg, world, index, frame, spawn, rng)) of the named backend."""
    if name == "kinematic":
        from .kinematic import KinematicController, KinematicWorld

        return SimpleNamespace(
            name=name,
            make_world=lambda cfg, n, params: KinematicWorld(cfg, n, params),
            make_robot=lambda cfg, world, i, frame, spawn, rng: KinematicController(cfg, world, i, frame, spawn, rng))
    if name == "mujoco":
        from .sim import SoccerController, World

        return SimpleNamespace(
            name=name,
            make_world=lambda cfg, n, params: World(cfg, n),
            make_robot=lambda cfg, world, i, frame, spawn, rng: SoccerController(cfg, world, i, frame, spawn, rng=rng))
    raise ValueError(f"unknown simulator backend {name!r}; expected one of {', '.join(BACKENDS)}")
