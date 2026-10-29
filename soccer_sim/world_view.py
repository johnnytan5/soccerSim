"""A brain's world-model snapshot (/brain/world_model JSON, in that robot's team frame) made ready for the browser map:
moved into the world (home) frame, rounded, and scored against the simulator's ground truth.

Pure functions, no ROS. The snapshot layout is defined by RCAP's src/brain/src/world_model/snapshot_json.cpp.
"""
from __future__ import annotations

import math

TEAMMATE_AS_OPPONENT_M = 0.5  # an "opponent" track this close to a real teammate is a misclassified teammate


def _r(v):
    return round(v, 3) if isinstance(v, float) else v


def _flip_xy(d: dict, flipped: bool) -> dict:
    """Copy with floats rounded; position and velocity rotated by 180 degrees for a flipped (away) team frame."""
    out = {k: _r(v) for k, v in d.items() if not isinstance(v, (dict, list))}
    if flipped:
        for k in ("x", "y", "vx", "vy"):
            if k in out:
                out[k] = -out[k]
        if "theta" in out:
            out["theta"] = _r(math.atan2(math.sin(d["theta"] + math.pi), math.cos(d["theta"] + math.pi)))
    return out
    # the covariance ellipse angle is unchanged by a 180 degree rotation


def to_world(snapshot: dict, flipped: bool) -> dict:
    """The snapshot in the world frame. `flipped` is the robot's TeamFrame.flipped (the away team, or any team after
    half-time)."""
    out = {k: snapshot[k] for k in ("schema_version", "player", "enabled", "method", "t") if k in snapshot}
    out["pose"] = _flip_xy(snapshot.get("pose", {}), flipped)
    local = snapshot.get("local_ball", {})
    out["local_ball"] = _flip_xy(local, flipped)
    fused = snapshot.get("fused_ball", {})
    out["fused_ball"] = {**_flip_xy(fused, flipped),
                         "sources": [_flip_xy(s, flipped) for s in fused.get("sources", [])]}
    out["teammates"] = [_flip_xy(t, flipped) for t in snapshot.get("teammates", [])]
    out["opponents_local"] = [_flip_xy(o, flipped) for o in snapshot.get("opponents_local", [])]
    out["opponents_fused"] = [
        {**_flip_xy(o, flipped), "sources": [_flip_xy(m, flipped) for m in o.get("sources", [])]}
        for o in snapshot.get("opponents_fused", [])]
    out["counters"] = snapshot.get("counters", {})
    out["packet"] = snapshot.get("packet", {})
    return out


def _nearest(x: float, y: float, points: list[tuple[float, float]]) -> float | None:
    return min((math.hypot(x - px, y - py) for px, py in points), default=None)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def truth_errors(view: dict, ball_xy: tuple[float, float], team: str, me: tuple[float, float],
                 robots: list[tuple[str, float, float]]) -> dict:
    """How wrong this robot's estimates are right now. `robots`: (team, x, y) of every robot in the world frame."""
    bx, by = ball_xy
    out: dict = {}
    for name in ("local_ball", "fused_ball"):
        b = view.get(name, {})
        out[f"{name}_err"] = _r(math.hypot(b["x"] - bx, b["y"] - by)) if b.get("valid") else None
    opponents = [(x, y) for t, x, y in robots if t != team]
    teammates = [(x, y) for t, x, y in robots if t == team and (x, y) != me]
    for name, key in (("opp_local_err", "opponents_local"), ("opp_fused_err", "opponents_fused")):
        errs = [d for o in view.get(key, []) if (d := _nearest(o["x"], o["y"], opponents)) is not None]
        out[name] = _mean(errs)
    out["teammates_as_opponents"] = sum(
        1 for o in view.get("opponents_local", [])
        if (d := _nearest(o["x"], o["y"], teammates)) is not None and d < TEAMMATE_AS_OPPONENT_M)
    return out


class Ema:
    """Exponential moving average of an error that may be missing (None) for a while."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float | None) -> float | None:
        if x is not None:
            self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        return None if self.value is None else round(self.value, 3)
