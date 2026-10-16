"""Validated robot placement requests (controller page: click on the field to put the robot somewhere)."""

import math

from .config import FieldDims

# Keeper's home: goal centre, 1 m in front of the line, facing the field (what "reset robot" always did).
HOME = (-10.0, 0.0, 0.0)


def robot_pose_from_dict(d: dict, field: FieldDims | None = None) -> tuple[float, float, float]:
    """{} -> HOME. Otherwise needs x, y (inside the pitch) and optionally yaw (radians, finite)."""
    field = field or FieldDims()
    if not d:
        return HOME
    unknown = set(d) - {"x", "y", "yaw"}
    if unknown:
        raise ValueError(f"unknown placement fields: {sorted(unknown)}")
    if "x" not in d or "y" not in d:
        raise ValueError("placement needs x and y")
    x, y, yaw = float(d["x"]), float(d["y"]), float(d.get("yaw", 0.0))
    if not all(map(math.isfinite, (x, y, yaw))):
        raise ValueError("x, y and yaw must be finite")
    if abs(x) > field.length / 2 or abs(y) > field.width / 2:
        raise ValueError(f"({x:.2f}, {y:.2f}) is outside the pitch")
    return x, y, math.atan2(math.sin(yaw), math.cos(yaw))
