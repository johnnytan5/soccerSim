"""Team frames: brain always thinks its own goal is at -x, so the away team sees the world rotated 180 degrees."""
import math
from dataclasses import dataclass

import numpy as np

from .geometry import wrap_pi
from .perception import Pose2D

_RZ_PI = np.diag([-1.0, -1.0, 1.0])


@dataclass
class TeamFrame:
    """World (home) frame <-> a team's own frame. `flipped` rotates by 180 degrees about the z axis.

    The pitch is point-symmetric, so posts, markings and lines need no conversion; only poses, the ball and
    the camera do. Mutable on purpose: the teams swap sides at half-time, which just toggles `flipped`.
    """

    flipped: bool = False

    def point(self, x: float, y: float) -> tuple[float, float]:
        return (-x, -y) if self.flipped else (x, y)

    def pose(self, p: Pose2D) -> Pose2D:
        """A world pose in this team's frame (the conversion is its own inverse)."""
        if not self.flipped:
            return p
        return Pose2D(-p.x, -p.y, wrap_pi(p.yaw + math.pi))

    def pose_tuple(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        p = self.pose(Pose2D(x, y, yaw))
        return p.x, p.y, p.yaw

    def vec3(self, v) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(3)
        return _RZ_PI @ v if self.flipped else v

    def rot(self, r) -> np.ndarray:
        r = np.asarray(r, dtype=float).reshape(3, 3)
        return _RZ_PI @ r if self.flipped else r

    def direction(self, yaw: float) -> float:
        return wrap_pi(yaw + math.pi) if self.flipped else yaw
