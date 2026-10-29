import math

from .geometry import world_to_robot_2d, wrap_pi
from .perception import Pose2D


class OdometryTracker:
    """Odometer zeroed at the spawn pose, like the robot's /odometer_state after boot.

    Without drift it is the ground-truth pose relative to the spawn pose. With drift (`drift_pct`: position error as a
    percentage of the distance walked, `yaw_drift_deg`: heading error per 10 m walked, and an `rng`) the pose is
    integrated from noisy increments: a bias drawn once per tracker plus noise that grows with the square root of the
    distance, so a robot that stands still does not move on the odometer.
    """

    def __init__(self, origin: Pose2D, drift_pct: float = 0.0, yaw_drift_deg: float = 0.0, rng=None):
        self.origin = origin
        self._frac = max(0.0, drift_pct) / 100.0
        self._yaw_per_m = math.radians(max(0.0, yaw_drift_deg)) / 10.0
        self._rng = rng
        self._noisy = rng is not None and (self._frac > 0.0 or self._yaw_per_m > 0.0)
        if self._noisy:
            self._bias_t = float(rng.normal(0.0, self._frac))
            self._bias_yaw = float(rng.normal(0.0, self._yaw_per_m))
        self._prev = origin  # last true pose (world)
        self._x = self._y = self._theta = 0.0  # integrated odometer pose in the origin frame

    def update(self, pose_w: Pose2D, noisy: bool = True) -> tuple[float, float, float]:
        """The odometer reading for the robot now at `pose_w`. noisy=False applies the move exactly (a teleport)."""
        o = self.origin
        if not self._noisy:
            x, y = world_to_robot_2d(pose_w.x, pose_w.y, o.x, o.y, o.yaw)
            return x, y, wrap_pi(pose_w.yaw - o.yaw)
        p = self._prev
        dxb, dyb = world_to_robot_2d(pose_w.x, pose_w.y, p.x, p.y, p.yaw)
        dth = wrap_pi(pose_w.yaw - p.yaw)
        d = math.hypot(dxb, dyb)
        if noisy and d > 0.0:
            scale = 1.0 + self._bias_t
            sigma = self._frac * 0.3 * math.sqrt(d)
            dxb = dxb * scale + float(self._rng.normal(0.0, sigma))
            dyb = dyb * scale + float(self._rng.normal(0.0, sigma))
            dth = dth * scale + self._bias_yaw * d + float(self._rng.normal(0.0, self._yaw_per_m * 0.3 * math.sqrt(d)))
        c, s = math.cos(self._theta), math.sin(self._theta)
        self._x += c * dxb - s * dyb
        self._y += s * dxb + c * dyb
        self._theta = wrap_pi(self._theta + dth)
        self._prev = pose_w
        return self._x, self._y, self._theta

    def teleport(self, pose_w: Pose2D) -> None:
        """The robot was moved by hand: the odometer follows exactly (brain's pose then follows it too)."""
        self.update(pose_w, noisy=False)
