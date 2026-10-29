"""Ball dynamics of the 2D kinematic simulator: rolling deceleration, flight with bounces, goalposts and the
push of walking robots. Pure numpy, one ball."""
import math

import numpy as np

from .kinematic_params import KinematicParams


class KinematicBall:
    def __init__(self, params: KinematicParams, radius: float, goal_height: float, post_radius: float,
                 posts: list[tuple[float, float]]):
        self.p = params
        self.r = radius
        self.goal_height = goal_height
        self.post_r = post_radius
        self.posts = posts
        self.pos = np.array([0.0, 0.0, radius])
        self.vel = np.zeros(3)

    def place(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0) -> None:
        self.pos[:] = (x, y, self.r)
        self.vel[:] = (vx, vy, vz)

    def step(self, dt: float, bodies: list["Body"]) -> None:
        n = max(1, self.p.ball_substeps)
        h = dt / n
        for _ in range(n):
            self._advance(h)
            self._posts()
            for b in bodies:
                self._body(b)

    # ---- free motion --------------------------------------------------------------------------------------------
    def _advance(self, h: float) -> None:
        p, r = self.p, self.r
        if self.pos[2] <= r + 1e-6 and self.vel[2] <= 1e-6:  # on the ground, rolling
            self.pos[2], self.vel[2] = r, 0.0
            speed = math.hypot(self.vel[0], self.vel[1])
            if speed > 0.0:
                ux, uy = self.vel[0] / speed, self.vel[1] / speed
                slow = speed - p.ball_roll_decel * h
                if slow <= 0.0:  # comes to rest inside this substep
                    distance, slow = speed * speed / (2.0 * p.ball_roll_decel), 0.0
                else:
                    distance = 0.5 * (speed + slow) * h
                self.pos[0] += ux * distance
                self.pos[1] += uy * distance
                self.vel[0], self.vel[1] = ux * slow, uy * slow
            return
        self.vel[2] -= p.ball_gravity * h
        self.pos += self.vel * h
        if self.pos[2] <= r and self.vel[2] < 0.0:  # bounce
            self.pos[2] = r
            self.vel[2] = -p.ball_restitution * self.vel[2]
            self.vel[:2] *= 1.0 - p.ball_bounce_friction
            if self.vel[2] < 0.5:
                self.vel[2] = 0.0

    def _posts(self) -> None:
        if self.pos[2] > self.goal_height + self.r:
            return
        reach = self.post_r + self.r
        for px, py in self.posts:
            dx, dy = self.pos[0] - px, self.pos[1] - py
            d = math.hypot(dx, dy)
            if d >= reach or d < 1e-9:
                continue
            nx, ny = dx / d, dy / d
            self.pos[0], self.pos[1] = px + nx * reach, py + ny * reach
            vn = self.vel[0] * nx + self.vel[1] * ny
            if vn < 0.0:
                bounce = (1.0 + self.p.ball_post_restitution) * vn
                self.vel[0] -= bounce * nx
                self.vel[1] -= bounce * ny

    # ---- walking robots push the ball ---------------------------------------------------------------------------
    def _body(self, b: "Body") -> None:
        if self.pos[2] > self.p.ball_contact_height:
            return
        reach = self.p.ball_contact_radius + self.r
        dx, dy = self.pos[0] - b.x, self.pos[1] - b.y
        d = math.hypot(dx, dy)
        if d >= reach:
            return
        if d < 1e-9:
            dx, dy, d = 1.0, 0.0, 1.0
        nx, ny = dx / d, dy / d
        self.pos[0], self.pos[1] = b.x + nx * reach, b.y + ny * reach
        rel = (self.vel[0] - b.vx) * nx + (self.vel[1] - b.vy) * ny
        if rel < 0.0:  # closing: leave at the body's normal speed plus a damped rebound
            push = -(1.0 + self.p.ball_contact_restitution) * rel
            self.vel[0] += push * nx
            self.vel[1] += push * ny


class Body:
    """What the ball sees of a robot: where it is and how fast it moves (world frame)."""

    __slots__ = ("x", "y", "vx", "vy")

    def __init__(self, x: float = 0.0, y: float = 0.0, vx: float = 0.0, vy: float = 0.0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
