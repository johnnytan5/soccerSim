"""2D kinematic simulator: the same World / SoccerController interface as sim.py, without MuJoCo, torch or a camera.

Robots are discs that track the brain's velocity commands with the lag and gains measured from the T2 walk policy,
the ball rolls, bounces and is pushed, robots block each other and can fall. Brains, gateways, UDP team comms, the
GameController and the web API run unchanged on top of it (see backend.py for how match_node picks it).

Sensors: odometry (with optional drift), IMU and head pose that look like the MuJoCo robot's, and ground-truth
detections from the head camera model in perception.py. Nothing here renders.
"""
from __future__ import annotations

import math
from collections import deque
from types import SimpleNamespace

import numpy as np

from .config import SHOT_SLIDERS, SimConfig
from .field_geometry import field_lines, goalposts, map_markings
from .frames import TeamFrame
from .geometry import wrap_pi, world_to_robot_2d
from .kick_model import KickRef, VisualKickModel
from .kinematic_ball import Body, KinematicBall
from .kinematic_params import (
    DEFAULT_JOINT_POS, HEAD_OFFSET, LYING_TRUNK_Z, STAND_TRUNK_PITCH, STAND_TRUNK_Z, KinematicParams,
)
from .loco_rpc import MOVING_MODES, LocoState
from .odometry import OdometryTracker
from .perception import Camera, Detection, LineObs, Pose2D, perceive_ground_truth
from .shots import Shot, shot_velocity

STANDING, FALLING, DOWN, GETTING_UP = range(4)
POSTURE_NAMES = ("standing", "falling", "down", "getting_up")
FALLEN_AMOUNT = 0.67  # share of the way to lying (60 degrees) beyond which the robot counts as fallen
POLICY_DT = 0.02


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    t = r[0, 0] + r[1, 1] + r[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class KinematicController:
    """One robot of a KinematicWorld; stands in for sim.SoccerController."""

    def __init__(self, sim_cfg: SimConfig, world: "KinematicWorld", index: int = 0, frame: TeamFrame | None = None,
                 spawn: tuple[float, float, float] | None = None, rng=None):
        self.sim_cfg = sim_cfg
        self.world = world
        self.index = index
        self.frame = frame if frame is not None else TeamFrame()
        self.p = world.params
        self.cfg = SimpleNamespace(policy_dt=world.policy_dt)
        self._rng = rng if rng is not None else np.random.default_rng(sim_cfg.seed)
        self.root_body_id = index  # the MuJoCo viewer's tint key; unused here
        world.robots.append(self)

        self._markings = map_markings(sim_cfg.field)
        self._lines = field_lines(sim_cfg.field)
        self._posts = goalposts(sim_cfg.field)
        self.kick_model = VisualKickModel()
        self._was_kicking = False

        self.x = self.y = self.yaw = 0.0           # world pose
        self.vx = self.vy = self.w = 0.0           # body-frame velocity
        self.head_yaw = self.head_pitch = 0.0
        self._head_target = (0.0, 0.0)             # (pitch, yaw) as commanded by the brain
        self._cmd = (0.0, 0.0, 0.0)
        self._fifo: deque = deque([(0.0, 0.0, 0.0)] * (self.p.cmd_delay_steps + 1), maxlen=self.p.cmd_delay_steps + 1)
        self.posture = STANDING
        self._posture_t = 0.0
        self._getup_dur = self.p.getup_secs
        self.fall_amount = 0.0                     # 0 standing .. 1 lying
        self.fall_dir = 0.0                        # body-frame direction of the fall, 0 = forward
        self._gyro = np.zeros(3)

        sx, sy, syaw = spawn if spawn is not None else sim_cfg.spawn_xy_yaw
        self.odometry = self._new_odometry(Pose2D(sx, sy, syaw))
        self.reset_robot(*self.frame.pose_tuple(sx, sy, syaw))

    # ---- resets -------------------------------------------------------------------------------------------------
    def reset_robot(self, x: float, y: float, yaw: float) -> None:
        """Stand the robot upright at a world-frame pose."""
        self.x, self.y, self.yaw = x, y, wrap_pi(yaw)
        self.vx = self.vy = self.w = 0.0
        self._cmd = (0.0, 0.0, 0.0)
        self._fifo.extend([(0.0, 0.0, 0.0)] * len(self._fifo))
        self.posture, self._posture_t, self.fall_amount = STANDING, 0.0, 0.0
        self._gyro[:] = 0.0
        self.odometry.teleport(self.team_pose())

    def restart_odometry(self) -> None:
        """Brain was picked up and put down: its odometer starts again from where the robot now stands."""
        self.odometry = self._new_odometry(self.team_pose())

    def _new_odometry(self, origin: Pose2D) -> OdometryTracker:
        c = self.sim_cfg
        return OdometryTracker(origin, c.odom_drift_pct, c.odom_yaw_drift_deg, self._rng)

    def reset_ball(self, x: float, y: float) -> None:
        self.world.reset_ball(x, y)

    # ---- inputs from brain --------------------------------------------------------------------------------------
    def apply(self, loco: LocoState, now: float, kick_ref: KickRef | None = None,
              manual: tuple[float, float, float] | None = None) -> None:
        if manual is not None and loco.mode in MOVING_MODES:  # held stick: firmware remote beats brain and kick
            vx, vy, vyaw = manual
            self.kick_model.reset()
        elif loco.kicking() and not self.fallen():  # a robot on the ground cannot kick
            if not self._was_kicking:
                self.kick_model.reset()
            pose, ball = self.robot_pose(), self.ball_position()
            true_ball = world_to_robot_2d(ball[0], ball[1], pose.x, pose.y, pose.yaw)
            out = self.kick_model.step(kick_ref, now, true_ball, loco.visual_kick_version)
            vx, vy, vyaw = out.vx, out.vy, out.vyaw
            if out.strike_speed > 0.0:
                self.world.launch_ball(out.strike_speed * self.p.ball_launch_efficiency, pose.yaw + out.strike_dir)
        else:
            vx, vy, vyaw = loco.effective_velocity()
        self._was_kicking = loco.kicking()
        self._cmd = (vx, vy, vyaw)
        self._head_target = (loco.head_pitch, loco.head_yaw)
        if loco.getup_requested:
            loco.getup_requested = False
            if self.posture == DOWN:
                self.posture, self._posture_t = GETTING_UP, 0.0
                jitter = 1.0 + self.p.getup_jitter * float(self._rng.uniform(-1.0, 1.0))
                self._getup_dur = max(1.0, self.p.getup_secs * jitter)

    # ---- one policy tick ----------------------------------------------------------------------------------------
    def integrate(self, dt: float) -> None:
        self._step_head(dt)
        p = self.p
        if self.posture == STANDING:
            self._fifo.append(self._cmd)
            cx, cy, cw = self._fifo[0]
            self.vx = self._lag(self.vx, cx * (p.gain_x_forward if cx >= 0 else p.gain_x_backward), p.tau_x,
                                p.tau_x_decel, p.accel_x, dt)
            self.vy = self._lag(self.vy, cy * p.gain_y, p.tau_y, p.tau_y, p.accel_y, dt)
            self.w = self._lag(self.w, cw * p.gain_yaw, p.tau_yaw, p.tau_yaw, p.accel_yaw, dt)
            self._gyro[:] = (0.0, 0.0, self.w)
            self._move(dt)
        elif self.posture == FALLING:
            self._posture_t += dt
            before = self.fall_amount
            self.fall_amount = self._ease(min(1.0, self._posture_t / p.fall_secs))
            self.vx *= 0.85
            self.vy *= 0.85
            self.w *= 0.85
            self._move(dt)
            self._set_fall_gyro((self.fall_amount - before) / dt)
            if self._posture_t >= p.fall_secs:
                self.posture, self.fall_amount = DOWN, 1.0
        elif self.posture == DOWN:
            self.vx = self.vy = self.w = 0.0
            self._gyro[:] = 0.0
        else:  # GETTING_UP: tilt returns to upright; gyro stays above the brain's "is moving" threshold
            self._posture_t += dt
            before = self.fall_amount
            # linear, not eased: the brain fails a get-up whose gyro stays below 0.1 rad/s while the trunk is tilted
            self.fall_amount = 1.0 - min(1.0, self._posture_t / self._getup_dur)
            self._set_fall_gyro((self.fall_amount - before) / dt)
            if self._posture_t >= self._getup_dur:
                self.posture, self.fall_amount = STANDING, 0.0
                self._gyro[:] = 0.0
                self._fifo.extend([(0.0, 0.0, 0.0)] * len(self._fifo))

    @staticmethod
    def _ease(t: float) -> float:
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _lag(v: float, target: float, tau: float, tau_slow: float, a_max: float, dt: float) -> float:
        slowing = abs(target) < abs(v) or target * v < 0.0
        t = tau_slow if slowing else tau
        nv = v + (target - v) * (1.0 - math.exp(-dt / t))
        nv = v + _clamp(nv - v, -a_max * dt, a_max * dt)
        return 0.0 if target == 0.0 and abs(nv) < 5e-3 else nv  # a stopped robot is stopped, not creeping

    def _move(self, dt: float) -> None:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * self.vx - s * self.vy) * dt
        self.y += (s * self.vx + c * self.vy) * dt
        self.yaw = wrap_pi(self.yaw + self.w * dt)

    def _step_head(self, dt: float) -> None:
        pitch_t, yaw_t = self._head_target
        k = 1.0 - math.exp(-dt / self.p.head_tau)
        limit = self.p.head_rate_max * dt
        self.head_pitch += _clamp((pitch_t - self.head_pitch) * k, -limit, limit)
        self.head_yaw += _clamp((yaw_t - self.head_yaw) * k, -limit, limit)

    def _set_fall_gyro(self, d_amount_dt: float) -> None:
        rate = d_amount_dt * math.pi / 2.0
        self._gyro[:] = (-rate * math.sin(self.fall_dir), rate * math.cos(self.fall_dir), 0.0)

    # ---- falling ------------------------------------------------------------------------------------------------
    def topple(self, push_world: tuple[float, float]) -> None:
        """Pushed by a hard contact: fall away from where the push came from (push is the world-frame direction the
        robot is shoved in)."""
        if self.posture != STANDING or not self.p.falls:
            return
        px, py = world_to_robot_2d(push_world[0], push_world[1], 0.0, 0.0, self.yaw)
        self.fall_dir = math.atan2(py, px)  # pushed forward -> falls forward (0)
        self.posture, self._posture_t = FALLING, 0.0

    @property
    def immovable(self) -> bool:
        return self.posture in (DOWN, GETTING_UP)

    def fallen(self) -> bool:
        return self.posture != STANDING and self.fall_amount >= FALLEN_AMOUNT

    @property
    def posture_name(self) -> str:
        return POSTURE_NAMES[self.posture]

    # ---- outputs to brain ---------------------------------------------------------------------------------------
    def robot_pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    def team_pose(self) -> Pose2D:
        return self.frame.pose(self.robot_pose())

    def world_velocity(self) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * self.vx - s * self.vy, s * self.vx + c * self.vy

    def odometer(self) -> tuple[float, float, float]:
        return self.odometry.update(self.team_pose())

    def _tilt(self) -> tuple[float, float]:
        """(roll, pitch) of the trunk: the standing lean, or the way down."""
        f = self.fall_amount
        return (-f * (math.pi / 2.0) * math.sin(self.fall_dir),
                (1.0 - f) * STAND_TRUNK_PITCH + f * (math.pi / 2.0) * math.cos(self.fall_dir))

    def _trunk(self) -> tuple[np.ndarray, np.ndarray]:
        """Trunk position and orientation in the world frame."""
        roll, pitch = self._tilt()
        z = (1.0 - self.fall_amount) * STAND_TRUNK_Z + self.fall_amount * LYING_TRUNK_Z
        return np.array([self.x, self.y, z]), _rot_z(self.yaw) @ _rot_y(pitch) @ _rot_x(roll)

    def _head(self) -> tuple[np.ndarray, np.ndarray]:
        """Head position and orientation in the world frame."""
        pos, r_trunk = self._trunk()
        return pos + r_trunk @ np.array(HEAD_OFFSET), r_trunk @ _rot_z(self.head_yaw) @ _rot_y(self.head_pitch)

    def imu(self):
        roll, pitch = self._tilt()
        rpy = (roll, pitch, wrap_pi(self.frame.direction(self.yaw)))
        _, r_trunk = self._trunk()
        acc = r_trunk.T @ np.array([0.0, 0.0, 9.81])
        return rpy, tuple(float(v) for v in self._gyro), tuple(float(v) for v in acc)

    def joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.array(DEFAULT_JOINT_POS, dtype=float)
        q[0], q[1] = self.head_yaw, self.head_pitch  # the brain reads head yaw / pitch from motors 0 and 1
        return q, np.zeros_like(q), np.zeros_like(q)

    def head_pose_in_base(self) -> tuple[np.ndarray, np.ndarray]:
        """Head in the ground base frame (trunk xy at z = 0, yaw only), like sim.SoccerController."""
        head_pos, head_rot = self._head()
        r_base_t = _rot_z(self.yaw).T
        pos = r_base_t @ (head_pos - np.array([self.x, self.y, 0.0]))
        return pos, _quat_wxyz(r_base_t @ head_rot)

    def camera(self) -> Camera:
        """The head camera in this robot's team frame."""
        head_pos, head_rot = self._head()
        c2h = self.sim_cfg.camera.cam_to_head_matrix()
        pos = head_pos + head_rot @ c2h[:3, 3]
        return Camera(self.sim_cfg.camera, self.frame.vec3(pos), self.frame.rot(head_rot @ c2h[:3, :3]))

    def perceive(self, rng=None, rgb=None, others: list[tuple[float, float]] | None = None) \
            -> tuple[list[Detection], list[LineObs]]:
        """What the camera would report; nothing while the robot is down. others: world-frame (x, y) of the other
        robots, seen as 'Opponent'."""
        if self.posture != STANDING:
            return [], []
        rng = rng if rng is not None else self._rng
        bx, by = self.frame.point(*self.ball_position()[:2])
        return perceive_ground_truth(
            self.camera(), self.team_pose(), (bx, by, self.ball_position()[2]),
            [self.frame.point(x, y) for x, y in others or []], self._posts, self._markings, self._lines, self.sim_cfg, rng)

    def render(self):
        raise NotImplementedError("the 2D simulator has no camera; use ball_detector=ground_truth")

    # ---- the shared ball and shot panel --------------------------------------------------------------------------
    def shot_panel_values(self) -> dict[str, float]:
        return self.world.shot_panel_values()

    def shoot(self, shot: Shot) -> None:
        self.world.shoot(shot)

    def ball_position(self) -> np.ndarray:
        return self.world.ball_position()

    def ball_velocity(self) -> np.ndarray:
        return self.world.ball_velocity()


class KinematicWorld:
    """Pitch, ball and robots: stands in for sim.World."""

    mj_model = None
    mj_data = None
    n_robot_ctrl = 0

    def __init__(self, sim_cfg: SimConfig, n_robots: int = 1, params: KinematicParams | None = None):
        self.sim_cfg = sim_cfg
        self.n_robots = n_robots
        self.params = params if params is not None else KinematicParams()
        self.policy_dt = POLICY_DT
        self.robots: list[KinematicController] = []
        self.ball = KinematicBall(self.params, sim_cfg.ball_radius, sim_cfg.goal_height, sim_cfg.goalpost_radius,
                                  goalposts(sim_cfg.field))
        self._contacts: set[tuple[int, int]] = set()
        self.reset_ball(*sim_cfg.ball_xy)

    # ---- ball ---------------------------------------------------------------------------------------------------
    def reset_ball(self, x: float, y: float) -> None:
        self.ball.place(x, y)

    def shoot(self, shot: Shot) -> None:
        self.ball.place(shot.x, shot.y, *shot_velocity(shot.speed, shot.direction_deg, shot.elevation_deg))

    def launch_ball(self, speed: float, world_dir: float) -> None:
        self.ball.vel[:] = (speed * math.cos(world_dir), speed * math.sin(world_dir), 0.0)

    def ball_position(self) -> np.ndarray:
        return self.ball.pos.copy()

    def ball_velocity(self) -> np.ndarray:
        return self.ball.vel.copy()

    def shot_panel_values(self) -> dict[str, float]:
        return {name: default for name, _lo, _hi, default in SHOT_SLIDERS}

    # ---- stepping -----------------------------------------------------------------------------------------------
    def start(self) -> None:
        pass

    def step(self) -> None:
        dt = self.policy_dt
        for r in self.robots:
            r.integrate(dt)
        if self.params.collisions:
            self._collide_robots()
        bodies = []
        for r in self.robots:
            vx, vy = (0.0, 0.0) if r.immovable else r.world_velocity()
            bodies.append(Body(r.x, r.y, vx, vy))
        self.ball.step(dt, bodies)

    def _collide_robots(self) -> None:
        p = self.params
        robots = self.robots
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                a, b = robots[i], robots[j]
                ra = p.lying_radius if a.posture != STANDING else p.robot_radius
                rb = p.lying_radius if b.posture != STANDING else p.robot_radius
                dx, dy = b.x - a.x, b.y - a.y
                dist = math.hypot(dx, dy)
                if dist >= ra + rb:
                    self._contacts.discard((i, j))
                    continue
                nx, ny = (dx / dist, dy / dist) if dist > 1e-6 else (1.0, 0.0)
                wa, wb = (0.0 if a.immovable else 1.0), (0.0 if b.immovable else 1.0)
                if wa + wb == 0.0:
                    continue
                overlap = ra + rb - dist
                a.x -= nx * overlap * wa / (wa + wb)
                a.y -= ny * overlap * wa / (wa + wb)
                b.x += nx * overlap * wb / (wa + wb)
                b.y += ny * overlap * wb / (wa + wb)
                vax, vay = a.world_velocity()
                vbx, vby = b.world_velocity()
                closing = (vax - vbx) * nx + (vay - vby) * ny  # > 0 while they approach
                if closing > 0.0:
                    if (i, j) not in self._contacts:
                        self._contacts.add((i, j))
                        self._maybe_topple(a, b, nx, ny, closing)
                    if wa:  # stop moving into the other robot
                        va_n = vax * nx + vay * ny
                        if va_n > 0.0:
                            self._set_world_velocity(a, vax - va_n * nx, vay - va_n * ny)
                    if wb:
                        vb_n = vbx * nx + vby * ny
                        if vb_n < 0.0:
                            self._set_world_velocity(b, vbx - vb_n * nx, vby - vb_n * ny)
                self._contacts.add((i, j))

    def _maybe_topple(self, a: KinematicController, b: KinematicController, nx: float, ny: float,
                      closing: float) -> None:
        p = self.params
        if not p.falls or closing <= p.v_safe:
            return
        prob = p.p_fall_max * min(1.0, (closing - p.v_safe) / max(1e-6, p.v_hard - p.v_safe))
        if a.posture == STANDING and float(a._rng.random()) < prob:
            a.topple((-nx, -ny))  # a is shoved away from b
        if b.posture == STANDING and float(b._rng.random()) < prob:
            b.topple((nx, ny))

    @staticmethod
    def _set_world_velocity(r: KinematicController, vx: float, vy: float) -> None:
        c, s = math.cos(r.yaw), math.sin(r.yaw)
        r.vx, r.vy = c * vx + s * vy, -s * vx + c * vy
