import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from booster_deploy.controllers.base_controller import BaseController
from tasks.locomotion.robots.t2 import T2WalkTaskCfg

from .config import SHOT_SLIDERS, SimConfig
from .field import OPTICAL_FROM_MUJOCO_CAM, ROBOT_VISUAL_GROUP, build_world, field_lines, goalposts, map_markings, robot_prefixes
from .frames import TeamFrame
from .geometry import rpy_from_quat_wxyz, wrap_pi, world_to_robot_2d, yaw_from_quat_wxyz
from .kick_model import KickRef, VisualKickModel
from .loco_rpc import MOVING_MODES, LocoState
from .odometry import OdometryTracker
from .perception import (
    Camera, Detection, LineObs, Pose2D, detect_ball, detect_goalposts, detect_markings, detect_robots, visible_lines,
)
from .shots import Shot, shot_velocity
from .yolo_ball import YoloBallDetector

HEAD_YAW_IDX, HEAD_PITCH_IDX = 0, 1  # T2_31DOF_CFG.joint_names order
UPRIGHT_MIN_ZZ = 0.5  # trunk z-axis . world z; same threshold as booster_deploy's safety fallback
HIDDEN_GEOM_GROUP = 4  # off in a default mjvOption; a robot's own visuals sit here while its camera renders


def _make_ball_detector(cfg: SimConfig) -> YoloBallDetector | None:
    if cfg.ball_detector == "ground_truth":
        return None
    if cfg.ball_detector != "yolo":
        raise ValueError(f"ball_detector must be 'yolo' or 'ground_truth', got {cfg.ball_detector!r}")
    if not Path(cfg.yolo_model_path).exists():
        raise FileNotFoundError(f"{cfg.yolo_model_path} missing; run `pixi run fetch-models`")
    return YoloBallDetector(cfg.yolo_model_path, cfg.yolo_ball_conf, cfg.yolo_nms_iou)


class World:
    """One MuJoCo model shared by every robot: pitch, ball, shot sliders, camera renderer, the physics step."""

    def __init__(self, sim_cfg: SimConfig, n_robots: int = 1):
        self.sim_cfg = sim_cfg
        self.n_robots = n_robots
        self.prefixes = robot_prefixes(n_robots)
        task = T2WalkTaskCfg()
        self.mj_model = build_world(sim_cfg, n_robots)
        self.mj_model.opt.timestep = task.mujoco.physics_dt
        # Elliptic cone: friction opposes the actual slip direction; pyramidal bends diagonal ball rolls toward 45°.
        self.mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.mj_data = mujoco.MjData(self.mj_model)
        self.decimation = task.mujoco.decimation
        self.n_robot_ctrl = n_robots * len(task.robot.joint_names)  # robot motors are ctrl[:n]; shot sliders follow
        self._sliders = {name: self.mj_model.actuator(name).id for name, *_ in SHOT_SLIDERS}
        for name, _lo, _hi, default in SHOT_SLIDERS:
            self.mj_data.ctrl[self._sliders[name]] = default
        ball_joint = self.mj_model.joint("ball_joint")
        self._ball_q = int(ball_joint.qposadr[0])
        self._ball_dq = int(ball_joint.dofadr[0])
        self._renderer: mujoco.Renderer | None = None
        self.yolo = _make_ball_detector(sim_cfg)  # one detector (and one ONNX session) for every robot
        self.robots: list[SoccerController] = []
        self._robot_geoms: np.ndarray | None = None
        self._robot_geoms_for: list = []
        self.reset_ball(*sim_cfg.ball_xy)

    def reset_ball(self, x: float, y: float) -> None:
        d = self.mj_data
        d.qpos[self._ball_q:self._ball_q + 7] = [x, y, self.sim_cfg.ball_radius, 1.0, 0.0, 0.0, 0.0]
        d.qvel[self._ball_dq:self._ball_dq + 6] = 0.0
        mujoco.mj_forward(self.mj_model, d)

    def shot_panel_values(self) -> dict[str, float]:
        return {name: float(self.mj_data.ctrl[i]) for name, i in self._sliders.items()}

    def shoot(self, shot: Shot) -> None:
        d = self.mj_data
        d.qpos[self._ball_q:self._ball_q + 7] = [shot.x, shot.y, self.sim_cfg.ball_radius, 1.0, 0.0, 0.0, 0.0]
        d.qvel[self._ball_dq:self._ball_dq + 3] = shot_velocity(shot.speed, shot.direction_deg, shot.elevation_deg)
        d.qvel[self._ball_dq + 3:self._ball_dq + 6] = 0.0
        mujoco.mj_forward(self.mj_model, d)

    def launch_ball(self, speed: float, world_dir: float) -> None:
        d = self.mj_data
        d.qvel[self._ball_dq:self._ball_dq + 3] = [speed * math.cos(world_dir), speed * math.sin(world_dir), 0.0]
        d.qvel[self._ball_dq + 3:self._ball_dq + 6] = 0.0

    def ball_position(self) -> np.ndarray:
        return self.mj_data.qpos[self._ball_q:self._ball_q + 3].copy()

    def ball_velocity(self) -> np.ndarray:
        return self.mj_data.qvel[self._ball_dq:self._ball_dq + 3].copy()

    def start(self) -> None:
        for r in self.robots:
            r.start()

    def step(self) -> None:
        """One policy tick for every robot: all policies first, then the PD substeps with a single physics step each."""
        targets = [r.next_targets() for r in self.robots]
        for _ in range(self.decimation):
            for r, t in zip(self.robots, targets):
                r.apply_pd(t)
            mujoco.mj_step(self.mj_model, self.mj_data)

    def robot_geoms(self) -> np.ndarray:
        if self._robot_geoms is None or len(self._robot_geoms_for) != len(self.robots):
            self._robot_geoms = np.unique(np.concatenate([r.visual_geoms for r in self.robots]))
            self._robot_geoms_for = list(self.robots)
        return self._robot_geoms

    def render(self, robot: "SoccerController") -> tuple[np.ndarray, np.ndarray]:
        cam_cfg = self.sim_cfg.camera
        if self._renderer is None:  # created lazily on the stepping thread (OpenGL context affinity)
            self._renderer = mujoco.Renderer(self.mj_model, height=cam_cfg.height, width=cam_cfg.width)
        r, m = self._renderer, self.mj_model
        # ponytail: every robot's body is hidden from the head camera — YOLO scores the chrome chest/hand meshes up to
        # 0.89 as "Ball" and brain has no self-filter. Costs self-occlusion; revisit if the real robot shows the same FP.
        # Other robots reach brain as ground-truth 'Opponent' detections instead (see perceive), and with no robot
        # meshes in view the frame also renders several times faster.
        own = self.robot_geoms()
        saved = m.geom_group[own].copy()
        m.geom_group[own] = HIDDEN_GEOM_GROUP
        try:
            r.disable_depth_rendering()
            r.update_scene(self.mj_data, camera=robot.cam_id)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.sim_cfg.render_shadows
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.sim_cfg.render_shadows
            rgb = r.render().copy()
            r.enable_depth_rendering()
            r.update_scene(self.mj_data, camera=robot.cam_id)
            depth = r.render().astype(np.float32)
            r.disable_depth_rendering()
        finally:
            m.geom_group[own] = saved
        return rgb, depth


class SoccerController(BaseController):
    """Runs Booster's T2 walk policy for one robot of a World (robot + its head camera, IMU and kick model).

    Mirrors booster_deploy's MujocoController PD loop, but slices robot joints out of a
    model that also contains a free-floating ball, overrides head targets with brain's
    RotateHead command, hands the chassis to VisualKickModel while a visual kick is
    active, renders the head camera, and never stops on a fall.

    Alone (no `world`) it builds a one-robot world, as the single-robot sim always did. In a match each robot
    gets `world`, its `index`, a `frame` (the away team sees the pitch rotated 180 degrees) and its own `spawn`
    pose, given in that frame.
    """

    def __init__(self, sim_cfg: SimConfig, world: World | None = None, index: int = 0,
                 frame: TeamFrame | None = None, spawn: tuple[float, float, float] | None = None):
        cfg = T2WalkTaskCfg()
        cfg.policy.enable_safety_fallback = False
        super().__init__(cfg)
        self.sim_cfg = sim_cfg
        self.world = world if world is not None else World(sim_cfg, 1)
        self.index = index
        self.frame = frame if frame is not None else TeamFrame()
        self.world.robots.append(self)

        self.mj_model = self.world.mj_model
        self.mj_data = self.world.mj_data
        self.decimation = self.world.decimation
        prefix = self.world.prefixes[index]
        m = self.mj_model

        nj = self.robot.num_joints
        root = m.joint(prefix + "world_joint")
        q0, dq0 = int(root.qposadr[0]), int(root.dofadr[0])
        self._root_q = q0
        self._nu_robot = nj
        self._act0 = index * nj  # this robot's motors are ctrl[_act0:_act0 + nj]
        self._q = slice(q0 + 7, q0 + 7 + nj)
        self._dq = slice(dq0 + 6, dq0 + 6 + nj)
        self.cam_id = m.camera(prefix + "head_rgb").id
        self._head_body = m.body(prefix + "head_pitch_link").id
        self._trunk_body = m.body(prefix + "trunk").id
        self._gyro = int(m.sensor(prefix + "imu_gyro").adr[0])
        self._acc = int(m.sensor(prefix + "imu_acc").adr[0])
        root_body = int(m.jnt_bodyid[root.id])
        self.root_body_id = int(m.body_rootid[root_body])  # what apply_look's tints are keyed on
        in_robot = m.body_rootid[m.geom_bodyid] == self.root_body_id
        self.visual_geoms = np.flatnonzero(in_robot & (m.geom_group == ROBOT_VISUAL_GROUP))

        self._kp = self.robot.joint_stiffness.numpy()
        self._kd = self.robot.joint_damping.numpy()
        self._effort = self.robot.effort_limit.numpy()
        self._head_target = (0.0, 0.0)  # (pitch, yaw)

        self._markings = map_markings(sim_cfg.field)
        self._lines = field_lines(sim_cfg.field)
        self._posts = goalposts(sim_cfg.field)

        self.kick_model = VisualKickModel()
        self._was_kicking = False
        self.yolo = self.world.yolo
        self._rng = np.random.default_rng()

        sx, sy, syaw = spawn if spawn is not None else sim_cfg.spawn_xy_yaw
        self.odometry = OdometryTracker(Pose2D(sx, sy, syaw))
        self.reset_robot(*self.frame.pose_tuple(sx, sy, syaw))

    # ---- resets -------------------------------------------------------------
    def reset_robot(self, x: float, y: float, yaw: float) -> None:
        """Stand the robot upright at a world-frame pose."""
        d, q0 = self.mj_data, self._root_q
        d.qpos[q0:q0 + 3] = [x, y, self.cfg.mujoco.init_pos[2]]
        d.qpos[q0 + 3:q0 + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qpos[self._q] = self.robot.default_joint_pos.numpy()
        d.qvel[self._dq.start - 6:self._dq.start] = 0.0
        d.qvel[self._dq] = 0.0
        mujoco.mj_forward(self.mj_model, d)
        self.update_state()
        if self.is_running:
            self.policy.reset()

    def restart_odometry(self) -> None:
        """Brain was picked up and put down: its odometer starts again from where the robot now stands."""
        self.odometry = OdometryTracker(self.team_pose())

    def reset_ball(self, x: float, y: float) -> None:
        self.world.reset_ball(x, y)

    # ---- booster_deploy controller contract ---------------------------------
    def update_state(self) -> None:
        d, data, q0 = self.mj_data, self.robot.data, self._root_q
        dq0 = self._dq.start - 6
        data.joint_pos = torch.from_numpy(d.qpos[self._q].astype(np.float32))
        data.joint_vel = torch.from_numpy(d.qvel[self._dq].astype(np.float32))
        data.feedback_torque = torch.from_numpy(d.qfrc_actuator[self._dq].astype(np.float32))
        data.root_pos_w = torch.from_numpy(d.qpos[q0:q0 + 3].astype(np.float32))
        data.root_quat_w = torch.from_numpy(d.qpos[q0 + 3:q0 + 7].astype(np.float32))
        data.root_lin_vel_b = torch.from_numpy(d.qvel[dq0:dq0 + 3].astype(np.float32))
        data.root_ang_vel_b = torch.from_numpy(d.qvel[dq0 + 3:dq0 + 6].astype(np.float32))

    def next_targets(self) -> np.ndarray:
        """Policy inference for this tick: joint position targets, head overridden by brain's RotateHead."""
        self.update_state()
        targets = self.policy_step().detach().cpu().numpy().astype(np.float64)
        targets[HEAD_PITCH_IDX], targets[HEAD_YAW_IDX] = self._head_target
        return targets

    def apply_pd(self, targets: np.ndarray) -> None:
        """One physics substep's motor torques (the world then steps once for everybody)."""
        d = self.mj_data
        q, dq = d.qpos[self._q], d.qvel[self._dq]
        # Only the robot motors; the shot sliders follow after the last robot's.
        d.ctrl[self._act0:self._act0 + self._nu_robot] = np.clip(
            self._kp * (targets - q) - self._kd * dq, -self._effort, self._effort
        )

    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        targets = dof_targets.detach().cpu().numpy().astype(np.float64)
        targets[HEAD_PITCH_IDX], targets[HEAD_YAW_IDX] = self._head_target
        for _ in range(self.decimation):
            self.apply_pd(targets)
            mujoco.mj_step(self.mj_model, self.mj_data)

    def step(self) -> None:
        """Alone in its world: one tick. In a match call World.step() instead (it steps every robot together)."""
        self.world.step()

    # ---- inputs from brain -------------------------------------------------
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
                self._launch_ball(out.strike_speed, pose.yaw + out.strike_dir)
        else:
            vx, vy, vyaw = loco.effective_velocity()
        self._was_kicking = loco.kicking()

        self.vel_command.lin_vel_x = vx
        self.vel_command.lin_vel_y = vy
        self.vel_command.ang_vel_yaw = vyaw
        self._head_target = (loco.head_pitch, loco.head_yaw)
        if loco.getup_requested:
            loco.getup_requested = False
            if self.fallen():
                pose = self.robot_pose()
                self.reset_robot(pose.x, pose.y, pose.yaw)

    # ---- the shared ball and shot panel (kept here so a lone robot reads like it always did) ----
    def shot_panel_values(self) -> dict[str, float]:
        return self.world.shot_panel_values()

    def shoot(self, shot: Shot) -> None:
        self.world.shoot(shot)

    def _launch_ball(self, speed: float, world_dir: float) -> None:
        self.world.launch_ball(speed, world_dir)

    def ball_position(self) -> np.ndarray:
        return self.world.ball_position()

    def ball_velocity(self) -> np.ndarray:
        return self.world.ball_velocity()

    @property
    def n_robot_ctrl(self) -> int:
        """Robot motors are ctrl[:n]; the shot-launcher sliders follow."""
        return self.world.n_robot_ctrl

    # ---- outputs to brain --------------------------------------------------
    def fallen(self) -> bool:
        trunk_zz = self.mj_data.xmat[self._trunk_body].reshape(3, 3)[2, 2]
        return bool(trunk_zz < UPRIGHT_MIN_ZZ)

    def robot_pose(self) -> Pose2D:
        """World-frame pose (the home team's frame)."""
        q = self.mj_data.qpos[self._root_q:self._root_q + 7]
        return Pose2D(float(q[0]), float(q[1]), yaw_from_quat_wxyz(q[3:7]))

    def team_pose(self) -> Pose2D:
        """The pose in this robot's own team frame, which is what its brain works in."""
        return self.frame.pose(self.robot_pose())

    def odometer(self) -> tuple[float, float, float]:
        return self.odometry.update(self.team_pose())

    def imu(self):
        d = self.mj_data
        rpy = rpy_from_quat_wxyz(d.qpos[self._root_q + 3:self._root_q + 7])
        rpy = (rpy[0], rpy[1], wrap_pi(self.frame.direction(rpy[2])))
        gyro = tuple(float(v) for v in d.sensordata[self._gyro:self._gyro + 3])
        acc = tuple(float(v) for v in d.sensordata[self._acc:self._acc + 3])
        return rpy, gyro, acc

    def joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.mj_data
        return d.qpos[self._q].copy(), d.qvel[self._dq].copy(), d.qfrc_actuator[self._dq].copy()

    def _base(self) -> tuple[np.ndarray, np.ndarray]:
        """Ground base frame: trunk xy at z = 0, trunk yaw only (brain's ground_height = 0)."""
        pose = self.robot_pose()
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return np.array([pose.x, pose.y, 0.0]), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def head_pose_in_base(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.mj_data
        origin, r_base = self._base()
        pos = r_base.T @ (d.xpos[self._head_body] - origin)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (r_base.T @ d.xmat[self._head_body].reshape(3, 3)).reshape(9))
        return pos, quat

    def camera(self) -> Camera:
        """The head camera in this robot's team frame."""
        d = self.mj_data
        r_opt = d.cam_xmat[self.cam_id].reshape(3, 3) @ OPTICAL_FROM_MUJOCO_CAM
        return Camera(self.sim_cfg.camera, self.frame.vec3(d.cam_xpos[self.cam_id]), self.frame.rot(r_opt))

    def perceive(self, rng=None, rgb: np.ndarray | None = None,
                 others: list[tuple[float, float]] | None = None) -> tuple[list[Detection], list[LineObs]]:
        """What the camera would report. others: world-frame (x, y) of every other robot, seen as 'Opponent'."""
        c = self.sim_cfg
        rng = rng if rng is not None else self._rng
        cam, robot = self.camera(), self.team_pose()
        dets: list[Detection] = []
        if self.yolo is not None and rgb is not None:
            dets += self.yolo.detect(rgb, cam, robot, c.ball_radius)
        else:
            bx, by = self.frame.point(*self.ball_position()[:2])
            ball = detect_ball(
                cam, robot, (bx, by, self.ball_position()[2]), c.ball_radius, c.max_ball_range, c.detection_noise_std, rng
            )
            if ball is not None:
                dets.append(ball)
        dets += detect_goalposts(
            cam, robot, self._posts, c.goal_height, c.goalpost_radius, c.max_goalpost_range, c.detection_noise_std, rng
        )
        dets += detect_markings(cam, robot, self._markings, c.max_marking_range, c.detection_noise_std, rng)
        if others:
            dets += detect_robots(cam, robot, [self.frame.point(x, y) for x, y in others], c.max_ball_range,
                                  c.detection_noise_std, rng)
        return dets, visible_lines(cam, robot, self._lines, c.max_line_range)

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        return self.world.render(self)
