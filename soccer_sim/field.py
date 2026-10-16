import math

import mujoco
import numpy as np
from booster_assets import BOOSTER_ASSETS_DIR
from booster_deploy.robots.t2 import T2_31DOF_CFG

from .config import SHOT_SLIDERS, FieldDims, SimConfig

Segment = tuple[tuple[float, float], tuple[float, float]]

# MuJoCo cameras look along -z with y up; the optical frame looks along +z with y down.
OPTICAL_FROM_MUJOCO_CAM = np.diag([1.0, -1.0, -1.0])
# Robot motors live in this actuator group; the viewer hides it so its Control panel shows only shot sliders.
ROBOT_ACTUATOR_GROUP = 3
ROBOT_VISUAL_GROUP = 1  # booster_assets T2 MJCF puts visual meshes in group 1 (collisions in 3, field/ball in 0)


def map_markings(fd: FieldDims) -> list[tuple[str, float, float]]:
    """Same 26 markings, same coordinates, as BrainConfig::calcMapMarkings()."""
    out: list[tuple[str, float, float]] = []
    for xs in (-1.0, 1.0):
        for ys in (1.0, -1.0):
            out += [
                ("LCross", xs * fd.length / 2, ys * fd.width / 2),
                ("TCross", xs * fd.length / 2, ys * fd.penalty_area_width / 2),
                ("TCross", xs * fd.length / 2, ys * fd.goal_area_width / 2),
                ("LCross", xs * (fd.length / 2 - fd.penalty_area_length), ys * fd.penalty_area_width / 2),
                ("LCross", xs * (fd.length / 2 - fd.goal_area_length), ys * fd.goal_area_width / 2),
            ]
    for ys in (1.0, -1.0):
        out += [("TCross", 0.0, ys * fd.width / 2), ("XCross", 0.0, ys * fd.circle_radius)]
    for xs in (-1.0, 1.0):
        out.append(("PenaltyPoint", xs * (fd.length / 2 - fd.penalty_dist), 0.0))
    return out


def field_lines(fd: FieldDims, circle_segments: int = 24) -> list[Segment]:
    half_l, half_w = fd.length / 2, fd.width / 2
    segs: list[Segment] = [
        ((-half_l, -half_w), (half_l, -half_w)),
        ((-half_l, half_w), (half_l, half_w)),
        ((-half_l, -half_w), (-half_l, half_w)),
        ((half_l, -half_w), (half_l, half_w)),
        ((0.0, -half_w), (0.0, half_w)),
    ]
    for xs in (-1.0, 1.0):
        for depth, box_half_w in (
            (fd.penalty_area_length, fd.penalty_area_width / 2),
            (fd.goal_area_length, fd.goal_area_width / 2),
        ):
            x_in = xs * (half_l - depth)
            segs += [
                ((xs * half_l, -box_half_w), (x_in, -box_half_w)),
                ((xs * half_l, box_half_w), (x_in, box_half_w)),
                ((x_in, -box_half_w), (x_in, box_half_w)),
            ]
    r = fd.circle_radius
    for i in range(circle_segments):
        a0 = 2 * math.pi * i / circle_segments
        a1 = 2 * math.pi * (i + 1) / circle_segments
        segs.append(((r * math.cos(a0), r * math.sin(a0)), (r * math.cos(a1), r * math.sin(a1))))
    return segs


def goalposts(fd: FieldDims) -> list[tuple[float, float]]:
    return [(xs * fd.length / 2, ys * fd.goal_width / 2) for xs in (-1.0, 1.0) for ys in (1.0, -1.0)]


def robot_mjcf_path() -> str:
    return T2_31DOF_CFG.mjcf_path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def robot_prefixes(n_robots: int) -> list[str]:
    """Name prefix of each robot in a match world: the first robot keeps the plain names."""
    return [""] + [f"r{i}_" for i in range(1, n_robots)]


def _add_robot_extras(spec: mujoco.MjSpec, cfg: SimConfig) -> None:
    """Head camera and IMU sensors, added to one robot's spec (before it is attached, if it is a child)."""
    cam_to_head = cfg.camera.cam_to_head_matrix()
    cam_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cam_quat, (cam_to_head[:3, :3] @ OPTICAL_FROM_MUJOCO_CAM).reshape(9))
    spec.body("head_pitch_link").add_camera(
        name="head_rgb",
        pos=cam_to_head[:3, 3].tolist(),
        quat=cam_quat.tolist(),
        fovy=math.degrees(2.0 * math.atan(cfg.camera.height / 2 / cfg.camera.fy)),
    )

    # The T2 MJCF already has an "imu" site at the trunk origin; add noise-free sensors on it.
    spec.add_sensor(name="imu_gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(
        name="imu_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu"
    )


def _strip_visual_meshes(spec: mujoco.MjSpec) -> None:
    """Drop a robot's ~1 M-triangle visual meshes (geom group 1) and the mesh assets only they used.

    The collision shapes stay, so bodies, joints, actuators and qpos are exactly as in the full model.
    """
    for geom in [g for g in spec.geoms if g.group == ROBOT_VISUAL_GROUP]:
        spec.delete(geom)
    used = {g.meshname for g in spec.geoms if g.meshname}
    for mesh in [m for m in spec.meshes if m.name not in used]:
        spec.delete(mesh)


def build_world(cfg: SimConfig, n_robots: int = 1, strip_visual: bool = False) -> mujoco.MjModel:
    """Field + ball + n robots. Robot 0 keeps the plain names; the others are attached with `robot_prefixes`.

    strip_visual: the viewer's copy of a match world. Same bodies, joints and qpos layout, but without the robots'
    visual meshes: the viewer copies the whole model on every sync, and ten robots are ~10 M triangles of it.
    """
    spec = mujoco.MjSpec.from_file(robot_mjcf_path())
    if strip_visual:
        _strip_visual_meshes(spec)
    world = spec.worldbody
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)

    def _material(name: str, texture: str, **kwargs):
        mat = spec.add_material(name=name, **kwargs)
        textures = [""] * len(mat.textures)
        textures[rgb_role] = texture
        mat.textures = textures

    # The YOLO model was trained on sim imagery with a patterned ball and textured grass;
    # a plain white sphere on flat green is never detected (2026-10-10 spike).
    spec.add_texture(
        name="turf_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        mark=mujoco.mjtMark.mjMARK_RANDOM, random=0.25, rgb1=[0.13, 0.5, 0.18], markrgb=[0.2, 0.62, 0.25],
        width=512, height=512,
    )
    _material("turf_mat", "turf_tex", texrepeat=[8.0, 6.0])
    spec.add_texture(
        name="ball_tex", type=mujoco.mjtTexture.mjTEXTURE_CUBE, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[1.0, 1.0, 1.0], rgb2=[0.05, 0.05, 0.05], width=64, height=64,
    )
    _material("ball_mat", "ball_tex")

    floor = spec.geom("floor")
    floor.material = "turf_mat"
    # Finite render size so the texture repeat is defined; planes still collide infinitely.
    floor.size = [cfg.field.length / 2 + 3.0, cfg.field.width / 2 + 3.0, 0.1]

    for i, ((x0, y0), (x1, y1)) in enumerate(field_lines(cfg.field)):
        length = math.hypot(x1 - x0, y1 - y0)
        world.add_geom(
            name=f"line_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[length / 2, cfg.line_width / 2, 0.0005],
            pos=[(x0 + x1) / 2, (y0 + y1) / 2, 0.0005],
            quat=_yaw_quat(math.atan2(y1 - y0, x1 - x0)),
            rgba=[1.0, 1.0, 1.0, 1.0],
            contype=0,
            conaffinity=0,
        )

    for i, (gx, gy) in enumerate(goalposts(cfg.field)):
        world.add_geom(
            name=f"goalpost_{i}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[cfg.goalpost_radius, cfg.goal_height / 2, 0.0],
            pos=[gx, gy, cfg.goal_height / 2],
            rgba=[1.0, 1.0, 1.0, 1.0],
        )
    for xs, name in ((-1.0, "own"), (1.0, "opp")):
        world.add_geom(
            name=f"crossbar_{name}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[cfg.goalpost_radius, cfg.field.goal_width / 2, 0.0],
            pos=[xs * cfg.field.length / 2, 0.0, cfg.goal_height],
            quat=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],  # cylinder axis z -> y
            rgba=[1.0, 1.0, 1.0, 1.0],
        )

    ball = world.add_body(name="ball", pos=[cfg.ball_xy[0], cfg.ball_xy[1], cfg.ball_radius])
    ball.add_freejoint(name="ball_joint")
    ball.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[cfg.ball_radius, 0.0, 0.0],
        mass=cfg.ball_mass,
        material="ball_mat",
        condim=6,
        # ponytail: rolling friction guessed for artificial turf; tune from real ball roll-out distance.
        friction=[0.8, 0.02, 0.01],
    )

    _add_robot_extras(spec, cfg)
    for i, prefix in enumerate(robot_prefixes(n_robots)[1:], start=1):
        child = mujoco.MjSpec.from_file(robot_mjcf_path())
        if strip_visual:
            _strip_visual_meshes(child)
        for geom in [g for g in child.geoms if g.name == "floor"]:
            child.delete(geom)  # one pitch for everybody
        for light in list(child.lights):
            child.delete(light)
        _add_robot_extras(child, cfg)
        frame = world.add_frame(pos=[-6.0 + i, -7.5, 0.0])
        spec.attach(child, prefix=prefix, frame=frame)

    for act in spec.actuators:  # only the robot motors (31 each) exist at this point
        act.group = ROBOT_ACTUATOR_GROUP
    for name, lo, hi, _default in SHOT_SLIDERS:
        # ponytail: zero-gain actuators used purely as viewer sliders; they never apply force.
        slider = spec.add_actuator(
            name=name,
            target="aa_head_yaw_joint",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            ctrlrange=[lo, hi],
            group=0,
        )
        slider.gainprm = [0.0] * len(slider.gainprm)
        slider.biasprm = [0.0] * len(slider.biasprm)
    return spec.compile()
