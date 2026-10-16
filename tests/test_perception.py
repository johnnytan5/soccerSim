import math

import numpy as np
import pytest

from soccer_sim.config import CameraCfg
from soccer_sim.geometry import rpy_from_quat_wxyz, world_to_robot_2d, wrap_pi
from soccer_sim.odometry import OdometryTracker
from soccer_sim.perception import (
    Camera, Pose2D, detect_ball, detect_goalposts, detect_markings, visible_lines,
)

CFG = CameraCfg()
ORIGIN = Pose2D(0.0, 0.0, 0.0)


def level_camera(x=0.0, y=0.0, z=1.2, yaw=0.0):
    """Camera looking horizontally along the robot heading; optical axes in world."""
    right = [math.sin(yaw), -math.cos(yaw), 0.0]
    down = [0.0, 0.0, -1.0]
    forward = [math.cos(yaw), math.sin(yaw), 0.0]
    return Camera(CFG, np.array([x, y, z]), np.column_stack([right, down, forward]))


def test_world_to_robot_2d_rotates_into_robot_frame():
    x, y = world_to_robot_2d(1.0, 4.0, 1.0, 1.0, math.pi / 2)
    assert (x, y) == pytest.approx((3.0, 0.0), abs=1e-9)


def test_rpy_from_quat_roll_only():
    roll = 0.3
    q = (math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0)
    assert rpy_from_quat_wxyz(q) == pytest.approx((0.3, 0.0, 0.0), abs=1e-9)


def test_odometry_is_spawn_relative():
    odom = OdometryTracker(Pose2D(-6.0, -7.5, math.pi / 2))
    assert odom.update(Pose2D(-6.0, -5.5, math.pi / 2)) == pytest.approx((2.0, 0.0, 0.0), abs=1e-9)


def test_odometry_theta_wraps():
    odom = OdometryTracker(Pose2D(0.0, 0.0, 3.0))
    _, _, theta = odom.update(Pose2D(0.0, 0.0, -3.0))
    assert -math.pi < theta <= math.pi
    assert theta == pytest.approx(wrap_pi(-6.0))


def test_ball_in_front_is_detected_in_robot_frame():
    det = detect_ball(level_camera(), ORIGIN, (3.0, 0.0, 0.11), 0.11, 8.0)
    assert det is not None and det.label == "Ball"
    assert det.pos_robot == pytest.approx((3.0, 0.0), abs=1e-9)
    xmin, ymin, xmax, ymax = det.bbox
    assert xmin < CFG.cx < xmax
    assert (ymin + ymax) / 2 > CFG.cy  # ball is below the camera's optical axis


def test_ball_with_rotated_robot():
    cam = level_camera(x=1.0, y=1.0, yaw=math.pi / 2)
    det = detect_ball(cam, Pose2D(1.0, 1.0, math.pi / 2), (1.0, 4.0, 0.11), 0.11, 8.0)
    assert det is not None
    assert det.pos_robot == pytest.approx((3.0, 0.0), abs=1e-9)


def test_ball_behind_camera_is_not_detected():
    assert detect_ball(level_camera(), ORIGIN, (-3.0, 0.0, 0.11), 0.11, 8.0) is None


def test_ball_outside_horizontal_fov_is_not_detected():
    assert detect_ball(level_camera(), ORIGIN, (1.0, 5.0, 0.11), 0.11, 8.0) is None


def test_ball_beyond_range_is_not_detected():
    assert detect_ball(level_camera(), ORIGIN, (9.0, 0.0, 0.11), 0.11, 8.0) is None


def test_point_exactly_on_right_border_is_outside():
    cam = level_camera()
    # u = fx * right / forward + cx == width  ->  right = (width - cx) * forward / fx; right = -y in world
    forward = 3.0
    right = (CFG.width - cam.cx) * forward / cam.fx
    assert cam.project((forward, -right, 1.2)) is None


def test_penalty_point_has_pixel_target():
    cam = level_camera()
    dets = detect_markings(cam, ORIGIN, [("PenaltyPoint", 3.0, 0.0)], 5.0)
    assert len(dets) == 1 and dets[0].label == "PenaltyPoint"
    u, v = dets[0].target_uv
    assert u == pytest.approx(cam.cx)
    assert v == pytest.approx(cam.cy + cam.fy * 1.2 / 3.0)


def test_goalpost_uses_ground_point():
    dets = detect_goalposts(level_camera(), ORIGIN, [(6.0, 1.3)], 1.2, 0.05, 12.0)
    assert len(dets) == 1 and dets[0].label == "Goalpost"
    assert dets[0].pos_robot == pytest.approx((6.0, 1.3), abs=1e-9)


def test_line_crossing_view_is_clipped_to_fov():
    cam = level_camera()
    lines = visible_lines(cam, ORIGIN, [((2.0, -10.0), (2.0, 10.0))], 5.0)
    assert len(lines) == 1
    x0, y0, x1, y1 = lines[0].robot
    half_width_at_2m = 2.0 * cam.cx / cam.fx
    assert x0 == pytest.approx(2.0) and x1 == pytest.approx(2.0)
    assert abs(y0) <= half_width_at_2m + 1e-6 and abs(y1) <= half_width_at_2m + 1e-6
    assert abs(y1 - y0) > 1.0


def test_line_behind_robot_is_dropped():
    assert visible_lines(level_camera(), ORIGIN, [((-2.0, -3.0), (-2.0, 3.0))], 5.0) == []


def test_all_outputs_finite():
    cam = level_camera()
    dets = [detect_ball(cam, ORIGIN, (2.0, 0.3, 0.11), 0.11, 8.0)]
    dets += detect_markings(cam, ORIGIN, [("LCross", 3.0, 1.0), ("TCross", 0.0, 0.0)], 5.0)
    lines = visible_lines(cam, ORIGIN, [((2.0, -1.0), (4.0, 1.0)), ((0.0, 0.0), (0.0, 0.0))], 5.0)
    for d in dets:
        assert d is not None
        assert all(math.isfinite(v) for v in (*d.bbox, *d.pos_robot))
    for line in lines:
        assert all(math.isfinite(v) for v in (*line.robot, *line.uv))


def _reference_visible_lines(cam, robot, segments, max_range, step=0.05):
    """The original point-by-point version, kept to prove the vectorised one matches."""
    from soccer_sim.perception import LineObs, _in_range
    out = []
    for (x0, y0), (x1, y1) in segments:
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
        run = []

        def flush():
            if len(run) >= 2:
                (ax, ay, au, av), (bx, by, bu, bv) = run[0], run[-1]
                pa = world_to_robot_2d(ax, ay, robot.x, robot.y, robot.yaw)
                pb = world_to_robot_2d(bx, by, robot.x, robot.y, robot.yaw)
                out.append(LineObs((pa[0], pa[1], pb[0], pb[1]), (au, av, bu, bv)))
            run.clear()

        for i in range(n):
            t = i / (n - 1)
            px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            p = cam.project((px, py, 0.0)) if _in_range(robot, px, py, max_range) else None
            if p is None:
                flush()
            else:
                run.append((px, py, p[0], p[1]))
        flush()
    return out


@pytest.mark.parametrize("seed", range(6))
def test_vectorised_lines_match_the_point_by_point_version(seed):
    from soccer_sim.field import field_lines
    from soccer_sim.config import FieldDims
    rng = np.random.default_rng(seed)
    segs = field_lines(FieldDims())
    x, y, yaw = rng.uniform(-9, 9), rng.uniform(-6, 6), rng.uniform(-math.pi, math.pi)
    pitch = rng.uniform(0.2, 0.9)
    forward = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    down = np.cross(forward, right)
    cam = Camera(CFG, np.array([x, y, 1.1]), np.column_stack([right, down, forward]))
    robot = Pose2D(x, y, yaw)
    a, b = visible_lines(cam, robot, segs, 5.0), _reference_visible_lines(cam, robot, segs, 5.0)
    assert len(a) == len(b) and len(a) > 0
    for p, q in zip(a, b):
        assert p.robot == pytest.approx(q.robot, abs=1e-6) and p.uv == pytest.approx(q.uv, abs=1e-6)
