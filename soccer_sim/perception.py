import math
from dataclasses import dataclass

import numpy as np

from .config import CameraCfg
from .field import Segment
from .geometry import world_to_robot_2d

MARKING_HALF_SIZE_M = 0.15
CONFIDENCE = 0.9
MIN_DEPTH_M = 0.05


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # xmin, ymin, xmax, ymax in pixels
    pos_robot: tuple[float, float]  # ground projection, robot (base) frame, metres
    target_uv: tuple[float, float] | None = None


@dataclass(frozen=True)
class LineObs:
    robot: tuple[float, float, float, float]  # x0, y0, x1, y1 robot frame
    uv: tuple[float, float, float, float]  # u0, v0, u1, v1 pixels


class Camera:
    """Pinhole camera in the optical convention: x right, y down, z forward."""

    def __init__(self, cfg: CameraCfg, pos_w, rot_w_optical):
        self.cfg = cfg
        self.pos = np.asarray(pos_w, dtype=float).reshape(3)
        self.rot = np.asarray(rot_w_optical, dtype=float).reshape(3, 3)
        self.fx, self.fy, self.cx, self.cy = cfg.fx, cfg.fy, cfg.cx, cfg.cy

    def to_cam(self, p_w) -> np.ndarray:
        return self.rot.T @ (np.asarray(p_w, dtype=float).reshape(3) - self.pos)

    def project(self, p_w) -> tuple[float, float, float] | None:
        d = self.to_cam(p_w)
        if d[2] <= MIN_DEPTH_M:
            return None
        u = self.fx * d[0] / d[2] + self.cx
        v = self.fy * d[1] / d[2] + self.cy
        if not (0.0 <= u < self.cfg.width and 0.0 <= v < self.cfg.height):
            return None
        return float(u), float(v), float(d[2])


def _bbox(cam: Camera, u: float, v_top: float, v_bottom: float, half_w: float) -> tuple[int, int, int, int]:
    w, h = cam.cfg.width - 1, cam.cfg.height - 1
    return (
        int(max(0.0, min(w, u - half_w))),
        int(max(0.0, min(h, v_top))),
        int(max(0.0, min(w, u + half_w))),
        int(max(0.0, min(h, v_bottom))),
    )


def _robot_xy(robot: Pose2D, x: float, y: float, noise_std: float, rng) -> tuple[float, float]:
    rx, ry = world_to_robot_2d(x, y, robot.x, robot.y, robot.yaw)
    if noise_std > 0.0 and rng is not None:
        rx += float(rng.normal(0.0, noise_std))
        ry += float(rng.normal(0.0, noise_std))
    return rx, ry


def _in_range(robot: Pose2D, x: float, y: float, max_range: float) -> bool:
    return math.hypot(x - robot.x, y - robot.y) <= max_range


def detect_ball(cam, robot, ball_w, radius, max_range, noise_std=0.0, rng=None) -> Detection | None:
    bx, by, bz = (float(v) for v in ball_w)
    if not _in_range(robot, bx, by, max_range):
        return None
    p = cam.project((bx, by, bz))
    if p is None:
        return None
    u, v, depth = p
    half_u, half_v = cam.fx * radius / depth, cam.fy * radius / depth
    return Detection(
        "Ball", CONFIDENCE, _bbox(cam, u, v - half_v, v + half_v, half_u), _robot_xy(robot, bx, by, noise_std, rng)
    )


def detect_goalposts(cam, robot, posts, height, radius, max_range, noise_std=0.0, rng=None) -> list[Detection]:
    out = []
    for gx, gy in posts:
        if not _in_range(robot, gx, gy, max_range):
            continue
        bottom = cam.project((gx, gy, 0.0))
        if bottom is None:
            continue
        u, v_bottom, depth = bottom
        top = cam.to_cam((gx, gy, height))
        v_top = cam.fy * top[1] / top[2] + cam.cy if top[2] > MIN_DEPTH_M else 0.0
        out.append(
            Detection(
                "Goalpost",
                CONFIDENCE,
                _bbox(cam, u, v_top, v_bottom, cam.fx * radius / depth),
                _robot_xy(robot, gx, gy, noise_std, rng),
            )
        )
    return out


ROBOT_HEIGHT_M = 1.15  # T2 standing height
ROBOT_HALF_WIDTH_M = 0.25


def detect_robots(cam, robot, others_xy, max_range, noise_std=0.0, rng=None) -> list[Detection]:
    """Other robots as brain's 'Opponent' objects: the real model cannot tell teammates from opponents either.

    others_xy are positions in the same frame as `cam` and `robot`; the robot itself is not in the list.
    """
    out = []
    for ox, oy in others_xy:
        if not _in_range(robot, ox, oy, max_range):
            continue
        feet = cam.to_cam((ox, oy, 0.0))
        head = cam.to_cam((ox, oy, ROBOT_HEIGHT_M))
        depth = float(max(feet[2], head[2]))
        if depth <= MIN_DEPTH_M or min(feet[2], head[2]) <= MIN_DEPTH_M:
            continue
        u = cam.fx * feet[0] / feet[2] + cam.cx
        if not 0.0 <= u < cam.cfg.width:
            continue
        v_feet = cam.fy * feet[1] / feet[2] + cam.cy
        v_head = cam.fy * head[1] / head[2] + cam.cy
        if max(v_feet, v_head) < 0.0 or min(v_feet, v_head) >= cam.cfg.height:
            continue
        out.append(
            Detection(
                "Opponent",
                CONFIDENCE,
                _bbox(cam, u, min(v_head, v_feet), max(v_head, v_feet), cam.fx * ROBOT_HALF_WIDTH_M / feet[2]),
                _robot_xy(robot, ox, oy, noise_std, rng),
            )
        )
    return out


def detect_markings(cam, robot, markings, max_range, noise_std=0.0, rng=None) -> list[Detection]:
    out = []
    for label, mx, my in markings:
        if not _in_range(robot, mx, my, max_range):
            continue
        p = cam.project((mx, my, 0.0))
        if p is None:
            continue
        u, v, depth = p
        half_u, half_v = cam.fx * MARKING_HALF_SIZE_M / depth, cam.fy * MARKING_HALF_SIZE_M / depth
        out.append(
            Detection(
                label,
                CONFIDENCE,
                _bbox(cam, u, v - half_v, v + half_v, half_u),
                _robot_xy(robot, mx, my, noise_std, rng),
                target_uv=(u, v),
            )
        )
    return out


_LINE_SAMPLES: dict[int, tuple] = {}


def _line_samples(segments: list[Segment], step: float):
    """Sample points along every segment, once per segment list: (points (N, 2), segment index of each point)."""
    cached = _LINE_SAMPLES.get(id(segments))
    if cached is not None and cached[0] is segments and cached[1] == step:
        return cached[2], cached[3]
    pts, seg = [], []
    for k, ((x0, y0), (x1, y1)) in enumerate(segments):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
        t = np.arange(n) / (n - 1)
        pts.append(np.column_stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]))
        seg.append(np.full(n, k))
    points, owner = np.concatenate(pts), np.concatenate(seg)
    _LINE_SAMPLES[id(segments)] = (segments, step, points, owner)
    return points, owner


def visible_lines(cam, robot, segments: list[Segment], max_range: float, step: float = 0.05) -> list[LineObs]:
    """Sample each segment; every contiguous visible run becomes one observed segment (vectorised: this runs
    for every robot of a match several times a second)."""
    pts, owner = _line_samples(segments, step)
    n = len(pts)
    d = (np.column_stack([pts, np.zeros(n)]) - cam.pos) @ cam.rot  # camera.rot.T @ (p - pos), for every sample
    z = d[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * d[:, 0] / z + cam.cx
        v = cam.fy * d[:, 1] / z + cam.cy
    ok = (
        (np.hypot(pts[:, 0] - robot.x, pts[:, 1] - robot.y) <= max_range)
        & (z > MIN_DEPTH_M) & (u >= 0.0) & (u < cam.cfg.width) & (v >= 0.0) & (v < cam.cfg.height)
    )
    same_prev = np.r_[False, owner[1:] == owner[:-1]]
    same_next = np.r_[owner[:-1] == owner[1:], False]
    starts = np.flatnonzero(ok & ~(np.r_[False, ok[:-1]] & same_prev))
    ends = np.flatnonzero(ok & ~(np.r_[ok[1:], False] & same_next))
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    out = []
    for a, b in zip(starts, ends):
        if b <= a:
            continue
        (ax, ay), (bx, by) = pts[a], pts[b]
        pa = (c * (ax - robot.x) + s * (ay - robot.y), -s * (ax - robot.x) + c * (ay - robot.y))
        pb = (c * (bx - robot.x) + s * (by - robot.y), -s * (bx - robot.x) + c * (by - robot.y))
        out.append(LineObs((pa[0], pa[1], pb[0], pb[1]), (float(u[a]), float(v[a]), float(u[b]), float(v[b]))))
    return out
