"""Field geometry that needs neither MuJoCo nor the robot assets: markings, lines and goalposts.

Kept apart from field.py (which builds the MuJoCo world) so the 2D simulator and the launcher-side code can use
them without importing mujoco, torch or booster_deploy.
"""
import math

from .config import FieldDims

Segment = tuple[tuple[float, float], tuple[float, float]]


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
