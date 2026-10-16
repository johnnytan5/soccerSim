import math

import pytest

from soccer_sim.placement import HOME, robot_pose_from_dict


def test_empty_payload_is_keepers_home():
    assert robot_pose_from_dict({}) == HOME


def test_valid_pose_and_yaw_is_wrapped():
    x, y, yaw = robot_pose_from_dict({"x": -3.0, "y": 2.5, "yaw": 3 * math.pi})
    assert (x, y) == (-3.0, 2.5) and yaw == pytest.approx(math.pi) or yaw == pytest.approx(-math.pi)
    assert robot_pose_from_dict({"x": 1, "y": 1})[2] == 0.0


@pytest.mark.parametrize("bad", [
    {"x": 1.0}, {"y": 1.0}, {"x": 99, "y": 0}, {"x": 0, "y": -50}, {"x": float("nan"), "y": 0},
    {"x": 0, "y": 0, "yaw": float("inf")}, {"x": 0, "y": 0, "z": 1},
])
def test_rejects(bad):
    with pytest.raises(ValueError):
        robot_pose_from_dict(bad)


def test_non_numeric_is_a_value_error():
    with pytest.raises(ValueError):
        robot_pose_from_dict({"x": "left", "y": 0})
