import numpy as np
import pytest
from brain.msg import Kick

from soccer_sim.config import CameraCfg
from soccer_sim.perception import Detection, LineObs
from soccer_sim.referee import RefereeSnapshot
from soccer_sim_bridge import msg_conv

pytestmark = pytest.mark.ros
STAMP = msg_conv.stamp_from_ns(1_500_000_000)


def test_detections_fill_the_fields_brain_reads():
    det = Detection("Ball", 0.9, (10, 20, 30, 40), (2.0, -0.5))
    mark = Detection("PenaltyPoint", 0.9, (1, 2, 3, 4), (3.0, 0.0), target_uv=(272.0, 300.0))
    msg = msg_conv.detections(STAMP, [det, mark])
    ball, pen = msg.detected_objects
    # Robotedge vision publishes confidence in percent (vision_node.cpp: confidence * 100) and brain
    # rejects < 50 (detectProcessMarkings, strategy.ball_confidence_threshold).
    assert ball.label == "Ball" and ball.confidence == pytest.approx(90.0)
    assert (ball.xmin, ball.ymin, ball.xmax, ball.ymax) == (10, 20, 30, 40)
    assert list(ball.position_projection) == pytest.approx([2.0, -0.5])
    assert list(ball.target_uv) == []
    assert list(pen.target_uv) == pytest.approx([272.0, 300.0])
    assert msg.header.stamp.sec == 1 and msg.header.stamp.nanosec == 500_000_000


def test_line_segments_are_flat_and_paired():
    lines = [LineObs((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)), LineObs((0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 2.0, 2.0))]
    msg = msg_conv.line_segments(STAMP, lines)
    assert list(msg.coordinates) == pytest.approx([1, 2, 3, 4, 0, 0, 1, 0])
    assert list(msg.coordinates_uv) == pytest.approx([5, 6, 7, 8, 1, 1, 2, 2])


def test_low_state_puts_head_yaw_and_pitch_first():
    q = np.arange(31, dtype=float) * 0.01
    msg = msg_conv.low_state((0.1, 0.2, 0.3), (0.0, 0.0, 0.5), (0.0, 0.0, 9.81), q, np.zeros(31), np.zeros(31))
    assert len(msg.motor_state_serial) == 31
    assert msg.motor_state_serial[0].q == pytest.approx(0.0)
    assert msg.motor_state_serial[1].q == pytest.approx(0.01)
    assert list(msg.imu_state.rpy) == pytest.approx([0.1, 0.2, 0.3])


def test_game_control_carries_state_set_play_and_teams():
    ref = RefereeSnapshot(state=3, set_play=6, kicking_team=1, stopped=True, secondary_time=25)
    msg = msg_conv.game_control(ref, team_id=131, opp_team_id=1, goalkeeper_id=1)
    assert (msg.state, msg.set_play, msg.kicking_team, msg.stopped, msg.secondary_time) == (3, 6, 1, True, 25)
    assert (msg.teams[0].team_number, msg.teams[0].goalkeeper) == (131, 1)
    assert msg.teams[1].team_number == 1
    assert len(msg.teams[0].players) == 20


def test_camera_info_uses_team_intrinsics():
    cfg = CameraCfg()
    msg = msg_conv.camera_info(STAMP, cfg)
    assert (msg.width, msg.height) == (544, 448)
    assert list(msg.k) == pytest.approx([cfg.fx, 0.0, cfg.cx, 0.0, cfg.fy, cfg.cy, 0.0, 0.0, 1.0])


def test_depth_image_is_millimetres_with_invalid_as_zero():
    depth = np.array([[2.5, np.inf], [70.0, np.nan]], dtype=np.float32)
    msg = msg_conv.depth_image(STAMP, depth, max_depth_m=10.0)
    assert msg.encoding == "16UC1" and (msg.height, msg.width) == (2, 2)
    assert msg.step == 4 and msg.is_bigendian == 0
    values = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(2, 2)
    assert values.tolist() == [[2500, 0], [0, 0]]


def test_rgb_image_layout():
    rgb = np.zeros((448, 544, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    msg = msg_conv.rgb_image(STAMP, rgb)
    assert msg.encoding == "rgb8" and msg.step == 544 * 3
    assert bytes(msg.data[:3]) == bytes([255, 0, 0])


def test_kick_ref_from_brain_kick_msg():
    msg = Kick(x=0.4, y=-0.1, dir=0.2, power=3.5)
    ref = msg_conv.kick_ref(msg, now=12.0)
    assert (ref.ball_x, ref.ball_y, ref.dir, ref.power, ref.stamp) == pytest.approx((0.4, -0.1, 0.2, 3.5, 12.0))


def test_joystick_from_dict_clamps_axes_and_sets_hat_centre():
    msg = msg_conv.joystick_from_dict({"lx": 3.0, "ly": -0.4, "lt": True, "b": True, "ignored_key": 1})
    assert (msg.lx, msg.ly, msg.rx, msg.ry) == pytest.approx((1.0, -0.4, 0.0, 0.0))
    assert msg.lt and msg.b and not msg.x
    assert msg.hat_c
    assert not msg_conv.joystick_from_dict({"hat_u": True}).hat_c


@pytest.mark.parametrize("bad", [{"b": 1}, {"lt": "true"}, {"lx": float("nan")}, {"lx": "left"}])
def test_joystick_from_dict_rejects_non_bool_buttons(bad):
    with pytest.raises(ValueError):
        msg_conv.joystick_from_dict(bad)


def test_game_control_match_reports_each_teams_remaining_packet_budget():
    from soccer_sim.game_controller import Snapshot
    from soccer_sim.roster import PlayerSpec

    snap = Snapshot(3, 0, "home", False, 0, True, 600, {"home": 1, "away": 0})
    roster = [PlayerSpec("home", 1, "goal_keeper"), PlayerSpec("away", 1, "goal_keeper")]
    msg = msg_conv.game_control_match(snap, roster, 1, {"home": 59000, "away": 0})
    assert (msg.teams[0].message_budget, msg.teams[1].message_budget) == (59000, 0)
    assert msg_conv.game_control_match(snap, roster, 1, {"home": 10 ** 6}).teams[0].message_budget == 65535
    assert msg_conv.game_control_match(snap, roster, 1).teams[0].message_budget == 0
