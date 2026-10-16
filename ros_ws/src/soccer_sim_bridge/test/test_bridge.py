import json
import time
import uuid
from types import SimpleNamespace

import pytest
import rclpy
from booster_interface.msg import Odometer, RemoteControllerState
from booster_msgs.msg import RpcReqMsg, RpcRespMsg
from brain.msg import Kick
from game_controller_interface.msg import GameControlData
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String
from vision_interface.msg import Detections

from soccer_sim_bridge.bridge_node import SimBridge

pytestmark = pytest.mark.ros


def make_harness(extra_params=()):
    rclpy.init()
    bridge = SimBridge(parameter_overrides=[
        Parameter("web_port", value=0),
        Parameter("joystick_timeout_s", value=0.2),
        *extra_params,
    ])
    probe = rclpy.create_node("probe")
    got = {k: [] for k in ("resp", "odom", "det", "rgb", "depth", "joy", "gc", "result")}
    probe.create_subscription(RpcRespMsg, "LocoApiTopicResp", got["resp"].append, 10)
    probe.create_subscription(Odometer, "/odometer_state", got["odom"].append, 10)
    probe.create_subscription(Detections, "/booster_vision/detection", got["det"].append, 10)
    probe.create_subscription(Image, "/boostercamera/head/rgb", got["rgb"].append, 10)
    probe.create_subscription(Image, "/boostercamera/head/depth", got["depth"].append, 10)
    probe.create_subscription(RemoteControllerState, "/remote_controller_state", got["joy"].append, 10)
    probe.create_subscription(GameControlData, "/robocup/game_controller", got["gc"].append, 10)
    probe.create_subscription(String, "/sim/shot_result", got["result"].append, 10)
    rpc_pub = probe.create_publisher(RpcReqMsg, "LocoApiTopicReq", 10)
    ex = SingleThreadedExecutor()
    ex.add_node(bridge)
    ex.add_node(probe)
    bridge.sim.start()

    def send(api_id, body="", header=None):
        msg = RpcReqMsg(uuid=str(uuid.uuid4()), header=header or json.dumps({"api_id": api_id}), body=body)
        rpc_pub.publish(msg)
        return msg.uuid

    def drain(rounds=30):
        # spin_once runs a single callback; the bridge publishes ~6 messages per tick.
        for _ in range(rounds):
            ex.spin_once(timeout_sec=0.0)

    def ticks(n):
        for _ in range(n):
            drain()
            bridge.tick()
        time.sleep(0.06)
        drain()

    deadline = time.monotonic() + 5.0
    while rpc_pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        drain()
        time.sleep(0.01)

    def close():
        ex.shutdown()
        bridge.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()

    ns = SimpleNamespace(
        bridge=bridge, got=got, send=send, ticks=ticks, close=close,
        kick_pub=probe.create_publisher(Kick, "/kick_ball", 10),
        shoot_pub=probe.create_publisher(Float64MultiArray, "/sim/shoot", 10),
    )
    ticks(5)  # let discovery settle
    return ns


@pytest.fixture
def h():
    ns = make_harness()
    yield ns
    ns.close()


@pytest.fixture
def booting():
    ns = make_harness([Parameter("boot_sequence", value=True)])
    yield ns
    ns.close()


def test_get_mode_response_echoes_uuid(h):
    req_id = h.send(2017)
    h.ticks(5)
    match = [r for r in h.got["resp"] if r.uuid == req_id]
    assert match and json.loads(match[0].header)["status"] == 0
    assert json.loads(match[0].body) == {"mode": 4}


def test_move_request_moves_the_odometer(h):
    for _ in range(3):
        h.send(2001, json.dumps({"vx": 0.5, "vy": 0.0, "vyaw": 0.0}))
        h.ticks(2)
    h.ticks(150)  # 3 s of sim
    assert h.got["odom"][-1].x > 0.6


def test_bridge_survives_garbage_rpc(h):
    bad = h.send(0, body="{", header="not json")
    h.ticks(5)
    match = [r for r in h.got["resp"] if r.uuid == bad]
    assert match and json.loads(match[0].header)["status"] != 0
    h.ticks(5)
    assert h.got["odom"]  # still publishing


def test_publishes_detections_and_camera_images(h):
    h.ticks(10)
    assert h.got["det"]
    assert h.got["rgb"] and h.got["rgb"][-1].encoding == "rgb8"
    assert h.got["depth"] and h.got["depth"][-1].encoding == "16UC1"


def test_visual_kick_rpc_and_kick_ball_reach_the_sim(h):
    h.send(2038, json.dumps({"start": True, "version": 1}))
    h.kick_pub.publish(Kick(x=1.0, y=0.0, dir=0.0, power=3.0))
    h.ticks(5)
    assert h.bridge.loco.kicking()
    assert h.bridge.kick_ref is not None and h.bridge.kick_ref.ball_x == pytest.approx(1.0)


def test_stale_joystick_releases_manual_drive(h):
    h.bridge.on_web_joystick({"ly": -1.0})
    h.ticks(1)
    assert h.bridge.status()["manual_drive"] is True
    assert any(j.ly == pytest.approx(-1.0) for j in h.got["joy"])  # forwarded to brain while fresh
    time.sleep(0.3)  # longer than joystick_timeout_s = 0.2 set in the fixture
    h.ticks(5)
    assert h.bridge.status()["manual_drive"] is False
    assert h.got["joy"][-1].ly == 0.0


def test_controller_tab_set_play_reaches_game_controller_topic(h):
    h.bridge.on_web_referee({"state": "PLAYING", "set_play": "CORNER_KICK", "kicking": "opponent", "stopped": False})
    time.sleep(0.6)  # referee publishes at 2 Hz
    h.ticks(5)
    last = h.got["gc"][-1]
    assert (last.state, last.set_play, last.kicking_team) == (3, 6, 1)
    with pytest.raises(ValueError):
        h.bridge.on_web_referee({"state": "PLAYING", "colour": "red"})


def test_boot_sequence_waits_for_lt_up_before_walking(booting):
    h = booting
    req = h.send(2017)
    h.ticks(5)
    assert json.loads([r for r in h.got["resp"] if r.uuid == req][0].body) == {"mode": 1}  # prepare: standing
    h.send(2001, json.dumps({"vx": 0.5, "vy": 0.0, "vyaw": 0.0}))
    h.ticks(50)
    assert abs(h.got["odom"][-1].x) < 0.2  # brain's Move is ignored until the robot is stood up
    h.bridge.on_web_joystick({"lt": True, "hat_u": True})
    h.ticks(3)
    h.bridge.on_web_joystick({})
    req = h.send(2017)
    h.ticks(5)
    assert json.loads([r for r in h.got["resp"] if r.uuid == req][0].body) == {"mode": 4}  # soccer


def test_scripted_shot_scores_and_reports_goal(h):
    h.shoot_pub.publish(Float64MultiArray(data=[-9.0, 0.0, 6.0, 180.0, 0.0]))
    h.ticks(60)  # 1.2 s of sim: 2 m to the line at 6 m/s
    assert any(m.data == "goal" for m in h.got["result"])
    assert h.bridge.status()["shots"]["goal"] == 1


def _overlay(h):
    left, right = h.bridge.overlay()
    return dict(zip(left.split("\n"), right.split("\n")))


def test_overlay_shows_brain_decision_and_walk_command(h):
    h.bridge.on_decision(String(data="save"))
    h.send(2001, json.dumps({"vx": 0.5, "vy": -0.25, "vyaw": 0.1}))  # kMove
    h.ticks(5)
    rows = _overlay(h)
    assert rows["brain decision"] == "save"
    assert rows["walk command"].startswith("vx +0.50  vy -0.25  yaw +0.10")
    assert rows["robot mode"].startswith("soccer")
    assert rows["ball speed"].endswith("m/s")


def test_overlay_uses_speak_for_decisions_only_without_the_decision_topic(h):
    h.bridge.on_speak(String(data="retreat"))
    h.bridge.on_speak(String(data="game state PLAYING"))
    rows = _overlay(h)
    assert rows["brain decision"] == "retreat" and rows["brain said"] == "game state PLAYING"
    h.bridge.on_decision(String(data="save"))
    h.bridge.on_speak(String(data="find"))  # stale, rate-limited: ignored once /brain/decision is live
    assert _overlay(h)["brain decision"] == "save"


def test_overlay_reports_the_shot_tally(h):
    h.bridge._status["shots"] = {"goal": 2, "no_goal": 1}
    h.bridge._status["last_shot"] = "goal"
    assert _overlay(h)["shots"] == "goal 2  /  no goal 1   last: goal"


def test_status_reports_pose_ball_decision_and_rate(h):
    h.ticks(3)
    st = h.bridge.status()
    assert set(st["robot"]) == {"x", "y", "yaw"} and set(st["ball"]) == {"x", "y"}
    sx, sy, _ = h.bridge.sim.sim_cfg.spawn_xy_yaw
    assert abs(st["robot"]["x"] - sx) < 0.5 and abs(st["robot"]["y"] - sy) < 0.5
    assert "decision" in st and "hz" in st and st["state"]  # old keys are still there


def test_controller_tab_places_the_robot_where_clicked_and_validates(h):
    h.bridge.on_web_reset_robot({"x": -3.0, "y": 2.0, "yaw": 1.0})
    h.ticks(1)
    st = h.bridge.status()
    assert abs(st["robot"]["x"] + 3.0) < 0.3 and abs(st["robot"]["y"] - 2.0) < 0.3
    h.bridge.on_web_reset_robot({})  # unchanged behaviour: back to the keeper's home
    h.ticks(1)
    assert abs(h.bridge.status()["robot"]["x"] + 10.0) < 0.3
    for bad in ({"x": 99, "y": 0}, {"x": 1}, {"x": float("nan"), "y": 0}):
        with pytest.raises(ValueError):
            h.bridge.on_web_reset_robot(bad)
