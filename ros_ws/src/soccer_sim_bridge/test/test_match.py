import time

import pytest
import rclpy
from booster_interface.msg import Odometer
from game_controller_interface.msg import GameControlData
from rclpy.executors import SingleThreadedExecutor
from vision_interface.msg import Detections

from soccer_sim.config import SimConfig
from soccer_sim.game_controller import MatchCfg
from soccer_sim.roster import AWAY, HOME, PlayerSpec
from soccer_sim_bridge.match_node import MatchBridge

pytestmark = pytest.mark.ros
BASE = 181  # far from the default domain and from the launcher's 70s


@pytest.fixture
def match():
    roster = [PlayerSpec(HOME, 1, "goal_keeper"), PlayerSpec(HOME, 2, "striker"), PlayerSpec(AWAY, 2, "striker")]
    bridge = MatchBridge(roster, SimConfig(ball_detector="ground_truth"),
                         MatchCfg(initial_secs=0.2, ready_secs=0.4, set_secs=0.2), BASE, web_port=0,
                         full_auto=False)
    probes = []
    for i in range(len(roster)):
        ctx = rclpy.Context()
        rclpy.init(args=[], context=ctx, domain_id=BASE + i)
        node = rclpy.create_node(f"probe{i}", context=ctx)
        got = {"odom": [], "gc": [], "det": []}
        node.create_subscription(Odometer, "/odometer_state", got["odom"].append, 10)
        node.create_subscription(GameControlData, "/robocup/game_controller", got["gc"].append, 10)
        node.create_subscription(Detections, "/booster_vision/detection", got["det"].append, 10)
        ex = SingleThreadedExecutor(context=ctx)
        ex.add_node(node)
        probes.append((ctx, ex, got))
    yield bridge, probes
    for ctx, ex, _ in probes:
        ex.shutdown()
        ctx.try_shutdown()
    bridge.close()


def run(bridge, probes, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        bridge.tick()
        for _, ex, _ in probes:
            ex.spin_once(timeout_sec=0.0)
        time.sleep(0.02)


def test_each_robot_publishes_only_into_its_own_domain_and_gets_the_game_state(match):
    bridge, probes = match
    run(bridge, probes, 2.5)
    for _, _, got in probes:
        assert got["odom"] and got["det"] and got["gc"]
    gc = probes[0][2]["gc"][-1]
    assert gc.teams[0].team_number == 131 and gc.teams[1].team_number == 1
    assert gc.players_per_team == 5
    assert gc.teams[0].players[0].penalty == 0 and gc.teams[0].players[2].penalty == 13  # home 3 is not there
    assert gc.teams[1].players[0].penalty == 13 and gc.teams[1].players[1].penalty == 0
    assert gc.packet_number > 0 and gc.state in (0, 1, 2, 3)
    packets = [m.packet_number for m in probes[1][2]["gc"]]
    assert packets == sorted(packets)  # brain drops anything that is not newer


def test_away_robot_odometer_starts_at_its_own_entry_spot_in_its_own_frame(match):
    bridge, probes = match
    run(bridge, probes, 0.5)
    home2 = probes[1][2]["odom"][0]
    away2 = probes[2][2]["odom"][0]
    assert (home2.x, home2.y) == pytest.approx((away2.x, away2.y), abs=0.2)


def test_status_lists_every_robot_and_the_score(match):
    bridge, probes = match
    run(bridge, probes, 0.5)
    st = bridge.status()
    assert [r["label"] for r in st["robots"]] == ["H1", "H2", "A2"]
    assert st["score"] == {"home": 0, "away": 0} and st["match"]
    bridge.on_web_select({"robot": 2})
    run(bridge, probes, 0.2)
    assert bridge.status()["selected"] == 2
    with pytest.raises(ValueError):
        bridge.on_web_select({"robot": 9})


def test_goal_changes_the_score_and_restarts_with_kickoff(match):
    bridge, probes = match
    run(bridge, probes, 1.5)  # INITIAL -> READY -> SET -> PLAYING
    assert bridge.status()["state"] == "PLAYING"
    bridge.world.reset_ball(11.3, 0.0)
    run(bridge, probes, 0.2)
    st = bridge.status()
    assert st["score"]["home"] == 1 and st["state"] == "READY" and st["kicking_team"] == "away"
    assert abs(st["ball"]["x"]) < 0.5


def _rpc(probes, i, api_id, body="{}"):
    import json
    import uuid
    from booster_msgs.msg import RpcReqMsg
    ctx, ex, got = probes[i]
    if "pub" not in got:
        node = list(ex.get_nodes())[0]
        got["pub"] = node.create_publisher(RpcReqMsg, "LocoApiTopicReq", 10)
        from booster_msgs.msg import RpcRespMsg
        got["resp"] = []
        node.create_subscription(RpcRespMsg, "LocoApiTopicResp", got["resp"].append, 10)
        time.sleep(0.5)
    got["pub"].publish(RpcReqMsg(uuid=str(uuid.uuid4()), header=json.dumps({"api_id": api_id}), body=body))


def test_loco_rpcs_reach_the_right_robot_through_its_gateway(match):
    import json
    bridge, probes = match
    run(bridge, probes, 0.3)
    _rpc(probes, 1, 2001, json.dumps({"vx": 0.5, "vy": 0.1, "vyaw": 0.0}))   # home 2
    _rpc(probes, 2, 2004, json.dumps({"pitch": 0.2, "yaw": 0.3}))            # away 2
    _rpc(probes, 1, 2017)
    run(bridge, probes, 0.6)
    home1, home2, away2 = (c.loco for c in bridge.channels)
    assert (home2.vx, home2.vy) == pytest.approx((0.5, 0.1))
    assert home1.vx == 0.0 and away2.vx == 0.0  # domains are separate: nobody else heard it
    assert away2.head_pitch == pytest.approx(0.2) and away2.head_yaw == pytest.approx(0.3)
    resp = probes[1][2]["resp"]
    assert resp and json.loads(resp[-1].body) == {"mode": 4}


def test_kick_reference_and_get_up_and_stand_up_travel_through_the_gateway(match):
    import json
    from brain.msg import Kick
    bridge, probes = match
    run(bridge, probes, 0.3)
    ex = probes[1][1]
    node = list(ex.get_nodes())[0]
    kick = node.create_publisher(Kick, "/kick_ball", 10)
    time.sleep(0.5)
    kick.publish(Kick(x=0.4, y=0.0, dir=0.1, power=3.0))
    run(bridge, probes, 0.4)
    ref = bridge.channels[1].kick_ref
    assert ref is not None and (ref.ball_x, ref.dir, ref.power) == pytest.approx((0.4, 0.1, 3.0))
    # stand up: the sim asks the gateway, which owns the mode
    ch = bridge.channels[1]
    _rpc(probes, 1, 2000, json.dumps({"mode": 1}))  # ChangeMode(prepare), like a freshly booted robot
    run(bridge, probes, 0.4)
    assert ch.loco.mode == 1
    ch.stand_up()
    run(bridge, probes, 0.4)
    assert ch.loco.mode == 4
    _rpc(probes, 1, 2008)  # GetUp
    run(bridge, probes, 0.4)
    assert ch._getup_seen >= 1.0


def test_team_packets_are_counted_and_shown_in_status_and_overlay(match):
    import socket
    from soccer_sim.team_udp import team_port

    bridge, probes = match
    run(bridge, probes, 0.2)
    before = bridge.status()["packets"]["home"]["used"]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for _ in range(4):
            s.sendto(b"team packet", ("127.0.0.1", team_port(HOME)))
    run(bridge, probes, 0.5)
    home = bridge.status()["packets"]["home"]
    assert home["used"] - before >= 4 and home["limit"] == 60000 and home["counting"] and not home["over"]
    assert f"home {home['used']}/60000" in bridge.overlay()[1]


def test_a_team_over_its_packet_budget_has_its_score_zeroed_and_reports_no_budget_left(match):
    bridge, probes = match
    run(bridge, probes, 1.5)
    bridge.world.reset_ball(11.3, 0.0)  # home scores
    run(bridge, probes, 0.2)
    assert bridge.status()["score"]["home"] == 1
    bridge.budget.limit = 100
    bridge.budget.add(HOME, 101)
    run(bridge, probes, 1.2)  # the game state goes out every 0.5 s
    st = bridge.status()
    assert st["score"] == {"home": 0, "away": 0} and st["raw_score"]["home"] == 1
    assert st["packets"]["home"]["over"] and not st["packets"]["away"]["over"]
    assert "OVER BUDGET" in bridge.overlay()[1]
    gc = probes[0][2]["gc"][-1]
    assert gc.teams[0].score == 0 and gc.teams[0].message_budget == 0 and gc.teams[1].message_budget > 0
