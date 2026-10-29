import math

from soccer_sim.world_view import Ema, to_world, truth_errors

SNAP = {
    "schema_version": 1, "player": 2, "enabled": True, "method": "ci", "t": 100.0,
    "pose": {"x": -3.0, "y": 1.0, "theta": 0.5, "vx": 0.4, "vy": 0.0},
    "local_ball": {"valid": True, "x": 1.234567, "y": -2.0, "vx": 0.5, "vy": 0.1, "smaj": 0.1, "smin": 0.05, "ang": 0.7,
                   "vsig": 0.2, "conf": 0.9, "unseen": 0.1},
    "fused_ball": {"valid": True, "x": 1.2, "y": -2.0, "vx": 0.5, "vy": 0.1, "smaj": 0.08, "smin": 0.04, "ang": 0.7,
                   "vsig": 0.1, "accepted": 2, "sources": [
                       {"player": 0, "track": 0, "w": 0.7, "rejected": False, "age": 0.1, "sigma": 0.07, "x": 1.23, "y": -2.0,
                        "vx": 0.5, "vy": 0.1},
                       {"player": 3, "track": 0, "w": 0.3, "rejected": False, "age": 0.2, "sigma": 0.1, "x": 1.1, "y": -2.1,
                        "vx": 0.5, "vy": 0.0}]},
    "teammates": [{"player": 3, "x": 2.0, "y": 3.0, "theta": 1.0, "vx": 0.0, "vy": 0.1, "sigma": 0.2, "age": 0.2, "rx_age": 0.2}],
    "opponents_local": [{"id": 1, "x": 5.0, "y": 1.0, "vx": 0.0, "vy": 0.0, "smaj": 0.2, "smin": 0.2, "ang": 0.0,
                         "vsig": 0.3, "unseen": 0.1}],
    "opponents_fused": [{"x": 5.0, "y": 1.0, "vx": 0.0, "vy": 0.0, "smaj": 0.2, "smin": 0.1, "ang": 0.3, "vsig": 0.2,
                         "n": 2, "self": True, "sources": [{"player": 0, "track": 1, "w": 1.0, "rejected": False,
                                                            "age": 0.1, "sigma": 0.2, "x": 5.0, "y": 1.0, "vx": 0, "vy": 0}]}],
    "counters": {"ball_accepted": 5}, "packet": {"bytes": 92},
}


def test_home_frame_snapshot_is_only_rounded():
    out = to_world(SNAP, flipped=False)
    assert out["local_ball"]["x"] == 1.235 and out["pose"]["theta"] == 0.5
    assert out["fused_ball"]["sources"][1]["player"] == 3 and out["packet"] == {"bytes": 92}


def test_away_frame_is_rotated_by_half_a_turn():
    out = to_world(SNAP, flipped=True)
    assert (out["pose"]["x"], out["pose"]["y"]) == (3.0, -1.0) and out["pose"]["vx"] == -0.4
    assert math.isclose(out["pose"]["theta"], 0.5 - math.pi, abs_tol=1e-3)
    assert (out["local_ball"]["x"], out["local_ball"]["y"], out["local_ball"]["vx"]) == (-1.235, 2.0, -0.5)
    assert out["fused_ball"]["ang"] == 0.7  # an ellipse looks the same after a 180 degree turn
    assert out["fused_ball"]["sources"][0]["x"] == -1.23 and out["opponents_fused"][0]["sources"][0]["x"] == -5.0
    assert out["teammates"][0]["x"] == -2.0 and out["opponents_local"][0]["y"] == -1.0


def test_missing_parts_do_not_crash():
    out = to_world({"player": 1, "local_ball": {"valid": False}}, flipped=True)
    assert out["local_ball"] == {"valid": False} and out["teammates"] == [] and out["fused_ball"]["sources"] == []


def test_truth_errors():
    view = to_world(SNAP, False)
    robots = [("home", -3.0, 1.0), ("home", 2.0, 3.0), ("away", 5.1, 1.0), ("away", -8.0, 0.0)]
    err = truth_errors(view, (1.2, -2.0), "home", (-3.0, 1.0), robots)
    assert err["fused_ball_err"] == 0.0 and err["local_ball_err"] == 0.035
    assert err["opp_local_err"] == 0.1 and err["opp_fused_err"] == 0.1 and err["teammates_as_opponents"] == 0
    robots[1] = ("home", 5.2, 1.0)  # a teammate stands where this robot tracks an "opponent"
    assert truth_errors(view, (1.2, -2.0), "home", (-3.0, 1.0), robots)["teammates_as_opponents"] == 1
    blind = to_world({"local_ball": {"valid": False}, "fused_ball": {"valid": False}}, False)
    assert truth_errors(blind, (0, 0), "home", (0, 0), robots)["fused_ball_err"] is None


def test_ema():
    e = Ema(0.5)
    assert e.update(None) is None and e.update(2.0) == 2.0 and e.update(4.0) == 3.0 and e.update(None) == 3.0
