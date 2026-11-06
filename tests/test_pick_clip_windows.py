import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("pick", Path(__file__).resolve().parent.parent / "scripts" / "pick_clip_windows.py")
pick = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pick)


def _rob(team, i, x, dec, fused=None):
    return {"id": i, "team": team, "x": x, "y": 0.0, "theta": 3.14, "decision": dec, "player_role": "striker", "outfield_rank": 1,
            "im_lead": False, "comm_mode": "healthy", "fused_ball": fused}


def trace(chase_fn, err_fn=lambda t: 0.0, seconds=60, bo=None, ball_fn=lambda t: (0.0, 0.0)):
    rows = []
    for k in range(seconds * 10 + 1):
        t = k / 10
        bx, by = ball_fn(t)
        fused = {"x": bx + err_fn(t), "y": by, "valid": True}
        a, b = chase_fn(t)
        rows.append({"t": round(t, 3), "ball": {"x": bx, "y": by}, "blackout": bool(bo and bo[0] <= t <= bo[1]),
                     "robots": [_rob("home", 1, -9, "-", fused), _rob("home", 2, -3, "chase" if a else "standby", fused),
                                _rob("home", 3, -2, "chase" if b else "standby", fused)]})
    return rows


def test_flips_picks_the_busy_window_of_the_old_trace():
    calm = trace(lambda t: (True, False))
    # old brain flickers between 30 s and 40 s (every 0.5 s)
    busy = trace(lambda t: ((int(t * 2) % 2 == 0) if 30 <= t < 40 else True, False))
    r = pick.pick_window(calm, busy, "flips", length=12.0)
    assert r["finding"] == "flips" and 28.0 <= r["t0"] <= 30.0 and r["t1"] == r["t0"] + 12.0
    assert r["score_old"] >= 18 and r["score_new"] == 0
    assert r == pick.pick_window(calm, busy, "flips", length=12.0)  # deterministic


def test_dup_counts_seconds_with_two_chasers():
    new = trace(lambda t: (True, False))
    old = trace(lambda t: (True, 10 <= t < 25))
    r = pick.pick_window(new, old, "dup", length=10.0)
    assert 10.0 <= r["t0"] <= 15.0 and abs(r["score_old"] - 10.0) < 0.3 and r["score_new"] == 0


def test_ball_gap_old_minus_new():
    new = trace(lambda t: (True, False), err_fn=lambda t: 0.2)
    old = trace(lambda t: (True, False), err_fn=lambda t: 3.0 if 40 <= t < 55 else 0.2)
    r = pick.pick_window(new, old, "ball", length=10.0)
    assert 40.0 <= r["t0"] <= 45.0 and r["score_old"] > 2.9 and abs(r["score_new"] - 0.2) < 1e-6


def test_reaction_needs_events_in_both_brains():
    def ball(t):  # teleports every 10 s starting at 5 s
        return (5.0 * (int((t + 5) // 10) % 2), 0.0)

    def chase(react_after):
        return lambda t: (False, ((t + 5) % 10) >= react_after)  # robot 3 is the nearest to the ball

    new, old = trace(chase(1.0), ball_fn=ball), trace(chase(2.0), ball_fn=ball)
    r = pick.pick_window(new, old, "reaction", length=22.0)
    assert r["score_new"] >= 2 and r["score_old"] >= 2


def test_blackout_window_is_fixed_around_the_flag():
    new = trace(lambda t: (True, False), bo=(30.0, 40.0))
    old = trace(lambda t: (True, False), bo=(30.0, 40.0))
    r = pick.pick_window(new, old, "blackout")
    assert (r["t0"], r["t1"]) == (25.0, 45.0)
    early = trace(lambda t: (True, False), bo=(2.0, 8.0))
    assert pick.pick_window(early, early, "blackout")["t0"] == 0.0
