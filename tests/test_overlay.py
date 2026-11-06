import math

from soccer_sim.metrics import MetricsRecorder
from soccer_sim.overlay import (ReactionTimer, RunningCounters, duplicate_teams, label_text, sample_from_row, short_decision,
                                trace_meta)


def _robot(team, i, x, y, decision="standby", theta=0.0, fused=None, mode="healthy", rank=None):
    return {"id": i, "team": team, "x": x, "y": y, "theta": theta, "decision": decision, "player_role": "goal_keeper" if i == 1 else "striker",
            "outfield_rank": rank, "im_lead": False, "comm_mode": mode, "fused_ball": fused}


def _row(t, robots, ball=(0.0, 0.0)):
    return {"t": round(t, 3), "ball": {"x": ball[0], "y": ball[1]}, "blackout": False, "robots": robots}


def _synthetic():
    """6 s at 10 Hz: home 2 and 3 take turns chasing, both chase for 1 s, nobody for 1 s; away never chases; ball belief off by 1 m."""
    rows = []
    for k in range(61):
        t = k / 10
        d2 = "chase" if (1.0 <= t < 3.0 or 4.0 <= t < 5.0) else "standby"
        d3 = "chase" if (2.0 <= t < 4.0) else "standby"
        fused = {"x": 1.0, "y": 0.0, "valid": True}
        rows.append(_row(t, [
            _robot("home", 1, -9, 0, fused=fused), _robot("home", 2, -3, 0, d2, fused=fused, mode="degraded"),
            _robot("home", 3, -3, 3, d3, fused=None),
            _robot("away", 1, 9, 0, fused=fused), _robot("away", 2, 3, 0, fused=fused, mode="isolated")]))
    return rows


def test_counters_agree_with_metrics():
    rows = _synthetic()
    rc, rec = RunningCounters(), MetricsRecorder()
    for r in rows:
        rc.update(r)
        rec.add(sample_from_row(r))
    s = rec.summary()
    for team in ("home", "away"):
        st, ref = rc.stats(team), s["teams"][team]
        assert abs(st["dup_s"] / st["play_s"] - ref["duplicate_decision_frac"]) < 1e-3
        assert abs(1 - st["no_chaser_s"] / st["play_s"] - ref["chaser_present_frac"]) < 1e-3
        assert abs((st["fused_err_m"] or 0) - (ref["fused_ball_err_m"] or 0)) < 1e-3
        assert st["chase_flips"] == round(ref["chase_flips_per_min"] * st["play_s"] / 60)


def test_counters_hand_values():
    rc = RunningCounters()
    for r in _synthetic():
        rc.update(r)
    h, a = rc.stats("home"), rc.stats("away")
    assert h["chase_flips"] == 6          # robot 2: on, off, on, off = 4 flips; robot 3: on, off = 2
    assert math.isclose(h["dup_s"], 1.0, abs_tol=0.11)        # t in [2, 3)
    assert math.isclose(h["no_chaser_s"], 2.0, abs_tol=0.11)  # t in [0, 1) and [5, 6)
    assert h["worst_mode"] == "degraded" and a["worst_mode"] == "isolated"
    assert a["chase_flips"] == 0 and math.isclose(a["no_chaser_s"], 6.0, abs_tol=0.11)
    assert math.isclose(a["fused_err_m"], 1.0)


def test_helpers():
    assert short_decision("auto_visual_kick") == "V-KICK" and short_decision("-") == ""
    assert label_text({"decision": "chase", "outfield_rank": 1}) == "CHASE #1"
    assert label_text({"decision": "-", "outfield_rank": -1}) == ""
    rob = [_robot("home", 2, 0, 0, "chase"), _robot("home", 3, 0, 0, "kick"), _robot("away", 2, 0, 0, "chase")]
    assert duplicate_teams(rob) == {"home"}


def _reaction_rows(react_at):
    """Ball teleports 5 m at t=1.0; the nearest home robot (id 2) faces away and stands by until react_at, then chases."""
    rows = []
    for k in range(121):
        t = k / 10
        ball = (0.0, 0.0) if t < 1.0 else (5.0, 0.0)
        dec = "chase" if react_at is not None and t >= react_at else "standby"
        rows.append(_row(t, [_robot("home", 1, -9, 0), _robot("home", 2, 5.0, 3.0, dec, theta=math.pi / 2),
                             _robot("away", 1, 9, 9), _robot("away", 2, 9, 9.5, "chase", theta=0.0)], ball))
    return rows


def test_reaction_timer_runs_then_holds_final():
    rt = ReactionTimer()
    seen = {}
    for r in _reaction_rows(2.5):
        rt.update(r)
        seen[r["t"]] = rt.display(r["t"])
    assert seen[0.5]["timers"] == [] and seen[0.5]["flashes"] == []
    d = seen[1.0]
    assert d["flashes"] and d["flashes"][0]["ball"] == (5.0, 0.0)
    run = [x for x in seen[1.8]["timers"] if x["team"] == "home"]
    assert run and run[0]["running"] and math.isclose(run[0]["elapsed"], 0.8, abs_tol=0.11) and run[0]["robot"] == "H2"
    fin = [x for x in seen[3.0]["timers"] if x["team"] == "home"]  # 0.5 s after it stopped
    assert fin and not fin[0]["running"] and math.isclose(fin[0]["elapsed"], 1.5, abs_tol=0.11)
    assert not [x for x in seen[3.9]["timers"] if x["team"] == "home"]  # shown ~1 s only
    rec = MetricsRecorder(True)
    for r in _reaction_rows(2.5):
        rec.add(sample_from_row(r))
    ev = [e for e in rec.events if e["team"] == "home"]
    assert len(ev) == 1 and math.isclose(ev[0]["reaction_s"], fin[0]["elapsed"], abs_tol=1e-6)


def test_reaction_timer_timeout():
    rt = ReactionTimer()
    last = None
    for r in _reaction_rows(None):
        rt.update(r)
        if r["t"] > 9.0 and last is None:
            last = rt.display(r["t"])
    home = [x for x in last["timers"] if x["team"] == "home"]
    assert home and home[0]["censored"] and not home[0]["running"] and home[0]["elapsed"] == 8.0


def test_trace_meta(tmp_path):
    (tmp_path / "m_0.json").write_text('{"meta": {"profile": "rcap_bad", "seed": 3, "home_brain": "current", "away_brain": "current"}}')
    assert trace_meta(str(tmp_path / "m_0.trace.jsonl")) == {"situation": "rcap_bad", "seed": 3, "brain": "current"}
    assert trace_meta(str(tmp_path / "none.trace.jsonl")) == {}


def test_one_timer_per_team_newest_event():
    rt = ReactionTimer()
    rows = _reaction_rows(None)
    for r in rows:
        if r["t"] >= 1.0:  # a second teleport at 3 s while the first is still running
            r["ball"] = {"x": 5.0 if r["t"] < 3.0 else -5.0, "y": 0.0}
            r["robots"][1]["x"] = 5.0 if r["t"] < 3.0 else -5.0
    for r in rows[:41]:
        rt.update(r)
    home = [x for x in rt.display(4.0)["timers"] if x["team"] == "home"]
    assert len(home) == 1 and home[0]["elapsed"] <= 1.1
