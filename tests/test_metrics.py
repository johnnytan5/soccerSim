import math

from soccer_sim import metrics
from soccer_sim.metrics import MetricsRecorder, RobotSample, Sample

DT = 0.1


def robot(key, decision="find", team="home", x=0.0, y=0.0, yaw=math.pi, **kw):
    return RobotSample(key, team, x, y, yaw, decision, **kw)


def feed(rec, seconds, make, t0=0.0, playing=True):
    """make(t) -> (ball, [RobotSample]); one sample per DT."""
    t = t0
    for _ in range(round(seconds / DT)):
        ball, robots = make(t)
        rec.add(Sample(t, playing, ball, robots))
        t += DT
    return t


def test_flips_per_minute_and_decision_flips():
    rec = MetricsRecorder()
    # 60 s of play; H2 alternates chase / standby every 10 s (5 role flips), H3 stays in standby.
    def make(t):
        d = "chase" if int(t // 10) % 2 == 0 else "standby"
        return (0.0, 0.0), [robot("H2", d), robot("H3", "standby")]
    feed(rec, 60.0, make)
    s = rec.summary()
    assert s["play_s"] == 59.9 or abs(s["play_s"] - 60.0) < 0.2
    h2 = s["robots"]["H2"]
    assert h2["role_flips"] == 5 and h2["decision_flips"] == 5
    assert s["robots"]["H3"]["role_flips"] == 0
    assert 4.9 < s["teams"]["home"]["role_flips_per_min"] < 5.2


def test_chase_to_kick_is_a_decision_flip_but_not_a_role_flip():
    rec = MetricsRecorder()
    feed(rec, 2.0, lambda t: ((0, 0), [robot("H2", "chase" if t < 1.0 else "kick")]))
    r = rec.summary()["robots"]["H2"]
    assert r["decision_flips"] == 1 and r["role_flips"] == 0


def test_stoppage_does_not_count_a_flip():
    rec = MetricsRecorder()
    t = feed(rec, 1.0, lambda t: ((0, 0), [robot("H2", "chase")]))
    t = feed(rec, 5.0, lambda t: ((0, 0), [robot("H2", "standby")]), t0=t, playing=False)
    feed(rec, 1.0, lambda t: ((0, 0), [robot("H2", "standby")]), t0=t)
    r = rec.summary()
    assert r["robots"]["H2"]["role_flips"] == 0 and r["play_s"] < 2.1


def test_missing_decisions_are_ignored():
    rec = MetricsRecorder()
    feed(rec, 3.0, lambda t: ((0, 0), [robot("H2", "-" if t < 1 else "chase")]))
    assert rec.summary()["robots"]["H2"]["decision_flips"] == 0


def test_duplicate_striker_fraction():
    rec = MetricsRecorder()
    # both home robots chase for the first 25 s of 100 s; the away robot chases alone all the time
    def make(t):
        both = t < 25.0
        return (0.0, 0.0), [robot("H2", "chase"), robot("H3", "chase" if both else "standby"),
                            robot("A2", "chase", team="away")]
    feed(rec, 100.0, make)
    teams = rec.summary()["teams"]
    assert abs(teams["home"]["duplicate_striker_frac"] - 0.25) < 0.01
    assert teams["away"]["duplicate_striker_frac"] == 0.0
    assert teams["home"]["chaser_present_frac"] == 1.0


def test_ball_error_pools_valid_samples_and_tolerates_missing_fields():
    rec = MetricsRecorder()
    feed(rec, 10.0, lambda t: ((0, 0), [
        robot("H2", fused_err=0.2, local_err=0.4),
        robot("H3", fused_err=0.4 if t < 5 else None, local_err=None),  # an old brain: no estimate at all later / ever
    ]))
    home = rec.summary()["teams"]["home"]
    assert abs(home["fused_ball_err_m"] - 0.267) < 0.01  # pooled over samples: (100*0.2 + 50*0.4) / 150
    assert abs(home["local_ball_err_m"] - 0.4) < 1e-6
    assert 0.7 < home["fused_ball_valid_frac"] < 0.8
    assert rec.summary()["robots"]["H3"]["local_ball_err_m"] is None


def test_comm_mode_share_with_unknown_for_old_brains():
    rec = MetricsRecorder()
    feed(rec, 10.0, lambda t: ((0, 0), [robot("H2", comm_mode="healthy" if t < 7 else "isolated"), robot("H3")]))
    s = rec.summary()
    share = s["robots"]["H2"]["comm_mode_share"]
    assert abs(share["healthy"] - 0.7) < 0.03 and abs(share["isolated"] - 0.3) < 0.03
    assert s["robots"]["H3"]["comm_mode_share"] == {"unknown": 1.0}
    assert abs(s["teams"]["home"]["comm_mode_share"]["unknown"] - 0.5) < 0.01


def test_teleport_reaction_by_heading():
    rec = MetricsRecorder()
    # ball teleports at t=2 to (5, 0). H2 (at origin, facing -x) turns to face it by t=3.5 (yaw crosses 30 deg of +x).
    def make(t):
        ball = (0.0, 5.0) if t < 2.0 else (5.0, 0.0)
        yaw = math.pi if t < 2.5 else max(0.0, math.pi - (t - 2.5) * 2.0)
        return ball, [robot("H2", "standby", x=0.0, y=0.0, yaw=yaw), robot("H3", "standby", x=-9.0, y=0.0, yaw=0.0)]
    feed(rec, 8.0, make)
    ev = [e for e in rec.summary()["ball_events"] if e["team"] == "home"]
    assert len(ev) == 1 and ev[0]["kind"] == "teleport" and ev[0]["robot"] == "H2"
    # facing within 30 deg when pi - (t - 2.5) * 2 <= pi/6  -> t = 2.5 + (5pi/6) / 2 = 3.81
    assert abs(ev[0]["reaction_s"] - 1.8) < 0.15


def test_reaction_by_decision_change():
    rec = MetricsRecorder()
    def make(t):
        ball = (0.0, 5.0) if t < 2.0 else (5.0, 0.0)
        return ball, [robot("H2", "chase" if t >= 3.0 else "standby", yaw=math.pi)]
    feed(rec, 6.0, make)
    ev = rec.summary()["ball_events"]
    assert len(ev) == 1 and abs(ev[0]["reaction_s"] - 1.0) < 0.15


def test_switch_between_chasing_decisions_is_not_a_reaction():
    rec = MetricsRecorder()
    # H2 chases facing away at the event, then flickers chase -> auto_visual_kick: still facing away, so censored
    def make(t):
        ball = (0.0, 5.0) if t < 2.0 else (5.0, 0.0)
        return ball, [robot("H2", "auto_visual_kick" if t >= 2.5 else "chase", yaw=math.pi)]
    feed(rec, 12.0, make)
    ev = rec.summary()["ball_events"]
    assert len(ev) == 1 and ev[0]["reaction_s"] is None


def test_unreacted_event_is_censored_and_keeper_is_not_the_nearest():
    rec = MetricsRecorder()
    def make(t):
        ball = (0.0, 5.0) if t < 2.0 else (5.0, 0.0)
        return ball, [robot("H1", "chase", x=5.0, y=0.5, yaw=math.pi, keeper=True),
                      robot("H2", "standby", x=-5.0, yaw=math.pi)]
    feed(rec, 12.0, make)
    s = rec.summary()
    ev = s["ball_events"]
    assert len(ev) == 1 and ev[0]["robot"] == "H2" and ev[0]["reaction_s"] is None
    assert s["teams"]["home"]["reaction_censored"] == 1 and s["teams"]["home"]["reaction_median_s"] is None


def test_kick_event_skips_the_kicker_and_needs_a_speed_rise():
    rec = MetricsRecorder()
    def make(t):
        # at rest until t=3, then rolls at 3 m/s along +x
        x = 0.0 if t < 3.0 else (t - 3.0) * 3.0
        return (x, 0.0), [robot("H2", "kick", x=-0.3, y=0.0, yaw=0.0),  # the kicker, next to the ball, already facing it
                          robot("H3", "standby", x=0.0, y=6.0, yaw=0.0)]
    feed(rec, 10.0, make)
    ev = rec.summary()["ball_events"]
    kicks = [e for e in ev if e["kind"] == "kick" and e["team"] == "home"]
    assert len(kicks) == 1 and kicks[0]["robot"] == "H3"  # H3 is the one that has to react; H2 kicked


def test_ball_rolling_steadily_is_not_an_event():
    rec = MetricsRecorder()
    feed(rec, 5.0, lambda t: ((t * 3.0, 0.0), [robot("H2")]))  # already fast from the first sample
    assert rec.summary()["ball_events"] == []


def test_csv_row_flat_and_complete():
    rec = MetricsRecorder()
    feed(rec, 5.0, lambda t: ((0, 0), [robot("H2", comm_mode="healthy"), robot("A2", team="away")]))
    row = metrics.csv_row({"profile": "good", "brain": "current", "goals_home": 1, "goals_away": 0, "goal_diff": 1},
                          rec.summary())
    assert set(row) == set(metrics.csv_columns())
    assert row["profile"] == "good" and row["home_mode_healthy"] == 1.0 and row["away_mode_unknown"] == 1.0
    assert row["home_duplicate_striker_frac"] == 0.0


def test_role_flips_and_lead_changes_use_brain_signals_when_published():
    rec = MetricsRecorder()
    # H2 swaps striker/defender every 10 s (5 flips in 60 s) while always chasing (no decision-based flip);
    # its lead flips with it; H3 never changes.
    def make(t):
        a = int(t // 10) % 2 == 0
        return (0.0, 0.0), [robot("H2", "chase", role="striker" if a else "defender", im_lead=a),
                            robot("H3", "standby", role="defender", im_lead=False)]
    feed(rec, 60.0, make)
    s = rec.summary()
    h2 = s["robots"]["H2"]
    assert h2["role_signal"] == "player_role" and h2["role_flips"] == 5 and h2["chase_flips"] == 0
    assert h2["lead_changes"] == 5 and s["robots"]["H3"]["role_flips"] == 0
    home = s["teams"]["home"]
    assert home["role_signal"] == "player_role" and 4.9 < home["role_flips_per_min"] < 5.2
    assert home["chase_flips_per_min"] == 0.0 and 4.9 < home["lead_changes_per_min"] < 5.2


def test_baseline_brain_falls_back_to_decisions():
    rec = MetricsRecorder()
    feed(rec, 20.0, lambda t: ((0, 0), [robot("H2", "chase" if t < 10 else "standby")]))
    home = rec.summary()["teams"]["home"]
    assert home["role_signal"] == "decision" and home["role_flips"] == 1
    assert home["duplicate_signal"] == "decision" and home["lead_changes"] is None
    assert home["duplicate_lead_frac"] is None and home["duplicate_role_frac"] is None


def test_duplicate_striker_is_lead_based_when_available_and_reports_the_others():
    rec = MetricsRecorder()
    # 100 s. Two outfield robots lead for the first 30 s; both chase for the first 50 s but only H3 is striker from 30 s.
    # The keeper "leads" all the time (ignored).
    def make(t):
        return (0.0, 0.0), [
            robot("H1", "standby", keeper=True, role="goal_keeper", im_lead=True),
            robot("H2", "chase", role="striker", im_lead=t < 30),
            robot("H3", "chase" if t < 50 else "standby", role="striker" if t < 30 else "defender", im_lead=t < 30)]
    feed(rec, 100.0, make)
    home = rec.summary()["teams"]["home"]
    assert home["duplicate_signal"] == "im_lead"
    assert abs(home["duplicate_striker_frac"] - 0.30) < 0.01 and home["duplicate_striker_frac"] == home["duplicate_lead_frac"]
    assert abs(home["duplicate_role_frac"] - 0.30) < 0.01  # >= 2 outfield robots with role striker and chasing
    assert abs(home["duplicate_decision_frac"] - 0.50) < 0.01  # decision based: H2 + H3 chase until 50 s (keeper idle)


def test_blackout_segments_and_window():
    rec = MetricsRecorder()
    # 60 s of play, blackout from 20 to 35 s. H2 flips its role once inside and once outside the blackout.
    def make(t):
        return (0.0, 0.0), [robot("H2", "chase", role="striker" if (t < 25 or t >= 50) else "defender")]
    t = 0.0
    for _ in range(600):
        ball, robots = make(t)
        rec.add(Sample(t, True, ball, robots, blackout=20.0 <= t < 35.0))
        t += DT
    s = rec.summary()
    w = s["blackout_window"]
    assert abs(w["start_play_s"] - 20.0) < 0.3 and abs(w["end_play_s"] - 35.0) < 0.3 and abs(w["duration_s"] - 15.0) < 0.3
    inside, outside = s["segments"]["blackout"], s["segments"]["outside"]
    assert abs(inside["play_s"] - 15.0) < 0.3 and abs(outside["play_s"] - 45.0) < 0.5
    assert inside["teams"]["home"]["role_flips"] == 1  # the flip at 25 s
    assert outside["teams"]["home"]["role_flips"] == 1  # the flip at 50 s; the one at 25 s is not outside
    row = metrics.csv_row({}, s)
    assert row["home_bo_role_flips_per_min"] is not None and row["home_out_role_flips_per_min"] is not None


def test_no_blackout_means_no_segments():
    rec = MetricsRecorder()
    feed(rec, 3.0, lambda t: ((0, 0), [robot("H2")]))
    s = rec.summary()
    assert "segments" not in s and "blackout_window" not in s
    assert metrics.csv_row({}, s)["home_bo_reaction_median_s"] is None


def test_event_is_excluded_when_the_nearest_robot_already_faces_it_and_chases():
    rec = MetricsRecorder()
    def make(t):
        ball = (0.0, 5.0) if t < 2.0 else (5.0, 0.0)
        # H2 stands at (3, 0) facing +x, chasing: already on the ball when it appears. A far H3 would be slower.
        return ball, [robot("H2", "chase", x=3.0, y=0.0, yaw=0.0), robot("H3", "standby", x=-8.0, yaw=math.pi)]
    feed(rec, 6.0, make)
    s = rec.summary()
    home = s["teams"]["home"]
    assert home["reaction_excluded"] == 1 and home["reactions"] == 0 and home["reaction_censored"] == 0
    assert home["reaction_median_s"] is None and home["ball_events"] == 1
    assert s["ball_events"][0]["excluded"] is True


def test_facing_without_chasing_is_still_scored():
    rec = MetricsRecorder()
    def make(t):
        ball = (0.0, 5.0) if t < 2.0 else (5.0, 0.0)
        return ball, [robot("H2", "standby", x=3.0, y=0.0, yaw=0.0)]
    feed(rec, 4.0, make)
    home = rec.summary()["teams"]["home"]
    assert home["reaction_excluded"] == 0 and home["reactions"] == 1 and home["reaction_median_s"] < 0.3


def test_reaction_median_p90_and_censored_count():
    rec = MetricsRecorder()
    for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 10.0]):
        rec.events.append({"t": i, "kind": "teleport", "team": "home", "robot": "H2", "reaction_s": None if v == 10.0 else v})
    rec.robots["H2"] = metrics._Robot("H2", "home")
    home = rec.summary()["teams"]["home"]
    assert home["reaction_median_s"] == 2.5 and home["reaction_p90_s"] == 3.7
    assert home["reaction_censored"] == 1 and home["reactions"] == 4


def test_world_view_passes_the_brain_role_signals_through():
    from soccer_sim.world_view import to_world
    snap = {"player": 2, "comm_mode": "isolated", "pose": {"x": 1.0, "y": 2.0, "theta": 0.0},
            "brain": {"player_role": "striker", "outfield_rank": 0, "im_lead": True, "ball_owner_id": 2,
                      "role_own_view": True}}
    for flipped in (False, True):
        out = to_world(snap, flipped)
        assert out["brain"] == snap["brain"] and out["comm_mode"] == "isolated"
    assert "brain" not in to_world({"player": 2}, False)


def test_batch_blackout_profile_parsing():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("batch_matches", pathlib.Path(__file__).parent.parent / "scripts" / "batch_matches.py")
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    assert bm.parse_blackout("blackout:15@20") == (15.0, 20.0)
    assert bm.parse_blackout("blackout:10") == (10.0, bm.DEFAULT_BLACKOUT_OFFSET_S)
    assert bm.parse_blackout("blackout") == (None, bm.DEFAULT_BLACKOUT_OFFSET_S)
    assert bm.parse_blackout("blackout:5@2.5") == (5.0, 2.5)
    assert bm.parse_blackout("good") is None and bm.valid_profile("good") and not bm.valid_profile("blackout:@3")


def test_window_state_ramps_are_neither_dark_nor_outside():
    from soccer_sim.metrics import window_state
    w = {"start": 20.0, "apply_s": 2.0, "revert_begin": 40.0, "end": 41.5, "revert_s": 1.5}
    states = {t: window_state(t, w) for t in (10, 20, 21.9, 22, 39.9, 40, 41.4, 41.5, 60)}
    assert states == {10: "outside", 20: "ramp", 21.9: "ramp", 22: "dark", 39.9: "dark", 40: "ramp", 41.4: "ramp",
                      41.5: "outside", 60: "outside"}
    assert window_state(5, None) == "outside"
    # apply still running (apply_s unknown): ramp; apply done, no revert yet (permanent blackout): dark for good
    assert window_state(25, {"start": 20.0, "apply_s": None}) == "ramp"
    assert window_state(1e6, {"start": 20.0, "apply_s": 2.0, "revert_begin": None, "end": None}) == "dark"
    # revert started but not finished: ramp until end; end - revert_s stands in for revert_begin
    assert window_state(40.5, {"start": 20.0, "apply_s": 2.0, "revert_begin": 40.0, "end": None}) == "ramp"
    assert window_state(38.9, {"start": 20.0, "apply_s": 2.0, "end": 41.0, "revert_s": 2.0}) == "dark"
    assert window_state(39.0, {"start": 20.0, "apply_s": 2.0, "end": 41.0, "revert_s": 2.0}) == "ramp"


def test_ramp_samples_count_in_neither_segment():
    from soccer_sim.metrics import window_state
    w = {"start": 20.0, "apply_s": 2.0, "revert_begin": 35.0, "end": 37.0, "revert_s": 2.0}
    rec = MetricsRecorder()
    t = 0.0
    for _ in range(600):
        ball, robots = (0.0, 0.0), [robot("H2", "chase")]
        st = window_state(t, w)
        rec.add(Sample(t, True, ball, robots, blackout=st == "dark", ramp=st == "ramp"))
        t += DT
    s = rec.summary()
    inside, outside = s["segments"]["blackout"], s["segments"]["outside"]
    assert abs(inside["play_s"] - 13.0) < 0.3                      # 22..35
    assert abs(outside["play_s"] - (60.0 - 17.0)) < 0.5            # 0..20 and 37..60; both ramps (2 s each) excluded
    assert abs(s["blackout_window"]["start_play_s"] - 22.0) < 0.3 and abs(s["blackout_window"]["end_play_s"] - 35.0) < 0.3
