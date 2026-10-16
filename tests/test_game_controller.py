import pytest

from soccer_sim.game_controller import (
    AWAY, FINISHED, HOME, INITIAL, PLAYING, READY, SET, Goal, GameController, MatchCfg, PlaceBall, SwapSides,
)
from soccer_sim.referee import SET_PLAYS

CFG = MatchCfg(half_secs=100.0, initial_secs=1.0, ready_secs=5.0, set_secs=1.0, halftime_secs=2.0,
               set_play_stop_secs=2.0, set_play_secs=10.0)


def run_to_playing(gc, t=0.0):
    for _ in range(100):
        t += 0.5
        gc.update(t, (0, 0, 0.1), None)
        if gc.state == PLAYING:
            return t
    raise AssertionError("never reached PLAYING")


def test_scripted_start_goes_initial_ready_set_playing():
    gc = GameController(CFG, 0.0)
    seen = []
    for i in range(40):
        gc.update(i * 0.5, (0, 0, 0.1), None)
        if not seen or seen[-1] != gc.state:
            seen.append(gc.state)
    assert seen == [INITIAL, READY, SET, PLAYING]


def test_hold_initial_waits_until_ready_is_pressed():
    gc = GameController(MatchCfg(**{**CFG.__dict__, "hold_initial": True}), 0.0)
    for i in range(20):
        gc.update(i * 1.0, (0, 0, 0.1), None)
    assert gc.state == INITIAL and gc.snapshot(20.0).awaiting_start
    gc.override(20.0, state="READY")
    for i in range(20, 60):
        gc.update(i * 0.5 + 10, (0, 0, 0.1), None)
    assert gc.state == PLAYING


def test_goal_scores_and_conceding_team_kicks_off():
    gc = GameController(CFG, 0.0)
    t = run_to_playing(gc)
    ev = gc.update(t + 0.1, (11.3, 0.0, 0.1), HOME)  # home attacks +x in the first half
    assert Goal(HOME) in ev and PlaceBall(0.0, 0.0, "kickoff") in ev
    assert gc.score == {HOME: 1, AWAY: 0}
    assert gc.state == READY and gc.kicking_team == AWAY


def test_ball_wide_of_the_post_is_not_a_goal():
    gc = GameController(CFG, 0.0)
    t = run_to_playing(gc)
    gc.update(t + 0.1, (11.3, 2.0, 0.1), HOME)
    assert gc.score == {HOME: 0, AWAY: 0}


def test_ball_over_the_bar_is_not_a_goal():
    gc = GameController(CFG, 0.0)
    t = run_to_playing(gc)
    gc.update(t + 0.1, (11.3, 0.0, 1.5), HOME)
    assert gc.score == {HOME: 0, AWAY: 0}


def test_touchline_gives_throw_in_to_the_other_team_at_the_line():
    gc = GameController(CFG, 0.0)
    t = run_to_playing(gc)
    ev = gc.update(t + 0.1, (3.0, 7.3, 0.1), HOME)
    assert gc.set_play == SET_PLAYS["THROW_IN"] and gc.kicking_team == AWAY and gc.stopped
    place = next(e for e in ev if isinstance(e, PlaceBall))
    assert place.x == pytest.approx(3.0) and place.y == pytest.approx(7.063)


def test_goal_line_out_off_defender_is_a_corner_off_attacker_a_goal_kick():
    gc = GameController(CFG, 0.0)
    t = run_to_playing(gc)
    gc.update(t + 0.1, (11.3, 3.0, 0.1), AWAY)  # away defends +x and touched it last -> corner for home
    assert gc.set_play == SET_PLAYS["CORNER_KICK"] and gc.kicking_team == HOME
    gc2 = GameController(CFG, 0.0)
    t = run_to_playing(gc2)
    ev = gc2.update(t + 0.1, (11.3, 3.0, 0.1), HOME)  # home shot wide -> goal kick for away
    assert gc2.set_play == SET_PLAYS["GOAL_KICK"] and gc2.kicking_team == AWAY
    place = next(e for e in ev if isinstance(e, PlaceBall))
    assert place.x == pytest.approx(11.0015 - 2.307) and place.y > 0


def test_set_play_stop_phase_then_execute_then_ends_when_ball_is_taken():
    gc = GameController(CFG, 0.0)
    t = run_to_playing(gc)
    gc.update(t + 0.1, (3.0, 7.3, 0.1), HOME)
    gc.update(t + 1.0, (3.0, 7.063, 0.1), None)
    assert gc.stopped
    gc.update(t + 2.5, (3.0, 7.063, 0.1), None)
    assert gc.set_play and not gc.stopped
    gc.update(t + 3.0, (3.0, 5.0, 0.1), None)
    assert gc.set_play == 0


def test_set_play_expires():
    gc = GameController(CFG, 0.0)
    t = run_to_playing(gc)
    gc.update(t + 0.1, (3.0, 7.3, 0.1), HOME)
    gc.update(t + 20.0, (3.0, 7.063, 0.1), None)
    assert gc.set_play == 0


def test_out_of_bounds_is_ignored_outside_play():
    gc = GameController(CFG, 0.0)
    gc.update(0.1, (3.0, 9.0, 0.1), HOME)
    assert gc.set_play == 0


def test_two_halves_swap_sides_and_finish():
    gc = GameController(CFG, 0.0)
    t = run_to_playing(gc)
    seen = []
    while gc.state != FINISHED:
        t += 1.0
        seen += gc.update(t, (0, 0, 0.1), None)
        assert t < 400
    assert gc.first_half and gc.state == FINISHED  # half time
    while gc.first_half:
        t += 1.0
        seen += gc.update(t, (0, 0, 0.1), None)
    assert any(isinstance(e, SwapSides) for e in seen)
    assert gc.state == INITIAL and gc.kicking_team == AWAY
    t = run_to_playing(gc, t)
    ev = gc.update(t + 0.1, (-11.3, 0.0, 0.1), HOME)
    assert Goal(HOME) in ev and gc.score[HOME] == 1  # sides swapped: -x is now the goal home attacks
    while gc.state != FINISHED:
        t += 1.0
        gc.update(t, (0, 0, 0.1), None)
        assert t < 2000
    assert not gc.first_half


def test_continuous_has_no_clock_limit():
    gc = GameController(MatchCfg(**{**CFG.__dict__, "continuous": True}), 0.0)
    t = run_to_playing(gc)
    gc.update(t + 5000.0, (0, 0, 0.1), None)
    assert gc.state == PLAYING and gc.snapshot(t).secs_remaining == 9999


def test_manual_override_holds_state_and_schedule_resumes():
    gc = GameController(CFG, 0.0)
    gc.override(0.1, state="SET")
    gc.update(50.0, (0, 0, 0.1), None)
    assert gc.state == SET
    gc.override(50.0, state="SCHEDULE")
    gc.update(52.0, (0, 0, 0.1), None)
    assert gc.state == PLAYING


def test_override_rejects_bad_input():
    gc = GameController(CFG, 0.0)
    with pytest.raises(ValueError):
        gc.override(0.0, state="LUNCH")
    with pytest.raises(ValueError):
        gc.override(0.0, kicking="nobody")


SOLO = MatchCfg(**{**CFG.__dict__, "teams": (HOME,)})


def test_with_no_opponents_the_home_team_kicks_off_again_after_its_own_goal():
    gc = GameController(SOLO, 0.0)
    t = run_to_playing(gc)
    gc.update(t + 0.1, (11.3, 0.0, 0.1), HOME)
    assert gc.score == {HOME: 1, AWAY: 0} and gc.kicking_team == HOME and gc.state == READY


def test_with_no_opponents_restarts_go_to_the_team_that_is_there():
    gc = GameController(SOLO, 0.0)
    t = run_to_playing(gc)
    gc.update(t + 0.1, (3.0, 7.3, 0.1), HOME)  # touchline: normally the other team's throw-in
    assert gc.set_play == SET_PLAYS["THROW_IN"] and gc.kicking_team == HOME
    gc2 = GameController(SOLO, 0.0)
    t = run_to_playing(gc2)
    gc2.update(t + 0.1, (11.3, 3.0, 0.1), HOME)  # shot wide at an undefended goal
    assert gc2.set_play == SET_PLAYS["GOAL_KICK"] and gc2.kicking_team == HOME
