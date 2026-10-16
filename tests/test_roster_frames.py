import math

import pytest

from soccer_sim.frames import TeamFrame
from soccer_sim.perception import Pose2D
from soccer_sim.roster import (
    AWAY, HOME, PlayerSpec, default_role, roster_from_json, roster_to_json, standard_team, validate_roster,
)


def test_frame_is_identity_for_home_and_an_involution_for_away():
    p = Pose2D(-6.0, -7.5, math.pi / 2)
    assert TeamFrame(False).pose(p) == p
    f = TeamFrame(True)
    q = f.pose(p)
    assert (q.x, q.y) == (6.0, 7.5) and q.yaw == pytest.approx(-math.pi / 2)
    r = f.pose(q)
    assert (r.x, r.y) == (-6.0, -7.5) and r.yaw == pytest.approx(p.yaw)


def test_frame_vectors_and_rotation():
    f = TeamFrame(True)
    assert list(f.vec3((1, 2, 3))) == [-1, -2, 3]
    assert f.rot(((1, 0, 0), (0, 1, 0), (0, 0, 1))).tolist() == [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    assert TeamFrame(False).point(1, 2) == (1, 2)


def test_standard_roster_is_valid_and_keeper_is_player_one():
    r = standard_team(HOME) + standard_team(AWAY)
    assert validate_roster(r) == []
    assert [p.role for p in standard_team(HOME)][0] == "goal_keeper"


@pytest.mark.parametrize("players, fragment", [
    ([], "at least one player"),
    ([PlayerSpec(AWAY, 2, "striker")], "home team needs"),
    ([PlayerSpec(HOME, 2, "striker"), PlayerSpec(HOME, 2, "defender")], "same id"),
    ([PlayerSpec(HOME, 1, "goal_keeper"), PlayerSpec(HOME, 2, "goal_keeper")], "only one goalkeeper"),
    ([PlayerSpec(HOME, 2, "goal_keeper")], "goalkeeper must be player 1"),
    ([PlayerSpec(HOME, 1, "striker")], "always the goalkeeper"),
    ([PlayerSpec(HOME, 6, "striker")], "ids are 1-5"),
    ([PlayerSpec(HOME, 2, "wizard")], "role must be one of"),
])
def test_roster_problems_are_plain_language(players, fragment):
    assert any(fragment in e for e in validate_roster(players))


def test_two_players_no_opponents_and_three_v_one_are_valid():
    assert validate_roster([PlayerSpec(HOME, 2, "striker"), PlayerSpec(HOME, 3, "defender")]) == []
    assert validate_roster(standard_team(HOME, (1, 2, 3)) + standard_team(AWAY, (2,))) == []


def test_json_round_trip():
    r = standard_team(HOME, (1, 2)) + standard_team(AWAY, (1,))
    assert roster_from_json(roster_to_json(r)) == r


def test_entry_poses_are_outside_the_touchline_facing_in_and_in_our_half():
    for i in range(1, 6):
        x, y, yaw = PlayerSpec(HOME, i, default_role(i)).entry_pose()
        assert -11.0 < x < -1.99 and abs(y) > 7.063
        assert (yaw > 0) == (y < 0)
    sides = {PlayerSpec(HOME, i, "striker").entry_side for i in range(1, 6)}
    assert sides == {"left", "right"}
