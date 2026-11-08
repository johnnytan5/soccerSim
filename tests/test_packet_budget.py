from soccer_sim.packet_budget import PacketBudget
from soccer_sim.roster import AWAY, HOME


def test_packets_are_counted_per_team():
    b = PacketBudget(100)
    b.add(HOME, 30)
    b.add(AWAY, 5)
    b.add(HOME, 2)
    assert (b.used(HOME), b.used(AWAY)) == (32, 5)
    assert (b.remaining(HOME), b.remaining(AWAY)) == (68, 95)


def test_a_team_is_over_only_past_the_limit():
    b = PacketBudget(10)
    b.add(HOME, 10)
    assert not b.over(HOME) and b.remaining(HOME) == 0
    b.add(HOME, 1)
    assert b.over(HOME) and not b.over(AWAY)
    assert b.remaining(HOME) == 0  # never negative


def test_zero_means_no_limit_but_still_counts():
    b = PacketBudget(0)
    b.add(HOME, 70000)
    assert b.used(HOME) == 70000 and not b.over(HOME)
    assert b.snapshot()[HOME]["limit"] == 0


def test_going_over_zeroes_only_that_teams_score():
    b = PacketBudget(5)
    b.add(AWAY, 6)
    score = {HOME: 2, AWAY: 3}
    assert b.effective_score(score) == {HOME: 2, AWAY: 0}
    assert score == {HOME: 2, AWAY: 3}  # the real goals are untouched


def test_unknown_teams_are_ignored_and_negative_adds_do_nothing():
    b = PacketBudget(5, teams=(HOME,))
    b.add(AWAY, 3)
    b.add(HOME, -4)
    assert b.used(HOME) == 0 and set(b.snapshot()) == {HOME}


def test_snapshot_shape():
    b = PacketBudget(50)
    b.add(HOME, 51)
    assert b.snapshot()[HOME] == {"used": 51, "limit": 50, "remaining": 0, "over": True}
    assert b.snapshot()[AWAY] == {"used": 0, "limit": 50, "remaining": 50, "over": False}


def test_newly_over_reports_each_team_once():
    b = PacketBudget(3)
    b.add(HOME, 3)
    assert b.newly_over() == []
    b.add(HOME, 1)
    assert b.newly_over() == [HOME]
    b.add(HOME, 1)
    assert b.newly_over() == []
