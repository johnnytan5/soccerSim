import importlib.util
import sys
from pathlib import Path

from soccer_launcher import command
from soccer_launcher.config import _match, _players

ROOT = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bm = _load("batch_matches_h2h", "scripts/batch_matches.py")
ml = _load("match_launch_h2h", "ros_ws/src/soccer_sim_bridge/launch/match.launch.py")


def test_brain_plan_uniform_keeps_old_behaviour():
    cur = bm.brain_plan("current", "current", "/cur", "/base", Path("/ws"), Path("/wsb"))
    assert cur == {"rcap_root": "/cur", "ws": Path("/ws"), "cfg": {}, "installs": {}}
    base = bm.brain_plan("baseline", "baseline", "/cur", "/base", Path("/ws"), Path("/wsb"))
    assert base == {"rcap_root": "/base", "ws": Path("/wsb"), "cfg": {}, "installs": {}}


def test_brain_plan_mixed_points_baseline_team_at_baseline_install():
    p = bm.brain_plan("current", "baseline", "/cur", "/base", Path("/ws"), Path("/wsb"))
    assert p["ws"] == Path("/ws") and p["rcap_root"] == "/cur"
    assert p["cfg"] == {"away_brain_root": "/base"} and p["installs"] == {"away": "/wsb/install"}
    q = bm.brain_plan("baseline", "current", "/cur", "/base", Path("/ws"), Path("/wsb"))
    assert q["cfg"] == {"home_brain_root": "/base"} and q["installs"] == {"home": "/wsb/install"}


def test_command_carries_per_team_brain_args_only_when_set():
    ids = (1, 2)
    cfg = _match(_players(ids, ids), continuous=True, network_profile="good", viewer=False)
    argv, _ = command.build_command(cfg, "/cur")
    assert not [a for a in argv if "brain_install" in a or "brain_root" in a]
    cfg.away_brain_root = "/base"
    argv, env = command.build_command(cfg, "/cur")
    assert "away_brain_root:=/base" in argv and any(a.startswith("away_brain_install:=") for a in argv)
    assert not [a for a in argv if a.startswith("home_brain")]
    plan = bm.brain_plan("current", "baseline", "/cur", "/base", Path("/ws"), Path("/wsb"))
    argv, env = bm.command_for(cfg, "/cur", plan)  # the baseline build, not ros_ws_brains/<name>
    assert "away_brain_install:=/wsb/install" in argv and "away_brain_root:=/base" in argv
    assert sum(a.startswith("away_brain_install:=") for a in argv) == 1 and env["SS_WS_DIR"] == "/ws"


def test_launch_team_brain_executable():
    assert ml.team_brain("home", "", "", "/cur") == (None, "/cur")
    exe, root = ml.team_brain("away", "/wsb/install", "/base", "/cur")
    assert exe == "/wsb/install/brain/lib/brain/brain_node" and root == "/base"


def test_team_overrides():
    over, hp, ap = bm.team_overrides("good", "", "blackout", 2)
    assert over == "away1=blackout,away2=blackout" and (hp, ap) == ("good", "blackout")
    assert bm.team_overrides("good", "", "", 5) == ("", "good", "good")
    cfg = _match(_players((1, 2), (1, 2)), continuous=True, network_profile="good", network_overrides=over)
    from soccer_launcher import netns
    assert netns.robot_profiles(cfg) == ["good", "good", "blackout", "blackout"]
    assert cfg.validate() == []


def test_args_default_to_brain():
    a = bm.parse_args(["--brain", "baseline"])
    assert bm.team_brains(a) == ("baseline", "baseline")
    a = bm.parse_args(["--away-brain", "baseline"])
    assert bm.team_brains(a) == ("current", "baseline")


def test_csv_has_h2h_columns_last():
    from soccer_sim import metrics
    cols = metrics.csv_columns()
    assert cols[-6:] == ["home_brain", "away_brain", "home_profile", "away_profile", "blackout_apply_s", "blackout_revert_s"]
    row = metrics.csv_row({"home_brain": "current", "away_brain": "baseline", "home_profile": "good", "away_profile": "blackout"}, {})
    assert row["away_profile"] == "blackout"


def test_parallel_needs_namespaced_robots():
    import pytest
    bm.validate_parallel(1, ["none"], "", "")                       # a single match may run on the host network
    bm.validate_parallel(3, ["good", "rcap_bad", "blackout:10"], "", "")
    with pytest.raises(SystemExit, match="namespace"):
        bm.validate_parallel(2, ["good", "none"], "", "")
    with pytest.raises(SystemExit, match="namespace"):
        bm.validate_parallel(2, ["good"], "none", "")
    with pytest.raises(SystemExit, match="namespace"):
        bm.validate_parallel(2, ["none"], "good", "")               # the away team would still be on the host
    bm.validate_parallel(2, ["none"], "good", "good")


def test_match_tag_includes_team_networks():
    assert bm.match_tag("good", "current", 2) == "good_current_2"
    assert bm.match_tag("blackout:10@20", "current", 0) == "blackout10@20_current_0"
    assert bm.match_tag("good", "current", 1, away_net="blackout") == "good_current_away-blackout_1"
    assert bm.match_tag("good", "a-vs-b", 1, "terrible", "blackout:5") == "good_a-vs-b_home-terrible_away-blackout5_1"
    assert bm.match_tag("good", "current", 1, home_net="blackout") != bm.match_tag("good", "current", 1, away_net="blackout")


def test_csv_header_check(tmp_path):
    import pytest
    cols = ["a", "b", "c"]
    bm.check_csv_header(tmp_path / "missing.csv", cols)
    ok = tmp_path / "ok.csv"
    ok.write_text("a,b,c\n1,2,3\n")
    bm.check_csv_header(ok, cols)
    old = tmp_path / "old.csv"
    old.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit, match="missing: c"):
        bm.check_csv_header(old, cols)


def _args(*a):
    return bm.parse_args(list(a))


def test_half_minutes_builds_a_two_half_match():
    a = _args("--half-minutes", "5", "--duration", "30")
    assert bm.play_secs(a) == 600.0 and bm.metrics_play_secs(a) == 660.0  # --duration ignored; FINISHED ends the match
    assert bm.game_secs(a) == 600.0 + 2 * (3.0 + 6.0 + 3.0) + 5.0  # two kick-off set-ups and the half-time pause
    ids = (1, 2)
    cfg = _match(_players(ids, ids), continuous=False, half_minutes=5.0, network_profile="good", viewer=False)
    margs = command.match_args(cfg)
    assert margs[margs.index("--half-minutes") + 1] == "5" and "--continuous" not in margs
    assert "--continuous" in command.match_args(_match(_players(ids, ids), continuous=True))
    assert cfg.validate() == []


def test_continuous_keeps_duration():
    a = _args("--duration", "90")
    assert a.half_minutes is None and bm.play_secs(a) == 90.0 and bm.metrics_play_secs(a) == 90.0
    assert bm.game_secs(a) == 96.0


def test_record_allowance_scales_with_frames():
    win = _args("--record", "20:40")
    assert bm.record_frames(win) == 500 and bm.record_allowance_s(win) == bm.RECORD_ALLOWANCE_S
    full = _args("--half-minutes", "5", "--record", "full")
    frames = bm.record_frames(full)
    assert frames == int(bm.game_secs(full) * 25) and frames > 15000
    assert bm.record_allowance_s(full) == 0.06 * frames > bm.RECORD_ALLOWANCE_S
    assert bm.match_limit_s(full, True) == bm.game_secs(full) + full.timeout_extra + 0.06 * frames
    assert bm.match_limit_s(full, False) == bm.game_secs(full) + full.timeout_extra


def test_match_over_waits_for_the_second_half():
    src = (ROOT / "ros_ws/src/soccer_sim_bridge/soccer_sim_bridge/match_node.py").read_text()
    assert "def match_over(st: dict)" in src  # rclpy is not importable here: check the rule on its source
    ns: dict = {}
    body = src[src.index("    @staticmethod\n    def match_over"):src.index("    def enable_trace")]
    exec("class M:\n" + body, ns)
    over = ns["M"].match_over
    assert not over({"state": "FINISHED", "first_half": True, "continuous": False})  # half-time
    assert over({"state": "FINISHED", "first_half": False, "continuous": False})  # full time
    assert over({"state": "FINISHED", "first_half": True, "continuous": True})
    assert not over({"state": "PLAYING", "first_half": False, "continuous": False})


def test_clip_recorder_full_window_matches_every_state():
    spec = importlib.util.spec_from_file_location(
        "clip_recorder_full", ROOT / "ros_ws/src/soccer_sim_bridge/soccer_sim_bridge/clip_recorder.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    full, win = cr.ClipRecorder.__new__(cr.ClipRecorder), cr.ClipRecorder.__new__(cr.ClipRecorder)
    full.window, win.window = None, (20.0, 40.0)
    assert full.full and not win.full
    assert full.wants(0.0, False) and full.wants(999.0, False) and full.wants(5.0, True)  # READY, half-time, FINISHED
    assert win.wants(30.0, True) and not win.wants(30.0, False) and not win.wants(10.0, True) and not win.wants(41.0, True)
