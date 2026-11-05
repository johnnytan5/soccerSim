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
    assert cols[-4:] == ["home_brain", "away_brain", "home_profile", "away_profile"]
    row = metrics.csv_row({"home_brain": "current", "away_brain": "baseline", "home_profile": "good", "away_profile": "blackout"}, {})
    assert row["away_profile"] == "blackout"
