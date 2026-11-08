import json
import subprocess

import pytest

from soccer_launcher import build, command, preflight, rcap
from soccer_launcher.config import PRESETS, LaunchConfig
from soccer_launcher.store import Store


# ---- config ---------------------------------------------------------------
def test_default_config_is_valid():
    assert LaunchConfig().validate() == []


@pytest.mark.parametrize("kwargs, fragment", [
    (dict(role="striker", player_id=1), "player 1 is always the keeper"),
    (dict(role="goal_keeper", player_id=3), "goal_keeper must be player id 1"),
    (dict(role="wizard"), "role must be"),
    (dict(referee_schedule="nonsense"), "referee schedule"),
    (dict(ros_domain_id=999), "ROS_DOMAIN_ID"),
    (dict(sim_vy_max=-1.0), "SIM_VY_MAX"),
    (dict(spawn_x=float("nan")), "finite"),
])
def test_validation_rejects(kwargs, fragment):
    errs = LaunchConfig(**kwargs).validate()
    assert any(fragment in e for e in errs), errs


def test_from_dict_is_tolerant():
    cfg = LaunchConfig.from_dict({"role": "striker", "player_id": "3", "bogus": 1,
                                  "viewer": "yes", "spawn_x": "1.5", "ros_domain_id": "7", "web_port": 9})
    assert (cfg.role, cfg.player_id, cfg.spawn_x, cfg.ros_domain_id) == ("striker", 3, 1.5, 7)
    assert cfg.viewer is True and not hasattr(cfg, "web_port")  # bad values fall back; retired keys are ignored


def test_roundtrip():
    cfg = LaunchConfig(role="striker", player_id=2, spawn_x=-3.0, ros_domain_id=4, referee_schedule="0:1,3:0")
    assert LaunchConfig.from_dict(cfg.to_dict()) == cfg


def test_every_preset_is_launchable_and_valid():
    for p in PRESETS:
        assert p.available, p.name
        assert p.config.validate() == [], p.name


def test_team_presets_have_the_right_lineups():
    by = {p.name: p.config for p in PRESETS}
    def lineup(name):
        c = by[name]
        return (sorted(q["player_id"] for q in c.players if q["team"] == "home"),
                sorted(q["player_id"] for q in c.players if q["team"] == "away"))
    assert lineup("Team vs team (5v5)") == ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert lineup("5 robots, no opponents") == ([1, 2, 3, 4, 5], [])
    assert lineup("Goalie vs 1 striker") == ([1], [2])
    assert lineup("Goalie vs 2 strikers") == ([1], [2, 3])
    assert lineup("Goalies only") == ([1], [1])
    custom = [p for p in PRESETS if p.custom]
    assert len(custom) == 1 and custom[0].config.is_match
    assert all(by[n].ball_detector == "ground_truth" for n in by if by[n].is_match)


# ---- match roster -----------------------------------------------------------
def player(team, pid, role):
    return {"team": team, "player_id": pid, "role": role}


def test_custom_rosters_are_accepted_when_sensible():
    for players in (
        [player("home", 2, "striker"), player("home", 3, "defender")],                       # 2 players, no opponents
        [player("home", 1, "goal_keeper"), player("home", 2, "striker"), player("home", 3, "striker"),
         player("away", 2, "striker")],                                                      # 3 v 1
        [player("home", 4, "defender")],
    ):
        assert LaunchConfig(players=players).validate() == [], players


def test_bad_rosters_are_rejected_with_a_reason():
    cases = {
        "only one goalkeeper": [player("home", 1, "goal_keeper"), player("home", 2, "goal_keeper")],
        "keeper must be player 1": [player("home", 3, "goal_keeper")],
        "same id": [player("home", 2, "striker"), player("home", 2, "defender")],
        "home team needs": [player("away", 2, "striker")],
    }
    for fragment, players in cases.items():
        errs = LaunchConfig(players=players).validate()
        assert any(fragment in e for e in errs), (fragment, errs)


def test_match_needs_a_sane_length_and_domain_room():
    base = [player("home", 2, "striker")]
    assert LaunchConfig(players=base, half_minutes=0).validate()
    assert LaunchConfig(players=base, half_minutes=0, continuous=True).validate() == []
    assert LaunchConfig(players=base * 1, ros_domain_id=232, ).validate() == []
    many = [player("home", i, "striker" if i > 1 else "goal_keeper") for i in range(1, 6)] + \
           [player("away", i, "striker" if i > 1 else "goal_keeper") for i in range(1, 6)]
    assert any("232" in e for e in LaunchConfig(players=many, ros_domain_id=230).validate())


def test_players_survive_the_json_roundtrip_and_junk_is_dropped():
    cfg = PRESETS[5].config
    assert LaunchConfig.from_dict(cfg.to_dict()) == cfg
    back = LaunchConfig.from_dict({"players": [player("home", 2, "striker"), "junk", {"team": "away", "player_id": "x"}]})
    assert len(back.players) == 2 and back.players[1]["player_id"] == 0
    assert LaunchConfig.from_dict({"players": "nope"}).players == []


def test_match_command_runs_the_match_script_with_the_roster():
    argv, env = command.build_command(PRESETS[5].config, "/rcap", 9000)
    assert argv[:3] == ["bash", "scripts/run_match.sh", "--viewer"]
    roster = json.loads(argv[3])
    assert len(roster) == 10 and roster[0] == player("home", 1, "goal_keeper")
    assert "tree:=game.xml" in argv and "domain_base:=70" in argv
    margs = env["MATCH_ARGS"].split()
    assert "--ball-detector" in margs and "ground_truth" in margs and "--web-port" in margs and "9000" in margs
    assert margs[margs.index("--start") + 1] == "quick" and margs[margs.index("--half-minutes") + 1] == "10"
    assert "--continuous" not in margs and env["ROBOTEDGE_ROOT"] == "/rcap" and "BRIDGE_ARGS" not in env
    cfg = LaunchConfig(**{**PRESETS[5].config.to_dict(), "match_mode": "full", "continuous": True, "ros_domain_id": 90})
    argv, env = command.build_command(cfg)
    assert "domain_base:=90" in argv
    m = env["MATCH_ARGS"].split()
    assert m[m.index("--start") + 1] == "full_auto" and "--continuous" in m


def test_single_robot_command_is_unchanged_by_the_match_fields():
    argv, env = command.build_command(LaunchConfig(role="striker", player_id=2))
    assert argv[1] == "scripts/run_sim.sh" and "BRIDGE_ARGS" in env and "MATCH_ARGS" not in env


# ---- command --------------------------------------------------------------
def test_existing_pixi_tasks_are_reproduced():
    argv, env = command.build_command(PRESETS[1].config)  # keeper-quick
    assert argv[:5] == ["bash", "scripts/run_sim.sh", "--viewer", "--quick", "tree:=game.xml"]
    assert "role:=goal_keeper" in argv and "player_id:=1" in argv
    argv, _ = command.build_command(PRESETS[2].config)  # striker
    assert "role:=striker" in argv and "player_id:=2" in argv and "--quick" in argv


def test_full_mode_and_no_viewer_drop_flags():
    argv, _ = command.build_command(LaunchConfig(match_mode="full", viewer=False))
    assert "--quick" not in argv and "--viewer" not in argv


def test_env_and_bridge_args():
    cfg = LaunchConfig(ball_detector="ground_truth", spawn_x=-3.5, ball_y=1.0, referee_schedule="0:1,3:0",
                       sim_vy_max=1.2, camera_gl="gpu", mujoco_gl="egl", ros_domain_id=5,
                       extra_bridge_args="-p team_id:=7", extra_launch_args="foo:=bar")
    argv, env = command.build_command(cfg, "/rcap", web_port=9000)
    assert argv[-1] == "foo:=bar"
    assert env["BRIDGE_ARGS"] == ("-p ball_detector:=ground_truth -p spawn_x:=-3.5 -p ball_y:=1.0 "
                                  "-p referee_schedule:=0:1,3:0 -p web_port:=9000 -p team_id:=7")
    assert (env["SIM_VY_MAX"], env["SIM_CAMERA_GL"], env["MUJOCO_GL"], env["ROS_DOMAIN_ID"],
            env["ROBOTEDGE_ROOT"]) == ("1.2", "gpu", "egl", "5", "/rcap")


def test_bridge_port_is_only_added_when_given():
    assert "web_port" not in command.build_command(LaunchConfig())[1]["BRIDGE_ARGS"]
    assert "-p web_port:=1234" in command.render(LaunchConfig(), None, 1234)


def test_defaults_set_no_optional_env():
    _, env = command.build_command(LaunchConfig())
    assert set(env) == {"BRIDGE_ARGS"}


# ---- store ----------------------------------------------------------------
def test_store_roundtrip_and_profiles(tmp_path):
    s = Store(tmp_path / ".launcher")
    s.load()
    assert s.last is None and s.profiles == {}
    s.set_last(LaunchConfig(role="striker", player_id=4))
    s.save_profile("demo", LaunchConfig(viewer=False))
    s2 = Store(tmp_path / ".launcher")
    s2.load()
    assert s2.last.player_id == 4 and s2.profiles["demo"].viewer is False
    s2.delete_profile("demo")
    s3 = Store(tmp_path / ".launcher")
    s3.load()
    assert s3.profiles == {}


def test_store_quarantines_corrupt_file(tmp_path):
    d = tmp_path / ".launcher"
    d.mkdir()
    (d / "state.json").write_text("{not json")
    s = Store(d)
    s.load()
    assert s.last is None and "unreadable" in s.load_warning
    assert (d / "state.json.bad").exists() and not (d / "state.json").exists()


def test_store_ignores_unknown_and_malformed_entries(tmp_path):
    d = tmp_path / ".launcher"
    d.mkdir()
    (d / "state.json").write_text(json.dumps({"version": 99, "last": {"role": "striker", "player_id": 2, "x": 1},
                                              "profiles": {"ok": {"viewer": False}, "bad": 3}}))
    s = Store(d)
    s.load()
    assert s.last.role == "striker" and list(s.profiles) == ["ok"]


# ---- rcap / build ---------------------------------------------------------
def _git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                   check=True, capture_output=True)


@pytest.fixture
def fake_rcap(tmp_path):
    root = tmp_path / "rcap"
    (root / "src" / "brain" / "src").mkdir(parents=True)
    (root / "src" / "brain" / "src" / "brain.cpp").write_text("int main(){}")
    _git(root, "init", "-q", "-b", "feature")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "first")
    return root


@pytest.fixture
def fake_local(tmp_path):
    local = tmp_path / "local"
    (local / "patches").mkdir(parents=True)
    (local / "patches" / "a.patch").write_text("x")
    (local / "scripts").mkdir()
    (local / "scripts" / "setup_ws.sh").write_text("#!/bin/sh")
    return local


def test_locate_order(tmp_path, monkeypatch):
    assert rcap.locate("/x/y", {}) == rcap.Path("/x/y")
    assert rcap.locate("", {"ROBOTEDGE_ROOT": "/env/root"}) == rcap.Path("/env/root")
    monkeypatch.setattr(rcap, "project_root", lambda: tmp_path / "soccerSim")
    (tmp_path / rcap.REPO_NAME / "src" / "brain").mkdir(parents=True)
    assert rcap.locate("", {}) == tmp_path / rcap.REPO_NAME


def test_inspect_reports_branch_dirty_and_invalid(fake_rcap, tmp_path):
    info = rcap.inspect(fake_rcap)
    assert info.valid and info.branch == "feature" and not info.dirty and info.sha
    (fake_rcap / "src" / "brain" / "src" / "brain.cpp").write_text("changed")
    assert rcap.inspect(fake_rcap).dirty
    assert not rcap.inspect(tmp_path / "nope").valid


def test_fingerprint_tracks_inputs_not_build_dirs(fake_rcap, fake_local):
    base = rcap.fingerprint(fake_rcap, fake_local)
    assert base == rcap.fingerprint(fake_rcap, fake_local)
    (fake_rcap / "src" / "brain" / "build").mkdir()
    (fake_rcap / "src" / "brain" / "build" / "junk.o").write_text("junk")
    assert rcap.fingerprint(fake_rcap, fake_local) == base
    (fake_rcap / "src" / "brain" / "src" / "brain.cpp").write_text("changed")
    changed = rcap.fingerprint(fake_rcap, fake_local)
    assert changed != base
    (fake_local / "patches" / "a.patch").write_text("y")
    assert rcap.fingerprint(fake_rcap, fake_local) != changed


def test_fingerprint_tracks_new_bridge_launch_files(fake_rcap, fake_local):
    # A pulled bridge change that adds a launch file needs colcon even though the Python is symlinked.
    base = rcap.fingerprint(fake_rcap, fake_local)
    launch = fake_local / "ros_ws" / "src" / "soccer_sim_bridge" / "launch"
    launch.mkdir(parents=True)
    (launch / "match.launch.py").write_text("x")
    assert rcap.fingerprint(fake_rcap, fake_local) != base


def test_build_staleness_lifecycle(fake_rcap, fake_local, tmp_path):
    ld = tmp_path / ".launcher"
    st = build.check(fake_rcap, ld, fake_local)
    assert st.stale and st.reason == "not built yet"
    (fake_local / "ros_ws" / "install").mkdir(parents=True)
    (fake_local / "ros_ws" / "install" / "setup.bash").write_text("")
    assert build.check(fake_rcap, ld, fake_local).reason == "no record of the last build"
    build.write_stamp(fake_rcap, st.fingerprint, ld)
    assert not build.check(fake_rcap, ld, fake_local).stale
    (fake_rcap / "src" / "brain" / "src" / "brain.cpp").write_text("edit")
    assert build.check(fake_rcap, ld, fake_local).reason == "rcap code changed since the last build"
    other = fake_rcap.parent / "other"
    other.mkdir()
    assert build.check(other, ld, fake_local).reason == "built from a different rcap clone"


# ---- preflight ------------------------------------------------------------
def test_preflight_flags(fake_rcap, fake_local):
    info = rcap.inspect(fake_rcap)
    checks = preflight.run(LaunchConfig(),info, True, fake_local, env={}, platform="Linux")
    texts = [c.text for c in preflight.blocking(checks)]
    assert any("missing" in t for t in texts)  # yolo model not in the fake tree
    assert [c for c in checks if c.fix == "fetch-models"]
    # ground_truth does not need the model
    ok = preflight.run(LaunchConfig(ball_detector="ground_truth", viewer=False), info, False,
                       fake_local, env={}, platform="Linux")
    assert preflight.blocking(ok) == []
    bad = preflight.run(LaunchConfig(), rcap.inspect(fake_local / "nope"), True, fake_local, env={})
    assert any("rcap clone not found" in c.text for c in preflight.blocking(bad))


def test_match_preflight_warns_about_yolo_with_many_robots_and_missing_network(tmp_path):
    from soccer_launcher.preflight import _has_default_route, run
    from soccer_launcher.rcap import RcapInfo
    info = RcapInfo(root=tmp_path, valid=True, branch="main", sha="abc", subject="", dirty=False, detached=False)
    cfg = LaunchConfig(**{**PRESETS[5].config.to_dict(), "ball_detector": "yolo"})
    texts = [c.text for c in run(cfg, info, False, tmp_path, env={"DISPLAY": ":0"}, platform="Linux")]
    assert any("10 robots" in t and "1.2 times a second" in t for t in texts)
    ok = [c.text for c in run(PRESETS[5].config, info, False, tmp_path, env={"DISPLAY": ":0"}, platform="Linux")]
    assert not any("YOLO" in t for t in ok)
    route = tmp_path / "route"
    route.write_text("Iface\tDestination\tGateway\tFlags\neth0\t00000000\t0102A8C0\t0003\n")
    assert _has_default_route(str(route))
    route.write_text("Iface\tDestination\tGateway\tFlags\nlo\t0000007F\t00000000\t0001\n")
    assert not _has_default_route(str(route))


def test_detailed_bodies_flag_roundtrips_and_reaches_the_bridge_only_when_set():
    cfg = PRESETS[5].config
    assert not cfg.detailed_bodies and "--detailed-bodies" not in command.build_command(cfg)[1]["MATCH_ARGS"]
    on = LaunchConfig(**{**cfg.to_dict(), "detailed_bodies": True})
    assert LaunchConfig.from_dict(on.to_dict()) == on
    assert "--detailed-bodies" in command.build_command(on)[1]["MATCH_ARGS"].split()


def test_detailed_bodies_preflight_warns_with_viewer_and_many_robots(tmp_path):
    from soccer_launcher.preflight import run
    from soccer_launcher.rcap import RcapInfo
    info = RcapInfo(root=tmp_path, valid=True, branch="main", sha="abc", subject="", dirty=False, detached=False)
    def texts(**kw):
        cfg = LaunchConfig(**{**PRESETS[5].config.to_dict(), **kw})
        return [c.text for c in run(cfg, info, False, tmp_path, env={"DISPLAY": ":0"}, platform="Linux")]
    assert any("detailed robot bodies" in t and "10 robots" in t for t in texts(detailed_bodies=True))
    assert not any("detailed robot bodies" in t for t in texts(detailed_bodies=True, viewer=False))
    assert not any("detailed robot bodies" in t for t in texts())


def test_graphics_defaults_to_gpu_and_only_cpu_reaches_the_sim():
    assert LaunchConfig().graphics == "gpu" and all(p.config.graphics == "gpu" for p in PRESETS)
    for cfg in (LaunchConfig(), PRESETS[5].config):
        assert "SIM_VIEWER_GL" not in command.build_command(cfg)[1]
        cpu = LaunchConfig(**{**cfg.to_dict(), "graphics": "cpu"})
        assert command.build_command(cpu)[1]["SIM_VIEWER_GL"] == "cpu"
        assert LaunchConfig.from_dict(cpu.to_dict()) == cpu
    assert LaunchConfig(graphics="vulkan").validate()


def test_cpu_graphics_preflight_notes(tmp_path):
    from soccer_launcher.preflight import run
    from soccer_launcher.rcap import RcapInfo
    info = RcapInfo(root=tmp_path, valid=True, branch="main", sha="abc", subject="", dirty=False, detached=False)
    cfg = LaunchConfig(graphics="cpu")
    linux = [c.text for c in run(cfg, info, False, tmp_path, env={"DISPLAY": ":0"}, platform="Linux")]
    mac = [c.text for c in run(cfg, info, False, tmp_path, env={"DISPLAY": ":0"}, platform="Darwin")]
    gpu = [c.text for c in run(LaunchConfig(), info, False, tmp_path, env={"DISPLAY": ":0"}, platform="Linux")]
    assert any("drawn by the CPU" in t for t in linux) and any("only applies on Linux" in t for t in mac)
    assert not any("CPU" in t and "viewer" in t for t in gpu)


def test_packet_budget_defaults_to_the_rcap_cap_and_is_range_checked():
    base = [player("home", 2, "striker")]
    assert LaunchConfig().packet_budget == 60000
    assert LaunchConfig(players=base, packet_budget=0).validate() == []  # 0 = no limit
    assert LaunchConfig(players=base, packet_budget=65535).validate() == []
    for bad in (-1, 65536):  # the GameController's message_budget is a uint16
        assert any("packet budget" in e for e in LaunchConfig(players=base, packet_budget=bad).validate()), bad


def test_packet_budget_survives_the_roundtrip_and_a_wrong_type_falls_back():
    cfg = LaunchConfig(players=[player("home", 2, "striker")], packet_budget=500)
    assert LaunchConfig.from_dict(cfg.to_dict()).packet_budget == 500
    assert LaunchConfig.from_dict({"packet_budget": "lots"}).packet_budget == 60000
    assert LaunchConfig.from_dict({"packet_budget": None}).packet_budget == 60000


def test_match_command_passes_the_packet_budget_to_the_bridge():
    argv, env = command.build_command(LaunchConfig(**{**PRESETS[5].config.to_dict(), "packet_budget": 500}))
    m = env["MATCH_ARGS"].split()
    assert m[m.index("--packet-budget") + 1] == "500"
    _, env = command.build_command(PRESETS[5].config)
    assert env["MATCH_ARGS"].split()[env["MATCH_ARGS"].split().index("--packet-budget") + 1] == "60000"
