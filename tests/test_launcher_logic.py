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


def test_available_presets_are_valid_and_future_ones_blocked():
    for p in PRESETS:
        if p.available:
            assert p.config.validate() == [], p.name
        else:
            assert "multi-robot" in p.blocked_reason
    assert sum(not p.available for p in PRESETS) == 4


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
