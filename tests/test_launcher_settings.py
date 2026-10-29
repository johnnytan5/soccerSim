import json

import pytest

from soccer_launcher import command, preflight, rcap, settings
from soccer_launcher.config import COMPETITION, DEFAULT_CONFIG, PRESETS, LaunchConfig
from soccer_launcher.rcap import RcapInfo

SCHEMA = {
    "schema_version": 1, "namespace": "world_model",
    "params": [
        {"key": "enabled", "group": "general", "type": "bool", "default": True, "unit": "", "help": "master"},
        {"key": "fusion.method", "group": "fusion", "type": "string", "default": "ci", "choices": ["legacy", "ci"],
         "unit": "", "help": "how to fuse"},
        {"key": "ball.reinit_after", "group": "ball", "type": "int", "default": 4, "min": 1, "max": 50, "unit": "frames",
         "help": "n"},
        {"key": "ball.gate_chi2", "group": "ball", "type": "double", "default": 13.8, "min": 1, "max": 100, "unit": "chi2",
         "help": "gate"},
    ],
}


@pytest.fixture
def rcap_root(tmp_path):
    (tmp_path / "src/brain/config").mkdir(parents=True)
    (tmp_path / "src/brain/config/world_model.schema.json").write_text(json.dumps(SCHEMA))
    return tmp_path


# ---- registry -------------------------------------------------------------------------------------------
def test_static_registry_is_consistent():
    keys = [s.key for s in settings.STATIC]
    assert len(keys) == len(set(keys))
    for s in settings.STATIC:
        assert s.type in settings.TYPES and s.target in (settings.BRAIN, settings.SIM)
        assert (s.target == settings.SIM) == s.key.startswith("sim."), s.key
        if s.lo is not None and s.hi is not None and s.default is not None:
            assert s.lo <= s.default <= s.hi, s.key


def test_schema_entries_come_from_the_rcap_checkout(rcap_root):
    reg = settings.registry_for(rcap_root)
    wm = {s.key: s for s in reg if s.source == "rcap"}
    assert set(wm) == {"world_model.enabled", "world_model.fusion.method", "world_model.ball.reinit_after",
                       "world_model.ball.gate_chi2"}
    assert wm["world_model.ball.gate_chi2"].type == "float" and wm["world_model.fusion.method"].choices == ("legacy", "ci")
    assert wm["world_model.ball.reinit_after"].param == "world_model.ball.reinit_after"
    assert settings.registry_for(None) == list(settings.STATIC)
    assert settings.load_schema(rcap_root / "nope") == []
    assert settings.schema_hash(rcap_root) and settings.schema_hash(None) == ""


def test_overrides_are_typed_for_ros(rcap_root):
    reg = settings.registry_for(rcap_root)
    vals = {"strategy.ball_confidence_threshold": 40, "obstacle_avoidance.robot_confirm_frames": 3.0, "enable_com": False,
            "world_model.ball.gate_chi2": 9, "world_model.ball.reinit_after": 6, "world_model.fusion.method": "legacy",
            "strategy.ball_confidence_decay_rate": 1.0, "sim.seed": 4, "unknown.key": 1}
    out = settings.brain_overrides(vals, reg)
    assert out["strategy.ball_confidence_threshold"] == 40.0 and isinstance(out["strategy.ball_confidence_threshold"], float)
    assert out["obstacle_avoidance.robot_confirm_frames"] == 3 and isinstance(out["obstacle_avoidance.robot_confirm_frames"], int)
    assert out["world_model.ball.gate_chi2"] == 9.0 and isinstance(out["world_model.ball.gate_chi2"], float)
    assert out["enable_com"] is False and out["world_model.fusion.method"] == "legacy"
    assert "strategy.ball_confidence_decay_rate" not in out  # the brain never reads it
    assert "sim.seed" not in out and "unknown.key" not in out


def test_sim_params_respect_the_simulator():
    vals = {"sim.detection_noise_std": 0.05, "sim.odom_drift_pct": 2.0, "sim.collisions": True, "sim.seed": 3}
    reg = list(settings.STATIC)
    assert dict(settings.sim_params(vals, reg, "2d")) == {"detection_noise_std": 0.05, "odom_drift_pct": 2.0,
                                                          "collisions": True, "seed": 3}
    assert dict(settings.sim_params(vals, reg, "mujoco")) == {"detection_noise_std": 0.05, "seed": 3}


def test_validation_and_cleaning(rcap_root):
    reg = settings.registry_for(rcap_root)
    assert settings.validate({"communication.team_broadcast_rate_hz": 3.0}, reg) == []
    errs = settings.validate({"communication.team_broadcast_rate_hz": 50, "world_model.fusion.method": "x",
                              "world_model.ball.reinit_after": 2.5}, reg)
    assert len(errs) == 3 and any("State rate" in e for e in errs)
    cleaned = settings.clean({"enable_com": "true", "communication.team_broadcast_rate_hz": "3", "junk": {"a": 1},
                              "sim.seed": "", "unknown": 5, "strategy.ball_memory_timeout": "abc"})
    assert cleaned == {"enable_com": True, "communication.team_broadcast_rate_hz": 3.0, "unknown": 5}


# ---- LaunchConfig --------------------------------------------------------------------------------------
def test_defaults_and_competition():
    assert LaunchConfig().validate() == []  # still a lone keeper
    assert DEFAULT_CONFIG == COMPETITION.config and PRESETS[0] is COMPETITION
    c = COMPETITION.config
    assert c.validate() == [] and c.simulator == "2d" and c.match_mode == "full" and not c.continuous
    assert len(c.players) == 10 and c.packet_budget == 60000 and c.half_minutes == 10.0
    assert c.settings["communication.team_broadcast_rate_hz"] == 3.0 and c.settings["sim.detection_noise_std"] > 0
    assert LaunchConfig.from_dict(c.to_dict()) == c


def test_old_saved_state_stays_mujoco():
    old = {"role": "striker", "player_id": 2, "viewer": True, "ball_detector": "yolo"}
    cfg = LaunchConfig.from_dict(old)
    assert cfg.simulator == "mujoco" and cfg.settings == {} and cfg.validate() == []


@pytest.mark.parametrize("kwargs, fragment", [
    (dict(simulator="3d"), "simulator must be"),
    (dict(simulator="2d"), "runs matches"),
    (dict(simulator="2d", players=COMPETITION.config.players, ball_detector="yolo"), "no camera"),
    (dict(settings={"communication.team_broadcast_rate_hz": 99}), "State rate"),
])
def test_simulator_and_settings_validation(kwargs, fragment):
    errs = LaunchConfig(**kwargs).validate()
    assert any(fragment in e for e in errs), errs


# ---- command line --------------------------------------------------------------------------------------
def test_2d_command_has_no_viewer_and_no_gl():
    argv, env = command.build_command(COMPETITION.config, "/rcap", 9000)
    assert argv[:2] == ["bash", "scripts/run_match.sh"] and "--viewer" not in argv
    assert "SIM_VIEWER_GL" not in env and "MUJOCO_GL" not in env and "SIM_CAMERA_GL" not in env
    match = env["MATCH_ARGS"].split()
    assert match[match.index("--sim-backend") + 1] == "kinematic"
    assert "--sim-param detection_noise_std=0.05" in env["MATCH_ARGS"] and "collisions=true" in env["MATCH_ARGS"]
    params = [a for a in argv if a.startswith("brain_params:=")]
    assert len(params) == 1
    assert json.loads(params[0].split(":=", 1)[1]) == {
        "communication.discovery_rate_hz": 1.0, "communication.team_broadcast_rate_hz": 3.0, "enable_com": True}


def test_mujoco_command_keeps_the_viewer_and_skips_2d_only_settings():
    cfg = LaunchConfig(**{**COMPETITION.config.to_dict(), "simulator": "mujoco", "graphics": "cpu"})
    argv, env = command.build_command(cfg, "/rcap", 9000)
    assert argv[:3] == ["bash", "scripts/run_match.sh", "--viewer"]
    assert "--sim-backend" not in env["MATCH_ARGS"] and "collisions" not in env["MATCH_ARGS"]
    assert env["SIM_VIEWER_GL"] == "cpu"
    cfg = LaunchConfig(**{**cfg.to_dict(), "viewer": False})
    assert "--viewer" not in command.build_command(cfg, "/rcap", 9000)[0]


def test_a_config_without_settings_builds_the_same_command_as_before():
    cfg = next(p for p in PRESETS if p.name == "Team vs team (5v5)").config
    argv, env = command.build_command(cfg, "/rcap", 9000)
    assert not any(a.startswith("brain_params:=") for a in argv) and "--sim-param" not in env["MATCH_ARGS"]
    assert "--sim-backend" not in env["MATCH_ARGS"]


def test_schema_settings_reach_the_brains(rcap_root):
    cfg = LaunchConfig(**{**COMPETITION.config.to_dict(), "settings": {
        "world_model.fusion.method": "legacy", "world_model.ball.reinit_after": 6}})
    argv, _ = command.build_command(cfg, str(rcap_root), 9000)
    params = json.loads(next(a for a in argv if a.startswith("brain_params:=")).split(":=", 1)[1])
    assert params == {"world_model.fusion.method": "legacy", "world_model.ball.reinit_after": 6}
    # a checkout without the schema drops them instead of passing unknown parameters to the brain
    argv, _ = command.build_command(cfg, "/nonexistent", 9000)
    assert not any(a.startswith("brain_params:=") for a in argv)


# ---- preflight / build fingerprint -----------------------------------------------------------------------
def test_preflight_for_2d(rcap_root, tmp_path):
    (rcap_root / "src/brain/x").touch()
    info = RcapInfo(root=rcap_root, valid=True, branch="fused-world-model", sha="abc", subject="", dirty=False, detached=False)
    cfg = LaunchConfig(**{**COMPETITION.config.to_dict(), "settings": {**COMPETITION.config.settings, "mystery": 1}})
    checks = preflight.run(cfg, info, False, tmp_path, env={}, platform="Linux")
    levels = {c.text: c.level for c in checks}
    assert not preflight.blocking(checks)
    assert any("2D simulator" in t for t in levels)
    assert not any("no display" in t or "viewer" in t.lower() and "2D" not in t for t in levels)
    assert any("mystery" in t and level == "warn" for t, level in levels.items())
    bare = LaunchConfig(settings={"enable_com": True})
    assert any("single-robot" in c.text for c in preflight.run(bare, info, False, tmp_path, env={}, platform="Linux"))


def test_fingerprint_ignores_files_read_at_launch(tmp_path):
    root, local = tmp_path / "rcap", tmp_path / "local"
    (root / "src/brain/config").mkdir(parents=True)
    (root / "src/brain/src").mkdir()
    local.mkdir()
    (root / "src/brain/src/a.cpp").write_text("int a;")
    (root / "src/brain/config/config.yaml").write_text("x: 1")
    before = rcap.fingerprint(root, local)
    (root / "src/brain/config/config.yaml").write_text("x: 2")
    (root / "src/brain/config/world_model.schema.json").write_text("{}")
    assert rcap.fingerprint(root, local) == before
    (root / "src/brain/src/a.cpp").write_text("int b;")
    assert rcap.fingerprint(root, local) != before
