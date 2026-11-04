import json
import subprocess

import pytest

from soccer_launcher import command, netns, preflight
from soccer_launcher.config import LaunchConfig, _match, _players


def _cfg(**kw):
    return _match(_players((1, 2), (3,)), **kw)


def test_default_is_no_namespaces_and_command_unchanged():
    cfg = _cfg()
    assert cfg.network_profile == "none" and not netns.active(cfg) and cfg.validate() == []
    argv, _ = command.build_command(cfg)
    assert not any(a.startswith(("netns:=", "game_control_ip:=")) for a in argv)
    assert not any("discovery_address" in a for a in argv)


def test_profile_sets_launch_args_and_broadcast_override():
    argv, _ = command.build_command(_cfg(network_profile="rcap_bad"))
    assert "netns:=true" in argv and f"game_control_ip:={netns.GATEWAY}" in argv
    params = json.loads(next(a for a in argv if a.startswith("brain_params:="))[len("brain_params:="):])
    assert params["communication.discovery_address"] == netns.BROADCAST


def test_explicit_discovery_setting_wins():
    cfg = _cfg(network_profile="good", settings={"communication.discovery_address": "10.77.0.200"})
    # not a registered setting, so it never reaches the brain: only the namespace address does
    argv, _ = command.build_command(cfg)
    assert netns.BROADCAST in next(a for a in argv if a.startswith("brain_params:="))


@pytest.mark.parametrize("text", ["none", "good", "rcap_bad", "terrible", "blackout", "blackout:10", "blackout:2.5"])
def test_valid_profiles(text):
    assert netns.valid_profile(text)


@pytest.mark.parametrize("text", ["", "bad", "blackout:", "blackout:-1", "terrible:3", "GOOD"])
def test_invalid_profiles(text):
    assert not netns.valid_profile(text)
    assert _cfg(network_profile=text).validate()


def test_overrides_map_to_roster_order():
    cfg = _cfg(network_profile="good", network_overrides="home2=terrible, away3=blackout:10")
    assert netns.robot_profiles(cfg) == ["good", "terrible", "blackout:10"]
    assert netns.parse_overrides("home2=terrible") == {("home", 2): "terrible"}


def test_per_robot_only_override_activates():
    cfg = _cfg(network_overrides="home1=terrible")
    assert netns.robot_profiles(cfg) == ["terrible", "none", "none"] and netns.active(cfg)


@pytest.mark.parametrize("text", ["home2", "robot2=good", "home2=bad", "home=good"])
def test_bad_overrides_rejected(text):
    with pytest.raises(ValueError):
        netns.parse_overrides(text)
    assert _cfg(network_profile="good", network_overrides=text).validate()


def test_profile_needs_a_match():
    assert any("only applies to matches" in e for e in LaunchConfig(network_profile="good").validate())


def test_config_roundtrip_and_tolerance():
    cfg = _cfg(network_profile="terrible", network_overrides="home1=good", network_ingress=True)
    assert LaunchConfig.from_dict(cfg.to_dict()) == cfg
    assert LaunchConfig.from_dict({"network_profile": 3, "network_ingress": "yes"}).network_ingress is False


def test_ingress_flag_and_env():
    assert netns.netem_env(_cfg(network_profile="good"), {})["SS_NETEM_INGRESS"] == "0"
    assert netns.netem_env(_cfg(network_profile="good", network_ingress=True), {})["SS_NETEM_INGRESS"] == "1"
    assert netns.netem_env(_cfg(network_profile="good"), {"SS_NETEM_INGRESS": "1"})["SS_ROBOTS"] == "3"


def test_missing_reports_each_gap(monkeypatch):
    monkeypatch.setattr(netns, "Path", lambda *a: type("P", (), {"is_dir": lambda self: False})())
    ok = lambda argv: True  # noqa: E731
    assert netns.missing(False, ok, lambda _n: "/bin/x") == []
    no_sudo = lambda argv: argv[:3] != ["sudo", "-n", "tc"]  # noqa: E731
    assert any("sudo -n tc" in m for m in netns.missing(False, no_sudo, lambda _n: "/bin/x"))
    assert netns.missing(False, ok, lambda _n: None) == ["iproute2 (tc / ip) not installed"]
    no_ifb = lambda argv: "ifb" not in argv  # noqa: E731
    assert netns.missing(False, no_ifb, lambda _n: "/bin/x") == []
    assert any("ifb" in m for m in netns.missing(True, no_ifb, lambda _n: "/bin/x"))


def test_session_falls_back_without_touching_anything(monkeypatch):
    calls, logs = [], []
    monkeypatch.setattr(netns, "missing", lambda ingress=False: ["passwordless `sudo -n tc` not available"])
    monkeypatch.setattr(netns, "_netem", lambda *a, **k: calls.append(a) or True)
    cfg = _cfg(network_profile="rcap_bad")
    with netns.session(cfg, logs.append) as run_cfg:
        assert run_cfg.network_profile == "none" and not netns.active(run_cfg)
    assert calls == [] and "WARNING" in logs[0]


def test_session_up_then_down_even_on_error(monkeypatch):
    calls = []
    monkeypatch.setattr(netns, "missing", lambda ingress=False: [])
    monkeypatch.setattr(netns, "_netem", lambda args, cfg, log, env=None: calls.append(args) or True)
    cfg = _cfg(network_profile="rcap_bad", network_overrides="home2=terrible")
    with pytest.raises(KeyboardInterrupt):
        with netns.session(cfg, lambda _l: None) as run_cfg:
            assert run_cfg is cfg
            raise KeyboardInterrupt
    assert calls == [["up"], ["apply", "rcap_bad", "1"], ["apply", "terrible", "2"], ["apply", "rcap_bad", "3"], ["down"]]


def test_session_inactive_does_nothing(monkeypatch):
    monkeypatch.setattr(netns, "_netem", lambda *a, **k: pytest.fail("must not run"))
    with netns.session(_cfg(), lambda _l: None) as run_cfg:
        assert run_cfg.network_profile == "none"


def test_preflight_warns_when_unsupported(monkeypatch, tmp_path):
    from soccer_launcher import rcap
    monkeypatch.setattr(netns, "missing", lambda ingress=False: ["kernel module sch_netem (netem) missing"])
    checks = preflight.run(_cfg(network_profile="good"), rcap.inspect(tmp_path), False, root=tmp_path, env={}, platform="Linux")
    assert any(c.level == "warn" and "network profile will be ignored" in c.text for c in checks)


def test_ns_exec_script_syntax():
    assert subprocess.run(["bash", "-n", "scripts/ns_exec.sh"]).returncode == 0
    assert subprocess.run(["bash", "-n", "scripts/netem.sh"]).returncode == 0
