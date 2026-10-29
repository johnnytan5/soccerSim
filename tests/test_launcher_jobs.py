import threading
import time

import pytest

from soccer_launcher import build
from soccer_launcher.config import LaunchConfig
from soccer_launcher.jobs import JobController
from soccer_launcher.store import Store


class FakeRunner:
    """Stands in for runner.Runner: records calls; the sim blocks until stop() like the real thing."""

    def __init__(self, local_root, exit_codes=None):
        self.local_root, self.calls, self.exit_codes = local_root, [], exit_codes or {}
        self.release = threading.Event()
        self.started = threading.Event()
        self.block_on = "scripts/run_sim.sh"
        self.was_stopped = False

    def run(self, argv, env, on_line):
        self.calls.append((argv, env))
        on_line("fake output")
        if argv[:2] == build.BUILD_ARGV:
            (self.local_root / "ros_ws" / "install").mkdir(parents=True, exist_ok=True)
            (self.local_root / "ros_ws" / "install" / "setup.bash").write_text("")
        if argv[1] == self.block_on:
            self.started.set()
            self.release.wait(10)
        return self.exit_codes.get(argv[1], 0)

    def stop(self, grace=5.0):
        self.was_stopped = True
        self.release.set()


@pytest.fixture
def world(tmp_path):
    local = tmp_path / "local"
    (local / "vendor" / "models").mkdir(parents=True)
    (local / "vendor" / "models" / "sim_data_det_0126.onnx").write_text("m")
    rcap = tmp_path / "rcap"
    (rcap / "src" / "brain").mkdir(parents=True)
    (rcap / "src" / "brain" / "a.cpp").write_text("x")
    store = Store(local / ".launcher")
    store.load()
    cfg = LaunchConfig(rcap_root=str(rcap), viewer=False, ros_domain_id=3, role="striker", player_id=2)
    return local, rcap, store, cfg


def wait_idle(jobs, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if jobs.state == "idle":
            return
        time.sleep(0.02)
    raise AssertionError(f"still {jobs.state}")


def test_launch_builds_then_runs_then_skips_build_next_time(world):
    local, rcap, store, cfg = world
    runner = FakeRunner(local)
    jobs = JobController(store, runner, local, port_picker=lambda: 4242)
    res = jobs.launch(cfg)
    assert res["ok"] and res["stale"]
    assert runner.started.wait(5)
    assert jobs.state == "running" and jobs.bridge_port == 4242
    assert [c[0][1] for c in runner.calls] == ["scripts/setup_ws.sh", "scripts/run_sim.sh"]
    assert runner.calls[0][1]["PACKAGES"] == "brain soccer_sim_bridge"
    sim_argv, sim_env = runner.calls[1]
    assert "role:=striker" in sim_argv and "-p web_port:=4242" in sim_env["BRIDGE_ARGS"]
    assert sim_env["ROBOTEDGE_ROOT"] == str(rcap) and sim_env["ROS_DOMAIN_ID"] == "3"
    assert store.last.player_id == 2
    runner.release.set()
    wait_idle(jobs)
    assert jobs.bridge_port is None

    runner2 = FakeRunner(local)
    runner2.release.set()
    jobs2 = JobController(store, runner2, local, port_picker=lambda: 1)
    res = jobs2.launch(cfg)
    assert res["ok"] and not res["stale"]
    wait_idle(jobs2)
    assert [c[0][1] for c in runner2.calls] == ["scripts/run_sim.sh"]  # build stamp recorded -> no rebuild

    runner3 = FakeRunner(local)
    runner3.release.set()
    jobs3 = JobController(store, runner3, local, port_picker=lambda: 1)
    assert jobs3.launch(cfg, force=True)["stale"]  # force rebuild
    wait_idle(jobs3)
    assert [c[0][1] for c in runner3.calls] == ["scripts/setup_ws.sh", "scripts/run_sim.sh"]


def test_failed_build_blocks_sim_and_keeps_no_stamp(world):
    local, _, store, cfg = world
    runner = FakeRunner(local, exit_codes={"scripts/setup_ws.sh": 1})
    jobs = JobController(store, runner, local)
    assert jobs.launch(cfg)["ok"]
    wait_idle(jobs)
    assert [c[0][1] for c in runner.calls] == ["scripts/setup_ws.sh"]
    assert any("build failed" in t for _, t in jobs.logs_after(0))
    assert jobs.check(cfg)["stale"]


def test_stop_ends_the_sim(world):
    local, _, store, cfg = world
    runner = FakeRunner(local)
    jobs = JobController(store, runner, local)
    jobs.launch(cfg)
    assert runner.started.wait(5)
    assert jobs.stop()
    assert jobs.state in ("stopping", "idle")  # the fake stops instantly
    wait_idle(jobs)
    assert runner.was_stopped and any(t == "sim stopped" for _, t in jobs.logs_after(0))
    assert not jobs.stop()  # nothing left to stop


def test_stop_during_build_never_starts_the_sim(world):
    local, _, store, cfg = world
    runner = FakeRunner(local)
    runner.block_on = "scripts/setup_ws.sh"
    jobs = JobController(store, runner, local)
    jobs.launch(cfg)
    assert runner.started.wait(5)
    jobs.stop()
    wait_idle(jobs)
    assert [c[0][1] for c in runner.calls] == ["scripts/setup_ws.sh"]


def test_busy_and_preflight_refusals(world):
    local, rcap, store, cfg = world
    runner = FakeRunner(local)
    jobs = JobController(store, runner, local)
    jobs.launch(cfg)
    assert runner.started.wait(5)
    assert jobs.launch(cfg)["error"].startswith("busy")
    assert jobs.run_tool("test")["error"].startswith("busy")
    runner.release.set()
    wait_idle(jobs)

    bad = LaunchConfig(rcap_root=str(rcap), role="striker", player_id=1)  # keeper id for a striker
    res = jobs.launch(bad)
    assert not res["ok"] and res["errors"] and jobs.state == "idle"
    res = jobs.launch(LaunchConfig(rcap_root=str(local / "missing")))
    assert not res["ok"] and any("rcap clone not found" in c["text"] for c in res["checks"])


def test_tools_and_unknown_tool(world):
    local, rcap, store, cfg = world
    runner = FakeRunner(local)
    jobs = JobController(store, runner, local)
    assert not jobs.run_tool("nope")["ok"]
    assert jobs.run_tool("build-sim", str(rcap))["ok"]
    wait_idle(jobs)
    assert runner.calls[0][0] == build.BUILD_ARGV and runner.calls[0][1]["ROBOTEDGE_ROOT"] == str(rcap)
    assert not jobs.check(cfg)["stale"]  # build-sim as a tool records the stamp too
    assert jobs.run_tool("test")["ok"]
    wait_idle(jobs)
    assert runner.calls[1][0] == ["pytest", "-m", "not slow and not ros"]


def test_event_wait_wakes_on_logs_and_state(world):
    local, _, store, cfg = world
    jobs = JobController(store, FakeRunner(local), local)
    jobs.log("one")
    lines, ver = jobs.wait(0, jobs.snapshot()["version"], timeout=0.1)
    assert lines == [(1, "one")]
    got = {}
    t = threading.Thread(target=lambda: got.update(r=jobs.wait(1, ver, timeout=5)))
    t.start()
    time.sleep(0.1)
    jobs.log("two")
    t.join(5)
    assert got["r"][0] == [(2, "two")]
    snap_before = jobs.snapshot()["version"]
    jobs.refresh("")  # first probe changes the cached rcap info
    assert jobs.snapshot()["version"] >= snap_before


def test_state_offers_the_settings_of_the_rcap_checkout_and_the_competition_default(world):
    import json

    from soccer_launcher.config import DEFAULT_CONFIG
    local, rcap, store, cfg = world
    jobs = JobController(store, FakeRunner(local), local)
    jobs.refresh(str(rcap))
    state = jobs.static_state()
    assert state["last"] == DEFAULT_CONFIG.to_dict() and state["default_config"] == DEFAULT_CONFIG.to_dict()
    assert not any(s["source"] == "rcap" for s in state["settings"]) and jobs.snapshot()["settings_hash"] == ""
    (rcap / "src/brain/config").mkdir()
    (rcap / "src/brain/config/world_model.schema.json").write_text(json.dumps({"namespace": "world_model", "params": [
        {"key": "enabled", "group": "general", "type": "bool", "default": True, "unit": "", "help": "on"}]}))
    jobs.refresh(str(rcap))
    state = jobs.static_state()
    assert [s["key"] for s in state["settings"] if s["source"] == "rcap"] == ["world_model.enabled"]
    assert jobs.snapshot()["settings_hash"]
