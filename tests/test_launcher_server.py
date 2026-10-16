import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from soccer_launcher import portguard
from soccer_launcher.config import LaunchConfig
from soccer_launcher.jobs import JobController
from soccer_launcher.server import LauncherServer
from soccer_launcher.store import Store
from test_launcher_jobs import FakeRunner, wait_idle


class FakeBridge:
    """Stands in for the bridge's ControlServer."""

    def __init__(self):
        self.posts = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"state": "PLAYING", "ball": {"x": 1, "y": 2}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.posts.append((self.path, self.headers["Content-Type"], data))
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class Srv(ThreadingHTTPServer):
            daemon_threads = True

        self.httpd = Srv(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def stack(tmp_path):
    local = tmp_path / "local"
    (local / "vendor" / "models").mkdir(parents=True)
    (local / "vendor" / "models" / "sim_data_det_0126.onnx").write_text("m")
    rcap = tmp_path / "rcap"
    (rcap / "src" / "brain").mkdir(parents=True)
    (rcap / "src" / "brain" / "a.cpp").write_text("x")
    store = Store(local / ".launcher")
    store.load()
    bridge = FakeBridge()
    runner = FakeRunner(local)
    jobs = JobController(store, runner, local, port_picker=lambda: bridge.port)
    server = LauncherServer(jobs, 0)
    server.start()
    cfg = LaunchConfig(rcap_root=str(rcap), viewer=False, ros_domain_id=3).to_dict()
    yield type("S", (), dict(server=server, jobs=jobs, runner=runner, bridge=bridge, cfg=cfg, rcap=rcap, store=store))
    runner.release.set()
    jobs.shutdown(5)
    server.stop()
    bridge.close()


def req(stack, method, path, body=None, headers=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", stack.server.port, timeout=5)
    h = dict(headers or {})
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, data, h)
    r = c.getresponse()
    payload = r.read()
    c.close()
    try:
        return r.status, json.loads(payload)
    except ValueError:
        return r.status, payload


def test_pages_and_state(stack):
    st, body = req(stack, "GET", "/")
    assert st == 200 and b"Soccer Sim Control Centre" in body
    assert req(stack, "GET", "/app.js")[0] == 200 and req(stack, "GET", "/style.css")[0] == 200
    for bad in ("/../pyproject.toml", "/web/app.js", "/server.py", "/%2e%2e/config.py"):
        assert req(stack, "GET", bad)[0] == 404, bad
    st, state = req(stack, "GET", "/api/launcher/state")
    assert st == 200 and state["state"] == "idle"
    assert any(p["name"] == "Lone keeper" and p["available"] for p in state["presets"])
    assert any(p["config"]["players"] for p in state["presets"]) and all(p["available"] for p in state["presets"])
    assert {t["key"] for t in state["tools"]} >= {"build-sim", "fetch-models"}
    assert req(stack, "GET", "/api/launcher/identity")[1]["app"] == "soccer-launcher"


def test_request_hardening(stack):
    assert req(stack, "GET", "/", headers={"Host": "evil.example"})[0] == 403
    assert req(stack, "GET", "/", headers={"Host": f"localhost:{stack.server.port}"})[0] == 200
    assert req(stack, "GET", "/", headers={"Host": f"127.0.0.1:{stack.server.port + 1}"})[0] == 403
    assert req(stack, "POST", "/api/launcher/stop", {}, headers={"Origin": "http://evil.example"})[0] == 403
    assert req(stack, "POST", "/api/launcher/stop", {}, headers={"Origin": f"http://127.0.0.1:{stack.server.port}"})[0] == 200
    assert req(stack, "POST", "/api/launcher/stop", raw=b"{}", headers={"Content-Type": "text/plain"})[0] == 415
    assert req(stack, "POST", "/api/launcher/stop", raw=b"{nope", headers={"Content-Type": "application/json"})[0] == 400
    assert req(stack, "POST", "/api/launcher/stop", raw=b"[1]", headers={"Content-Type": "application/json"})[0] == 400
    assert req(stack, "POST", "/api/launcher/check", raw=b"x" * 70000, headers={"Content-Type": "application/json"})[0] == 413
    assert req(stack, "POST", "/api/shoot", raw=b"x" * 5000, headers={"Content-Type": "application/json"})[0] == 413
    assert req(stack, "POST", "/api/nope", {})[0] == 404
    s = socket.create_connection(("127.0.0.1", stack.server.port), timeout=3)
    s.sendall(b"POST /api/launcher/stop HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: -1\r\n\r\n")
    assert b" 400 " in s.recv(200).split(b"\r\n")[0]
    s.close()


def test_check_launch_stop_flow(stack):
    st, res = req(stack, "POST", "/api/launcher/check", {"config": stack.cfg})
    assert st == 200 and res["stale"] and not res["blocking"] and "run_sim.sh" in res["preview"]
    st, res = req(stack, "POST", "/api/launcher/launch", {"config": stack.cfg})
    assert st == 200 and res["ok"]
    assert stack.runner.started.wait(5)
    assert req(stack, "POST", "/api/launcher/launch", {"config": stack.cfg})[0] == 409
    assert req(stack, "POST", "/api/launcher/tool", {"key": "test"})[0] == 409
    assert req(stack, "GET", "/api/launcher/state")[1]["state"] == "running"
    assert req(stack, "POST", "/api/launcher/stop", {})[1]["ok"] is True
    wait_idle(stack.jobs)
    bad = {**stack.cfg, "role": "striker", "player_id": 1}
    st, res = req(stack, "POST", "/api/launcher/launch", {"config": bad})
    assert st == 422 and not res["ok"] and res["errors"]


def test_profiles_roundtrip(stack):
    st, res = req(stack, "POST", "/api/launcher/profile", {"name": "demo", "config": stack.cfg})
    assert st == 200 and "demo" in res["profiles"]
    assert "demo" in req(stack, "GET", "/api/launcher/state")[1]["profiles"]
    assert req(stack, "POST", "/api/launcher/profile", {"name": "", "config": stack.cfg})[0] == 400
    assert req(stack, "POST", "/api/launcher/profile", {"name": "x" * 41, "config": stack.cfg})[0] == 400
    assert req(stack, "POST", "/api/launcher/profile", {"name": "bad", "config": {"role": "striker", "player_id": 1}})[0] == 422
    assert req(stack, "POST", "/api/launcher/profile/delete", {"name": "demo"})[1]["profiles"] == {}


def test_proxy_when_down_and_up(stack):
    assert req(stack, "GET", "/api/status") == (200, {})
    st, res = req(stack, "POST", "/api/joystick", {"lx": 0})
    assert st == 503 and "not running" in res["error"]
    req(stack, "POST", "/api/launcher/launch", {"config": stack.cfg})
    assert stack.runner.started.wait(5)
    assert req(stack, "GET", "/api/status")[1]["state"] == "PLAYING"
    assert req(stack, "POST", "/api/joystick", {"lx": 0.5})[0] == 200
    assert req(stack, "POST", "/api/shoot", {"x": 1, "y": 2, "speed": 0, "direction_deg": 0, "elevation_deg": 0})[0] == 200
    assert req(stack, "POST", "/api/reset_robot", {})[0] == 200
    paths = [p[0] for p in stack.bridge.posts]
    assert paths == ["/api/joystick", "/api/shoot", "/api/reset_robot"]
    assert stack.bridge.posts[0][1] == "application/json" and stack.bridge.posts[0][2] == {"lx": 0.5}


def test_proxy_reports_a_vanished_sim_controller(stack):
    with socket.socket() as s:  # a loopback port nothing listens on: the bridge died but the job is still "running"
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    stack.jobs._port_picker = None
    stack.jobs._pick_port = lambda: dead_port
    req(stack, "POST", "/api/launcher/launch", {"config": stack.cfg})
    assert stack.runner.started.wait(5)
    assert req(stack, "POST", "/api/joystick", {"lx": 0})[0] == 502
    assert req(stack, "GET", "/api/status") == (200, {})  # status degrades quietly while the sim warms up or dies


def test_sse_replays_logs_and_pushes_state(stack):
    stack.jobs.log("hello")
    c = http.client.HTTPConnection("127.0.0.1", stack.server.port, timeout=5)
    c.request("GET", "/api/launcher/events")
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    seen = []
    deadline = time.time() + 5
    while time.time() < deadline and not any("event: state" in s for s in seen):
        seen.append(r.fp.readline().decode())
    text = "".join(seen)
    assert "event: log" in text and '"hello"' in text and "event: state" in text
    stack.jobs.log("live line")
    deadline = time.time() + 5
    while time.time() < deadline and '"live line"' not in text:
        text += r.fp.readline().decode()
    assert '"live line"' in text
    c.close()


# ---- port guard ----------------------------------------------------------------
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_claim_free_port_is_a_noop():
    portguard.claim(free_port())


def test_claim_refuses_to_touch_other_programs():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        with pytest.raises(portguard.PortBusy, match="another program"):
            portguard.claim(port, wait=1)
        assert not portguard.identify(port).is_launcher
    finally:
        s.close()


def test_claim_does_not_trust_a_fake_identity_for_a_foreign_pid(tmp_path):
    # something answers like a launcher but reports a pid whose command line isn't soccer_launcher
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"app": "soccer-launcher", "pid": os.getppid()}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert not portguard.identify(httpd.server_address[1]).is_launcher
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_new_launcher_replaces_a_previous_one(tmp_path):
    port = free_port()
    env = {**os.environ, "SOCCER_LAUNCHER_DIR": str(tmp_path / "state")}
    root = Path(__file__).resolve().parent.parent
    old = subprocess.Popen([sys.executable, "-m", "soccer_launcher", "--port", str(port), "--no-open"], cwd=root, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not portguard.identify(port).is_launcher:
            assert old.poll() is None, "first launcher died"
            time.sleep(0.3)
        assert portguard.identify(port).pid == old.pid
        portguard.claim(port, wait=20, say=lambda *_: None)
        assert old.wait(5) == 0  # clean shutdown on SIGTERM
        assert portguard.port_free(port)
    finally:
        if old.poll() is None:
            old.kill()
