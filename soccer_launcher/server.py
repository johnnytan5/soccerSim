"""Always-on web server: the Control Centre page, the launcher API, and a proxy to the running sim's controller.

The sim's own ControlServer (inside the bridge process) only exists while a sim runs and listens on an
internal port; the page talks to this server only, so every call stays same-origin.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .config import LaunchConfig
from .jobs import JobController

APP_ID = "soccer-launcher"
WEB_DIR = Path(__file__).with_name("web")
MAX_BODY = 64 * 1024  # launcher routes
MAX_PROXY_BODY = 4096  # same cap the sim's own server enforces
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
PROXY_POST = ("/api/joystick", "/api/referee", "/api/shoot", "/api/reset_robot")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_NAME_RE = re.compile(r"^[^\x00-\x1f]{1,40}$")


class LauncherServer:
    def __init__(self, jobs: JobController, port: int = 8765, host: str = "127.0.0.1"):
        self.jobs = jobs
        self.closing = threading.Event()
        self._upstream = threading.local()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "SoccerLauncher"

            def log_message(self, *_a):
                pass

            # ---- plumbing ----
            def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, obj) -> None:
                self._send(code, json.dumps(obj).encode())

            def _err(self, code: int, msg: str) -> None:
                self._json(code, {"error": msg})

            def _host_ok(self) -> bool:
                try:
                    u = urlsplit("//" + self.headers.get("Host", ""))
                    name, port = u.hostname, u.port
                except ValueError:
                    return False
                return name in LOCAL_HOSTS and port in (None, outer.port)

            def _origin_ok(self) -> bool:
                origin = self.headers.get("Origin")
                return origin is None or urlsplit(origin).netloc == self.headers.get("Host")

            def _read_json(self, limit: int) -> dict | None:
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if ctype != "application/json":
                    self._err(415, "Content-Type must be application/json")
                    return None
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._err(400, "bad Content-Length")
                    return None
                if length < 0:
                    self._err(400, "bad Content-Length")
                    return None
                if length > limit:
                    self.close_connection = True  # body left unread
                    self._err(413, f"body over {limit} bytes")
                    return None
                try:
                    data = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    self._err(400, "body is not valid JSON")
                    return None
                if not isinstance(data, dict):
                    self._err(400, "body must be a JSON object")
                    return None
                return data

            # ---- verbs ----
            def do_GET(self):
                if not self._host_ok():
                    return self._err(403, "bad Host header")
                path = urlsplit(self.path).path
                if path in STATIC:
                    name, ctype = STATIC[path]
                    f = WEB_DIR / name
                    return self._send(200, f.read_bytes(), ctype) if f.is_file() else self._err(404, "not found")
                if path == "/api/launcher/identity":
                    return self._json(200, {"app": APP_ID, "pid": os.getpid()})
                if path == "/api/launcher/state":
                    return self._json(200, {**outer.jobs.static_state(), **outer.jobs.snapshot()})
                if path == "/api/launcher/events":
                    return self._events()
                if path == "/api/status":
                    return self._proxy("GET", path, None)
                self._err(404, "not found")

            def do_POST(self):
                if not self._host_ok():
                    return self._err(403, "bad Host header")
                if not self._origin_ok():
                    return self._err(403, "cross-origin request refused")
                path = urlsplit(self.path).path
                if path in PROXY_POST:
                    body = self._read_json(MAX_PROXY_BODY)
                    return None if body is None else self._proxy("POST", path, body)
                route = outer.routes.get(path)
                if route is None:
                    return self._err(404, "not found")
                body = self._read_json(MAX_BODY)
                if body is None:
                    return None
                try:
                    code, obj = route(body)
                except (ValueError, TypeError, KeyError) as e:
                    return self._err(400, str(e))
                self._json(code, obj)

            # ---- proxy to the bridge's ControlServer ----
            def _proxy(self, method: str, path: str, body: dict | None) -> None:
                port = outer.jobs.bridge_port
                live = ("running", "stopping") if method == "GET" else ("running",)
                if port is None or outer.jobs.state not in live:
                    return self._json(200, {}) if method == "GET" else self._err(503, "sim not running")
                payload = json.dumps(body).encode() if body is not None else None
                headers = {"Content-Type": "application/json"} if payload is not None else {}
                for attempt in (0, 1):
                    conn = outer._conn(port, fresh=attempt == 1)
                    try:
                        conn.request(method, path, payload, headers)
                        resp = conn.getresponse()
                        data = resp.read()
                        return self._send(resp.status, data)
                    except (OSError, http.client.HTTPException):
                        outer._drop_conn()
                if method == "GET":
                    return self._json(200, {})  # sim still warming up
                self._err(502, "sim controller not reachable yet")

            # ---- server-sent events: log lines + state changes ----
            def _events(self) -> None:
                try:
                    last = int(self.headers.get("Last-Event-ID") or urlsplit(self.path).query.partition("after=")[2] or 0)
                except ValueError:
                    last = 0
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                jobs = outer.jobs
                version = -1
                try:
                    while not outer.closing.is_set():
                        lines, new_version = jobs.wait(last, version, 15.0)
                        out = []
                        for n, text in lines:
                            out.append(f"id: {n}\nevent: log\ndata: {json.dumps(text)}\n\n")
                            last = n
                        if new_version != version:
                            version = new_version
                            out.append(f"event: state\ndata: {json.dumps(jobs.snapshot())}\n\n")
                        self.wfile.write("".join(out or [": ping\n\n"]).encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        self.routes = {
            "/api/launcher/check": self._r_check,
            "/api/launcher/launch": self._r_launch,
            "/api/launcher/stop": self._r_stop,
            "/api/launcher/tool": self._r_tool,
            "/api/launcher/refresh": self._r_refresh,
            "/api/launcher/profile": self._r_profile,
            "/api/launcher/profile/delete": self._r_profile_delete,
        }

        class Server(ThreadingHTTPServer):
            daemon_threads = True

        self.httpd = Server((host, port), Handler)
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.closing.set()
        with self.jobs._cond:
            self.jobs._cond.notify_all()  # wake SSE threads so they exit
        self.httpd.shutdown()
        self.httpd.server_close()

    # --------------------------------------------------- upstream connection
    def _conn(self, port: int, fresh: bool = False) -> http.client.HTTPConnection:
        c = getattr(self._upstream, "conn", None)
        if fresh or c is None or getattr(self._upstream, "port", None) != port:
            self._drop_conn()
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            self._upstream.conn, self._upstream.port = c, port
        return c

    def _drop_conn(self) -> None:
        c = getattr(self._upstream, "conn", None)
        if c is not None:
            try:
                c.close()
            except OSError:
                pass
        self._upstream.conn = None

    # ----------------------------------------------------------------- routes
    @staticmethod
    def _cfg(body: dict) -> LaunchConfig:
        cfg = body.get("config", {})
        if not isinstance(cfg, dict):
            raise ValueError("config must be an object")
        return LaunchConfig.from_dict(cfg)

    def _r_check(self, body: dict):
        return 200, self.jobs.check(self._cfg(body), bool(body.get("force")))

    def _r_launch(self, body: dict):
        res = self.jobs.launch(self._cfg(body), bool(body.get("force")))
        if res["ok"]:
            return 200, res
        return (409 if res.get("error", "").startswith("busy") else 422), res

    def _r_stop(self, _body: dict):
        return 200, {"ok": self.jobs.stop()}

    def _r_tool(self, body: dict):
        res = self.jobs.run_tool(str(body.get("key", "")), str(body.get("rcap_root", "")))
        return (200 if res["ok"] else 409), res

    def _r_refresh(self, body: dict):
        self.jobs.refresh(str(body.get("rcap_root", "")))
        return 200, {"ok": True}

    def _r_profile(self, body: dict):
        name = str(body.get("name", "")).strip()
        if not _NAME_RE.match(name):
            raise ValueError("profile name must be 1-40 characters")
        cfg = self._cfg(body)
        errs = cfg.validate()
        if errs:
            return 422, {"ok": False, "error": errs[0]}
        self.jobs.store.save_profile(name, cfg)
        self.jobs._bump()
        return 200, {"ok": True, "profiles": {k: v.to_dict() for k, v in sorted(self.jobs.store.profiles.items())}}

    def _r_profile_delete(self, body: dict):
        self.jobs.store.delete_profile(str(body.get("name", "")))
        self.jobs._bump()
        return 200, {"ok": True, "profiles": {k: v.to_dict() for k, v in sorted(self.jobs.store.profiles.items())}}
