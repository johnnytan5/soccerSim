"""Controller tab HTTP server: page + JSON API, stdlib only, bound to 127.0.0.1."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

MAX_BODY_BYTES = 4096
PAGE_PATH = Path(__file__).with_name("controller.html")


class ControlServer:
    def __init__(self, port: int, on_joystick: Callable[[dict], None], on_referee: Callable[[dict], None],
                 on_shoot: Callable[[dict], None], status: Callable[[], dict], host: str = "127.0.0.1",
                 on_reset_robot: Callable[[dict], None] | None = None,
                 extra_routes: dict[str, Callable[[dict], None]] | None = None):
        routes = {"/api/joystick": on_joystick, "/api/referee": on_referee, "/api/shoot": on_shoot}
        if on_reset_robot is not None:
            routes["/api/reset_robot"] = on_reset_robot
        routes.update(extra_routes or {})
        page = PAGE_PATH.read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):  # keep the ROS console clean
                pass

            def _reply(self, code: int, body: bytes, ctype: str = "application/json") -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _error(self, code: int, message: str) -> None:
                self._reply(code, json.dumps({"error": message}).encode())

            def do_GET(self):
                if self.path == "/":
                    self._reply(200, page, "text/html; charset=utf-8")
                elif self.path == "/api/status":
                    self._reply(200, json.dumps(status()).encode())
                else:
                    self._error(404, "not found")

            def do_POST(self):
                handler = routes.get(self.path)
                if handler is None:
                    return self._error(404, "not found")
                # Requiring a JSON content type forces a CORS preflight for any cross-site request,
                # which this server never answers, so other web pages cannot drive the sim.
                if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                    return self._error(415, "Content-Type must be application/json")
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return self._error(400, "bad Content-Length")
                if length < 0:
                    return self._error(400, "bad Content-Length")
                if length > MAX_BODY_BYTES:
                    return self._error(413, "body too large")
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("expected a JSON object")
                    handler(payload)
                except (ValueError, TypeError, KeyError, ArithmeticError) as exc:  # huge ints overflow float()
                    return self._error(400, str(exc))
                self._reply(200, b'{"ok": true}')

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
