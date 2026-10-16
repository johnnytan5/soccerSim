"""UI-independent orchestration: probe rcap -> pre-flight -> build if stale -> run the sim, or run a tool.

All state lives here so the HTTP server (and tests) just call methods and read `snapshot()`.
"""
from __future__ import annotations

import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from . import build, command, preflight, rcap
from .config import PRESETS, LaunchConfig
from .rcap import RcapInfo
from .runner import TOOLS, TOOLS_BY_KEY, Runner
from .store import Store, project_root

LOG_LINES = 5000
ACTIVE = ("building", "starting", "running", "tool")


def pick_free_port() -> int:
    """An unused loopback port for the bridge's own controller server (the launcher proxies it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class JobController:
    def __init__(self, store: Store, runner: Runner | None = None, local_root: Path | None = None,
                 port_picker: Callable[[], int] = pick_free_port):
        self.local_root = Path(local_root) if local_root else project_root()
        self.store = store
        self.runner = runner or Runner(self.local_root)
        self._pick_port = port_picker
        self._cond = threading.Condition()
        self._log: deque[tuple[int, str]] = deque(maxlen=LOG_LINES)
        self._seq = 0
        self._version = 0
        self._state = "idle"
        self._label = ""
        self._bridge_port: int | None = None
        self._stop_requested = False
        self._info: RcapInfo | None = None
        self._status: build.BuildStatus | None = None
        self._probe_key: str | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------- log + state
    def log(self, text: str) -> None:
        with self._cond:
            self._seq += 1
            self._log.append((self._seq, text))
            self._cond.notify_all()

    def _bump(self) -> None:
        with self._cond:
            self._version += 1
            self._cond.notify_all()

    def _set_state(self, state: str, label: str | None = None) -> None:
        with self._cond:
            if state == "running" and self._stop_requested:
                return
            self._state = state
            if label is not None:
                self._label = label
            if state == "idle":
                self._label, self._bridge_port, self._stop_requested = "", None, False
            self._version += 1
            self._cond.notify_all()

    @property
    def state(self) -> str:
        return self._state

    @property
    def bridge_port(self) -> int | None:
        return self._bridge_port

    def logs_after(self, seq: int) -> list[tuple[int, str]]:
        with self._cond:
            return [(n, t) for n, t in self._log if n > seq]

    def wait(self, seq: int, version: int, timeout: float = 15.0) -> tuple[list[tuple[int, str]], int]:
        """Block until there are log lines after `seq` or the state version moved past `version`."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= seq and self._version == version:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(left)
            return [(n, t) for n, t in self._log if n > seq], self._version

    # ----------------------------------------------------------------- probing
    def refresh(self, rcap_root_text: str = "") -> None:
        """Re-read rcap git info and the build fingerprint (blocking: git + hashing)."""
        info = rcap.inspect(rcap.locate(rcap_root_text))
        status = build.check(info.root, self.store.dir, self.local_root) if info.valid else None
        changed = (info, status) != (self._info, self._status) or rcap_root_text != self._probe_key
        self._info, self._status, self._probe_key = info, status, rcap_root_text
        if changed:
            self._bump()

    def _ensure_probe(self, rcap_root_text: str) -> tuple[RcapInfo, build.BuildStatus | None]:
        if self._info is None or self._probe_key != rcap_root_text:
            self.refresh(rcap_root_text)
        return self._info, self._status  # type: ignore[return-value]

    def background_refresh(self, stop: threading.Event, every: float = 10.0) -> None:
        """Keep the rcap card current while idle (the user edits rcap in another window)."""
        while not stop.wait(every):
            if self._state == "idle":
                try:
                    self.refresh(self._probe_key or "")
                except Exception as e:  # never let the poller die
                    self.log(f"rcap refresh failed: {e}")

    # -------------------------------------------------------------- read models
    def check(self, cfg: LaunchConfig, force: bool = False) -> dict:
        info, status = self._ensure_probe(cfg.rcap_root)
        stale = force or status is None or status.stale
        checks = preflight.run(cfg, info, stale, self.local_root)
        errs = cfg.validate()
        return {
            "errors": errs,
            "checks": [asdict(c) for c in checks],
            "blocking": bool(preflight.blocking(checks)),
            "stale": stale,
            "preview": command.render(cfg, str(info.root) if info.valid else None) if not errs else "",
        }

    def snapshot(self) -> dict:
        info, status = self._info, self._status
        with self._cond:
            return {
                "state": self._state,
                "label": self._label,
                "bridge_port": self._bridge_port,
                "version": self._version,
                "log_seq": self._seq,
                "rcap": None if info is None else {
                    "root": str(info.root), "valid": info.valid, "branch": info.branch, "sha": info.sha,
                    "subject": info.subject, "dirty": info.dirty, "detached": info.detached},
                "build": None if status is None else {"stale": status.stale, "reason": status.reason},
            }

    def static_state(self) -> dict:
        """Things that only change when the user edits them: config, presets, profiles, tools."""
        last = self.store.last or LaunchConfig()
        return {
            "last": last.to_dict(),
            "profiles": {k: v.to_dict() for k, v in sorted(self.store.profiles.items())},
            "presets": [{"name": p.name, "description": p.description, "available": p.available,
                         "blocked_reason": p.blocked_reason, "custom": p.custom,
                         "config": p.config.to_dict() if p.config else None} for p in PRESETS],
            "tools": [{"key": t.key, "label": t.label, "description": t.description} for t in TOOLS],
            "load_warning": self.store.load_warning,
        }

    # ------------------------------------------------------------------- jobs
    def _begin(self, state: str, label: str) -> bool:
        with self._cond:
            if self._state != "idle":
                return False
            self._state, self._label, self._stop_requested = state, label, False
            self._version += 1
            self._cond.notify_all()
            return True

    def launch(self, cfg: LaunchConfig, force: bool = False) -> dict:
        """Validate + pre-flight synchronously, then build (if stale) and run on a worker thread."""
        if self._state != "idle":
            return {"ok": False, "error": f"busy ({self._state}); stop the current job first"}
        self.refresh(cfg.rcap_root)  # the user may have just edited rcap
        result = self.check(cfg, force)
        if result["errors"] or result["blocking"]:
            return {"ok": False, "error": "pre-flight failed", **result}
        if not self._begin("building" if result["stale"] else "starting", "sim"):
            return {"ok": False, "error": "busy; stop the current job first"}
        try:
            self.store.set_last(cfg)
        except OSError as e:
            self.log(f"could not save last config: {e}")
        info, status = self._info, self._status
        self._thread = threading.Thread(target=self._launch_worker, args=(cfg, info, status, result["stale"]),
                                        daemon=True)
        self._thread.start()
        return {"ok": True, **result}

    def _rcap_env(self, info: RcapInfo | None) -> dict[str, str]:
        return {"ROBOTEDGE_ROOT": str(info.root)} if info is not None and info.valid else {}

    def _launch_worker(self, cfg: LaunchConfig, info: RcapInfo, status, stale: bool) -> None:
        try:
            if stale:
                self.log("$ " + " ".join(build.BUILD_ARGV))
                code = self.runner.run(list(build.BUILD_ARGV), {**build.BUILD_ENV, **self._rcap_env(info)}, self.log)
                if code != 0 or self._stop_requested:
                    self.log("build stopped" if self._stop_requested else f"build failed (exit {code}); sim not started")
                    return
                if status is not None:
                    build.write_stamp(info.root, status.fingerprint, self.store.dir)
            if self._stop_requested:
                return
            port = self._pick_port()
            argv, env = command.build_command(cfg, str(info.root), port)
            with self._cond:
                self._bridge_port = port
            self._set_state("running", "sim")
            self.log("$ " + " ".join(argv))
            code = self.runner.run(argv, env, self.log)
            self.log("sim stopped" if self._stop_requested else f"sim exited with code {code}")
        except Exception as e:  # surface instead of dying silently on the worker thread
            self.log(f"launcher error: {type(e).__name__}: {e}")
        finally:
            self._set_state("idle")
            self._safe_refresh()

    def run_tool(self, key: str, rcap_root_text: str = "") -> dict:
        tool = TOOLS_BY_KEY.get(key)
        if tool is None:
            return {"ok": False, "error": f"unknown tool '{key}'"}
        if not self._begin("tool", tool.label):
            return {"ok": False, "error": f"busy ({self._state}); stop the current job first"}
        self._thread = threading.Thread(target=self._tool_worker, args=(tool, rcap_root_text), daemon=True)
        self._thread.start()
        return {"ok": True}

    def _tool_worker(self, tool, rcap_root_text: str) -> None:
        try:
            self.refresh(rcap_root_text)
            info, status = self._info, self._status
            self.log(f"== {tool.label}: {tool.description}")
            code = self.runner.run(list(tool.argv), {**self._rcap_env(info), **dict(tool.env)}, self.log)
            self.log(f"== {tool.label}: " + ("stopped" if self._stop_requested else f"exit {code}"))
            if code == 0 and tool.key == "build-sim" and status is not None and info is not None:
                build.write_stamp(info.root, status.fingerprint, self.store.dir)
        except Exception as e:
            self.log(f"launcher error: {type(e).__name__}: {e}")
        finally:
            self._set_state("idle")
            self._safe_refresh()

    def _safe_refresh(self) -> None:
        try:
            self.refresh(self._probe_key or "")
        except Exception as e:
            self.log(f"rcap refresh failed: {e}")

    def stop(self) -> bool:
        """Ask the running job to end (SIGINT -> SIGTERM -> SIGKILL on its process group). Non-blocking."""
        with self._cond:
            if self._state not in ACTIVE:
                return False
            self._stop_requested = True
            self._state = "stopping"
            self._version += 1
            self._cond.notify_all()
        self.log("stopping…")
        threading.Thread(target=self._stopper, daemon=True).start()
        return True

    def _stopper(self) -> None:
        # Retry until the worker is gone: a stop that lands between two phases (build -> sim) finds no process.
        t = self._thread
        while t is not None and t.is_alive():
            self.runner.stop()
            t.join(0.3)

    def shutdown(self, timeout: float = 15.0) -> None:
        """Stop whatever runs and wait for the worker (used on SIGTERM/exit)."""
        self.stop()
        t = self._thread
        if t is not None:
            t.join(timeout)
