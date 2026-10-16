"""Process supervisor: runs one command in its own process group and streams its output (thread-based)."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .build import BUILD_ARGV, BUILD_ENV
from .store import project_root

LineSink = Callable[[str], None]


@dataclass(frozen=True)
class Tool:
    key: str
    label: str
    description: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()


TOOLS: tuple[Tool, ...] = (
    Tool("build-sim", "Build rcap (sim)", "pixi run build-sim: copy+patch rcap brain, colcon build brain + bridge",
         tuple(BUILD_ARGV), tuple(BUILD_ENV.items())),
    Tool("build-ws", "Build brain only", "pixi run build-ws: colcon build the brain package only",
         tuple(BUILD_ARGV)),
    Tool("fetch-models", "Fetch YOLO model", "pixi run fetch-models: download the ball detector ONNX",
         ("bash", "scripts/fetch_models.sh")),
    Tool("test", "Tests (fast)", "pixi run test: pytest -m 'not slow and not ros'",
         ("pytest", "-m", "not slow and not ros")),
    Tool("test-ros", "Tests (ROS)", "pixi run test-ros: bridge tests against the built workspace",
         ("bash", "-c", "source ros_ws/install/setup.bash && pytest -m 'ros and not slow' "
                        "ros_ws/src/soccer_sim_bridge/test -p no:cacheprovider")),
    Tool("test-slow", "Tests (slow)", "pixi run test-slow: pytest -m slow (steps MuJoCo)",
         ("pytest", "-m", "slow")),
    Tool("setup", "First-time setup", "scripts/setup.sh: submodules, pixi install, models, build, tests",
         ("bash", "scripts/setup.sh")),
)
TOOLS_BY_KEY = {t.key: t for t in TOOLS}


class Runner:
    """One job at a time. `run` blocks (call it from a worker thread) and returns the exit code;
    `stop` is safe to call from any other thread."""

    def __init__(self, cwd: Path | None = None):
        self.cwd = Path(cwd) if cwd else project_root()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stopping = False

    @property
    def running(self) -> bool:
        p = self._proc
        return p is not None and p.poll() is None

    @property
    def was_stopped(self) -> bool:
        return self._stopping

    def run(self, argv: list[str], env: dict[str, str], on_line: LineSink) -> int:
        with self._lock:
            if self._proc is not None:
                raise RuntimeError("a job is already running")
            self._stopping = False
            try:
                proc = subprocess.Popen(
                    argv, cwd=self.cwd, env={**os.environ, **env}, start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except OSError as e:
                on_line(f"could not start {argv[0]}: {e}")
                return 127
            self._proc = proc
        reader = threading.Thread(target=self._pump, args=(proc, on_line), daemon=True)
        reader.start()
        try:
            code = proc.wait()
            self._signal_group(proc.pid, signal.SIGTERM)  # bridge/viewer children must not outlive the job
            reader.join(timeout=3)
            return code
        finally:
            with self._lock:
                self._proc = None

    @staticmethod
    def _pump(proc: subprocess.Popen, on_line: LineSink) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            on_line(raw.decode("utf-8", "replace").rstrip("\r\n"))

    @staticmethod
    def _signal_group(pid: int, sig: int) -> bool:
        try:
            os.killpg(pid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def stop(self, grace: float = 5.0) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self._stopping = True
        for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
            if not self._signal_group(proc.pid, sig):
                return
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue
