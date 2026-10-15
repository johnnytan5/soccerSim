"""Claiming the launcher's port: replace a previous launcher instance, never touch anything else."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .server import APP_ID


class PortBusy(Exception):
    pass


@dataclass(frozen=True)
class Holder:
    is_launcher: bool
    pid: int | None = None
    detail: str = ""


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _ss_holder(port: int) -> str:
    """Best-effort 'who listens here' for the error message (Linux `ss`)."""
    try:
        out = subprocess.run(["ss", "-H", "-ltnp", f"sport = :{port}"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.strip().splitlines()[0] if out.strip() else ""


def identify(port: int, timeout: float = 1.5) -> Holder:
    """Ask whatever listens on `port` whether it is a soccer launcher (and verify its pid)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/launcher/identity", timeout=timeout) as r:
            data = json.loads(r.read(4096))
    except (OSError, ValueError, urllib.error.URLError):
        return Holder(False, detail=_ss_holder(port))
    pid = data.get("pid") if isinstance(data, dict) and data.get("app") == APP_ID else None
    if not isinstance(pid, int) or pid <= 1 or pid == os.getpid():
        return Holder(False, detail=_ss_holder(port))
    cmd = _cmdline(pid)
    if "soccer_launcher" not in cmd:  # its self-reported pid must really be a launcher process
        return Holder(False, detail=_ss_holder(port))
    return Holder(True, pid, cmd)


def claim(port: int, wait: float = 25.0, say=print) -> None:
    """Return when `port` can be bound. Terminates a previous launcher on it; raises PortBusy for anything else."""
    if port_free(port):
        return
    holder = identify(port)
    if not holder.is_launcher:
        who = f" ({holder.detail})" if holder.detail else ""
        raise PortBusy(f"port {port} is in use by another program{who}. I only replace a previous soccer launcher, "
                       f"so stop that program or use `pixi run launch --port <other>`.")
    say(f"port {port} is held by a previous launcher (pid {holder.pid}); stopping it (it stops its sim first)…")
    try:
        os.kill(holder.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if port_free(port):
            return
        time.sleep(0.25)
    raise PortBusy(f"the previous launcher (pid {holder.pid}) did not release port {port} within {wait:.0f}s")
