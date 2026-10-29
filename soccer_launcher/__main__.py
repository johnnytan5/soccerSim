"""python -m soccer_launcher [--last]   (pixi run launch)"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import build, command, rcap
from .config import DEFAULT_CONFIG
from . import settings
from .store import Store, project_root


def _relaunch_last() -> int:
    """--last: no TUI, just build if stale and run the last saved config in the foreground."""
    store = Store()
    store.load()
    cfg = store.last or DEFAULT_CONFIG
    errs = cfg.validate(settings.registry_for(rcap.locate(cfg.rcap_root)))
    if errs:
        print("last config is invalid: " + "; ".join(errs), file=sys.stderr)
        return 2
    root = rcap.locate(cfg.rcap_root)
    info = rcap.inspect(root)
    if not info.valid:
        print(f"rcap clone not found at {root} (no src/brain)", file=sys.stderr)
        return 2
    status = build.check(root, store.dir)
    env_root = {"ROBOTEDGE_ROOT": str(root)}
    cwd = project_root()
    if status.stale:
        print(f"rcap build is stale ({status.reason}); building…")
        if subprocess.call(build.BUILD_ARGV, cwd=cwd, env={**os.environ, **build.BUILD_ENV, **env_root}) != 0:
            return 1
        build.write_stamp(root, status.fingerprint, store.dir)
    argv, env = command.build_command(cfg, str(root))
    return subprocess.call(argv, cwd=cwd, env={**os.environ, **env})


DEFAULT_PORT = 8765


def _open_browser(url: str) -> None:
    import shutil
    import webbrowser
    try:
        if os.environ.get("WSL_DISTRO_NAME") and shutil.which("wslview"):
            subprocess.run(["wslview", url], timeout=10, capture_output=True)
        else:
            webbrowser.open(url)
    except Exception:
        pass  # best effort: the URL is printed anyway


def _serve(port: int, open_browser: bool) -> int:
    import signal
    import threading

    from . import portguard
    from .jobs import JobController
    from .server import LauncherServer

    try:
        portguard.claim(port)
    except portguard.PortBusy as e:
        print(f"soccer launcher: {e}", file=sys.stderr)
        return 1
    store = Store()
    store.load()
    jobs = JobController(store)
    jobs.refresh(store.last.rcap_root if store.last else "")
    try:
        server = LauncherServer(jobs, port)
    except OSError as e:
        print(f"soccer launcher: cannot listen on port {port}: {e}", file=sys.stderr)
        return 1
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    pidfile = store.dir / "server.pid"
    store.dir.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(os.getpid()))
    server.start()
    threading.Thread(target=jobs.background_refresh, args=(stop,), daemon=True).start()
    url = f"http://127.0.0.1:{server.port}/"
    print(f"Soccer Sim Control Centre: {url}   (Ctrl-C to quit; a running sim is stopped first)")
    if open_browser:
        _open_browser(url)
    try:
        stop.wait()
    finally:
        print("shutting down…")
        jobs.shutdown()
        server.stop()
        try:
            pidfile.unlink()
        except OSError:
            pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="soccer_launcher", description="soccerSim web Control Centre")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to serve on (default {DEFAULT_PORT})")
    ap.add_argument("--no-open", action="store_true", help="don't try to open a browser")
    ap.add_argument("--last", action="store_true", help="skip the web UI and run the last saved config")
    args = ap.parse_args()
    if args.last:
        return _relaunch_last()
    return _serve(args.port, not args.no_open)


if __name__ == "__main__":
    sys.exit(main())
