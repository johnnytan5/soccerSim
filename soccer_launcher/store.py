"""Persistence: last-launched config and named profiles in <project>/.launcher/state.json."""
from __future__ import annotations

import functools
import json
import os
import tempfile
import threading
from pathlib import Path

from .config import LaunchConfig

VERSION = 1


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        with self._lock:
            return fn(self, *a, **kw)
    return wrapper


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class Store:
    def __init__(self, directory: Path | None = None):
        self.dir = Path(directory or os.environ.get("SOCCER_LAUNCHER_DIR") or project_root() / ".launcher")
        self.path = self.dir / "state.json"
        self.last: LaunchConfig | None = None
        self.profiles: dict[str, LaunchConfig] = {}
        self.load_warning = ""
        self._lock = threading.RLock()

    @_locked
    def load(self) -> None:
        self.last, self.profiles, self.load_warning = None, {}, ""
        try:
            raw = json.loads(self.path.read_text())
            if not isinstance(raw, dict):
                raise ValueError("state is not an object")
        except FileNotFoundError:
            return
        except (OSError, ValueError) as e:
            self._quarantine(e)
            return
        if isinstance(raw.get("last"), dict):
            self.last = LaunchConfig.from_dict(raw["last"])
        profiles = raw.get("profiles")
        if isinstance(profiles, dict):
            self.profiles = {str(k): LaunchConfig.from_dict(v) for k, v in profiles.items() if isinstance(v, dict)}

    def _quarantine(self, err: Exception) -> None:
        backup = self.path.with_suffix(".json.bad")
        try:
            os.replace(self.path, backup)
            self.load_warning = f"state.json was unreadable ({err}); moved to {backup.name}"
        except OSError:
            self.load_warning = f"state.json was unreadable ({err})"

    @_locked
    def save(self) -> None:
        data = {
            "version": VERSION,
            "last": self.last.to_dict() if self.last else None,
            "profiles": {k: v.to_dict() for k, v in sorted(self.profiles.items())},
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".state-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @_locked
    def set_last(self, cfg: LaunchConfig) -> None:
        self.last = cfg
        self.save()

    @_locked
    def save_profile(self, name: str, cfg: LaunchConfig) -> None:
        self.profiles[name.strip()] = cfg
        self.save()

    @_locked
    def delete_profile(self, name: str) -> None:
        self.profiles.pop(name, None)
        self.save()
