"""Minimal `.env` support for the launcher: KEY=VALUE lines, no extra dependency.

Existing environment variables always win, so `SOCCER_LAUNCHER_ALLOWED_HOSTS=... pixi run launch`
keeps working and a committed `.env` can never override an explicit shell export.
"""
from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _KEY_RE.match(key):
            continue  # malformed line: ignore rather than break the launcher
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = _strip_comment(value)
        values[key] = value
    return values


def _strip_comment(value: str) -> str:
    """Drop an unquoted trailing `# comment` (only when `#` follows whitespace)."""
    for i, ch in enumerate(value):
        if ch == "#" and i > 0 and value[i - 1].isspace():
            return value[:i].rstrip()
    return value


def load_dotenv(path: Path, environ: MutableMapping[str, str] | None = None) -> None:
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError:
        return
    env = os.environ if environ is None else environ
    for key, value in parse_dotenv(text).items():
        env.setdefault(key, value)
