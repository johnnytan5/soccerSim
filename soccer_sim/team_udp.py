"""Counts the datagrams each team puts on its team port, the way the GameController does. Stdlib only.

The sim has no UDP of its own: every robot's brain broadcasts on 10000 + team number, with SO_REUSEADDR. A second
SO_REUSEADDR socket on the same port is handed every broadcast too, so it sees exactly what the GameController would.
"""
from __future__ import annotations

import select
import socket
import threading
from collections.abc import Callable

from .roster import TEAM_NUMBERS

MAX_PACKET_BYTES = 512  # the official GameController accepts at most this on the team channel


def team_port(team: str) -> int:
    """Mirrors discoveryPortForTeam in the brain's brain_communication.cpp."""
    return 10000 + TEAM_NUMBERS[team]


class TeamPacketCounter:
    """`ports` maps team -> UDP port. `on_packet(team, n)` runs on the counter's thread for every batch received."""

    def __init__(self, ports: dict[str, int], on_packet: Callable[[str, int], None],
                 warn: Callable[[str], None] = print):
        self._on_packet = on_packet
        self._socks: dict[socket.socket, str] = {}
        for team, port in ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.bind(("0.0.0.0", port))
                sock.setblocking(False)
            except OSError as e:
                sock.close()
                warn(f"[packets] {team}: cannot listen on UDP {port} ({e}); its packets will not be counted")
                continue
            self._socks[sock] = team
        self.active = set(self._socks.values())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="team-packet-counter", daemon=True)
        if self._socks:
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select(list(self._socks), [], [], 0.1)
            for sock in ready:
                n = 0
                while True:  # drain what is queued, then report once
                    try:
                        sock.recv(MAX_PACKET_BYTES + 1)
                    except BlockingIOError:
                        break
                    except OSError:
                        return
                    n += 1
                if n:
                    self._on_packet(self._socks[sock], n)

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for sock in self._socks:
            sock.close()
        self._socks = {}
