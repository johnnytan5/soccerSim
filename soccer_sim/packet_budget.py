"""The GameController's per-team UDP packet budget. Pure Python, no I/O: counting lives in team_udp.py.

In the real competition every datagram a team puts on its team port counts against one whole-match total
(discovery and state alike, no reset at half-time) and a team that goes over has its score zeroed. A limit of 0
means no limit: packets are still counted and shown, nobody goes over.
"""
from __future__ import annotations

import threading

from .roster import TEAMS


class PacketBudget:
    def __init__(self, limit: int, teams: tuple = TEAMS):
        self.limit = max(0, int(limit))
        self._used = {t: 0 for t in teams}
        self._reported: set[str] = set()
        self._lock = threading.Lock()  # the UDP counter adds from its own thread

    def add(self, team: str, n: int = 1) -> None:
        with self._lock:
            if team in self._used and n > 0:
                self._used[team] += n

    def used(self, team: str) -> int:
        with self._lock:
            return self._used.get(team, 0)

    def remaining(self, team: str) -> int:
        return max(0, self.limit - self.used(team))

    def over(self, team: str) -> bool:
        return self.limit > 0 and self.used(team) > self.limit

    def newly_over(self) -> list[str]:
        """Teams that crossed the limit since the last call, each reported once (for a single log line)."""
        fresh = [t for t in self._used if self.over(t) and t not in self._reported]
        self._reported.update(fresh)
        return fresh

    def effective_score(self, score: dict) -> dict:
        """The score the GameController shows: zero for a team that is over budget. The input is not changed."""
        return {t: (0 if self.over(t) else s) for t, s in score.items()}

    def snapshot(self) -> dict:
        return {t: {"used": self.used(t), "limit": self.limit, "remaining": self.remaining(t), "over": self.over(t)}
                for t in self._used}
