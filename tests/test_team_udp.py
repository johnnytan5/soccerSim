import socket
import time

from soccer_sim.roster import AWAY, HOME
from soccer_sim.team_udp import TeamPacketCounter, team_port


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def send(port: int, n: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for _ in range(n):
            s.sendto(b"x" * 40, ("127.0.0.1", port))


def wait_for(cond, timeout=2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_the_team_port_is_the_gamecontrollers_10000_plus_team_number():
    assert team_port(HOME) == 10131 and team_port(AWAY) == 10001


def test_every_datagram_is_counted_on_its_own_teams_port():
    got = {HOME: 0, AWAY: 0}
    ports = {HOME: free_port(), AWAY: free_port()}
    counter = TeamPacketCounter(ports, lambda team, n: got.__setitem__(team, got[team] + n))
    try:
        assert counter.active == {HOME, AWAY}
        send(ports[HOME], 7)
        send(ports[AWAY], 3)
        assert wait_for(lambda: got == {HOME: 7, AWAY: 3}), got
    finally:
        counter.close()


def test_a_port_that_cannot_be_bound_is_skipped_not_fatal():
    port = free_port()
    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # no SO_REUSEADDR: the port stays exclusive
    blocker.bind(("0.0.0.0", port))
    warnings = []
    try:
        counter = TeamPacketCounter({HOME: port}, lambda team, n: None, warn=warnings.append)
        assert counter.active == set() and warnings and str(port) in warnings[0]
        counter.close()
    finally:
        blocker.close()


def test_close_stops_counting_and_is_idempotent():
    got = []
    port = free_port()
    counter = TeamPacketCounter({HOME: port}, lambda team, n: got.append(n))
    counter.close()
    counter.close()
    send(port, 3)
    time.sleep(0.1)
    assert got == []
