import threading
import time

import pytest

from soccer_sim_bridge import gateway

pytestmark = pytest.mark.ros


@pytest.fixture
def shared(tmp_path):
    return gateway.Shared(str(tmp_path / "state.bin"), 3, create=True)


def test_nothing_published_reads_as_none(shared):
    assert shared.get_wm(0) is None


def test_roundtrip_and_per_robot_isolation(shared):
    assert shared.set_wm(1, b'{"a": 1}')
    stamp, data = shared.get_wm(1)
    assert data == b'{"a": 1}' and stamp > 0
    assert shared.get_wm(0) is None and shared.get_wm(2) is None
    assert shared.set_wm(1, b"x")
    assert shared.get_wm(1)[1] == b"x"  # a shorter snapshot replaces, it does not append


def test_oversized_snapshot_is_refused_and_the_old_one_stays(shared):
    shared.set_wm(0, b"old")
    assert not shared.set_wm(0, b"x" * (gateway.WM_BYTES + 1))
    assert shared.get_wm(0)[1] == b"old"
    assert shared.set_wm(0, b"y" * gateway.WM_BYTES)


def test_other_slots_are_not_disturbed(shared):
    shared.set_text(0, gateway.DECISION, "chase")
    shared.set_wm(0, b"{}")
    shared.rows[0][gateway.VX] = 0.5
    assert shared.get_text(0, gateway.DECISION) == "chase" and shared.rows[0][gateway.VX] == 0.5


def test_a_reader_never_sees_a_torn_snapshot(shared):
    stop = threading.Event()

    def writer():
        n = 0
        while not stop.is_set():
            n += 1
            shared.set_wm(0, (b"%d" % n) * 200)
            time.sleep(0.0002)  # a writer that never yields can starve the reader on a loaded machine

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        seen = 0
        for _ in range(3000):
            got = shared.get_wm(0)
            if got is None:
                continue
            seen += 1
            body = got[1]
            digits = body[: len(body) // 200]
            assert body == digits * 200  # always one writer's whole snapshot
        assert seen > 0
    finally:
        stop.set()
        thread.join()
