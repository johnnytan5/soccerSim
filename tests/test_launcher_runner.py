import os
import sys
import threading
import time

from soccer_launcher.runner import Runner


def test_streams_output_and_exit_code(tmp_path):
    lines = []
    code = Runner(tmp_path).run(
        [sys.executable, "-c", "import os;print('hi', os.environ['X']);import sys;sys.exit(3)"], {"X": "1"}, lines.append)
    assert code == 3 and lines == ["hi 1"]


def test_missing_binary_reports(tmp_path):
    lines = []
    assert Runner(tmp_path).run(["definitely-not-a-binary"], {}, lines.append) == 127
    assert "could not start" in lines[0]


def test_stop_kills_whole_process_group(tmp_path):
    pidfile = tmp_path / "child.pid"
    # parent shell spawns a long-lived child and waits, like run_sim.sh with its bridge
    script = f"sleep 300 & echo $! > {pidfile}; wait"
    r = Runner(tmp_path)
    result = {}
    t = threading.Thread(target=lambda: result.update(code=r.run(["bash", "-c", script], {}, lambda _l: None)))
    t.start()
    for _ in range(50):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.1)
    assert r.running
    r.stop(grace=2)
    t.join(10)
    assert not t.is_alive() and r.was_stopped and result["code"] != 0
    child = int(pidfile.read_text())
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("child outlived the job")


def test_second_run_while_busy_is_refused(tmp_path):
    r = Runner(tmp_path)
    t = threading.Thread(target=lambda: r.run(["sleep", "5"], {}, lambda _l: None))
    t.start()
    time.sleep(0.3)
    try:
        r.run(["true"], {}, lambda _l: None)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    r.stop(grace=1)
    t.join(10)
