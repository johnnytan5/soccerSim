"""UDP probe through a netem profile (the method of docs/netem.md): robot 1 broadcasts numbered 64 B datagrams at a fixed
rate from port 30001 to the team port, robot 2 receives them; loss, burst length and one-way delay are computed from the
sequence numbers and timestamps (CLOCK_MONOTONIC: the wall clock of this WSL steps by seconds during a run). Needs `scripts/netem.sh up` and the profile applied (to robot 1).

  pixi run python scripts/netem_probe.py --profile lat100_loss5 [--count 4000] [--rate 200] [--slot ""]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 10131

SENDER = r"""
import socket, struct, sys, time
dst, count, rate = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("", 30001))
t0 = time.monotonic()
for i in range(count):
    s.sendto(struct.pack("!Id", i, time.monotonic()).ljust(64, b"\0"), (dst, %d))
    d = t0 + (i + 1) / rate - time.monotonic()
    if d > 0: time.sleep(d)
""" % PORT

RECEIVER = r"""
import socket, struct, sys, time, json
dur = float(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("", %d)); s.settimeout(0.5)
got = []; end = time.monotonic() + dur
while time.monotonic() < end:
    try: b = s.recv(2048)
    except socket.timeout: continue
    i, ts = struct.unpack("!Id", b[:12]); got.append((i, (time.monotonic() - ts) * 1e3))
print(json.dumps(got))
""" % PORT


def analyse(got: list, count: int) -> dict:
    seen = {i for i, _ in got}
    delays = [d for _, d in got]
    lost = [i not in seen for i in range(count)]
    bursts, run = [], 0
    for x in lost:
        if x:
            run += 1
        elif run:
            bursts.append(run)
            run = 0
    if run:
        bursts.append(run)
    srt = sorted(delays)
    return {"duplicates": len(got) - len(seen), "delay_p50_ms": round(srt[len(srt) // 2], 1) if srt else None,
            "delay_p99_ms": round(srt[int(len(srt) * 0.99)], 1) if srt else None,
            "sent": count, "received": len(seen), "loss_pct": round(100 * sum(lost) / count, 2),
            "mean_burst": round(statistics.mean(bursts), 2) if bursts else 0.0,
            "delay_mean_ms": round(statistics.mean(delays), 1) if delays else None,
            "delay_sd_ms": round(statistics.pstdev(delays), 1) if delays else None,
            "delay_max_ms": round(max(delays), 1) if delays else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--count", type=int, default=4000)
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--slot", default=os.environ.get("SS_SLOT", ""))
    ap.add_argument("--keep-up", action="store_true", help="leave the namespaces up afterwards")
    a = ap.parse_args()
    env = {**os.environ, "SS_SLOT": a.slot, "SS_ROBOTS": "3"}
    sh = lambda *args: subprocess.run(["bash", str(ROOT / "scripts" / "netem.sh"), *args], env=env, check=True,  # noqa: E731
                                      capture_output=True, text=True).stdout
    names = dict(kv.split("=", 1) for kv in sh("names").split())
    ns1, ns2 = names["ns1"], names["ns1"][:-1] + "2"
    sh("up")
    try:
        sh("apply", a.profile, "1")
        run = lambda ns, code, *args: subprocess.Popen(["sudo", "-n", "ip", "netns", "exec", ns, sys.executable, "-c", code, *args],  # noqa: E731
                                                       stdout=subprocess.PIPE, text=True)
        dur = a.count / a.rate + 3.0
        rx = run(ns2, RECEIVER, str(dur))
        time.sleep(0.5)
        run(ns1, SENDER, names["broadcast"], str(a.count), str(a.rate)).wait()
        got = json.loads(rx.communicate()[0])
        res = {"profile": a.profile, "netem": sh("status").count("netem"), **analyse(got, a.count)}
        print(json.dumps(res)); print("outliers", [(i, round(d)) for i, d in got if abs(d) > 1000][:10])
    finally:
        if not a.keep_up:
            sh("down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
