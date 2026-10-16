"""How many robots can the sim tick in real time? Usage: pixi run python scripts/bench_match.py [N ...]

Reports milliseconds per 20 ms tick for the walk policies + physics, for ground-truth perception, and for one
rendered head-camera frame (RGB + depth). A tick has a 20 ms budget at 50 Hz.
"""
import math
import sys
import time

from soccer_sim.config import SimConfig
from soccer_sim.frames import TeamFrame
from soccer_sim.loco_rpc import LocoState
from soccer_sim.sim import SoccerController, World


def bench(n: int) -> None:
    cfg = SimConfig(ball_detector="ground_truth", ball_xy=(0.0, 0.0))
    world = World(cfg, n)
    robots = [SoccerController(cfg, world, i, TeamFrame(i >= (n + 1) // 2), (-6.0 + 0.5 * (i % 5), -7.5, math.pi / 2))
              for i in range(n)]
    world.start()
    locos = [LocoState(vx=0.3) for _ in robots]
    for _ in range(25):
        for r, loco in zip(robots, locos):
            r.apply(loco, 0.0)
        world.step()
    t = time.perf_counter()
    ticks = 100
    for _ in range(ticks):
        for r, loco in zip(robots, locos):
            r.apply(loco, 0.0)
        world.step()
    step_ms = (time.perf_counter() - t) / ticks * 1000
    others = [(r.robot_pose().x, r.robot_pose().y) for r in robots]
    t = time.perf_counter()
    for _ in range(20):
        for r in robots:
            r.perceive(others=others)
    gt_ms = (time.perf_counter() - t) / 20 * 1000
    world.render(robots[0])
    t = time.perf_counter()
    for _ in range(10):
        world.render(robots[0])
    cam_ms = (time.perf_counter() - t) / 10 * 1000
    print(f"{n:2d} robots: step {step_ms:6.2f} ms/tick  ground-truth perception {gt_ms:6.2f} ms/tick (all robots, "
          f"every tick)  one camera frame {cam_ms:5.1f} ms  -> {1000 / (step_ms + gt_ms / 2):.0f} Hz with perception every 2nd tick")


if __name__ == "__main__":
    for count in [int(a) for a in sys.argv[1:]] or [1, 2, 5, 10]:
        bench(count)
