"""Measure the MuJoCo T2 walk policy and print the numbers soccer_sim/kinematic_params.py is built from.

    pixi run calibrate-kinematics

Runs the real walk policy (needs MuJoCo; no ROS) and reports, per axis, the steady speed reached for a commanded speed
(gain), the time to 63 % of it, the time to fall to 37 % after the command stops, and the peak acceleration; plus the
standing trunk height and the head pose. Single runs: gait oscillation makes each figure good to about 10 %.
Nothing is written: copy what changed into kinematic_params.py.
"""
import math

import numpy as np

from soccer_sim.config import SimConfig
from soccer_sim.loco_rpc import MODE_SOCCER, LocoState
from soccer_sim.sim import SoccerController, World


def main() -> None:
    cfg = SimConfig(ball_detector="ground_truth")
    world = World(cfg, 1)
    robot = SoccerController(cfg, world, 0, spawn=(0.0, 0.0, 0.0))
    robot.start()
    loco = LocoState(mode=MODE_SOCCER)
    dt, clock = robot.cfg.policy_dt, [0.0]

    def run(seconds, cmd=(0.0, 0.0, 0.0), rec=None):
        loco.vx, loco.vy, loco.vyaw = cmd
        for _ in range(int(seconds / dt)):
            robot.apply(loco, clock[0])
            world.step()
            clock[0] += dt
            if rec is not None:
                p = robot.robot_pose()
                rec.append((clock[0], p.x, p.y, p.yaw))

    run(3.0)
    pos, _ = robot.head_pose_in_base()
    print(f"standing trunk z {float(world.mj_data.xpos[robot._trunk_body][2]):.4f} m, "
          f"trunk pitch {robot.imu()[0][1]:.3f} rad, head in base {pos.round(4).tolist()}")
    for name, axis, cmd in (("vx", 0, (0.5, 0, 0)), ("vx", 0, (1.0, 0, 0)), ("vx", 0, (-0.5, 0, 0)),
                            ("vy", 1, (0, 0.4, 0)), ("vyaw", 2, (0, 0, 0.8))):
        robot.reset_robot(0.0, 0.0, 0.0)
        run(2.0)
        rec: list = []
        run(5.0, cmd, rec)
        run(3.0, rec=rec)
        t0, speeds = rec[0][0], []
        for a, b in zip(rec[:-1], rec[1:]):
            h = b[0] - a[0]
            c, s = math.cos(a[3]), math.sin(a[3])
            dx, dy = b[1] - a[1], b[2] - a[2]
            v = ((c * dx + s * dy) / h, (-s * dx + c * dy) / h, ((b[3] - a[3] + math.pi) % (2 * math.pi) - math.pi) / h)
            speeds.append((b[0] - t0, v[axis]))
        target = cmd[axis]
        steady = float(np.mean([v for t, v in speeds if 3.0 < t < 5.0]))
        first = lambda pred: next((round(t, 2) for t, v in speeds if pred(v)), None)  # noqa: E731
        t63 = first(lambda v: abs(v) > 0.632 * abs(steady))
        t37 = next((round(t - 5.0, 2) for t, v in speeds if t >= 5.0 and abs(v) < 0.368 * abs(steady)), None)
        acc = max(abs(speeds[i + 5][1] - speeds[i][1]) / (speeds[i + 5][0] - speeds[i][0])
                  for i in range(len(speeds) - 5) if speeds[i][0] < 3.0)
        print(f"{name} cmd {target:+.2f}: steady {steady:+.3f} (gain {steady / target:.2f}), "
              f"t63 {t63} s, decel t37 {t37} s, peak accel {acc:.2f}")


if __name__ == "__main__":
    main()
