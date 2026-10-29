"""The 2D simulator against the MuJoCo walk policy it was calibrated on: same commands, similar motion."""
import math

import numpy as np
import pytest

from soccer_sim.config import SimConfig
from soccer_sim.frames import TeamFrame
from soccer_sim.kinematic import KinematicController, KinematicWorld
from soccer_sim.kinematic_params import KinematicParams
from soccer_sim.loco_rpc import MODE_SOCCER, LocoState

pytestmark = pytest.mark.slow
CFG = SimConfig(ball_detector="ground_truth")


def _run(world, robot, script):
    loco, t, dt, track = LocoState(mode=MODE_SOCCER), 0.0, 0.02, []
    for secs, cmd in script:
        loco.vx, loco.vy, loco.vyaw = cmd
        for _ in range(int(secs / dt)):
            robot.apply(loco, t)
            world.step()
            t += dt
            p = robot.robot_pose()
            track.append((p.x, p.y, p.yaw))
    return np.array(track)


@pytest.mark.parametrize("script, tol_xy", [
    ([(3.0, (0, 0, 0)), (4.0, (0.5, 0, 0)), (1.0, (0, 0, 0))], 0.35),
    ([(3.0, (0, 0, 0)), (4.0, (1.0, 0, 0)), (1.0, (0, 0, 0))], 0.6),
    # walking while turning is the weakest part of the model: the policy couples forward speed and turn rate, the
    # kinematic robot does not (measured: about 0.46 m apart after a 2 m arc)
    ([(3.0, (0, 0, 0)), (4.0, (0.4, 0, 0.5)), (1.0, (0, 0, 0))], 0.55),
])
def test_same_commands_give_similar_tracks(script, tol_xy):
    from soccer_sim.sim import SoccerController, World

    mj_world = World(CFG, 1)
    mj_robot = SoccerController(CFG, mj_world, 0, spawn=(0.0, 0.0, 0.0))
    mj_robot.start()
    mj = _run(mj_world, mj_robot, script)

    k_world = KinematicWorld(CFG, 1, KinematicParams())
    k_robot = KinematicController(CFG, k_world, 0, TeamFrame(), (0.0, 0.0, 0.0), np.random.default_rng(0))
    k = _run(k_world, k_robot, script)

    assert np.hypot(*(mj[-1, :2] - k[-1, :2])) < tol_xy, (mj[-1], k[-1])
    assert abs(math.atan2(math.sin(mj[-1, 2] - k[-1, 2]), math.cos(mj[-1, 2] - k[-1, 2]))) < 0.4
