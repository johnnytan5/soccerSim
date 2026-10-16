import math

import numpy as np
import pytest

from soccer_sim.config import SimConfig
from soccer_sim.frames import TeamFrame
from soccer_sim.loco_rpc import LocoState
from soccer_sim.sim import SoccerController, World

pytestmark = pytest.mark.slow
DT = 0.02


def make(n_home=1, n_away=1):
    cfg = SimConfig(ball_detector="ground_truth", ball_xy=(0.0, 0.0))
    world = World(cfg, n_home + n_away)
    home, away = TeamFrame(False), TeamFrame(True)
    robots = []
    for i in range(n_home + n_away):
        is_home = i < n_home
        spawn = (-6.0, -7.5, math.pi / 2)  # in its own team frame: the same entry spot for both teams
        robots.append(SoccerController(cfg, world, i, home if is_home else away, spawn))
    world.start()
    return world, robots


def run(world, robots, locos, seconds):
    t = 0.0
    for _ in range(int(seconds / DT)):
        for r, loco in zip(robots, locos):
            r.apply(loco, t)
        world.step()
        t += DT


def test_two_robots_share_one_world_and_stand_independently():
    world, (a, b) = make(1, 1)
    assert a.mj_data is b.mj_data and world.n_robot_ctrl == 62
    run(world, [a, b], [LocoState(), LocoState()], 2.0)
    assert not a.fallen() and not b.fallen()
    pa, pb = a.robot_pose(), b.robot_pose()
    assert (pa.x, pa.y) != pytest.approx((pb.x, pb.y))


def test_away_robot_spawns_mirrored_but_its_own_frame_matches_home():
    world, (a, b) = make(1, 1)
    pa, pb = a.robot_pose(), b.robot_pose()
    assert (pb.x, pb.y) == pytest.approx((-pa.x, -pa.y), abs=0.01)  # the away entry spot is the home one, rotated
    ta, tb = a.team_pose(), b.team_pose()
    assert (ta.x, ta.y) == pytest.approx((tb.x, tb.y), abs=0.01) and ta.yaw == pytest.approx(tb.yaw, abs=0.05)


def test_walking_forward_is_forward_in_each_teams_odometer():
    world, (a, b) = make(1, 1)
    run(world, [a, b], [LocoState(vx=0.5), LocoState(vx=0.5)], 4.0)
    assert not a.fallen() and not b.fallen()
    for r in (a, b):
        x, y, _ = r.odometer()
        assert x > 1.0 and abs(y) < 0.6
    # Same command, mirrored start: they end up mirrored in the world.
    pa, pb = a.robot_pose(), b.robot_pose()
    assert pb.x == pytest.approx(-pa.x, abs=0.4) and pb.y == pytest.approx(-pa.y, abs=0.4)


def test_each_robot_sees_the_ball_in_its_own_frame_and_the_other_as_opponent():
    world, (a, b) = make(1, 1)
    world.reset_ball(-6.0, -4.0)
    run(world, [a, b], [LocoState(), LocoState()], 1.0)
    a.reset_robot(-6.0, -6.0, math.pi / 2)  # looking at the ball and at nothing else
    dets, _ = a.perceive(others=[(0.0, 0.0)])
    ball = next(d for d in dets if d.label == "Ball")
    assert ball.pos_robot[0] > 1.0 and abs(ball.pos_robot[1]) < 0.3
    # b stands where the home robot's mirror is; put it in front of a and a sees an Opponent
    pa = a.robot_pose()
    dets, _ = a.perceive(others=[(pa.x, pa.y + 2.0)])
    opp = [d for d in dets if d.label == "Opponent"]
    assert len(opp) == 1 and opp[0].pos_robot[0] == pytest.approx(2.0, abs=0.2)
    # the away robot at the mirrored spot sees the mirrored ball at the same robot-relative place
    b.reset_robot(6.0, 6.0, -math.pi / 2)
    world.reset_ball(6.0, 4.0)  # the mirror image of what a saw
    dets_b, _ = b.perceive()
    ball_b = next(d for d in dets_b if d.label == "Ball")
    assert ball_b.pos_robot == pytest.approx(ball.pos_robot, abs=0.05)


def test_camera_hides_every_robot_body_and_restores_the_scene():
    world, (a, b) = make(1, 1)
    m = world.mj_model
    before = m.geom_group.copy()
    seen = {}
    real_update = world._renderer if world._renderer else None
    world.render(a)  # creates the renderer
    import mujoco
    orig = mujoco.Renderer.update_scene

    def spy(self, data, camera=-1, scene_option=None):
        seen["hidden"] = set(np.flatnonzero(m.geom_group == 4))
        return orig(self, data, camera=camera, scene_option=scene_option)

    mujoco.Renderer.update_scene = spy
    try:
        world.render(b)
    finally:
        mujoco.Renderer.update_scene = orig
    assert set(a.visual_geoms) | set(b.visual_geoms) <= seen["hidden"]
    assert set(a.visual_geoms).isdisjoint(set(b.visual_geoms)) and len(a.visual_geoms) > 10
    assert (m.geom_group == before).all()  # restored afterwards


def test_ball_is_shared_and_kick_launches_it():
    world, (a, b) = make(1, 1)
    world.shoot(__import__("soccer_sim.shots", fromlist=["Shot"]).Shot(0.0, 0.0, 3.0, 0.0, 0.0))
    assert a.ball_velocity()[0] == pytest.approx(3.0) and b.ball_position()[0] == pytest.approx(0.0)
    assert np.allclose(a.ball_position(), b.ball_position())
