"""The macOS run loop (the viewer lives in the sim process), driven here with a stand-in viewer."""
import contextlib
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from soccer_sim.config import SimConfig
from soccer_sim.field import build_world
from soccer_sim_bridge import match_node
from soccer_sim_bridge.viewer_link import LocalMirror

pytestmark = pytest.mark.ros


class FakeViewer:
    def __init__(self, frames):
        self.frames, self.synced, self.texts, self.locked = frames, 0, [], 0

    def is_running(self):
        return self.synced < self.frames

    @contextlib.contextmanager
    def lock(self):
        self.locked += 1
        yield

    def set_texts(self, *t):
        self.texts.append(t)

    def sync(self):
        self.synced += 1


def test_run_loop_mirrors_poses_to_the_viewer_model_and_input_back_to_the_sim():
    world_model = build_world(SimConfig(), 2)
    world = SimpleNamespace(mj_data=mujoco.MjData(world_model), n_robot_ctrl=world_model.nu - 6, policy_dt=0.0)
    n = world.n_robot_ctrl
    world.mj_data.qpos[:] = world_model.qpos0
    mirror = LocalMirror(build_world(SimConfig(), 2, strip_visual=True), n, world.mj_data.ctrl[n:])
    ticks = []

    def tick():
        ticks.append(1)
        world.mj_data.qpos[0] = 0.5 * len(ticks)   # the sim moves robot 0
        mirror.data.ctrl[n + 2] = 9.0              # the person drags a slider in the window

    bridge = SimpleNamespace(
        channels=[SimpleNamespace(sim=SimpleNamespace(cfg=SimpleNamespace(policy_dt=0.0)))], world=world,
        tick=tick, note_tick=lambda: None, overlay=lambda: ("left", "right"))
    viewer = FakeViewer(frames=6)
    match_node._run_realtime(bridge, viewer, None, mirror=mirror)
    assert len(ticks) == 6 and viewer.synced == 6 and viewer.locked >= 12  # always under the viewer's lock
    assert mirror.data.qpos[0] == pytest.approx(3.0)             # the viewer's model shows the sim's latest pose
    assert world.mj_data.ctrl[n + 2] == 9.0                       # and the slider reached the sim
    assert viewer.texts and viewer.texts[0][0][2:] == ("left", "right")
