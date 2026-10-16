import numpy as np
import pytest

from soccer_sim_bridge.viewer_link import TEXT_BYTES, _State

pytestmark = pytest.mark.ros


def _pair(tmp_path, nq=7, n_sliders=3, nbody=4):
    path = str(tmp_path / "state.bin")
    return _State(path, nq, n_sliders, nbody, create=True), _State(path, nq, n_sliders, nbody, create=False)


def test_pose_goes_sim_to_viewer(tmp_path):
    sim, viewer = _pair(tmp_path)
    sim.qpos[:] = np.arange(7)
    assert viewer.qpos.tolist() == list(range(7))


def test_sliders_go_viewer_to_sim(tmp_path):
    sim, viewer = _pair(tmp_path)
    viewer.sliders[:] = [1.0, 2.0, 3.0]
    assert sim.sliders.tolist() == [1.0, 2.0, 3.0]
    assert sim.qpos.tolist() == [0.0] * 7  # the two regions don't overlap


def test_drag_force_goes_viewer_to_sim(tmp_path):
    sim, viewer = _pair(tmp_path)
    viewer.xfrc[2] = [45.0, 0.0, 22.5, 0.0, 0.0, 0.0]
    assert sim.xfrc.shape == (4, 6)
    assert sim.xfrc[2].tolist() == [45.0, 0.0, 22.5, 0.0, 0.0, 0.0]
    assert sim.xfrc[[0, 1, 3]].sum() == 0.0
    assert sim.sliders.tolist() == [0.0] * 3  # the regions don't overlap


def test_overlay_text_round_trips_and_shrinks(tmp_path):
    sim, viewer = _pair(tmp_path)
    sim.set_text("brain decision\nball speed", "retreat\n3.2 m/s")
    assert viewer.get_text() == ("brain decision\nball speed", "retreat\n3.2 m/s")
    sim.set_text("a", "b")
    assert viewer.get_text() == ("a", "b")


def test_overlay_text_is_truncated_not_overflowed(tmp_path):
    sim, viewer = _pair(tmp_path)
    sim.set_text("x" * (2 * TEXT_BYTES), "y")
    left, right = viewer.get_text()
    assert len(left) == TEXT_BYTES and right == ""
    assert sim.sliders.tolist() == [0.0] * 3  # text never spills into the numeric regions


def test_viewer_gets_the_gpu_unless_cpu_is_asked_for(monkeypatch):
    from soccer_sim_bridge import viewer_link
    monkeypatch.setattr(viewer_link.sys, "platform", "linux")
    camera_env = {"DISPLAY": ":0", "GALLIUM_DRIVER": "llvmpipe", "LIBGL_ALWAYS_SOFTWARE": "1"}  # what the camera switch leaves
    gpu = viewer_link.viewer_environment(camera_env, "gpu")
    assert "GALLIUM_DRIVER" not in gpu and "LIBGL_ALWAYS_SOFTWARE" not in gpu and gpu["DISPLAY"] == ":0"
    cpu = viewer_link.viewer_environment({"DISPLAY": ":0"}, "cpu")
    assert cpu["GALLIUM_DRIVER"] == "llvmpipe" and cpu["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert viewer_link.viewer_environment({"GALLIUM_DRIVER": "zink"}, "gpu")["GALLIUM_DRIVER"] == "zink"  # not ours to touch
    assert camera_env["GALLIUM_DRIVER"] == "llvmpipe"  # the sim's own environment is left alone


def test_macos_viewer_environment_is_never_changed(monkeypatch):
    from soccer_sim_bridge import viewer_link
    monkeypatch.setattr(viewer_link.sys, "platform", "darwin")
    base = {"HOME": "/Users/x"}
    assert viewer_link.viewer_environment(base, "cpu") == base == viewer_link.viewer_environment(base, "gpu")


def test_local_mirror_carries_poses_in_and_the_persons_input_out():
    import mujoco
    import numpy as np
    from soccer_sim.config import SimConfig
    from soccer_sim.field import build_world
    from soccer_sim_bridge.viewer_link import LocalMirror
    sim_model = build_world(SimConfig(), 2)
    view_model = build_world(SimConfig(), 2, strip_visual=True)
    sim = mujoco.MjData(sim_model)
    n = sim_model.nu - 6
    sliders = np.array([-6.0, 0.0, 4.0, 180.0, 0.0, 0.0])
    mirror = LocalMirror(view_model, n, sliders)
    assert np.allclose(mirror.data.ctrl[n:], sliders)
    sim.qpos[:] = sim_model.qpos0
    sim.qpos[0:3] = (3.0, -2.0, 0.7)
    mirror.from_sim(sim)
    assert np.allclose(mirror.data.qpos, sim.qpos)
    assert mirror.data.xpos[1][:2] == pytest.approx((3.0, -2.0), abs=0.2)  # the trunk follows
    mirror.data.ctrl[n + 2] = 7.5          # dragged the speed slider in the window
    mirror.data.xfrc_applied[1, :3] = (10, 0, 0)   # Ctrl+right-drag on a robot
    mirror.to_sim(sim)
    assert sim.ctrl[n + 2] == 7.5 and sim.xfrc_applied[1, 0] == 10
