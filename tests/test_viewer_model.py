import mujoco
import pytest

from soccer_sim.config import SimConfig
from soccer_sim.field import build_world


@pytest.mark.parametrize("n", [1, 3])
def test_stripped_viewer_model_has_the_same_layout_but_almost_no_mesh_data(n):
    full, lite = build_world(SimConfig(), n), build_world(SimConfig(), n, strip_visual=True)
    for attr in ("nq", "nv", "nu", "nbody", "njnt", "nsensor", "ncam"):
        assert getattr(full, attr) == getattr(lite, attr), attr
    names = lambda m, kind, k: [mujoco.mj_id2name(m, kind, i) for i in range(k)]
    assert names(full, mujoco.mjtObj.mjOBJ_JOINT, full.njnt) == names(lite, mujoco.mjtObj.mjOBJ_JOINT, lite.njnt)
    assert names(full, mujoco.mjtObj.mjOBJ_BODY, full.nbody) == names(lite, mujoco.mjtObj.mjOBJ_BODY, lite.nbody)
    assert int(full.mesh_facenum.sum()) > 900_000 * n and int(lite.mesh_facenum.sum()) < 2_000 * n
    assert (lite.geom_group == 1).sum() == 0 and (lite.geom_group == 3).sum() == 21 * n  # collision shapes stay


def test_a_pose_from_the_sim_model_poses_the_stripped_model_identically():
    import numpy as np
    full, lite = build_world(SimConfig(), 2), build_world(SimConfig(), 2, strip_visual=True)
    d, dl = mujoco.MjData(full), mujoco.MjData(lite)
    d.qpos[:] = full.qpos0
    d.qpos[0:3] = (1.0, 2.0, 0.7); d.qpos[7] = 0.4
    dl.qpos[:] = d.qpos
    mujoco.mj_forward(full, d); mujoco.mj_forward(lite, dl)
    assert np.allclose(d.xpos, dl.xpos)
