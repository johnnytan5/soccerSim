"""MuJoCo viewer in its own process, fed by the sim through a small memory-mapped file.

Drawing the viewer inside the sim process makes the two fight over the GIL and the CPU (and, on Linux, over
the GL driver the head camera needs), and the sim then falls well below real time. Here the sim only writes the
robot/ball pose (qpos); the viewer process replays it with mj_forward and hands the shot sliders back.

State file layout (float64 header, then utf-8 text):
    [0]               sim heartbeat (counts up on every write)
    [1 : 1 + nq]      qpos, sim -> viewer
    [1 + nq : ...]    slider values (ctrl[n_robot:]), viewer -> sim
    [... : +6*nbody]  xfrc_applied, viewer -> sim: the mouse-drag force (Ctrl+right-drag) the viewer computes
    [... : +TEXT]     overlay: two utf-8 columns joined by a NUL byte, NUL padded; sim -> viewer
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np

TEXT_BYTES = 2048


class _State:
    def __init__(self, path: str, nq: int, n_sliders: int, nbody: int, create: bool):
        n_float = 1 + nq + n_sliders + 6 * nbody
        size = n_float * 8 + TEXT_BYTES
        if create:
            with open(path, "wb") as f:
                f.truncate(size)
        mm = np.memmap(path, dtype=np.uint8, mode="r+", shape=(size,))
        self._mm = mm
        self.floats = mm[: n_float * 8].view(np.float64)
        self.text = mm[n_float * 8 :]
        self.qpos = self.floats[1 : 1 + nq]
        self.sliders = self.floats[1 + nq : 1 + nq + n_sliders]
        self.xfrc = self.floats[1 + nq + n_sliders :].reshape(nbody, 6)

    def set_text(self, left: str, right: str) -> None:
        raw = (left.encode() + b"\0" + right.encode())[:TEXT_BYTES]
        self.text[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        self.text[len(raw) :] = 0

    def get_text(self) -> tuple[str, str]:
        left, _, right = bytes(self.text).rstrip(b"\0").partition(b"\0")
        return left.decode(errors="replace"), right.decode(errors="replace")


class ViewerLink:
    """Sim side: spawns the viewer process and exchanges state with it."""

    def __init__(self, model, n_robot_ctrl: int, sliders: np.ndarray, env: dict | None = None):
        """env: environment for the viewer process (default: ours); the sim may have changed the GL driver for itself."""
        import mujoco

        self._dir = tempfile.mkdtemp(prefix="soccer_sim_viewer_")
        model_path = os.path.join(self._dir, "world.mjb")
        state_path = os.path.join(self._dir, "state.bin")
        mujoco.mj_saveModel(model, model_path)
        n_sliders = model.nu - n_robot_ctrl
        self._state = _State(state_path, model.nq, n_sliders, model.nbody, create=True)
        self._state.sliders[:] = sliders
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "soccer_sim_bridge.viewer_link", model_path, state_path,
             str(model.nq), str(n_robot_ctrl), str(n_sliders), str(os.getpid())],
            env=env,
        )
        self._beat = 0.0

    def is_running(self) -> bool:
        return self._proc.poll() is None

    def sliders(self) -> np.ndarray:
        return self._state.sliders

    def xfrc(self) -> np.ndarray:
        """Mouse-drag forces from the viewer, one row (force, torque) per body."""
        return self._state.xfrc

    def sync(self, qpos: np.ndarray, left: str | None = None, right: str = "") -> None:
        """Publish the pose; left/right (two text columns) update the overlay along the top when given."""
        self._beat += 1.0
        self._state.qpos[:] = qpos
        self._state.floats[0] = self._beat
        if left is not None:
            self._state.set_text(left, right)

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        for name in ("world.mjb", "state.bin"):
            try:
                os.remove(os.path.join(self._dir, name))
            except OSError:
                pass
        try:
            os.rmdir(self._dir)
        except OSError:
            pass


def _viewer_main(model_path, state_path, nq, n_robot, n_sliders, parent_pid) -> None:
    import mujoco
    import mujoco.viewer

    from soccer_sim.field import ROBOT_ACTUATOR_GROUP

    model = mujoco.MjModel.from_binary_path(model_path)
    data = mujoco.MjData(model)
    state = _State(state_path, nq, n_sliders, model.nbody, create=False)
    data.qpos[:] = state.qpos
    data.ctrl[n_robot:] = state.sliders
    mujoco.mj_forward(model, data)

    period = 1.0 / 60.0
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True) as viewer:
        with viewer.lock():
            # Hide the robot motors so the Control panel shows only the shot sliders.
            viewer.opt.actuatorgroup[ROBOT_ACTUATOR_GROUP] = 0
        last_text = None
        next_t = time.monotonic()
        # Leave when the window closes or the sim is gone (the sim's pid stops being our parent).
        while viewer.is_running() and os.getppid() == parent_pid:
            with viewer.lock():
                data.qpos[:] = state.qpos
                mujoco.mj_forward(model, data)
                state.sliders[:] = data.ctrl[n_robot:]
                state.xfrc[:] = data.xfrc_applied  # written by viewer.sync() from a Ctrl+right-drag
                viewer.cam.lookat[:] = data.qpos[0:3]
            text = state.get_text()
            if text != last_text:
                last_text = text
                viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, *text))
            viewer.sync()
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()


if __name__ == "__main__":
    _viewer_main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
