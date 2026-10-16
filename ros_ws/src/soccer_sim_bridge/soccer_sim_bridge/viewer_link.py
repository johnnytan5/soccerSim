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

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

TEXT_BYTES = 2048


def viewer_environment(base: dict | None = None, gl: str = "gpu") -> dict:
    """The environment for the viewer process: your graphics card (gl="gpu") or Mesa's software rasteriser ("cpu").

    The sim switches its own head camera to llvmpipe on WSL (see bridge_node._pick_camera_gl); the viewer must not
    inherit that, or the window is drawn by the CPU. Only Linux has a software choice: macOS always uses the
    system's graphics, whatever `gl` says.
    """
    env = dict(os.environ if base is None else base)
    if sys.platform.startswith("linux"):
        if gl == "cpu":
            env["GALLIUM_DRIVER"] = "llvmpipe"
            env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        else:
            if env.get("GALLIUM_DRIVER") == "llvmpipe":
                del env["GALLIUM_DRIVER"]
            env.pop("LIBGL_ALWAYS_SOFTWARE", None)
    return env


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


class LocalMirror:
    """macOS: the viewer lives in the sim's own process, but still gets a model and data of its own (a small model
    without the visual meshes), because the viewer copies the whole model on every sync. Kept in step like the
    memory-mapped link does for a separate process: poses in, shot sliders and mouse-drag force out."""

    def __init__(self, model, n_robot_ctrl: int, sliders: np.ndarray):
        import mujoco

        self._mujoco = mujoco
        self.model, self.n_robot = model, n_robot_ctrl
        self.data = mujoco.MjData(model)
        self.data.ctrl[n_robot_ctrl:] = sliders
        self.data.qpos[:] = 0.0
        mujoco.mj_forward(model, self.data)

    def to_sim(self, sim_data) -> None:
        """What the person did in the viewer window (caller holds the viewer's lock)."""
        sim_data.ctrl[self.n_robot:] = self.data.ctrl[self.n_robot:]
        sim_data.xfrc_applied[:] = self.data.xfrc_applied

    def from_sim(self, sim_data) -> None:
        """Show the sim's latest poses (caller holds the viewer's lock)."""
        self.data.qpos[:] = sim_data.qpos
        self._mujoco.mj_forward(self.model, self.data)


class ViewerLink:
    """Sim side: spawns the viewer process and exchanges state with it."""

    def __init__(self, model, n_robot_ctrl: int, sliders: np.ndarray, env: dict | None = None, overview: bool = False,
                 tints: dict | None = None, detailed: bool = False, gl: str = "gpu"):
        """env: environment for the viewer process (default: ours); the sim may have changed the GL driver for itself.

        overview: look down at the whole pitch instead of following the first robot (matches).
        gl: "gpu" (default) or "cpu": which renderer the viewer window uses, see `viewer_environment`.
        tints: robot root body id -> RGBA. Given (a match), the viewer draws cheap team-coloured bodies with shadows off
        (see soccer_sim.viewer_look); `detailed` keeps the real meshes instead.
        """
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
             str(model.nq), str(n_robot_ctrl), str(n_sliders), str(os.getpid()), "overview" if overview else "follow",
             json.dumps({"tints": {str(k): list(v) for k, v in tints.items()}, "detailed": detailed}) if tints else ""],
            env=viewer_environment(env, gl),
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


def _viewer_main(model_path, state_path, nq, n_robot, n_sliders, parent_pid, mode="follow", look="") -> None:
    import mujoco
    import mujoco.viewer

    from soccer_sim.field import ROBOT_ACTUATOR_GROUP

    from soccer_sim.viewer_look import apply_look

    model = mujoco.MjModel.from_binary_path(model_path)
    groups = {}
    if look:
        spec = json.loads(look)
        groups = apply_look(model, {int(k): v for k, v in spec["tints"].items()}, spec["detailed"])
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
            for group, shown in groups.items():
                viewer.opt.geomgroup[group] = shown
            if mode == "overview":
                viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
                viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 26.0, -55.0, 90.0
        last_text = None
        next_t = time.monotonic()
        # Leave when the window closes or the sim is gone (the sim's pid stops being our parent).
        while viewer.is_running() and os.getppid() == parent_pid:
            with viewer.lock():
                data.qpos[:] = state.qpos
                mujoco.mj_forward(model, data)
                state.sliders[:] = data.ctrl[n_robot:]
                state.xfrc[:] = data.xfrc_applied  # written by viewer.sync() from a Ctrl+right-drag
                if mode != "overview":
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
    _viewer_main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
                 sys.argv[7] if len(sys.argv) > 7 else "follow", sys.argv[8] if len(sys.argv) > 8 else "")
