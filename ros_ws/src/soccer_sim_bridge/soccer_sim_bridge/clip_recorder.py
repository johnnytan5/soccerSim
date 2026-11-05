"""MuJoCo 3D clip of a match window (`match_node --record out.mp4 --record-window a:b`, PLAY seconds).

The sim never renders: like the viewer (viewer_link.py) it hands qpos to another process, here one that collects the frames
during the match and renders them when it is over (at ~65 ms per frame on llvmpipe, 250 ms on this WSL's D3D12 GPU path,
rendering live cannot keep 25 fps, and a slowed-down sim is the worse trade). It renders the stripped
viewer model offscreen from a free camera above the pitch (oblique), draws a per-robot label (id, role, LEAD, comm mode)
and a legend with cv2, and pipes BGR frames to ffmpeg (h264_nvenc, libx264 fallback). The sim side is `ClipRecorder.offer`:
a pickle into a bounded queue, never blocking the sim loop (a frame is dropped and counted when the renderer lags).
GL: SIM_RECORD_GL=cpu (default: Mesa llvmpipe) or gpu (the system driver; D3D12 on WSL, slower here, see docs/batch_matches.md).

Child process: python -m soccer_sim_bridge.clip_recorder MODEL.mjb OUT.mp4 OPTS_JSON   (frames on stdin, stats JSON on stdout)
"""
from __future__ import annotations

import json
import math
import os
import pickle
import queue
import subprocess
import sys
import threading
import time

import numpy as np

W, H, FPS = 1280, 720, 25
CAMERA = {"lookat": (0.0, -0.6, 0.0), "distance": 18.5, "elevation": -46.0, "azimuth": 90.0}
TEAM_BGR = {"home": (178, 114, 0), "away": (0, 159, 230)}      # Okabe-Ito blue / orange, as scripts/render_replay.py
TEAM_TEXT = {"home": (255, 255, 255), "away": (0, 0, 0)}
MODE_BGR = {"healthy": (115, 158, 0), "degraded": (66, 228, 240), "isolated": (0, 94, 213)}
MODE_TEXT = {"healthy": (255, 255, 255), "degraded": (0, 0, 0), "isolated": (255, 255, 255)}
MODE_RANK = {"healthy": 0, "degraded": 1, "isolated": 2}
STOP = None


class ClipRecorder:
    """Parent side: spawns the renderer, forwards frames from a thread so the sim loop only ever does a queue put."""

    def __init__(self, viewer_model, tints: dict, out: str, window: tuple[float, float], log=print, gl: str = "cpu"):
        import tempfile

        import mujoco

        from soccer_sim.viewer_look import apply_look

        from .viewer_link import viewer_environment

        self.window, self.out, self._log = window, out, log
        groups = apply_look(viewer_model, tints, False)
        self._dir = tempfile.mkdtemp(prefix="soccer_sim_clip_")
        path = os.path.join(self._dir, "world.mjb")
        mujoco.mj_saveModel(viewer_model, path)
        opts = {"groups": {str(k): v for k, v in groups.items()}, "camera": CAMERA}
        self._proc = subprocess.Popen([sys.executable, "-m", "soccer_sim_bridge.clip_recorder", path, out, json.dumps(opts)],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=viewer_environment(None, gl))
        self._q: queue.Queue = queue.Queue(maxsize=6)
        self.offered = self.dropped = 0
        self.stats: dict = {}
        self._thread = threading.Thread(target=self._pump, name="clip-recorder", daemon=True)
        self._thread.start()

    def in_window(self, play_s: float) -> bool:
        return self.window[0] <= play_s <= self.window[1]

    def offer(self, qpos: np.ndarray, info: dict) -> None:
        self.offered += 1
        try:
            self._q.put_nowait((qpos.copy(), info))
        except queue.Full:
            self.dropped += 1

    def _pump(self) -> None:
        while True:
            item = self._q.get()
            try:
                pickle.dump(item, self._proc.stdin, protocol=pickle.HIGHEST_PROTOCOL)
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                return
            if item is STOP:
                return

    def close(self) -> dict:
        """Finish the file; returns the renderer's timing statistics."""
        try:
            self._q.put(STOP, timeout=30)
        except queue.Full:  # the renderer died: nothing drains the queue
            pass
        self._thread.join(timeout=60)
        try:
            out, _ = self._proc.communicate(timeout=900)
            self.stats = json.loads(out.decode().strip().splitlines()[-1]) if out.strip() else {}
        except (subprocess.TimeoutExpired, ValueError):
            self._proc.kill()
        self.stats.update(offered=self.offered, dropped=self.dropped)
        self._log(f"clip {self.out}: {self.stats}")
        return self.stats


# ---- renderer process --------------------------------------------------------------------------------------
def project(scene, p: np.ndarray, width: int, height: int, fovy_deg: float = 45.0):
    """Pixel (u, v) of the world point p in the rendered scene's first camera (vertical field of view fovy), or None when it
    is behind the camera."""
    cam = scene.camera[0]
    fwd, up, pos = np.array(cam.forward), np.array(cam.up), np.array(cam.pos)
    right = np.cross(fwd, up)
    v = np.asarray(p, float) - pos
    z = float(v @ fwd)
    if z <= 0.1:
        return None
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    x_n, y_n = float(v @ right) / z * f * height / width, float(v @ up) / z * f
    return int(round((x_n + 1) / 2 * width)), int(round((1 - y_n) / 2 * height))


def draw_overlay(img: np.ndarray, scene, info: dict) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    robots = info["robots"]
    for r in robots:
        pt = project(scene, (r["x"], r["y"], 1.25), img.shape[1], img.shape[0], info.get("fovy", 45.0))
        if pt is None:
            continue
        role = (r.get("player_role") or "?")
        text = f"{'H' if r['team'] == 'home' else 'A'}{r['id']} {role.replace('goal_keeper', 'keeper')}" + (" LEAD" if r.get("im_lead") else "")
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        x0, y0 = pt[0] - tw // 2 - 4, pt[1] - th - 6
        cv2.rectangle(img, (x0, y0), (x0 + tw + 8, y0 + th + 8), TEAM_BGR[r["team"]], -1)
        cv2.rectangle(img, (x0, y0), (x0 + tw + 8, y0 + th + 8), (255, 255, 255) if r.get("im_lead") else (0, 0, 0),
                      3 if r.get("im_lead") else 1)
        cv2.putText(img, text, (x0 + 4, y0 + th + 3), font, 0.5, TEAM_TEXT[r["team"]], 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (30, 30, 30), -1)
    cv2.putText(img, f"t = {info['t']:.1f} s", (14, 31), font, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
    x = 190
    for team in ("home", "away"):
        modes = [r.get("comm_mode") for r in robots if r["team"] == team and r.get("comm_mode")]
        worst = max(modes, key=lambda m: MODE_RANK.get(m, 3)) if modes else "unknown"
        label = f"{team.upper()} comm: {worst}"
        (tw, _), _ = cv2.getTextSize(label, font, 0.6, 1)
        cv2.rectangle(img, (x, 6), (x + tw + 36, 38), MODE_BGR.get(worst, (120, 120, 120)), -1)
        cv2.rectangle(img, (x, 6), (x + 12, 38), TEAM_BGR[team], -1)
        cv2.putText(img, label, (x + 22, 28), font, 0.6, MODE_TEXT.get(worst, (255, 255, 255)), 1, cv2.LINE_AA)
        x += tw + 52
    if info.get("blackout"):
        cv2.rectangle(img, (img.shape[1] - 300, 6), (img.shape[1] - 10, 38), (30, 30, 200), -1)
        cv2.putText(img, "TEAM LINK BLACKOUT", (img.shape[1] - 290, 29), font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def _child(model_path: str, out: str, opts: dict) -> None:
    import cv2
    import mujoco

    from soccer_sim.field import ROBOT_ACTUATOR_GROUP
    from soccer_sim.video import encoder_args, find_ffmpeg, open_encoder

    model = mujoco.MjModel.from_binary_path(model_path)
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    opt = mujoco.MjvOption()
    opt.actuatorgroup[ROBOT_ACTUATOR_GROUP] = 0
    for g, shown in opts["groups"].items():
        opt.geomgroup[int(g)] = shown
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    c = opts["camera"]
    cam.lookat[:] = c["lookat"]
    cam.distance, cam.elevation, cam.azimuth = c["distance"], c["elevation"], c["azimuth"]
    ffmpeg = find_ffmpeg()
    codec_args, codec = encoder_args(ffmpeg)
    enc = open_encoder(ffmpeg, out, codec_args, W, H, FPS)
    t_render, t_over, t_write, n = [], [], [], 0
    gl = os.environ.get("GALLIUM_DRIVER", "system default (GPU)")
    # Frames only queue up while the match runs (one pickle every 40 ms); rendering starts at the end, so it never competes
    # with the sim and brains for CPU/GPU, and slow rendering never drops or retimes a frame.
    frames = []
    stdin = sys.stdin.buffer
    while True:
        try:
            item = pickle.load(stdin)
        except EOFError:
            break
        if item is STOP:
            break
        frames.append(item)
    for qpos, info in frames:
        a = time.perf_counter()
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        rgb = renderer.render()
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        b = time.perf_counter()
        draw_overlay(img, renderer.scene, info)
        c_ = time.perf_counter()
        enc.stdin.write(img.tobytes())
        d = time.perf_counter()
        t_render.append((b - a) * 1e3)
        t_over.append((c_ - b) * 1e3)
        t_write.append((d - c_) * 1e3)
        n += 1
    enc.stdin.close()
    enc.wait()
    med = lambda xs: round(float(np.median(xs)), 2) if xs else None  # noqa: E731
    size = os.path.getsize(out) if os.path.exists(out) else 0
    print(json.dumps({"frames": n, "codec": codec, "gl": gl, "render_ms_median": med(t_render),
                      "render_ms_p95": round(float(np.percentile(t_render, 95)), 2) if t_render else None,
                      "overlay_ms_median": med(t_over), "encode_write_ms_median": med(t_write), "bytes": size}), flush=True)


if __name__ == "__main__":
    _child(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
