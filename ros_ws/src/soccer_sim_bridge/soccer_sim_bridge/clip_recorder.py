"""MuJoCo 3D clip of a match window (`match_node --record out.mp4 --record-window a:b`, PLAY seconds).

The sim never renders: like the viewer (viewer_link.py) it hands qpos to another process, here one that collects the frames
during the match and renders them when it is over (at ~65 ms per frame on llvmpipe, 250 ms on this WSL's D3D12 GPU path,
rendering live cannot keep 25 fps, and a slowed-down sim is the worse trade). It renders the stripped
viewer model offscreen from a free camera above the pitch (oblique), draws the status overlays of soccer_sim/overlay.py (the
same as scripts/render_replay.py: per-robot label with decision word and rank, chaser marker, duplicate ring on the ground,
fused-ball belief dots, ball-event flash and reaction timer, per-team HUD, caption, legend) with cv2, and pipes BGR frames to ffmpeg (h264_nvenc, libx264 fallback). The sim side is `ClipRecorder.offer`:
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
STOP = None


class ClipRecorder:
    """Parent side: spawns the renderer, forwards frames from a thread so the sim loop only ever does a queue put."""

    def __init__(self, viewer_model, tints: dict, out: str, window: tuple[float, float], log=print, gl: str = "cpu",
                 caption: dict | None = None):
        import tempfile

        import mujoco

        from soccer_sim.viewer_look import apply_look

        from .viewer_link import viewer_environment

        self.window, self.out, self._log = window, out, log
        groups = apply_look(viewer_model, tints, False)
        self._dir = tempfile.mkdtemp(prefix="soccer_sim_clip_")
        path = os.path.join(self._dir, "world.mjb")
        mujoco.mj_saveModel(viewer_model, path)
        opts = {"groups": {str(k): v for k, v in groups.items()}, "camera": CAMERA, "caption": caption or {}}
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
        """Queue one frame, never blocking. `info` is the trace row (match_node._trace_row): t, ball {x, y}, blackout and per
        robot id, team, x, y, theta, decision, player_role, outfield_rank, im_lead, comm_mode, fused_ball, ... (all that
        draw_overlay needs; a robot missing a field just gets no mark for it)."""
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


class OverlayState:
    """What the overlay remembers between frames: the running counters and the reaction timer (fed once per frame, in order)
    and the caption. The renderer creates one per clip."""

    def __init__(self, caption: dict | None = None):
        from soccer_sim import overlay as ov

        cap = caption or {}
        self.counters, self.timer = ov.RunningCounters(), ov.ReactionTimer()
        self.label, self.situation_line = ov.caption_text(cap.get("label") or "", cap.get("situation"), cap.get("seed"))


def draw_overlay(img: np.ndarray, scene, info: dict, state: OverlayState | None = None) -> None:
    import cv2

    from soccer_sim import overlay as ov

    state = state or OverlayState()
    height, width = img.shape[:2]
    robots, ball = info["robots"], info["ball"]
    state.counters.update(info)
    state.timer.update(info)
    fovy = info.get("fovy", 45.0)

    def to_px(x, y, z=0.0):
        return project(scene, (x, y, z), width, height, fovy)

    ov.belief_marks(img, to_px, robots, ball)
    dup = ov.duplicate_teams(robots)
    for r in robots:
        if ov.is_chaser(r) and r["team"] in dup:  # red ring on the ground under the robot
            ov.ground_ring(img, to_px, r["x"], r["y"], 0.75, (0, 0, 0), 7)
            ov.ground_ring(img, to_px, r["x"], r["y"], 0.75, ov.DUP_RING, 4)
    top = ov.caption_height(width, state.label, state.situation_line)
    placer = ov.LabelPlacer((0, top + 2, width, height - 2))
    pts = {id(r): to_px(r["x"], r["y"], 1.25) for r in robots}
    texts = {}
    for r in robots:
        role = (r.get("player_role") or "?").replace("goal_keeper", "keeper")
        words = ov.label_text(r)
        texts[id(r)] = f"{ov.robot_key(r)} {words or role}" + (" LEAD" if r.get("im_lead") else "")
    for r in sorted(robots, key=lambda q: not ov.is_chaser(q)):
        pt = pts[id(r)]
        if pt is None:
            continue
        text = texts[id(r)]
        tw, th = ov.text_size(text)
        bw, bh = tw + 10, th + 8
        x0, y0 = placer.place(bw, bh, [(pt[0] - bw // 2, pt[1] - bh - 6 - k * (bh + 4)) for k in range(4)])
        lead = bool(r.get("im_lead"))
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), ov.TEAM[r["team"]], -1)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255) if lead else (0, 0, 0), 3 if lead else 1)
        cv2.putText(img, text, (x0 + 5, y0 + bh - 6), ov.FONT, ov.TEXT_SCALE, ov.TEAM_TEXT[r["team"]], 1, cv2.LINE_AA)
        cv2.line(img, (x0 + bw // 2, y0 + bh), (pt[0], pt[1] + 6), (0, 0, 0), 1, cv2.LINE_AA)
        if ov.is_chaser(r):  # filled triangle above the label
            ov.chaser_marker(img, (x0 + bw // 2, y0 - 3), r["team"])
    # caption (top), HUD per team (bottom corners) with time and legend between
    ov.draw_caption(img, 0, 0, width, state.label, state.situation_line)
    teams = ("home", "away")
    block_w, hud_h = 310, ov.HUD_ROW * 5
    base = height - hud_h - 8
    for team, x in zip(teams, (8, width - block_w - 8)):
        plate = img[base - 4:base + hud_h + 4, x - 4:x + block_w + 4]
        plate[:] = (plate * 0.35).astype(np.uint8)
        ov.draw_team_hud(img, x, base, block_w, team, state.counters.stats(team))
    mid = block_w + 24
    legend_w = width - 2 * mid
    leg_h = ov.legend_height(legend_w)
    plate = img[height - leg_h - 52:height - 4, mid:width - mid]
    plate[:] = (plate * 0.35).astype(np.uint8)
    t_txt = f"t = {info['t']:.1f} s"
    ov.put_text(img, t_txt, (width // 2 - ov.text_size(t_txt, 1.0, 2)[0] // 2, height - leg_h - 18), 1.0, ov.TEXT, 2)
    ov.draw_legend(img, mid + 8, height - leg_h - 8, legend_w - 16)
    ov.event_marks(img, to_px, state.timer.display(info["t"]), (0, top, width, height), ball)
    if info.get("blackout"):
        cv2.rectangle(img, (width - 300, top + 6), (width - 10, top + 38), (30, 30, 200), -1)
        cv2.putText(img, "TEAM LINK BLACKOUT", (width - 290, top + 29), ov.FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


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
    state = OverlayState(opts.get("caption"))
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
        draw_overlay(img, renderer.scene, info, state)
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
