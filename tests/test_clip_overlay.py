import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "clip_recorder", Path(__file__).resolve().parent.parent / "ros_ws" / "src" / "soccer_sim_bridge" / "soccer_sim_bridge" / "clip_recorder.py")
cr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cr)
W, H = 1280, 720


def fake_scene():
    """A camera 12 m up and 10 m back from the pitch centre, looking at it (what project() reads from the rendered scene)."""
    pos = np.array([0.0, -10.0, 12.0])
    fwd = -pos / np.linalg.norm(pos)
    right = np.cross(fwd, [0, 0, 1.0])
    up = np.cross(right, fwd)
    return SimpleNamespace(camera=[SimpleNamespace(forward=fwd, up=up, pos=pos)])


def robot(team, i, x, y, decision, rank, lead=False, fused=(0.5, 0.0)):
    return {"id": i, "team": team, "x": x, "y": y, "theta": 0.0, "decision": decision, "player_role": "striker", "outfield_rank": rank,
            "im_lead": lead, "comm_mode": "degraded", "fused_ball": {"x": fused[0], "y": fused[1], "valid": True}}


def info(t, home_decisions, ball=(0.0, 0.0), blackout=False):
    return {"t": t, "ball": {"x": ball[0], "y": ball[1]}, "blackout": blackout, "robots": [
        robot("home", 2, -2.0, 1.0, home_decisions[0], 1, True), robot("home", 3, -3.0, -2.0, home_decisions[1], 2),
        robot("away", 2, 3.0, 0.0, "standby", 1), robot("away", 1, 9.0, 0.0, "-", -1)]}


def test_project_centre_is_image_centre_ish():
    u, v = cr.project(fake_scene(), (0.0, 0.0, 0.0), W, H)
    assert abs(u - W / 2) < 2 and 0 < v < H
    assert cr.project(fake_scene(), (0.0, -30.0, 0.0), W, H) is None  # behind the camera


def run(frames, caption=None):
    state = cr.OverlayState(caption or {"label": "new brain", "situation": "rcap_bad", "seed": 1})
    imgs = []
    for f in frames:
        img = np.full((H, W, 3), 80, np.uint8)
        cr.draw_overlay(img, fake_scene(), f, state)
        imgs.append(img)
    return state, imgs


def test_draw_overlay_marks_chasers_duplicates_and_counts():
    from soccer_sim.overlay import DUP_RING

    single, _ = run([info(0.0, ("chase", "standby"))])
    frames = [info(k / 25, ("chase", "kick")) for k in range(26)]
    state, imgs = run(frames)
    red = np.array(DUP_RING, np.uint8)
    one = _count(run([info(0.0, ("chase", "standby"))])[1][0], red)
    two = _count(imgs[0], red)
    assert two - one > 200                       # ground rings under both home chasers
    st = state.counters.stats("home")
    assert st["dup_s"] > 0.9 and st["chase_flips"] == 0 and st["worst_mode"] == "degraded"
    assert abs(st["fused_err_m"] - 0.5) < 1e-6
    assert state.label == "new brain" and "Network: Congested" in state.situation_line
    assert imgs[0].shape == (H, W, 3) and (imgs[0] != 80).any()


def _count(img, colour):
    return int((img == colour).all(axis=2).sum())


def test_draw_overlay_without_optional_fields_and_with_state_default():
    f = info(0.0, ("chase", "standby"), blackout=True)
    for r in f["robots"]:
        r.update(fused_ball=None, outfield_rank=None, comm_mode=None, im_lead=None, player_role=None)
    img = np.full((H, W, 3), 80, np.uint8)
    cr.draw_overlay(img, fake_scene(), f)         # no state: a fresh one
    assert (img != 80).any()


def test_ball_event_flash_and_timer_drawn():
    frames = [info(k / 10, ("standby", "standby"), ball=(0.0, 0.0) if k < 10 else (4.0, 0.0)) for k in range(16)]
    state, imgs = run(frames)
    assert state.timer.display(1.4)["timers"]
    from soccer_sim.overlay import FLASH
    assert _count(imgs[10], np.array(FLASH, np.uint8)) > 100 and math.isfinite(state.timer.t)
