import importlib.util
import json
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("render_replay", Path(__file__).resolve().parent.parent / "scripts" / "render_replay.py")
rr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rr)


def _row(t, x, blackout=False):
    robots = [{"id": i, "team": team, "x": x + i, "y": 0.0, "theta": 0.0, "decision": "-", "player_role": "striker",
               "im_lead": i == 2, "outfield_rank": 1, "role_own_view": False, "comm_mode": "healthy", "links": [
                   {"player": j, "health": 1.0, "latency_ms": 5.0, "loss": 0.0} for j in (1, 2) if j != i],
               "fused_ball": None, "own_ball": None} for team in ("home", "away") for i in (1, 2)]
    return {"t": t, "ball": {"x": x, "y": 1.0, "vx": 0, "vy": 0}, "blackout": blackout, "robots": robots}


def _trace(tmp_path, rows):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return rr.load_trace(str(p))


def test_state_interpolates_between_rows(tmp_path):
    tr = _trace(tmp_path, [_row(0.0, 0.0), _row(0.1, 1.0, blackout=True)])
    st = rr.state_at(tr, 0.05)
    assert abs(st["ball"]["x"] - 0.5) < 1e-9 and not st["blackout"] and st["covered"]
    assert abs(st["robots"][0]["x"] - 1.5) < 1e-9
    assert not rr.state_at(tr, 5.0)["covered"]


def test_panel_draws_robots_and_blackout_tint(tmp_path):
    tr = _trace(tmp_path, [_row(0.0, 0.0), _row(0.1, 0.0, blackout=True)])
    panel = rr.Panel(0, rr.W, "x")
    plain, dark = (np.zeros((rr.H, rr.W, 3), np.uint8) for _ in range(2))
    panel.draw(plain, rr.state_at(tr, 0.0), 0.0)
    panel.draw(dark, rr.state_at(tr, 0.1), 0.1)
    assert plain.any() and not np.array_equal(plain, dark)
    assert dark[..., 2].astype(int).sum() > plain[..., 2].astype(int).sum()  # the blackout tint is red
