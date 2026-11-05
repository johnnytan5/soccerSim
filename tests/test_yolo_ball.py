from pathlib import Path

import numpy as np
import pytest

from soccer_sim.config import DEFAULT_YOLO_MODEL, CameraCfg
from soccer_sim.perception import Camera, Pose2D
from soccer_sim.yolo_ball import Letterbox, YoloBallDetector, decode_balls, ground_point_at_height, letterbox

CFG = CameraCfg()


def level_camera(z=1.2):
    return Camera(CFG, np.array([0.0, 0.0, z]), np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))


def test_letterbox_shape_scale_and_padding():
    rgb = np.full((448, 544, 3), 200, np.uint8)
    x, lb = letterbox(rgb)
    assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32
    assert lb.scale == pytest.approx(320 / 544)
    assert lb.pad_x == 0 and lb.pad_y == (320 - round(448 * 320 / 544)) // 2
    assert x[0, :, 0, 0] == pytest.approx([114 / 255] * 3)  # top padding row
    assert x[0, :, 160, 160] == pytest.approx([200 / 255] * 3, abs=1e-6)


def test_decode_maps_boxes_back_to_image_and_applies_threshold_and_nms():
    out = np.zeros((13, 4), np.float32)
    # columns: cx, cy, w, h, then 9 class scores (class 0 = Ball, 1 = Goalpost)
    out[:4, 0], out[4, 0] = [160, 160, 20, 20], 0.9  # ball, kept
    out[:4, 1], out[4, 1] = [162, 160, 20, 20], 0.8  # overlapping ball, suppressed by NMS
    out[:4, 2], out[4, 2] = [60, 60, 10, 10], 0.3  # ball below threshold
    out[:4, 3], out[4, 3], out[5, 3] = [250, 250, 10, 10], 0.5, 0.9  # top class Goalpost, ignored
    boxes = decode_balls(out, Letterbox(scale=0.5, pad_x=0, pad_y=0), 1000, 1000, 0.4, 0.4)
    assert len(boxes) == 1
    conf, box = boxes[0]
    assert conf == pytest.approx(0.9)
    assert box == pytest.approx((300.0, 300.0, 340.0, 340.0))


def test_ground_point_at_ball_height():
    cam = level_camera(z=1.2)
    radius, forward = 0.105, 2.0
    v = cam.cy + cam.fy * (1.2 - radius) / forward
    p = ground_point_at_height(cam, cam.cx, v, radius)
    assert p == pytest.approx([forward, 0.0, radius], abs=1e-6)


def test_ray_at_or_above_horizon_returns_none():
    cam = level_camera()
    assert ground_point_at_height(cam, cam.cx, cam.cy, 0.105) is None  # horizontal ray
    assert ground_point_at_height(cam, cam.cx, cam.cy - 50, 0.105) is None  # upward ray


@pytest.mark.skipif(not Path(DEFAULT_YOLO_MODEL).exists(), reason="run `pixi run fetch-models`")
def test_real_model_sees_no_ball_in_empty_turf():
    det = YoloBallDetector(DEFAULT_YOLO_MODEL)
    rng = np.random.default_rng(0)
    turf = np.clip(rng.normal([40, 140, 50], 12, (448, 544, 3)), 0, 255).astype(np.uint8)
    assert det.detect(turf, level_camera(), Pose2D(0.0, 0.0, 0.0), 0.105) == []


def test_select_providers():
    import pytest

    from soccer_sim.yolo_ball import select_providers

    cpu, both = ["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert select_providers("auto", cpu) == (["CPUExecutionProvider"], 4)
    assert select_providers("auto", both) == (both, 1)
    assert select_providers("", both) == (both, 1)
    assert select_providers("cpu", both) == (["CPUExecutionProvider"], 4)
    assert select_providers("CUDA", both) == (both, 1)
    with pytest.raises(RuntimeError):
        select_providers("cuda", cpu)
    with pytest.raises(ValueError):
        select_providers("tpu", cpu)
