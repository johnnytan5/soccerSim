"""Ball detection with Booster's sim-trained YOLOv8 (robocup_demo sim_data_det_0126.onnx).

Mirrors Robotedge vision for the ball: per-class confidence >= 0.4, NMS IoU 0.4, and
range from the bbox-centre ray intersected with the plane z = ball radius
(BallPoseEstimator::EstimateByColor; use_depth is false for the ball in vision.yaml).
"""

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from .geometry import world_to_robot_2d
from .perception import Camera, Detection, Pose2D

BALL_CLASS = 0
INPUT_SIZE = 320
PAD_VALUE = 114


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int


def letterbox(rgb: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, Letterbox]:
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    pad_y, pad_x = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, Letterbox(scale, pad_x, pad_y)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def decode_balls(output, lb: Letterbox, width: int, height: int, conf_thresh: float, nms_iou: float):
    preds = np.asarray(output).T  # (N, 4 + classes)
    scores = preds[:, 4:]
    conf = scores[:, BALL_CLASS]
    keep = np.where((scores.argmax(1) == BALL_CLASS) & (conf >= conf_thresh))[0]
    boxes: list[tuple[float, tuple[float, float, float, float]]] = []
    for i in keep[np.argsort(-conf[keep])]:
        cx, cy, w, h = (float(v) for v in preds[i, :4])
        box = (
            max(0.0, (cx - w / 2 - lb.pad_x) / lb.scale),
            max(0.0, (cy - h / 2 - lb.pad_y) / lb.scale),
            min(width - 1.0, (cx + w / 2 - lb.pad_x) / lb.scale),
            min(height - 1.0, (cy + h / 2 - lb.pad_y) / lb.scale),
        )
        if all(_iou(box, kept) <= nms_iou for _, kept in boxes):
            boxes.append((float(conf[i]), box))
    return boxes


def ground_point_at_height(cam: Camera, u: float, v: float, z: float) -> np.ndarray | None:
    ray = cam.rot @ np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
    if ray[2] >= -1e-6:  # at or above the horizon: never meets the plane in front of the camera
        return None
    t = (z - cam.pos[2]) / ray[2]
    return cam.pos + t * ray if t > 0.0 else None


class YoloBallDetector:
    def __init__(self, model_path: str, conf_thresh: float = 0.4, nms_iou: float = 0.4):
        # ORT defaults to one spinning thread per core, which starves the sim, brain and renderer on a busy desktop.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        names = self.session.get_modelmeta().custom_metadata_map.get("names", "")
        if "'Ball'" not in names.split(",")[0]:
            raise ValueError(f"{model_path}: class 0 is not 'Ball' (names={names})")
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thresh = conf_thresh
        self.nms_iou = nms_iou

    def detect(self, rgb: np.ndarray, cam: Camera, robot: Pose2D, ball_radius: float) -> list[Detection]:
        tensor, lb = letterbox(rgb)
        output = self.session.run(None, {self.input_name: tensor})[0][0]
        h, w = rgb.shape[:2]
        out = []
        for conf, (x0, y0, x1, y1) in decode_balls(output, lb, w, h, self.conf_thresh, self.nms_iou):
            p = ground_point_at_height(cam, (x0 + x1) / 2, (y0 + y1) / 2, ball_radius)
            if p is None:
                continue
            pos = world_to_robot_2d(float(p[0]), float(p[1]), robot.x, robot.y, robot.yaw)
            out.append(Detection("Ball", conf, (int(x0), int(y0), int(x1), int(y1)), pos))
        return out
