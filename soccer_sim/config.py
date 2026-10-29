from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

import numpy as np

DEFAULT_YOLO_MODEL = str(Path(__file__).resolve().parents[1] / "vendor" / "models" / "sim_data_det_0126.onnx")

# (name, min, max, default) — drawn as sliders in the MuJoCo viewer's top-right Control panel.
SHOT_SLIDERS: tuple[tuple[str, float, float, float], ...] = (
    ("shot_origin_x", -11.0, 11.0, -6.0),
    ("shot_origin_y", -7.0, 7.0, 0.0),
    ("shot_speed", 0.0, 10.0, 4.0),  # m/s
    ("shot_direction_deg", -180.0, 180.0, 180.0),  # field frame; 180 = toward our goal (-x)
    ("shot_elevation_deg", 0.0, 45.0, 0.0),
    ("shot_fire", 0.0, 1.0, 0.0),  # drag above 0.5 to shoot, back below 0.5 to re-arm
)


@dataclass(frozen=True)
class FieldDims:
    """Mirror of Robotedge FD_ROBOLEAGUE (src/brain/include/types.h)."""

    length: float = 22.003
    width: float = 14.126
    penalty_dist: float = 3.635
    goal_width: float = 2.6
    circle_radius: float = 1.99
    penalty_area_length: float = 5.221
    penalty_area_width: float = 8.121
    goal_area_length: float = 2.307
    goal_area_width: float = 5.083


@dataclass(frozen=True)
class CameraCfg:
    """Mirror of $ROBOTEDGE_ROOT/src/vision/config/vision.yaml camera.intrin / camera.extrin.

    cam_to_head maps the optical frame (x right, y down, z forward) into the head frame,
    which the sim takes to be MuJoCo body head_pitch_link.
    """

    width: int = 544
    height: int = 448
    fx: float = 208.712936
    fy: float = 208.883377
    cx: float = 272.0
    cy: float = 224.0
    cam_to_head: tuple[tuple[float, float, float, float], ...] = (
        (0.0, -0.5716, 0.8206, 0.08684),
        (-1.0, 0.0, 0.0, 0.0349),
        (0.0, -0.8206, -0.5716, 0.08245),
        (0.0, 0.0, 0.0, 1.0),
    )

    def cam_to_head_matrix(self) -> np.ndarray:
        return np.array(self.cam_to_head, dtype=float)


@dataclass(frozen=True)
class SimConfig:
    field: FieldDims = dc_field(default_factory=FieldDims)
    camera: CameraCfg = dc_field(default_factory=CameraCfg)
    ball_radius: float = 0.105  # size 5, same as vision.yaml ball_pose_estimator.ball_size
    ball_mass: float = 0.45
    goalpost_radius: float = 0.05
    goal_height: float = 1.2
    line_width: float = 0.05
    # Default spawn = brain's "enter_field" prior for player_start_pos "right":
    # own half, just outside the right touchline, facing into the field.
    spawn_xy_yaw: tuple[float, float, float] = (-6.0, -7.5, 1.5707963267948966)
    ball_xy: tuple[float, float] = (-6.0, -4.0)
    max_ball_range: float = 8.0
    max_goalpost_range: float = 12.0
    # ponytail: real-vision detection range for crosses is unmeasured; brain accepts up to 15 m and needs
    # >= 4 per frame to localise, which from the sideline spawn requires ~8.1 m. Calibrate from robot logs.
    max_marking_range: float = 9.0
    max_line_range: float = 5.0
    max_depth_m: float = 10.0
    # ponytail: shadows/reflections cost ~12 ms per RGB frame on an M-series Mac and push camera ticks past the
    # 20 ms real-time budget; YOLO still finds the ball without them (conf ~0.5-0.7 vs ~0.7-0.8 with).
    render_shadows: bool = False
    # ponytail: ground-truth goalposts/markings/lines are perfect; raise to emulate vision noise.
    detection_noise_std: float = 0.0
    # sigma = detection_noise_std + detection_noise_range_gain * range: far objects are noisier
    detection_noise_range_gain: float = 0.0
    # Odometer error (both simulators): position error as % of the distance walked, heading error in degrees per
    # 10 m walked. A bias is drawn once per robot; noise accumulates with distance, so a standing robot stays put.
    odom_drift_pct: float = 0.0
    odom_yaw_drift_deg: float = 0.0
    seed: int | None = None  # fixes every random stream (noise, drift, falls); None = fresh each run
    ball_detector: str = "yolo"  # "yolo" (real model on rendered frames) | "ground_truth"
    yolo_model_path: str = DEFAULT_YOLO_MODEL
    yolo_ball_conf: float = 0.4  # vision.yaml post_process.confidence_thresholds.Ball
    yolo_nms_iou: float = 0.4  # vision.yaml detection_model.nms_threshold
    initial_mode: int = 4  # RobotMode.kSoccer
