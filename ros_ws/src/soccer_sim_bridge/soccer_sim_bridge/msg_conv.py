"""Pure builders between soccer_sim values and the ROS messages brain_node uses."""

import math

import numpy as np
from booster_interface.msg import ImuState, LowState, MotorState, Odometer, RemoteControllerState
from builtin_interfaces.msg import Time
from game_controller_interface.msg import GameControlData
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from vision_interface.msg import DetectedObject, Detections, LineSegments

from soccer_sim.config import CameraCfg
from soccer_sim.kick_model import KickRef
from soccer_sim.perception import Detection, LineObs
from soccer_sim.game_controller import Snapshot
from soccer_sim.referee import RefereeSnapshot
from soccer_sim.roster import GOALKEEPER_ID, MAX_PLAYERS, TEAM_NUMBERS, TEAMS, PlayerSpec

CAMERA_FRAME = "head_camera"
BASE_FRAME = "base"
JOY_AXES = ("lx", "ly", "rx", "ry")
JOY_BUTTONS = ("a", "b", "x", "y", "lb", "rb", "lt", "rt", "ls", "rs", "back", "start",
               "hat_u", "hat_d", "hat_l", "hat_r")


def stamp_from_ns(ns: int) -> Time:
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def _header(stamp: Time, frame_id: str) -> Header:
    return Header(stamp=stamp, frame_id=frame_id)


def odometer(x: float, y: float, theta: float) -> Odometer:
    return Odometer(x=float(x), y=float(y), theta=float(theta))


def low_state(rpy, gyro, acc, q, dq, tau) -> LowState:
    msg = LowState()
    msg.imu_state = ImuState(rpy=[float(v) for v in rpy], gyro=[float(v) for v in gyro], acc=[float(v) for v in acc])
    msg.motor_state_serial = [
        MotorState(q=float(q[i]), dq=float(dq[i]), tau_est=float(tau[i])) for i in range(len(q))
    ]
    msg.motor_state_parallel = []
    return msg


def head_pose(stamp: Time, pos, quat_wxyz) -> PoseStamped:
    msg = PoseStamped()
    msg.header = _header(stamp, BASE_FRAME)
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in pos)
    w, x, y, z = (float(v) for v in quat_wxyz)
    msg.pose.orientation.w, msg.pose.orientation.x = w, x
    msg.pose.orientation.y, msg.pose.orientation.z = y, z
    return msg


def detections(stamp: Time, dets: list[Detection]) -> Detections:
    msg = Detections(header=_header(stamp, CAMERA_FRAME))
    for d in dets:
        obj = DetectedObject()
        obj.label = d.label
        obj.confidence = float(d.confidence) * 100.0  # percent, as Robotedge vision publishes it
        obj.xmin, obj.ymin, obj.xmax, obj.ymax = (int(v) for v in d.bbox)
        obj.position_projection = [float(d.pos_robot[0]), float(d.pos_robot[1])]
        obj.position = [float(d.pos_robot[0]), float(d.pos_robot[1]), 0.0]
        obj.target_uv = [float(v) for v in d.target_uv] if d.target_uv is not None else []
        msg.detected_objects.append(obj)
    return msg


def line_segments(stamp: Time, lines: list[LineObs]) -> LineSegments:
    msg = LineSegments(header=_header(stamp, CAMERA_FRAME))
    msg.coordinates = [float(v) for line in lines for v in line.robot]
    msg.coordinates_uv = [float(v) for line in lines for v in line.uv]
    return msg


def camera_info(stamp: Time, cfg: CameraCfg) -> CameraInfo:
    msg = CameraInfo(header=_header(stamp, CAMERA_FRAME), width=cfg.width, height=cfg.height)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [cfg.fx, 0.0, cfg.cx, 0.0, cfg.fy, cfg.cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [cfg.fx, 0.0, cfg.cx, 0.0, 0.0, cfg.fy, cfg.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def rgb_image(stamp: Time, rgb: np.ndarray) -> Image:
    h, w, _ = rgb.shape
    return Image(
        header=_header(stamp, CAMERA_FRAME), height=h, width=w, encoding="rgb8",
        is_bigendian=0, step=w * 3, data=np.ascontiguousarray(rgb, dtype=np.uint8).tobytes(),
    )


def depth_image(stamp: Time, depth_m: np.ndarray, max_depth_m: float) -> Image:
    """16UC1 millimetres (brain's vision.depth_scale_16u = 0.001); 0 marks invalid pixels."""
    d = np.where(np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_depth_m), depth_m, 0.0)
    mm = np.round(d * 1000.0).astype("<u2")
    h, w = mm.shape
    return Image(
        header=_header(stamp, CAMERA_FRAME), height=h, width=w, encoding="16UC1",
        is_bigendian=0, step=w * 2, data=mm.tobytes(),
    )


def game_control(ref: RefereeSnapshot, team_id: int, opp_team_id: int, goalkeeper_id: int) -> GameControlData:
    # header/version are validated by Robotedge's UDP game_controller node, not by brain; left default.
    msg = GameControlData()
    msg.players_per_team = 5
    msg.state = ref.state
    msg.set_play = ref.set_play
    msg.kicking_team = ref.kicking_team
    msg.stopped = ref.stopped
    msg.secondary_time = ref.secondary_time
    msg.first_half = True
    msg.secs_remaining = 600
    msg.teams[0].team_number = team_id
    msg.teams[0].goalkeeper = goalkeeper_id
    msg.teams[1].team_number = opp_team_id
    msg.teams[1].goalkeeper = 1
    return msg


PENALTY_SUBSTITUTE = 13  # RoboCupGameControlData: a player who is not on the pitch


def game_control_match(snap: Snapshot, roster: list[PlayerSpec], packet_number: int) -> GameControlData:
    """The simulated GameController's whole state as brain reads it. Teams: [0] home, [1] away.

    Players who are not in the roster are reported as substitutes, so brain does not count them as live
    teammates or opponents (role switching, goalkeeper substitution).
    """
    msg = GameControlData()
    msg.packet_number = packet_number % 256  # brain drops a packet that is not newer than the last one it saw
    msg.players_per_team = MAX_PLAYERS
    msg.state = snap.state
    msg.set_play = snap.set_play
    msg.kicking_team = TEAM_NUMBERS[snap.kicking_team]
    msg.stopped = snap.stopped
    msg.secondary_time = snap.secondary_time
    msg.first_half = snap.first_half
    msg.secs_remaining = max(-32768, min(32767, snap.secs_remaining))
    for slot, team in enumerate(TEAMS):
        info = msg.teams[slot]
        info.team_number = TEAM_NUMBERS[team]
        info.goalkeeper = GOALKEEPER_ID
        info.score = snap.score.get(team, 0)
        present = {p.player_id for p in roster if p.team == team}
        for i in range(MAX_PLAYERS):
            info.players[i].penalty = 0 if (i + 1) in present else PENALTY_SUBSTITUTE
    return msg


def joystick_from_dict(d: dict) -> RemoteControllerState:
    """Controller-tab JSON -> RemoteControllerState. Axes clamped to [-1, 1]; unknown keys ignored."""
    msg = RemoteControllerState()
    for key in JOY_AXES:
        value = float(d.get(key, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        setattr(msg, key, max(-1.0, min(1.0, value)))
    for key in JOY_BUTTONS:
        value = d.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true/false")
        setattr(msg, key, value)
    msg.hat_c = not (msg.hat_u or msg.hat_d or msg.hat_l or msg.hat_r)
    return msg


def kick_ref(msg, now: float) -> KickRef:
    """brain/msg/Kick -> KickRef stamped on the bridge's clock at receipt."""
    return KickRef(ball_x=float(msg.x), ball_y=float(msg.y), dir=float(msg.dir), power=float(msg.power), stamp=now)
