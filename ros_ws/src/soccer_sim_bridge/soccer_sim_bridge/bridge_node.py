"""ROS 2 stand-in for the Booster T2 SDK, head camera, and vision stack, backed by soccer_sim."""

import os
import sys
import threading
import time

import mujoco
import numpy as np
import rclpy
from booster_interface.msg import LowState, Odometer, RemoteControllerState
from booster_msgs.msg import RpcReqMsg, RpcRespMsg
from brain.msg import Kick
from game_controller_interface.msg import GameControlData
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray, String
from vision_interface.msg import Detections, LineSegments

from soccer_sim.config import SimConfig
from soccer_sim.field import ROBOT_ACTUATOR_GROUP
from soccer_sim.joystick import FirmwareRemote, manual_velocity
from soccer_sim.kick_model import KickRef
from soccer_sim.loco_rpc import MODE_DAMPING, MODE_PREPARE, MODE_SOCCER, LocoState, handle_request
from soccer_sim.referee import SET_PLAYS, STATES, Referee, name_of, parse_schedule
from soccer_sim.shots import SHOT_FIELDS, Shot, ShotPanel, ShotTracker, shot_from_dict
from soccer_sim.sim import SoccerController

from . import msg_conv
from .viewer_link import ViewerLink
from .web import ControlServer

REFEREE_KEYS = {"state", "set_play", "kicking", "stopped"}
MODE_NAMES = {0: "damping", 1: "prepare", 2: "walking", 3: "custom", 4: "soccer"}


class SimBridge(Node):
    def __init__(self, **node_kwargs):
        super().__init__("soccer_sim_bridge", **node_kwargs)
        d = SimConfig()
        sx, sy, syaw = d.spawn_xy_yaw
        defaults = {
            "spawn_x": sx, "spawn_y": sy, "spawn_yaw": syaw,
            "ball_x": d.ball_xy[0], "ball_y": d.ball_xy[1],
            "ball_detector": d.ball_detector,
            "team_id": 131, "opp_team_id": 1, "goalkeeper_id": 1,
            "referee_schedule": "0:3,1:20,2:3,3:0",
            "camera_every_n_ticks": 2,
            "web_port": 8765,
            "joystick_timeout_s": 0.5,
            "boot_sequence": False,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

        def g(name):
            return self.get_parameter(name).value

        cfg = SimConfig(
            spawn_xy_yaw=(float(g("spawn_x")), float(g("spawn_y")), float(g("spawn_yaw"))),
            ball_xy=(float(g("ball_x")), float(g("ball_y"))),
            ball_detector=str(g("ball_detector")),
        )
        self.sim = SoccerController(cfg)
        # boot_sequence: like a freshly powered T2, stand in prepare mode until LT+Up on the remote.
        self.loco = LocoState(mode=MODE_PREPARE if bool(g("boot_sequence")) else cfg.initial_mode)
        self.firmware_remote = FirmwareRemote()
        self.kick_ref: KickRef | None = None
        self.lock = threading.Lock()
        self.team_id, self.opp_team_id = int(g("team_id")), int(g("opp_team_id"))
        self.goalkeeper_id = int(g("goalkeeper_id"))
        self.camera_every = max(1, int(g("camera_every_n_ticks")))
        self.joystick_timeout = float(g("joystick_timeout_s"))
        self.referee = Referee(
            parse_schedule(str(g("referee_schedule"))), self.team_id, self.opp_team_id, time.monotonic()
        )
        self.shot_panel = ShotPanel()
        self.shot_tracker = ShotTracker(cfg.field, cfg.goal_height, ball_radius=cfg.ball_radius)
        self.last_shot = ""
        self.tick_count = 0
        self._joy: tuple[RemoteControllerState, float] | None = None
        self._pending_shots: list[Shot] = []
        self._fire_panel = False
        self._reset_robot = False
        self._status: dict = {}
        self._decision = "-"  # brain's current decision ("retreat", "save", ...)
        self._decision_topic = False  # True once /brain/decision (patches/brain-decision-topic.patch) has published
        self._event = ""  # last longer announcement brain made ("game state PLAYING", ...)
        self._rate_t, self._rate_ticks, self.tick_hz = time.monotonic(), 0, 0.0

        self.create_subscription(RpcReqMsg, "LocoApiTopicReq", self.on_rpc, 10)
        self.create_subscription(Kick, "/kick_ball", self.on_kick_ball, 10)
        self.create_subscription(Float64MultiArray, "/sim/shoot", self.on_sim_shoot, 10)
        self.create_subscription(String, "/speak", self.on_speak, 10)
        self.create_subscription(String, "/brain/decision", self.on_decision, 10)
        self.resp_pub = self.create_publisher(RpcRespMsg, "LocoApiTopicResp", 10)
        self.odom_pub = self.create_publisher(Odometer, "/odometer_state", 10)
        self.low_pub = self.create_publisher(LowState, "/low_state", 10)
        self.head_pub = self.create_publisher(PoseStamped, "/head_pose_stamped", 10)
        self.det_pub = self.create_publisher(Detections, "/booster_vision/detection", 10)
        self.line_pub = self.create_publisher(LineSegments, "/booster_vision/line_segments", 10)
        self.rgb_pub = self.create_publisher(Image, "/boostercamera/head/rgb", 5)
        self.depth_pub = self.create_publisher(Image, "/boostercamera/head/depth", 5)
        self.caminfo_pub = self.create_publisher(CameraInfo, "/boostercamera/head/rgb/camera_info", 10)
        self.gc_pub = self.create_publisher(GameControlData, "/robocup/game_controller", 10)
        self.joy_pub = self.create_publisher(RemoteControllerState, "/remote_controller_state", 10)
        self.shot_result_pub = self.create_publisher(String, "/sim/shot_result", 10)
        self.create_timer(0.5, self.publish_referee)
        self.create_timer(1.0, self.publish_camera_info)
        self.create_timer(0.05, self.publish_joystick)

        self.web: ControlServer | None = None
        port = int(g("web_port"))
        if port > 0:
            self.web = ControlServer(port, self.on_web_joystick, self.on_web_referee, self.on_web_shoot, self.status,
                                     on_reset_robot=self.on_web_reset_robot)
            self.web.start()
            self.get_logger().info(f"controller tab: http://127.0.0.1:{self.web.port}/")

    # ---- inputs (executor / HTTP threads; only touch state under self.lock) ----
    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _fresh_joystick(self, now: float) -> RemoteControllerState | None:
        """Caller holds self.lock."""
        if self._joy is None or now - self._joy[1] > self.joystick_timeout:
            return None
        return self._joy[0]

    def on_rpc(self, msg: RpcReqMsg) -> None:
        with self.lock:
            reply = handle_request(self.loco, msg.header, msg.body)
        self.resp_pub.publish(RpcRespMsg(uuid=msg.uuid, header=reply.header, body=reply.body))

    def on_kick_ball(self, msg: Kick) -> None:
        ref = msg_conv.kick_ref(msg, time.monotonic())
        with self.lock:
            self.kick_ref = ref

    def on_speak(self, msg: String) -> None:
        # Brain's espeak announcements: sentences are events. Single words are decisions, but /speak is
        # rate-limited, so they only count when the patched brain's /brain/decision isn't there.
        with self.lock:
            if " " in msg.data.strip():
                self._event = msg.data
            elif not self._decision_topic:
                self._decision = msg.data

    def on_decision(self, msg: String) -> None:
        with self.lock:
            self._decision_topic = True
            self._decision = msg.data or "-"

    def on_sim_shoot(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != len(SHOT_FIELDS):
            self.get_logger().warning(f"/sim/shoot needs {len(SHOT_FIELDS)} values {SHOT_FIELDS}")
            return
        try:
            shot = shot_from_dict(dict(zip(SHOT_FIELDS, msg.data)))
        except ValueError as exc:
            self.get_logger().warning(f"/sim/shoot rejected: {exc}")
            return
        with self.lock:
            self._pending_shots.append(shot)

    def on_web_joystick(self, payload: dict) -> None:
        msg = msg_conv.joystick_from_dict(payload)
        with self.lock:
            self._joy = (msg, time.monotonic())

    def on_web_referee(self, payload: dict) -> None:
        unknown = set(payload) - REFEREE_KEYS
        if unknown:
            raise ValueError(f"unknown referee fields: {sorted(unknown)}")
        with self.lock:
            self.referee.override(time.monotonic(), **payload)

    def on_web_shoot(self, payload: dict) -> None:
        shot = shot_from_dict(payload) if payload else None  # empty body = use the MuJoCo slider values
        with self.lock:
            if shot is None:
                self._fire_panel = True
            else:
                self._pending_shots.append(shot)

    def on_web_reset_robot(self, _payload: dict) -> None:
        with self.lock:
            self._reset_robot = True

    def status(self) -> dict:
        with self.lock:
            return dict(self._status)

    # ---- main-thread tick: the only code that touches mj_data or the renderer ----
    def tick(self) -> None:
        now = time.monotonic()
        with self.lock:
            joy = self._fresh_joystick(now)
            stand_up = self.firmware_remote.update(
                joy is not None and joy.lt and not joy.rt, joy is not None and joy.hat_u
            )
            if stand_up and self.loco.mode in (MODE_DAMPING, MODE_PREPARE):
                self.loco.mode = MODE_SOCCER
                self.get_logger().info("remote LT+Up: robot stood up into soccer mode")
            manual = manual_velocity(joy.lx, joy.ly, joy.rx, joy.ry) if joy is not None else None
            self.sim.apply(self.loco, now, self.kick_ref, manual)
            shots, self._pending_shots = self._pending_shots, []
            fire_panel, self._fire_panel = self._fire_panel, False
            reset_robot, self._reset_robot = self._reset_robot, False

        if reset_robot:  # keeper's home: goal centre, 1 m in front of the line, facing the field
            self.sim.reset_robot(-10.0, 0.0, 0.0)

        panel = self.sim.shot_panel_values()
        panel_shot = self.shot_panel.read(panel)
        if panel_shot is not None:
            shots.append(panel_shot)
        if fire_panel:
            shots.append(Shot.from_panel(panel))
        for shot in shots:
            self.sim.shoot(shot)
            if shot.speed > 0:  # speed 0 just places the ball; keep it out of the goal tally
                self.shot_tracker.start(now)
            self.get_logger().info(f"shot: {shot}")

        self.sim.step()

        outcome = self.shot_tracker.update(now, self.sim.ball_position(), self.sim.ball_velocity())
        if outcome is not None:
            self.last_shot = outcome
            self.shot_result_pub.publish(String(data=outcome))
            self.get_logger().info(f"shot result: {outcome}  tally={self.shot_tracker.tally}")

        stamp = self._stamp()
        self.odom_pub.publish(msg_conv.odometer(*self.sim.odometer()))
        rpy, gyro, acc = self.sim.imu()
        self.low_pub.publish(msg_conv.low_state(rpy, gyro, acc, *self.sim.joint_state()))
        self.head_pub.publish(msg_conv.head_pose(stamp, *self.sim.head_pose_in_base()))
        if self.tick_count % self.camera_every == 0:
            rgb, depth = self.sim.render()
            dets, lines = self.sim.perceive(rgb=rgb)  # ball via YOLO on this frame, the rest ground truth
            self.rgb_pub.publish(msg_conv.rgb_image(stamp, rgb))
            self.depth_pub.publish(msg_conv.depth_image(stamp, depth, self.sim.sim_cfg.max_depth_m))
            self.det_pub.publish(msg_conv.detections(stamp, dets))
            self.line_pub.publish(msg_conv.line_segments(stamp, lines))

        with self.lock:
            ref = self.referee.snapshot(now)
            self._status = {
                "state": name_of(STATES, ref.state),
                "set_play": name_of(SET_PLAYS, ref.set_play),
                "kicking": "us" if ref.kicking_team == self.team_id else "opponent",
                "stopped": ref.stopped,
                "set_play_secs_left": ref.secondary_time,
                "robot_mode": self.loco.mode,
                "visual_kick": self.loco.kicking(),
                "manual_drive": manual is not None,
                "shots": dict(self.shot_tracker.tally),
                "last_shot": self.last_shot,
            }
        self.tick_count += 1

    def overlay(self) -> tuple[str, str]:
        """Two text columns for the top of the viewer: labels, values."""
        with self.lock:
            st = dict(self._status)
            decision, event = self._decision, self._event
            mode, kicking = self.loco.mode, self.loco.kicking()
            vx, vy, vyaw = self.loco.effective_velocity()
        speed = float(np.linalg.norm(self.sim.ball_velocity()))
        shots = st.get("shots", {})
        left = ["brain decision", "walk command", "ball speed", "robot mode", "referee", "brain said", "shots", "sim rate"]
        right = [
            decision,
            f"vx {vx:+.2f}  vy {vy:+.2f}  yaw {vyaw:+.2f}" + ("   [manual]" if st.get("manual_drive") else ""),
            f"{speed:.1f} m/s",
            MODE_NAMES.get(mode, str(mode)) + ("  (kicking)" if kicking else ""),
            f"{st.get('state', '-')}  set play {st.get('set_play', '-')}  kick: {st.get('kicking', '-')}",
            event or "-",
            f"goal {shots.get('goal', 0)}  /  no goal {shots.get('no_goal', 0)}"
            + (f"   last: {st['last_shot']}" if st.get("last_shot") else ""),
            f"{self.tick_hz:.0f} Hz" + ("" if self.tick_hz >= 45 else "  (target 50, running slow)"),
        ]
        return "\n".join(left), "\n".join(right)

    def note_tick(self) -> None:
        """Measure the achieved tick rate; warn when the sim can't keep up (brain then sees a slowed world)."""
        self._rate_ticks += 1
        now = time.monotonic()
        if now - self._rate_t >= 1.0:
            self.tick_hz = self._rate_ticks / (now - self._rate_t)
            self._rate_t, self._rate_ticks = now, 0
            if self.tick_hz < 45 and self.tick_count > 250:  # not while the first frames warm up
                self.get_logger().warning(
                    f"sim ticking at {self.tick_hz:.0f} Hz (target 50): brain sees a slowed-down world",
                    throttle_duration_sec=15.0,
                )

    # ---- timers ----
    def publish_referee(self) -> None:
        with self.lock:
            ref = self.referee.snapshot(time.monotonic())
        self.gc_pub.publish(msg_conv.game_control(ref, self.team_id, self.opp_team_id, self.goalkeeper_id))

    def publish_camera_info(self) -> None:
        self.caminfo_pub.publish(msg_conv.camera_info(self._stamp(), self.sim.sim_cfg.camera))

    def publish_joystick(self) -> None:
        with self.lock:
            joy = self._fresh_joystick(time.monotonic())
        self.joy_pub.publish(joy if joy is not None else msg_conv.joystick_from_dict({}))

    def destroy_node(self):
        if self.web is not None:
            self.web.stop()
            self.web = None
        super().destroy_node()


def _run_realtime(node: SimBridge, viewer, link: ViewerLink | None = None) -> None:
    """viewer: MuJoCo's passive-viewer handle (macOS, drawn in this process); link: separate viewer process."""
    period = node.sim.cfg.policy_dt
    next_t = time.monotonic()
    n_robot = node.sim.n_robot_ctrl
    frame = 0
    while rclpy.ok() and (viewer is None or viewer.is_running()) and (link is None or link.is_running()):
        if link is not None:
            node.sim.mj_data.ctrl[n_robot:] = link.sliders()  # the shot sliders live in the viewer process
            node.sim.mj_data.xfrc_applied[:] = link.xfrc()  # so does the mouse drag (Ctrl+right-drag)
        node.tick()
        node.note_tick()
        frame += 1
        text = node.overlay() if (viewer or link) and frame % 5 == 0 else None  # 10 Hz is plenty for text
        if viewer is not None:
            viewer.cam.lookat[:] = node.sim.mj_data.qpos[0:3]
            if text is not None:
                viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, *text))
            viewer.sync()
        if link is not None:
            link.sync(node.sim.mj_data.qpos, *(text or (None,)))
        next_t += period
        delay = next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -5 * period:
            node.get_logger().warning(f"sim running {-delay:.2f}s behind real time; resyncing", throttle_duration_sec=5.0)
            next_t = time.monotonic()


def _pick_camera_gl() -> None:
    """Choose Mesa's software rasteriser for the head camera on WSL, where the GPU path is slower.

    The camera frame is small (544x448); WSL's D3D12 driver stalls ~25 ms per frame reading it back, ~5x slower
    than llvmpipe. SIM_CAMERA_GL=software forces llvmpipe on any Linux (e.g. when EGL finds no usable GPU),
    SIM_CAMERA_GL=gpu keeps whatever driver the system picks. Default "auto" is software on WSL only.
    """
    if not sys.platform.startswith("linux") or os.environ.get("MUJOCO_GL") != "egl":
        return
    mode = os.environ.get("SIM_CAMERA_GL", "auto")
    on_wsl = "microsoft" in open("/proc/version").read().lower() if os.path.exists("/proc/version") else False
    if mode == "software" or (mode == "auto" and on_wsl):
        os.environ["GALLIUM_DRIVER"] = "llvmpipe"


def main(argv=None) -> None:
    argv = sys.argv if argv is None else argv
    use_viewer = "--viewer" in argv
    viewer_env = dict(os.environ)  # the viewer keeps the desktop's GL driver; only the camera is switched
    _pick_camera_gl()
    rclpy.init(args=[a for a in argv if a != "--viewer"])
    node = SimBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    node.sim.start()
    link = None
    try:
        if use_viewer and sys.platform == "darwin":
            import mujoco.viewer  # needs mjpython on macOS, which drives the viewer from this process

            with mujoco.viewer.launch_passive(
                node.sim.mj_model, node.sim.mj_data, show_left_ui=False, show_right_ui=True
            ) as viewer:
                with viewer.lock():
                    # Hide the robot motors so the top-right Control panel shows only the shot sliders.
                    viewer.opt.actuatorgroup[ROBOT_ACTUATOR_GROUP] = 0
                _run_realtime(node, viewer)
        elif use_viewer:
            link = ViewerLink(
                node.sim.mj_model, node.sim.n_robot_ctrl, node.sim.mj_data.ctrl[node.sim.n_robot_ctrl:], env=viewer_env
            )
            _run_realtime(node, None, link)
        else:
            _run_realtime(node, None)
    except KeyboardInterrupt:
        pass
    finally:
        if link is not None:
            link.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
