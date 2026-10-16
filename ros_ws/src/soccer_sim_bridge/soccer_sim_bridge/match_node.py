"""A whole match in one process: up to 10 simulated T2s, each with its own ROS domain, and a GameController.

Brain's topic names are absolute and shared, so every robot gets its own ROS domain (base + index): the brain
for that robot runs with ROS_DOMAIN_ID set to the same number (see launch/match.launch.py). One MuJoCo world
steps everybody together; one main loop publishes each robot's sensors into its own domain.
"""

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import subprocess
import tempfile

import mujoco
import numpy as np
import rclpy
from booster_interface.msg import LowState, Odometer, RemoteControllerState
from game_controller_interface.msg import GameControlData
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from vision_interface.msg import Detections, LineSegments

from soccer_sim.config import SimConfig
from soccer_sim.field import ROBOT_ACTUATOR_GROUP
from soccer_sim.frames import TeamFrame
from soccer_sim.game_controller import AWAY, HOME, SET_PLAYS, GameController, Goal, MatchCfg, PlaceBall, SwapSides
from soccer_sim.joystick import FirmwareRemote, manual_velocity
from soccer_sim.kick_model import KickRef
from soccer_sim.loco_rpc import MODE_DAMPING, MODE_PREPARE, MODE_SOCCER, LocoState
from soccer_sim.placement import robot_pose_from_dict
from soccer_sim.referee import STATES, name_of
from soccer_sim.roster import PlayerSpec, roster_from_json, validate_roster
from soccer_sim.shots import Shot, ShotPanel, shot_from_dict
from soccer_sim.field import build_world
from soccer_sim.sim import SoccerController, World
from soccer_sim.viewer_look import TEAM_RGBA, apply_look

from . import gateway, msg_conv
from .viewer_link import LocalMirror, ViewerLink
from .web import ControlServer

MODE_NAMES = {0: "damping", 1: "prepare", 2: "walking", 3: "custom", 4: "soccer"}
REFEREE_KEYS = {"state", "set_play", "kicking", "stopped", "action"}
TOUCH_DIST_M = 0.45  # a robot this close to the ball (trunk to ball) counts as the last to touch it
# Full-procedure stand-in for the person at the remote: LT+Up, LT+A (recalibrate), LT+B (auto), 0.4 s each.
BOOT_PLAN = ((2.0, "hat_u"), (7.0, "a"), (10.0, "b"))
BOOT_HOLD_S = 0.4
MAX_ROBOTS = 10


@dataclass
class Channel:
    """One robot, sim side: its own ROS domain and publishers. What brain commands arrives through the gateway
    process (see gateway.py), read from the shared state file once per tick."""

    index: int
    spec: PlayerSpec
    domain: int
    sim: SoccerController
    loco: LocoState
    lock: threading.Lock
    shared: gateway.Shared
    context: object = None
    node: object = None
    kick_ref: KickRef | None = None
    joy: tuple[RemoteControllerState, float] | None = None
    remote: FirmwareRemote = field(default_factory=FirmwareRemote)
    manual: bool = False
    boot_t0: float | None = None
    _getup_seen: float = 0.0

    def __post_init__(self):
        self.context = rclpy.Context()
        rclpy.init(args=[], context=self.context, domain_id=self.domain)
        # Publish-only node: no executor, no parameter services, no /rosout.
        self.node = Node(
            f"soccer_sim_{self.spec.team}{self.spec.player_id}", context=self.context,
            start_parameter_services=False, enable_rosout=False)
        n = self.node
        self.odom_pub = n.create_publisher(Odometer, "/odometer_state", 10)
        self.low_pub = n.create_publisher(LowState, "/low_state", 10)
        self.head_pub = n.create_publisher(PoseStamped, "/head_pose_stamped", 10)
        self.det_pub = n.create_publisher(Detections, "/booster_vision/detection", 10)
        self.line_pub = n.create_publisher(LineSegments, "/booster_vision/line_segments", 10)
        self.rgb_pub = n.create_publisher(Image, "/boostercamera/head/rgb", 5)
        self.depth_pub = n.create_publisher(Image, "/boostercamera/head/depth", 5)
        self.caminfo_pub = n.create_publisher(CameraInfo, "/boostercamera/head/rgb/camera_info", 10)
        self.gc_pub = n.create_publisher(GameControlData, "/robocup/game_controller", 10)
        self.joy_pub = n.create_publisher(RemoteControllerState, "/remote_controller_state", 10)

    @property
    def row(self):
        return self.shared.rows[self.index]

    def read_commands(self) -> None:
        """Copy brain's latest commands (written by the gateway) into this tick's LocoState and kick reference."""
        r, loco = self.row, self.loco
        loco.mode = int(r[gateway.MODE])
        loco.vx, loco.vy, loco.vyaw = float(r[gateway.VX]), float(r[gateway.VY]), float(r[gateway.VYAW])
        loco.head_pitch, loco.head_yaw = float(r[gateway.HEAD_PITCH]), float(r[gateway.HEAD_YAW])
        loco.visual_kick_active = bool(r[gateway.KICK_ACTIVE])
        loco.visual_kick_version = int(r[gateway.KICK_VERSION])
        if r[gateway.GETUP] != self._getup_seen:  # a counter: every GetUp request bumps it
            self._getup_seen = float(r[gateway.GETUP])
            loco.getup_requested = True
        self.kick_ref = (KickRef(ball_x=float(r[gateway.KICK_X]), ball_y=float(r[gateway.KICK_Y]),
                                 dir=float(r[gateway.KICK_DIR]), power=float(r[gateway.KICK_POWER]),
                                 stamp=float(r[gateway.KICK_STAMP])) if r[gateway.KICK_VALID] else None)

    def stand_up(self) -> None:
        """The remote's LT+Up: the gateway (which owns the loco mode) switches PREPARE/DAMPING to SOCCER."""
        self.row[gateway.STAND_UP] = 1.0
        if self.loco.mode in (MODE_DAMPING, MODE_PREPARE):
            self.loco.mode = MODE_SOCCER

    @property
    def decision(self) -> str:
        return self.shared.get_text(self.index, gateway.DECISION) or "-"

    @property
    def event(self) -> str:
        return self.shared.get_text(self.index, gateway.EVENT)

    def stamp(self):
        return self.node.get_clock().now().to_msg()

    def fresh_joystick(self, now: float, timeout: float) -> RemoteControllerState | None:
        """Caller holds the lock."""
        if self.joy is None or now - self.joy[1] > timeout:
            return None
        return self.joy[0]

    def boot_press(self, now: float) -> dict | None:
        """The remote buttons the full-procedure script holds right now (caller holds the lock)."""
        if self.boot_t0 is None:
            return None
        t = now - self.boot_t0
        for at, button in BOOT_PLAN:
            if at <= t < at + BOOT_HOLD_S:
                return {"lt": True, button: True}
        return None

    def close(self) -> None:
        try:
            self.node.destroy_node()
        finally:
            self.context.try_shutdown()


def _fmt_clock(secs: float) -> str:
    secs = max(0, int(secs))
    return f"{secs // 60}:{secs % 60:02d}"


class MatchBridge:
    def __init__(self, roster: list[PlayerSpec], cfg: SimConfig, match: MatchCfg, domain_base: int,
                 web_port: int, full_auto: bool, camera_every: int = 2):
        problems = validate_roster(roster)
        if problems:
            raise ValueError("; ".join(problems))
        if len(roster) > MAX_ROBOTS:
            raise ValueError(f"at most {MAX_ROBOTS} robots")
        self.roster = roster
        self.cfg = cfg
        self.joystick_timeout = 0.5
        self.lock = threading.Lock()
        self.frames = {HOME: TeamFrame(False), AWAY: TeamFrame(True)}
        self.world = World(cfg, len(roster))
        self._dir = tempfile.mkdtemp(prefix="soccer_sim_match_")
        self._state_path = os.path.join(self._dir, "state.bin")
        self.shared = gateway.Shared(self._state_path, len(roster), create=True)
        self.channels: list[Channel] = []
        for i, spec in enumerate(roster):
            sim = SoccerController(cfg, self.world, i, self.frames[spec.team], spec.entry_pose())
            mode = MODE_PREPARE if full_auto else cfg.initial_mode
            self.shared.rows[i][gateway.MODE] = mode
            self.channels.append(Channel(i, spec, domain_base + i, sim, LocoState(mode=mode), self.lock, self.shared))
        self.world.start()
        self._start_gateways(domain_base)
        self.full_auto = full_auto
        self.camera_every = max(1, camera_every)
        self.use_camera = cfg.ball_detector == "yolo"
        self.gc = GameController(match, time.monotonic())
        self.packet = 0
        self.last_touch: str | None = None
        self.selected = 0
        self.tick_count = 0
        self.tick_hz = 0.0
        self._rate_t, self._rate_ticks = time.monotonic(), 0
        self._t_gc = self._t_info = -1e9
        self._pending_shots: list[Shot] = []
        self._fire_panel = False
        self._resets: dict[int, tuple[float, float, float]] = {}
        self._status: dict = {}
        self.shot_panel = ShotPanel()
        self._log = lambda text: print(f"[match] {text}", flush=True)
        self.web: ControlServer | None = None
        if web_port >= 0:
            self.web = ControlServer(
                web_port, self.on_web_joystick, self.on_web_referee, self.on_web_shoot, self.status,
                on_reset_robot=self.on_web_reset_robot, extra_routes={"/api/select": self.on_web_select},
            )
            self.web.start()
            self._log(f"controller tab: http://127.0.0.1:{self.web.port}/")
        self.world.reset_ball(*cfg.ball_xy)
        self._log(f"{len(roster)} robots, ROS domains {domain_base}..{domain_base + len(roster) - 1}")

    def _start_gateways(self, domain_base: int) -> None:
        """One gateway process per robot, then wait until each answers (brains must not talk to nobody)."""
        self._gateways = [
            subprocess.Popen([sys.executable, "-m", "soccer_sim_bridge.gateway", self._state_path, str(len(self.channels)),
                              str(ch.index), str(ch.domain), f"gateway_{ch.spec.team}{ch.spec.player_id}"])
            for ch in self.channels
        ]
        deadline = time.monotonic() + 40.0
        while time.monotonic() < deadline:
            if all(self.shared.rows[ch.index][gateway.ALIVE] > 0 for ch in self.channels):
                return
            if any(p.poll() is not None for p in self._gateways):
                raise RuntimeError("a robot gateway process exited during start-up")
            time.sleep(0.05)
        raise RuntimeError("robot gateway processes did not start in 40 s")

    # ---- web handlers (HTTP thread) ----
    def _robot(self, payload: dict) -> Channel:
        idx = int(payload.pop("robot", self.selected))
        if not 0 <= idx < len(self.channels):
            raise ValueError(f"robot must be 0-{len(self.channels) - 1}")
        return self.channels[idx]

    def on_web_select(self, payload: dict) -> None:
        self.selected = self._robot(payload).index

    def on_web_joystick(self, payload: dict) -> None:
        ch = self._robot(payload)
        msg = msg_conv.joystick_from_dict(payload)
        with self.lock:
            ch.joy = (msg, time.monotonic())

    def on_web_referee(self, payload: dict) -> None:
        unknown = set(payload) - REFEREE_KEYS
        if unknown:
            raise ValueError(f"unknown referee fields: {sorted(unknown)}")
        payload = dict(payload)
        action = payload.pop("action", None)
        now = time.monotonic()
        with self.lock:
            if action == "kickoff":
                self.world.reset_ball(0.0, 0.0)
                self.gc.override(now, state="READY")
                self.gc.override(now, state="SCHEDULE")
            elif action == "next_half":
                self.gc.finish_half(now)
            elif action == "finish":
                self.gc.override(now, state="FINISHED")
            elif action is not None:
                raise ValueError(f"unknown referee action {action!r}")
            if payload:
                self.gc.override(now, **payload)

    def on_web_shoot(self, payload: dict) -> None:
        shot = shot_from_dict(payload) if payload else None
        with self.lock:
            if shot is None:
                self._fire_panel = True
            else:
                self._pending_shots.append(shot)

    def on_web_reset_robot(self, payload: dict) -> None:
        payload = dict(payload)
        ch = self._robot(payload)
        if payload:
            pose = robot_pose_from_dict(payload, self.cfg.field)
        else:  # {} = back to its own entry spot
            pose = ch.sim.frame.pose_tuple(*ch.spec.entry_pose())
        with self.lock:
            self._resets[ch.index] = pose

    def status(self) -> dict:
        with self.lock:
            return dict(self._status)

    # ---- main-thread tick: the only code that touches mj_data or the renderer ----
    def tick(self) -> None:
        now = time.monotonic()
        world = self.world
        with self.lock:
            for ch in self.channels:
                joy = ch.fresh_joystick(now, self.joystick_timeout)
                press = ch.boot_press(now)
                if press is not None and joy is None:
                    joy = msg_conv.joystick_from_dict(press)
                ch.read_commands()
                if ch.remote.update(joy is not None and joy.lt and not joy.rt, joy is not None and joy.hat_u):
                    ch.stand_up()
                manual = manual_velocity(joy.lx, joy.ly, joy.rx, joy.ry) if joy is not None else None
                ch.manual = manual is not None
                ch.sim.apply(ch.loco, now, ch.kick_ref, manual)
            shots, self._pending_shots = self._pending_shots, []
            fire_panel, self._fire_panel = self._fire_panel, False
            resets, self._resets = self._resets, {}
            if self.full_auto and self.tick_count == 0:
                for ch in self.channels:
                    ch.boot_t0 = now

        for idx, pose in resets.items():
            self.channels[idx].sim.reset_robot(*pose)
        panel = world.shot_panel_values()
        panel_shot = self.shot_panel.read(panel)
        if panel_shot is not None:
            shots.append(panel_shot)
        if fire_panel:
            shots.append(Shot.from_panel(panel))
        for shot in shots:
            world.shoot(shot)

        world.step()
        self._referee(now)
        self._publish(now)
        self._update_status(now)
        self.tick_count += 1

    def _referee(self, now: float) -> None:
        ball = self.world.ball_position()
        dists = []
        for ch in self.channels:
            pose = ch.sim.robot_pose()
            dists.append((math.hypot(pose.x - ball[0], pose.y - ball[1]), ch))
        near, ch = min(dists, key=lambda d: d[0])
        if near < TOUCH_DIST_M and ball[2] < 0.4:
            self.last_touch = ch.spec.team
        with self.lock:
            events = self.gc.update(now, (float(ball[0]), float(ball[1]), float(ball[2])), self.last_touch)
        for ev in events:
            if isinstance(ev, PlaceBall):
                self.world.reset_ball(ev.x, ev.y)
                self.last_touch = None
                self._log(f"ball to ({ev.x:.1f}, {ev.y:.1f}) for {ev.why}")
            elif isinstance(ev, Goal):
                self._log(f"GOAL for {ev.team}: {self.gc.score[HOME]}-{self.gc.score[AWAY]}")
            elif isinstance(ev, SwapSides):
                self._swap_sides()

    def _swap_sides(self) -> None:
        """Half-time: the teams change ends and every robot is put back on its sideline, odometry restarted."""
        self.frames[HOME].flipped = not self.frames[HOME].flipped
        self.frames[AWAY].flipped = not self.frames[AWAY].flipped
        for ch in self.channels:
            ch.sim.reset_robot(*ch.sim.frame.pose_tuple(*ch.spec.entry_pose()))
            ch.sim.restart_odometry()
            if self.full_auto:
                ch.stand_up()
        self._log("half-time: sides swapped")

    def _publish(self, now: float) -> None:
        world = self.world
        n = len(self.channels)
        poses = [(ch.sim.robot_pose().x, ch.sim.robot_pose().y) for ch in self.channels]
        # With YOLO the sim renders one frame per `frame_gap` ticks in total, round-robin over the robots (a frame is
        # ~8 ms of rendering + ~10 ms of YOLO; one per 2 ticks is what a lone robot holds 50 Hz with, and with 6+
        # robots the physics and publishing leave room for one per 4). Ground truth is cheap: every robot is
        # perceived every `camera_every` ticks.
        frame_gap = self.camera_every * (1 if n <= 5 else 2)
        period = frame_gap * n if self.use_camera else self.camera_every
        for ch in self.channels:
            sim, stamp = ch.sim, ch.stamp()
            ch.odom_pub.publish(msg_conv.odometer(*sim.odometer()))
            rpy, gyro, acc = sim.imu()
            ch.low_pub.publish(msg_conv.low_state(rpy, gyro, acc, *sim.joint_state()))
            ch.head_pub.publish(msg_conv.head_pose(stamp, *sim.head_pose_in_base()))
            due = (self.tick_count % period) == (ch.index * frame_gap if self.use_camera else 0)
            if not due:
                continue
            others = [p for j, p in enumerate(poses) if j != ch.index]
            if self.use_camera:
                rgb, depth = sim.render()
                dets, lines = sim.perceive(rgb=rgb, others=others)
                ch.rgb_pub.publish(msg_conv.rgb_image(stamp, rgb))
                ch.depth_pub.publish(msg_conv.depth_image(stamp, depth, self.cfg.max_depth_m))
            else:
                dets, lines = sim.perceive(others=others)
            ch.det_pub.publish(msg_conv.detections(stamp, dets))
            ch.line_pub.publish(msg_conv.line_segments(stamp, lines))
        if now - self._t_gc >= 0.5:
            self._t_gc = now
            with self.lock:
                snap = self.gc.snapshot(now)
            self.packet += 1
            msg = msg_conv.game_control_match(snap, self.roster, self.packet)
            for ch in self.channels:
                ch.gc_pub.publish(msg)
        if now - self._t_info >= 1.0:
            self._t_info = now
            for ch in self.channels:
                ch.caminfo_pub.publish(msg_conv.camera_info(ch.stamp(), self.cfg.camera))
        if self.tick_count % 3 == 0:  # ~17 Hz, brain wants a regular remote state
            for ch in self.channels:
                with self.lock:
                    joy = ch.fresh_joystick(now, self.joystick_timeout)
                    press = ch.boot_press(now)
                if joy is None and press is not None:
                    joy = msg_conv.joystick_from_dict(press)
                ch.joy_pub.publish(joy if joy is not None else msg_conv.joystick_from_dict({}))

    def _update_status(self, now: float) -> None:
        with self.lock:
            snap = self.gc.snapshot(now)
            ball = self.world.ball_position()
            robots = []
            for ch in self.channels:
                pose = ch.sim.robot_pose()
                robots.append({
                    "index": ch.index, "team": ch.spec.team, "player_id": ch.spec.player_id, "role": ch.spec.role,
                    "label": f"{'H' if ch.spec.team == HOME else 'A'}{ch.spec.player_id}",
                    "x": pose.x, "y": pose.y, "yaw": pose.yaw, "mode": ch.loco.mode, "decision": ch.decision,
                    "kicking": ch.loco.kicking(), "manual": ch.manual, "fallen": ch.sim.fallen(),
                })
            sel = robots[min(self.selected, len(robots) - 1)]
            state = name_of(STATES, snap.state)
            play = next((k for k, v in SET_PLAYS.items() if v == snap.set_play), "NONE")
            self._status = {
                "match": True,
                "state": state, "set_play": play, "kicking_team": snap.kicking_team,
                "kicking": "us" if snap.kicking_team == HOME else "opponent",
                "stopped": snap.stopped, "set_play_secs_left": snap.secondary_time,
                "score": snap.score, "first_half": snap.first_half, "secs_remaining": snap.secs_remaining,
                "continuous": self.gc.cfg.continuous, "awaiting_start": snap.awaiting_start,
                "play_secs": self.gc.play_elapsed, "event": self.gc.last_event,
                "ball": {"x": float(ball[0]), "y": float(ball[1])},
                "robots": robots, "selected": sel["index"],
                "robot": {"x": sel["x"], "y": sel["y"], "yaw": sel["yaw"]},
                "robot_mode": sel["mode"], "visual_kick": sel["kicking"], "manual_drive": sel["manual"],
                "decision": sel["decision"], "shots": {"goal": 0, "no_goal": 0}, "last_shot": "",
                "hz": round(self.tick_hz, 1),
            }

    def overlay(self) -> tuple[str, str]:
        """Two text columns for the top of the viewer: labels, values."""
        with self.lock:
            st = dict(self._status)
            ch = self.channels[min(self.selected, len(self.channels) - 1)]
            decision, event = ch.decision, ch.event
            vx, vy, vyaw = ch.loco.effective_velocity()
        score = st.get("score", {HOME: 0, AWAY: 0})
        clock = "continuous" if st.get("continuous") else _fmt_clock(st.get("secs_remaining", 0))
        speed = float(np.linalg.norm(self.world.ball_velocity()))
        left = ["score", "clock", "referee", f"robot {self.selected}", "brain decision", "walk command", "ball speed",
                "brain said", "sim rate"]
        right = [
            f"home {score.get(HOME, 0)} - {score.get(AWAY, 0)} away",
            f"{clock}   half {1 if st.get('first_half', True) else 2}",
            f"{st.get('state', '-')}  set play {st.get('set_play', '-')}  kick: {st.get('kicking_team', '-')}",
            f"{ch.spec.team} {ch.spec.player_id} ({ch.spec.role})",
            decision,
            f"vx {vx:+.2f}  vy {vy:+.2f}  yaw {vyaw:+.2f}",
            f"{speed:.1f} m/s",
            event or "-",
            f"{self.tick_hz:.0f} Hz" + ("" if self.tick_hz >= 45 else "  (target 50, running slow)"),
        ]
        return "\n".join(left), "\n".join(right)

    def note_tick(self) -> None:
        self._rate_ticks += 1
        now = time.monotonic()
        if now - self._rate_t >= 1.0:
            self.tick_hz = self._rate_ticks / (now - self._rate_t)
            self._rate_t, self._rate_ticks = now, 0
            if self.tick_hz < 45 and self.tick_count > 250:
                self._log(f"sim ticking at {self.tick_hz:.0f} Hz (target 50): brains see a slowed-down world")

    def close(self) -> None:
        if self.web is not None:
            self.web.stop()
            self.web = None
        self.shared.rows[0][gateway.STOP] = 1.0
        for p in self._gateways:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        for ch in self.channels:
            ch.close()
        for name in ("state.bin",):
            try:
                os.remove(os.path.join(self._dir, name))
            except OSError:
                pass
        try:
            os.rmdir(self._dir)
        except OSError:
            pass


def _run_realtime(bridge: MatchBridge, viewer, link: ViewerLink | None, profile_secs: float = 0.0,
                  profile_file: str = "", mirror: LocalMirror | None = None) -> None:
    profiler = None
    if profile_secs > 0:
        import cProfile
        profiler = cProfile.Profile()
    t_start, profiling = time.monotonic(), False
    period = bridge.channels[0].sim.cfg.policy_dt
    next_t = time.monotonic()
    n_robot = bridge.world.n_robot_ctrl
    frame = 0
    while (viewer is None or viewer.is_running()) and (link is None or link.is_running()):
        if link is not None:
            bridge.world.mj_data.ctrl[n_robot:] = link.sliders()
            bridge.world.mj_data.xfrc_applied[:] = link.xfrc()
        if mirror is not None:
            with viewer.lock():
                mirror.to_sim(bridge.world.mj_data)
        if profiler is not None:
            since = time.monotonic() - t_start
            if not profiling and 20.0 <= since < 20.0 + profile_secs:
                profiler.enable()
                profiling = True
            elif profiling and since >= 20.0 + profile_secs:
                profiler.disable()
                import pstats
                out = open(profile_file, "w") if profile_file else sys.stderr
                pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(30)
                if profile_file:
                    out.close()
                profiler = None
        bridge.tick()
        bridge.note_tick()
        frame += 1
        text = bridge.overlay() if (viewer or link) and frame % 5 == 0 else None
        if viewer is not None:
            if mirror is not None:
                with viewer.lock():
                    mirror.from_sim(bridge.world.mj_data)
            if text is not None:
                viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, *text))
            viewer.sync()
        if link is not None:
            link.sync(bridge.world.mj_data.qpos, *(text or (None,)))
        next_t += period
        delay = next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -5 * period:
            next_t = time.monotonic()


def _pick_camera_gl() -> None:
    """Same choice as bridge_node: Mesa's software rasteriser for the head camera on WSL."""
    from .bridge_node import _pick_camera_gl as pick
    pick()


def parse_args(argv):
    p = argparse.ArgumentParser(description="Simulated match: several robots, a GameController, one MuJoCo world")
    p.add_argument("--roster", required=True, help="JSON list of {team, player_id, role}")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--domain-base", type=int, default=70, help="ROS domain of the first robot; the rest follow")
    p.add_argument("--ball-detector", choices=("yolo", "ground_truth"), default="ground_truth")
    p.add_argument("--start", choices=("quick", "full_auto"), default="quick")
    p.add_argument("--half-minutes", type=float, default=10.0)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--ready-secs", type=float, default=30.0)
    p.add_argument("--camera-every", type=int, default=2)
    p.add_argument("--web-port", type=int, default=8765)
    p.add_argument("--ball-x", type=float, default=0.0)
    p.add_argument("--ball-y", type=float, default=0.0)
    p.add_argument("--detailed-bodies", action="store_true",
                   help="viewer: real T2 meshes (~1 M triangles each, slow with many robots) instead of simple team-coloured bodies")
    p.add_argument("--profile", type=float, default=0.0, metavar="SECS",
                   help="profile the main loop for this long after 20 s, print the hot spots (to stderr, or --profile-file)")
    p.add_argument("--profile-file", default="")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    roster = roster_from_json(args.roster)
    # Ten tiny walk policies on torch's default 8-thread pool spend more time spinning than computing.
    import torch
    torch.set_num_threads(1)
    viewer_env = dict(os.environ)  # the viewer must not inherit the camera's software GL (see viewer_environment)
    _pick_camera_gl()
    cfg = SimConfig(ball_detector=args.ball_detector, ball_xy=(args.ball_x, args.ball_y))
    match = MatchCfg(half_secs=args.half_minutes * 60.0, continuous=args.continuous, ready_secs=args.ready_secs,
                     hold_initial=args.start == "full_auto",
                     teams=tuple(t for t in (HOME, AWAY) if any(p.team == t for p in roster)))
    bridge = MatchBridge(roster, cfg, match, args.domain_base, args.web_port, args.start == "full_auto",
                         args.camera_every)
    tints = {ch.sim.root_body_id: TEAM_RGBA[ch.spec.team] for ch in bridge.channels}
    w = bridge.world
    # The viewer copies its whole model on every sync, and ten robots' visual meshes are ~10 M triangles of it (66 ms a
    # sync). Simple bodies therefore get a model without them; the physics model is untouched.
    viewer_model = (build_world(cfg, len(roster), strip_visual=True)
                    if args.viewer and not args.detailed_bodies else w.mj_model)
    link, mirror = None, None
    try:
        if args.viewer and sys.platform == "darwin":
            import mujoco.viewer

            groups = apply_look(viewer_model, tints, args.detailed_bodies)
            data = w.mj_data
            if viewer_model is not w.mj_model:
                mirror = LocalMirror(viewer_model, w.n_robot_ctrl, w.mj_data.ctrl[w.n_robot_ctrl:])
                data = mirror.data
            with mujoco.viewer.launch_passive(viewer_model, data, show_left_ui=False, show_right_ui=True) as viewer:
                with viewer.lock():
                    viewer.opt.actuatorgroup[ROBOT_ACTUATOR_GROUP] = 0
                    for group, shown in groups.items():
                        viewer.opt.geomgroup[group] = shown
                    viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
                    viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 26.0, -55.0, 90.0
                _run_realtime(bridge, viewer, None, args.profile, args.profile_file, mirror)
        elif args.viewer:
            link = ViewerLink(viewer_model, w.n_robot_ctrl, w.mj_data.ctrl[w.n_robot_ctrl:], env=viewer_env, overview=True,
                              tints=tints, detailed=args.detailed_bodies, gl=os.environ.get("SIM_VIEWER_GL", "gpu"))
            _run_realtime(bridge, None, link, args.profile, args.profile_file)
        else:
            _run_realtime(bridge, None, None, args.profile, args.profile_file)
    except KeyboardInterrupt:
        pass
    finally:
        if link is not None:
            link.close()
        bridge.close()


if __name__ == "__main__":
    main()
