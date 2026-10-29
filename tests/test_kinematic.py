import math
import subprocess
import sys

import numpy as np
import pytest

from soccer_sim.backend import load_backend, resolve_params, robot_rngs
from soccer_sim.config import SimConfig
from soccer_sim.frames import TeamFrame
from soccer_sim.kick_model import KickParams
from soccer_sim.kinematic import DOWN, GETTING_UP, STANDING, KinematicController, KinematicWorld
from soccer_sim.kinematic_params import KinematicParams
from soccer_sim.loco_rpc import MODE_SOCCER, LocoState
from soccer_sim.odometry import OdometryTracker
from soccer_sim.perception import Pose2D
from soccer_sim.shots import Shot

DT = 0.02
CFG = SimConfig(ball_detector="ground_truth")


def make(n=1, spawns=None, frame=None, seed=1, **kin):
    world = KinematicWorld(CFG, n, KinematicParams(**kin))
    spawns = spawns or [(0.0, 0.0, 0.0)] * n
    robots = [KinematicController(CFG, world, i, frame or TeamFrame(), spawns[i], np.random.default_rng(seed + i))
              for i in range(n)]
    return world, robots, [LocoState(mode=MODE_SOCCER) for _ in robots]


def run(world, robots, locos, secs, cmds=None):
    now = 0.0
    for _ in range(int(round(secs / DT))):
        for r, loco, cmd in zip(robots, locos, cmds or [None] * len(robots)):
            if cmd is not None:
                loco.vx, loco.vy, loco.vyaw = cmd
            r.apply(loco, now)
        world.step()
        now += DT


# ---- walking -----------------------------------------------------------------------------------------------------
def test_forward_walk_follows_the_measured_response():
    world, (r,), (loco,) = make()
    run(world, [r], [loco], 6.0, [(0.5, 0, 0)])
    assert r.vx == pytest.approx(0.5 * KinematicParams().gain_x_forward, rel=0.02)  # the policy overshoots forward
    assert r.x > 2.5 and abs(r.y) < 1e-6
    # time to 63 % is about tau_x (0.55 s) plus the command delay
    world, (r,), (loco,) = make()
    reached = None
    for k in range(200):
        loco.vx = 0.5
        r.apply(loco, k * DT)
        world.step()
        if reached is None and r.vx >= 0.632 * 0.5 * KinematicParams().gain_x_forward:
            reached = (k + 1) * DT
    assert 0.5 < reached < 0.8


def test_acceleration_is_capped_and_stopping_is_quicker_than_starting():
    world, (r,), (loco,) = make()
    peak = 0.0
    prev = 0.0
    for k in range(60):
        loco.vx = 3.0
        r.apply(loco, k * DT)
        world.step()
        peak = max(peak, (r.vx - prev) / DT)
        prev = r.vx
    assert peak <= KinematicParams().accel_x + 1e-6
    run(world, [r], [loco], 4.0, [(1.0, 0, 0)])
    v0 = r.vx
    run(world, [r], [loco], 0.4, [(0, 0, 0)])
    assert r.vx < 0.45 * v0  # tau_x_decel = 0.4 s: below 37 % after about 0.4 s plus the delay
    run(world, [r], [loco], 3.0, [(0, 0, 0)])
    assert abs(r.vx) < 1e-3


def test_strafing_is_weak_and_turning_is_fast():
    world, (r,), (loco,) = make()
    run(world, [r], [loco], 3.0, [(0, 0.4, 0)])
    assert r.vy == pytest.approx(0.4 * KinematicParams().gain_y, rel=0.02)
    run(world, [r], [loco], 1.0, [(0, 0, 0.8)])
    assert r.w == pytest.approx(0.8, rel=0.02)


def test_commands_are_ignored_outside_a_moving_mode():
    world, (r,), (loco,) = make()
    loco.mode = 1  # PREPARE
    run(world, [r], [loco], 2.0, [(1.0, 0, 0)])
    assert r.vx == 0.0


def test_head_lags_and_is_rate_limited():
    world, (r,), (loco,) = make()
    loco.head_yaw, loco.head_pitch = 1.0, 0.3
    run(world, [r], [loco], 0.1)
    assert 0 < r.head_yaw < 0.5
    run(world, [r], [loco], 2.0)
    assert r.head_yaw == pytest.approx(1.0, abs=0.02) and r.head_pitch == pytest.approx(0.3, abs=0.02)


# ---- sensors -----------------------------------------------------------------------------------------------------
def test_standing_head_and_camera_match_the_mujoco_measurements():
    world, (r,), _ = make()
    pos, quat = r.head_pose_in_base()
    assert pos == pytest.approx([0.0446, 0.0, 1.2791], abs=0.005)
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    assert r.camera().pos == pytest.approx([0.1142, 0.0343, 1.3633], abs=0.03)
    rpy, gyro, acc = r.imu()
    assert rpy[1] == pytest.approx(-0.03, abs=0.005) and abs(rpy[0]) < 1e-9 and acc[2] == pytest.approx(9.8, abs=0.1)
    q, dq, _ = r.joint_state()
    assert len(q) == 31 and (q[0], q[1]) == (0.0, 0.0)


def test_head_rotation_composes_like_the_mujoco_quaternion():
    world, (r,), (loco,) = make()
    loco.head_pitch, loco.head_yaw = 0.2, 0.5
    run(world, [r], [loco], 3.0)
    _, quat = r.head_pose_in_base()
    assert quat == pytest.approx([0.9648, -0.0304, 0.0885, 0.2457], abs=0.02)  # measured in MuJoCo, same command


def test_perception_sees_the_ball_ahead_and_other_robots_as_opponents():
    world, robots, locos = make(2, [(0.0, 0.0, 0.0), (3.0, 0.0, math.pi)])
    world.reset_ball(2.0, 0.0)
    run(world, robots, locos, 0.2)
    dets, lines = robots[0].perceive(others=[(3.0, 0.0)])
    labels = [d.label for d in dets]
    assert "Ball" in labels and "Opponent" in labels
    ball = next(d for d in dets if d.label == "Ball")
    assert ball.pos_robot == pytest.approx((2.0, 0.0), abs=1e-6)
    world.reset_ball(-2.0, 0.0)  # behind the robot
    assert "Ball" not in [d.label for d in robots[0].perceive()[0]]


def test_detection_noise_grows_with_range():
    cfg = SimConfig(ball_detector="ground_truth", detection_noise_std=0.0, detection_noise_range_gain=0.05)
    world = KinematicWorld(cfg, 1, KinematicParams())
    r = KinematicController(cfg, world, 0, TeamFrame(), (0.0, 0.0, 0.0), np.random.default_rng(3))
    errs = {}
    for dist in (1.5, 6.0):
        world.reset_ball(dist, 0.0)
        e = [np.hypot(*(np.array(d.pos_robot) - (dist, 0.0))) for _ in range(300)
             for d in r.perceive()[0] if d.label == "Ball"]
        errs[dist] = np.mean(e)
    assert errs[1.5] > 0.0 and errs[6.0] > 3.0 * errs[1.5]


def test_the_away_frame_flips_pose_and_imu_yaw():
    world, (r,), _ = make(frame=TeamFrame(flipped=True), spawns=[(-6.0, -7.5, math.pi / 2)])
    assert r.robot_pose().x == pytest.approx(6.0) and r.robot_pose().y == pytest.approx(7.5)
    assert r.team_pose().x == pytest.approx(-6.0)
    assert r.imu()[0][2] == pytest.approx(math.pi / 2)
    ox, oy, oth = r.odometer()
    assert (ox, oy, oth) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


# ---- ball ---------------------------------------------------------------------------------------------------------
def test_ball_rolling_deceleration_matches_the_kick_model():
    assert KinematicParams().ball_roll_decel == KickParams().ball_decel
    world = KinematicWorld(CFG, 1, KinematicParams())  # no robot: it would shove a ball that starts under it
    world.shoot(Shot(0.0, 0.0, 4.0, 0.0, 0.0))
    for _ in range(int(7.0 / DT)):
        world.step()
    assert world.ball_position()[0] == pytest.approx(4.0**2 / (2 * 0.8), rel=0.02)
    assert np.linalg.norm(world.ball_velocity()) == 0.0


def test_lofted_ball_bounces_with_decaying_height_and_settles():
    world, _, _ = make()
    world.shoot(Shot(0.0, 0.0, 6.0, 0.0, 45.0))
    peaks, z_prev, vz_prev = [], 0.0, 1.0
    for _ in range(int(6.0 / DT)):
        world.step()
        z = world.ball_position()[2]
        if vz_prev > 0 and world.ball_velocity()[2] <= 0:
            peaks.append(z)
        vz_prev = world.ball_velocity()[2]
    assert len(peaks) >= 2 and peaks[1] < 0.5 * peaks[0]
    assert world.ball_position()[2] == pytest.approx(CFG.ball_radius)


def test_ball_bounces_off_a_goalpost():
    world, _, _ = make()
    gx, gy = CFG.field.length / 2, CFG.field.goal_width / 2
    world.shoot(Shot(gx - 1.0, gy, 3.0, 0.0, 0.0))  # straight at the post, centred
    for _ in range(int(1.5 / DT)):
        world.step()
    assert world.ball_velocity()[0] < 0.0 and world.ball_position()[0] < gx


def test_walking_robot_pushes_the_ball_ahead_of_it():
    world, (r,), (loco,) = make()
    world.reset_ball(1.0, 0.0)
    run(world, [r], [loco], 4.0, [(0.5, 0, 0)])
    bx = world.ball_position()[0]
    assert bx > r.x + 0.25 and bx > 1.5 and abs(world.ball_position()[1]) < 0.1


def test_a_kick_launches_the_ball():
    world, (r,), (loco,) = make()
    world.reset_ball(0.4, 0.0)
    from soccer_sim.kick_model import KickRef
    loco.visual_kick_active = True
    for k in range(30):
        r.apply(loco, k * DT, KickRef(0.4, 0.0, 0.0, 5.0, k * DT))
        world.step()
    assert world.ball_velocity()[0] > 2.0


# ---- robots -------------------------------------------------------------------------------------------------------
def test_robots_do_not_overlap_and_the_pusher_is_blocked():
    world, robots, locos = make(2, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)], falls=False)
    run(world, robots, locos, 5.0, [(0.6, 0, 0), (0, 0, 0)])
    gap = math.hypot(robots[1].x - robots[0].x, robots[1].y - robots[0].y)
    assert gap >= 2 * KinematicParams().robot_radius - 1e-6
    assert robots[0].x < robots[1].x


def test_without_collisions_robots_walk_through_each_other():
    world, robots, locos = make(2, [(0.0, 0.0, 0.0), (1.0, 0.0, math.pi)], collisions=False)
    run(world, robots, locos, 3.0, [(0.5, 0, 0), (0.5, 0, 0)])
    assert robots[0].x > robots[1].x


HARD = dict(p_fall_max=1.0, v_safe=0.1, v_hard=0.5)


def test_a_hard_collision_topples_both_robots_in_the_brains_terms():
    world, robots, locos = make(2, [(-2.0, 0.0, 0.0), (2.0, 0.0, math.pi)], **HARD)
    tilted = []
    now = 0.0
    for _ in range(int(8.0 / DT)):
        for r, loco in zip(robots, locos):
            loco.vx = 1.0 if r.posture == STANDING else 0.0
            r.apply(loco, now)
        world.step()
        now += DT
        rpy, gyro, _ = robots[0].imu()
        tilted.append((max(abs(rpy[0]), abs(rpy[1] + 0.03)), np.linalg.norm(gyro[:2]), robots[0].posture))
    assert robots[0].posture == DOWN and robots[1].posture == DOWN and robots[0].fallen()
    assert max(t[0] for t in tilted) > 0.99   # brain: |roll| or |pitch| > 0.99 rad means IS_FALLING
    assert tilted[-1][1] < 1e-9               # lying still: "has fallen"
    # shoved backward by the head-on hit: pitch strongly negative (the brain reads that as falling backward)
    assert robots[0].imu()[0][1] < -1.0
    assert robots[0].perceive()[0] == []      # a robot on the ground sees nothing
    assert robots[0].world_velocity() == (0.0, 0.0)


def test_getup_takes_the_brains_request_and_keeps_the_gyro_moving():
    world, robots, locos = make(2, [(-2.0, 0.0, 0.0), (2.0, 0.0, math.pi)], **HARD)
    run(world, robots, locos, 8.0, [(1.0, 0, 0), (1.0, 0, 0)])
    r, loco = robots[0], locos[0]
    assert r.posture == DOWN
    loco.getup_requested = True
    r.apply(loco, 0.0)
    assert r.posture == GETTING_UP and not loco.getup_requested
    moving_while_tilted, t = [], 0.0
    while r.posture == GETTING_UP and t < 10.0:
        r.apply(loco, t)
        world.step()
        t += DT
        rpy, gyro, _ = r.imu()
        if max(abs(rpy[0]), abs(rpy[1])) > 0.99:
            moving_while_tilted.append(np.linalg.norm(gyro[:2]))
    assert r.posture == STANDING and 4.0 < t < 6.0
    assert moving_while_tilted and min(moving_while_tilted) > 0.1  # brain: tilted and gyro < 0.1 means failed


def test_getup_is_ignored_unless_the_robot_is_lying_down():
    world, (r,), (loco,) = make()
    loco.getup_requested = True
    r.apply(loco, 0.0)
    assert r.posture == STANDING and not loco.getup_requested


def test_falls_can_be_switched_off():
    world, robots, locos = make(2, [(-2.0, 0.0, 0.0), (2.0, 0.0, math.pi)], falls=False, **HARD)
    run(world, robots, locos, 8.0, [(1.0, 0, 0), (1.0, 0, 0)])
    assert robots[0].posture == STANDING and robots[1].posture == STANDING


def test_reset_stands_a_fallen_robot_up_where_it_is_put():
    world, robots, locos = make(2, [(-2.0, 0.0, 0.0), (2.0, 0.0, math.pi)], **HARD)
    run(world, robots, locos, 8.0, [(1.0, 0, 0), (1.0, 0, 0)])
    robots[0].reset_robot(1.0, 2.0, 0.5)
    assert robots[0].posture == STANDING and not robots[0].fallen()
    assert (robots[0].x, robots[0].y, robots[0].yaw) == (1.0, 2.0, 0.5)


def test_same_seed_same_falls_different_seed_may_differ():
    def outcome(seed):
        world, robots, locos = make(2, [(-2.0, 0.0, 0.0), (2.0, 0.0, math.pi)], seed=seed,
                                    p_fall_max=0.5, v_safe=0.1, v_hard=0.5)
        run(world, robots, locos, 6.0, [(1.0, 0, 0), (1.0, 0, 0)])
        return tuple(r.posture for r in robots)
    assert outcome(5) == outcome(5)
    assert len({outcome(s) for s in range(12)}) > 1


# ---- odometry ------------------------------------------------------------------------------------------------------
def test_odometer_without_drift_is_the_true_displacement():
    world, (r,), (loco,) = make()
    run(world, [r], [loco], 3.0, [(0.5, 0.0, 0.3)])
    x, y, th = r.odometer()
    p = r.team_pose()
    assert (x, y, th) == pytest.approx((p.x, p.y, p.yaw), abs=1e-9)


def test_odometry_drift_grows_with_distance_and_not_while_standing():
    cfg = SimConfig(ball_detector="ground_truth", odom_drift_pct=3.0, odom_yaw_drift_deg=3.0)
    errors = []
    for seed in range(12):
        world = KinematicWorld(cfg, 1, KinematicParams())
        r = KinematicController(cfg, world, 0, TeamFrame(), (0.0, 0.0, 0.0), np.random.default_rng(seed))
        loco = LocoState(mode=MODE_SOCCER)
        loco.vx = 0.6
        for k in range(int(10.0 / DT)):
            r.apply(loco, k * DT)
            world.step()
            r.odometer()
        x, y, th = r.odometer()
        errors.append(math.hypot(x - r.x, y - r.y))
        rest = r.odometer()
        loco.vx = 0.0
        for k in range(int(3.0 / DT)):
            r.apply(loco, k * DT)
            world.step()
        before = r.odometer()
        for k in range(int(4.0 / DT)):
            r.apply(loco, k * DT)
            world.step()
            r.odometer()
        after = r.odometer()
        assert after == pytest.approx(before, abs=1e-9)  # standing still: the odometer stays put
    assert 0.05 < np.mean(errors) < 1.0 and max(errors) > 0.1


def test_odometry_drift_is_seeded_and_teleports_are_exact():
    def final(seed):
        t = OdometryTracker(Pose2D(0, 0, 0), 5.0, 5.0, np.random.default_rng(seed))
        for k in range(1, 200):
            out = t.update(Pose2D(0.02 * k, 0.0, 0.0))
        return out
    assert final(4) == final(4) and final(4) != final(5)
    t = OdometryTracker(Pose2D(0, 0, 0), 5.0, 5.0, np.random.default_rng(1))
    t.update(Pose2D(1.0, 0.0, 0.0))
    x0, y0, _ = t.update(Pose2D(1.0, 0.0, 0.0))
    t.teleport(Pose2D(4.0, 0.0, 0.0))
    x1, y1, _ = t.update(Pose2D(4.0, 0.0, 0.0))
    # exact in distance (the odometer frame itself may have drifted in heading, which rotates the displacement)
    assert math.hypot(x1 - x0, y1 - y0) == pytest.approx(3.0, abs=1e-9)


# ---- seam ------------------------------------------------------------------------------------------------------------
def test_sim_params_resolve_to_the_right_dataclass_and_reject_junk():
    cfg, kin = resolve_params(SimConfig(), ["detection_noise_std=0.1", "collisions=false", "seed=7", "falls=true",
                                            "odom_drift_pct=2"])
    assert cfg.detection_noise_std == 0.1 and cfg.seed == 7 and cfg.odom_drift_pct == 2.0
    assert kin.collisions is False and kin.falls is True
    for bad in ("nonsense=1", "collisions=maybe", "detection_noise_std=abc", "field=1", "noequals"):
        with pytest.raises(ValueError):
            resolve_params(SimConfig(), [bad])


def test_robot_streams_are_independent_and_reproducible():
    a = [g.random() for g in robot_rngs(3, 3)]
    assert a == [g.random() for g in robot_rngs(3, 3)] and len(set(a)) == 3


def test_backend_factory():
    b = load_backend("kinematic")
    world = b.make_world(SimConfig(), 2, KinematicParams())
    robot = b.make_robot(SimConfig(), world, 0, TeamFrame(), (0.0, 0.0, 0.0), np.random.default_rng(0))
    assert isinstance(world, KinematicWorld) and isinstance(robot, KinematicController) and world.policy_dt == 0.02
    with pytest.raises(ValueError):
        load_backend("gazebo")


def test_the_2d_stack_imports_nothing_heavy():
    code = (
        "import sys\n"
        "import soccer_sim.backend, soccer_sim.kinematic, soccer_sim.perception, soccer_sim.frames, soccer_sim.odometry\n"
        "import soccer_sim.game_controller, soccer_sim.shots, soccer_sim.placement, soccer_sim.roster, soccer_sim.config\n"
        "soccer_sim.backend.load_backend('kinematic')\n"
        "bad = [m for m in ('torch', 'mujoco', 'booster_deploy', 'onnxruntime', 'cv2') if m in sys.modules]\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[1])})
    assert out.returncode == 0, out.stdout + out.stderr
