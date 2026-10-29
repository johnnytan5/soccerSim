"""Constants of the 2D kinematic simulator, in one place.

Standing geometry is read from the T2 MJCF (vendor/booster_assets .../T2_31dof.xml) and a MuJoCo run of the walk
policy; the walking response was measured from `t2_walk` step responses (scripts/calibrate_kinematics.py prints the
same numbers, refit them there when the policy changes). Everything marked "guess" has no measurement behind it.
"""
from dataclasses import dataclass

# ---- standing robot ------------------------------------------------------------------------------------------
STAND_TRUNK_Z = 0.9752        # m, trunk height after 3 s of standing under the walk policy
STAND_TRUNK_PITCH = -0.03     # rad, the policy leans back slightly
LYING_TRUNK_Z = 0.15          # m, guess
HEAD_OFFSET = (0.0537, 0.0, 0.30246)  # m, head_pitch_link origin in the trunk frame (MJCF; independent of head angles)

# Joint positions while standing (rad); the brain only reads [0] (head yaw) and [1] (head pitch) from /low_state.
DEFAULT_JOINT_POS = (
    -0.0, 0.0093, 0.3004, -1.2718, 0.2269, -0.6583, 0.0035, 0.0552, -0.0251, 0.2846, 1.2699, 0.2646, 0.6452, 0.0034,
    -0.0545, 0.0241, -0.0024, -0.0003, -0.0023, -0.0033, 0.0733, 0.1063, 0.1664, -0.124, -0.0684, -0.0032, -0.0721,
    -0.1059, 0.1703, -0.1281, 0.0671,
)


@dataclass(frozen=True)
class KinematicParams:
    # ---- walking: command -> achieved body velocity (first-order lag, acceleration-limited) -------------------
    # achieved steady speed = gain * commanded speed (the policy overshoots forward, strafes poorly)
    gain_x_forward: float = 1.17
    gain_x_backward: float = 0.86
    gain_y: float = 0.50
    gain_yaw: float = 1.00
    tau_x: float = 0.55           # s, speeding up
    tau_x_decel: float = 0.40     # s, slowing down
    tau_y: float = 0.15
    tau_yaw: float = 0.07
    accel_x: float = 3.0          # m/s^2 cap
    accel_y: float = 3.0
    accel_yaw: float = 8.0        # rad/s^2
    cmd_delay_steps: int = 2      # policy ticks between a command and its first effect
    # ---- head -----------------------------------------------------------------------------------------------
    head_tau: float = 0.12        # s, guess
    head_rate_max: float = 4.0    # rad/s, guess
    # ---- bodies ---------------------------------------------------------------------------------------------
    robot_radius: float = 0.28    # m, collision disc (shoulder width about 0.5 m), guess
    lying_radius: float = 0.40    # m, a fallen robot is a longer obstacle, guess
    ball_contact_radius: float = 0.23  # m: ball centre farther than this + ball radius from the trunk centre is free
    ball_contact_height: float = 0.40  # m: a ball above this flies over the legs
    ball_contact_restitution: float = 0.30
    # ---- ball -----------------------------------------------------------------------------------------------
    ball_roll_decel: float = 0.8  # m/s^2, equal to kick_model.KickParams.ball_decel (a test pins them together)
    ball_gravity: float = 9.81
    ball_restitution: float = 0.55   # vertical bounce, guess
    ball_bounce_friction: float = 0.10  # horizontal speed lost per bounce, guess
    ball_post_restitution: float = 0.50
    ball_launch_efficiency: float = 1.0  # fraction of the kick model's strike speed the ball leaves with
    ball_substeps: int = 4
    # ---- contact, falls -------------------------------------------------------------------------------------
    collisions: bool = True       # robots block and push each other
    falls: bool = True            # hard contact can topple a robot
    v_safe: float = 0.6           # m/s closing speed below which contact never topples anyone, guess
    v_hard: float = 1.6           # m/s closing speed at which toppling is most likely, guess
    p_fall_max: float = 0.7       # probability at v_hard, guess
    fall_secs: float = 0.6        # standing -> lying
    getup_secs: float = 5.0       # lying -> standing once the brain asks (brain itself waits >= 6.5 s)
    getup_jitter: float = 0.10    # +- fraction
