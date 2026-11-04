"""Start one unmodified Robotedge brain_node per robot of a match, each in its own ROS domain.

roster:      JSON list of {team, player_id, role}; team is "home" or "away".
domain_base: ROS domain of the first robot; robot i uses domain_base + i, the same as the match bridge.
home_brain_install / away_brain_install: colcon install dir of another brain build for that team's robots (empty = the
             sourced one). Its brain_node is run by absolute path; the rest of the environment is shared.
home_brain_root / away_brain_root: Robotedge checkout whose config.yaml, behaviour trees and vision config that team's
             brains use (empty = $ROBOTEDGE_ROOT). Set together with the install dir for a head-to-head match.
netns:       "true" runs robot i's brain inside network namespace ss-r{i+1} (scripts/netem.sh up, scripts/ns_exec.sh).
brain_params: JSON object of extra ROS parameter overrides for every brain (dotted names, typed values), e.g.
             {"communication.team_broadcast_rate_hz": 3.0, "world_model.fusion.method": "ci"}. Wins over
             enable_com:= and config.yaml. The launcher builds it from its settings.
"""

import json
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

TEAM_NUMBERS = {"home": 131, "away": 1}  # mirrors soccer_sim.roster.TEAM_NUMBERS
# brain's enter_field prior depends on the touchline the robot starts at: mirrors soccer_sim.roster
START_SIDE = {1: "right", 2: "left", 3: "right", 4: "left", 5: "right"}


def team_brain(team, install, root, default_root):
    """(brain_node executable or None for the sourced one, Robotedge root) of a team."""
    exe = os.path.join(install, "brain", "lib", "brain", "brain_node") if install else None
    return exe, root or default_root


def _brains(context):
    default_root = os.environ.get("ROBOTEDGE_ROOT", os.path.expanduser("~/Robotedge-5v5-RCAP2026"))
    roster = json.loads(LaunchConfiguration("roster").perform(context))
    base = int(LaunchConfiguration("domain_base").perform(context))
    tree = LaunchConfiguration("tree").perform(context)
    enable_com = LaunchConfiguration("enable_com").perform(context).lower() == "true"
    gc_ip = LaunchConfiguration("game_control_ip").perform(context)
    team_cfg = {}
    for team in ("home", "away"):
        team_cfg[team] = team_brain(team, LaunchConfiguration(f"{team}_brain_install").perform(context),
                                    LaunchConfiguration(f"{team}_brain_root").perform(context), default_root)
    use_netns = LaunchConfiguration("netns").perform(context).lower() == "true"  # scripts/netem.sh namespaces are up
    ns_exec = os.path.join(os.environ.get("SOCCER_SIM_ROOT", os.getcwd()), "scripts", "ns_exec.sh")
    extra = json.loads(LaunchConfiguration("brain_params").perform(context) or "{}")
    if not isinstance(extra, dict):
        raise ValueError("brain_params must be a JSON object")
    nodes = []
    for i, p in enumerate(roster):
        exe, root = team_cfg[p["team"]]
        brain_dir = os.path.join(root, "src", "brain")
        overrides = {
            "tree_file_path": os.path.join(brain_dir, "behavior_trees", tree),
            "vision_config_path": os.path.join(root, "src", "vision", "config", "vision.yaml"),
            "vision_config_local_path": os.path.join(root, "src", "vision", "config", "vision_local.yaml"),
            "rerunLog.enable_file": False,
            "rerunLog.enable_tcp": False,
            "odomLog.enable": False,
            "enable_com": enable_com,
            "game.team_id": TEAM_NUMBERS[p["team"]],
            "game.player_id": int(p["player_id"]),
            "game.player_role": p["role"],
            "game.player_start_pos": START_SIDE[int(p["player_id"])],
        }
        if gc_ip:
            overrides["game_control_ip"] = gc_ip
        overrides.update(extra)
        nodes.append(
            Node(
                package="brain" if exe is None else None,
                executable="brain_node" if exe is None else exe,
                name="brain_node",
                output="screen",
                parameters=[os.path.join(brain_dir, "config", "config.yaml"), overrides],
                additional_env={"ROS_DOMAIN_ID": str(base + i)},
                # robot i runs in network namespace ss-r{i+1}, as the same user with the same environment
                prefix=f"bash {ns_exec} ss-r{i + 1}" if use_netns else None,
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("roster"),
            DeclareLaunchArgument("domain_base", default_value="70"),
            DeclareLaunchArgument("tree", default_value="game.xml"),
            DeclareLaunchArgument("enable_com", default_value="true"),
            DeclareLaunchArgument("game_control_ip", default_value=""),
            DeclareLaunchArgument("home_brain_install", default_value=""),
            DeclareLaunchArgument("away_brain_install", default_value=""),
            DeclareLaunchArgument("home_brain_root", default_value=""),
            DeclareLaunchArgument("away_brain_root", default_value=""),
            DeclareLaunchArgument("brain_params", default_value="{}"),
            DeclareLaunchArgument("netns", default_value="false"),
            OpaqueFunction(function=_brains),
        ]
    )
