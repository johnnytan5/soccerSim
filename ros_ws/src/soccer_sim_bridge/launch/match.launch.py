"""Start one unmodified Robotedge brain_node per robot of a match, each in its own ROS domain.

roster:      JSON list of {team, player_id, role}; team is "home" or "away".
domain_base: ROS domain of the first robot; robot i uses domain_base + i, the same as the match bridge.
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


def _brains(context):
    root = os.environ.get("ROBOTEDGE_ROOT", os.path.expanduser("~/Robotedge-5v5-RCAP2026"))
    brain_dir = os.path.join(root, "src", "brain")
    roster = json.loads(LaunchConfiguration("roster").perform(context))
    base = int(LaunchConfiguration("domain_base").perform(context))
    tree = LaunchConfiguration("tree").perform(context)
    enable_com = LaunchConfiguration("enable_com").perform(context).lower() == "true"
    gc_ip = LaunchConfiguration("game_control_ip").perform(context)
    nodes = []
    for i, p in enumerate(roster):
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
        nodes.append(
            Node(
                package="brain",
                executable="brain_node",
                name="brain_node",
                output="screen",
                parameters=[os.path.join(brain_dir, "config", "config.yaml"), overrides],
                additional_env={"ROS_DOMAIN_ID": str(base + i)},
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
            OpaqueFunction(function=_brains),
        ]
    )
