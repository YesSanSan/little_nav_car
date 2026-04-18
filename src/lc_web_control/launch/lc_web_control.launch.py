import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_workspace = os.environ.get("LITTLE_NAV_CAR_WS", str(Path.cwd()))
    default_state_file = str(Path.home() / ".little_nav_car" / "web_control" / "return_pose.json")

    bind_host = LaunchConfiguration("bind_host")
    port = LaunchConfiguration("port")
    workspace_dir = LaunchConfiguration("workspace_dir")
    state_file = LaunchConfiguration("state_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bind_host",
                default_value="0.0.0.0",
                description="HTTP bind address for the local web dashboard",
            ),
            DeclareLaunchArgument(
                "port",
                default_value="8080",
                description="HTTP port for the local web dashboard",
            ),
            DeclareLaunchArgument(
                "workspace_dir",
                default_value=default_workspace,
                description="Workspace root used for calling slam.sh/nav.sh",
            ),
            DeclareLaunchArgument(
                "state_file",
                default_value=default_state_file,
                description="Path used to persist the marked return-home pose",
            ),
            Node(
                package="lc_web_control",
                executable="web_control",
                name="lc_web_control",
                output="screen",
                parameters=[
                    {
                        "bind_host": bind_host,
                        "port": port,
                        "workspace_dir": workspace_dir,
                        "state_file": state_file,
                    }
                ],
            ),
        ]
    )
