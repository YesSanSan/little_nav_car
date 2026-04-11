import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("lc_vision")

    default_sdk_root = os.path.expanduser(
        "~/AstraSDK-v2.1.3-Linux-arm/AstraSDK-v2.1.3-94bca0f52e-20210611T023312Z-Linux-aarch64"
    )
    default_params_file = os.path.join(package_share, "config", "vision.yaml")

    sdk_root = LaunchConfiguration("sdk_root")
    params_file = LaunchConfiguration("params_file")

    declare_sdk_root = DeclareLaunchArgument(
        "sdk_root",
        default_value=default_sdk_root,
        description="Path to the Astra SDK root directory.",
    )

    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to the lc_vision parameter YAML file.",
    )

    lc_vision_node = Node(
        package="lc_vision",
        executable="lc_vision_node",
        namespace="",
        output="screen",
        emulate_tty=True,
        parameters=[params_file, {"sdk_root": sdk_root}],
        cwd=PathJoinSubstitution([sdk_root, "lib"]),
    )

    return LaunchDescription(
        [
            declare_sdk_root,
            declare_params_file,
            lc_vision_node,
        ]
    )
