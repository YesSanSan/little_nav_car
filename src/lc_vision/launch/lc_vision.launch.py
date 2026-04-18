import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("lc_vision")

    default_sdk_root = os.environ.get(
        "ASTRA_SDK_ROOT",
        "/home/cmls/sanwu/AstraSDK-v2.1.3-Ubuntu-x86_64/AstraSDK-v2.1.3-94bca0f52e-20210608T062039Z-Ubuntu18.04-x86_64",
    )
    default_sdk_root = os.path.expanduser(default_sdk_root)
    default_params_file = os.path.join(package_share, "config", "vision.yaml")

    sdk_root = LaunchConfiguration("sdk_root")
    params_file = LaunchConfiguration("params_file")
    camera_x = LaunchConfiguration("camera_x")
    camera_y = LaunchConfiguration("camera_y")
    camera_z = LaunchConfiguration("camera_z")
    camera_roll = LaunchConfiguration("camera_roll")
    camera_pitch = LaunchConfiguration("camera_pitch")
    camera_yaw = LaunchConfiguration("camera_yaw")

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

    declare_camera_x = DeclareLaunchArgument(
        "camera_x",
        default_value="-0.08",
        description="Camera mount X offset from base_link in meters.",
    )

    declare_camera_y = DeclareLaunchArgument(
        "camera_y",
        default_value="0.0",
        description="Camera mount Y offset from base_link in meters.",
    )

    declare_camera_z = DeclareLaunchArgument(
        "camera_z",
        default_value="0.0",
        description="Camera mount Z offset from base_link in meters.",
    )

    declare_camera_roll = DeclareLaunchArgument(
        "camera_roll",
        default_value="1.570796",
        description="Camera mount roll in radians.",
    )

    declare_camera_pitch = DeclareLaunchArgument(
        "camera_pitch",
        default_value="-0.261799",
        description="Camera mount pitch in radians.",
    )

    declare_camera_yaw = DeclareLaunchArgument(
        "camera_yaw",
        default_value="0.0",
        description="Camera mount yaw in radians.",
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

    lc_vision_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            '--x', camera_x,
            '--y', camera_y,
            '--z', camera_z,
            '--yaw', camera_yaw,
            '--pitch', camera_pitch,
            '--roll', camera_roll,
            '--frame-id', 'base_link',
            '--child-frame-id', 'cam_link',
        ],
    )

    lc_vision_optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--yaw', '-1.5707963267948966',
            '--pitch', '0.0',
            '--roll', '-1.5707963267948966',
            '--frame-id', 'cam_link',
            '--child-frame-id', 'cam_optical_frame',
        ],
    )

    return LaunchDescription(
        [
            declare_sdk_root,
            declare_params_file,
            declare_camera_x,
            declare_camera_y,
            declare_camera_z,
            declare_camera_roll,
            declare_camera_pitch,
            declare_camera_yaw,
            lc_vision_node,
            lc_vision_tf,
            lc_vision_optical_tf,
        ]
    )
