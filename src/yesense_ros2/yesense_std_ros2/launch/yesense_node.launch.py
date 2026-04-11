from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('yesense_std_ros2'),
        'config',
        'yesense_config.yaml',
    ),

    return LaunchDescription([
        Node(
            package='yesense_std_ros2',
            executable='yesense_node_publisher',
            name='yesense_pub',
            parameters=[config],
            output='screen',
        ),
        # Node(
        #     package="tf2_ros",
        #     executable="static_transform_publisher",
        #     arguments=['0','0','0','3.14159265358979','0','-1.57079632679','base_link','imu_link'],
        # ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=[
                '--x', '0.0',
                '--y', '0.0',
                '--z', '0.0',
                '--yaw', '0.0',
                '--pitch', '0.0',
                '--roll', '3.14159265358979',
                '--frame-id', 'base_link',
                '--child-frame-id', 'imu_link',
            ],
        ),
    ])
