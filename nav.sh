#!/bin/bash

set -euo pipefail

WORKSPACE_DIR="/home/betty/help_ws/little_nav_car"

export STACK_SESSION_NAME="nav"
export STACK_WORKSPACE_DIR="${WORKSPACE_DIR}"
export STACK_MODE_LABEL="ROS2 navigation"
export STACK_ENTRY_SCRIPT_NAME="nav.sh"

export STACK_TASK_NAMES="lidar|yesense|serial|localization|navigation"
export STACK_TASK_CMDS="ros2 launch lslidar_driver lslidar_launch.py|ros2 launch yesense_std_ros2 yesense_node.launch.py|ros2 launch rm_serial_driver serial_driver.launch.py|ros2 launch lc_localization ekf.launch.py|ros2 launch lc_navigation bringup_launch.py"
export STACK_TASK_START_DELAYS="0|0|0|0|2"

export STACK_IMU_TOPIC="/base/imu0"
export STACK_IMU_STABLE_DURATION_SEC="2.0"
export STACK_IMU_MAX_GAP_SEC="0.30"
export STACK_IMU_WAIT_SCRIPT="${WORKSPACE_DIR}/scripts/wait_for_imu_stable.py"

exec "${WORKSPACE_DIR}/scripts/start_stack.sh" "$@"
