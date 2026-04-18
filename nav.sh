#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"
DEFAULT_OPENVINO_ROOT="${HOME}/intel/openvino_2026"
DEFAULT_ASTRA_SDK_ROOT="/home/cmls/sanwu/AstraSDK-v2.1.3-Ubuntu-x86_64/AstraSDK-v2.1.3-94bca0f52e-20210608T062039Z-Ubuntu18.04-x86_64"
USE_LC_VISION="true"
STACK_MODE_ARG="--detach"

export STACK_SESSION_NAME="nav"
export STACK_WORKSPACE_DIR="${WORKSPACE_DIR}"
export STACK_MODE_LABEL="ROS2 navigation"
export STACK_ENTRY_SCRIPT_NAME="nav.sh"

usage() {
  cat <<EOF
Usage: ./nav.sh [--attach|--detach|--stop|--status|--help] [--no-vision]

  --no-vision  Do not launch lc_vision.
EOF
}

while (($# > 0)); do
  case "$1" in
    --no-vision)
      USE_LC_VISION="false"
      shift
      ;;
    --attach|--detach|--stop|--status)
      STACK_MODE_ARG="$1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${WORKSPACE_DIR}/install/setup.bash" ]]; then
  echo "Missing ${WORKSPACE_DIR}/install/setup.bash. Please build the workspace first with colcon build." >&2
  exit 1
fi

export OPENVINO_ROOT="${OPENVINO_ROOT:-${DEFAULT_OPENVINO_ROOT}}"
export ASTRA_SDK_ROOT="${ASTRA_SDK_ROOT:-${DEFAULT_ASTRA_SDK_ROOT}}"

NAVIGATION_CMD="ros2 launch lc_navigation bringup_launch.py use_lc_vision:=True use_rviz:=False"
if [[ "${USE_LC_VISION}" != "true" ]]; then
  NAVIGATION_CMD="ros2 launch lc_navigation bringup_launch.py use_lc_vision:=False use_rviz:=False"
fi

export STACK_TASK_NAMES="lidar|yesense|serial|localization|navigation"
export STACK_TASK_CMDS="ros2 launch lslidar_driver lslidar_launch.py|ros2 launch yesense_std_ros2 yesense_node.launch.py|ros2 launch rm_serial_driver serial_driver.launch.py|ros2 launch lc_localization ekf.launch.py|${NAVIGATION_CMD}"
export STACK_TASK_START_DELAYS="0|0|0|0|2"

export STACK_IMU_TOPIC="/base/imu0"
export STACK_IMU_STABLE_DURATION_SEC="2.0"
export STACK_IMU_MAX_GAP_SEC="0.30"
export STACK_IMU_WAIT_SCRIPT="${WORKSPACE_DIR}/scripts/wait_for_imu_stable.py"

exec "${WORKSPACE_DIR}/scripts/start_stack.sh" "${STACK_MODE_ARG}"
