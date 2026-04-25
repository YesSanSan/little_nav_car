#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"
STACK_MODE_ARG="--detach"

export STACK_SESSION_NAME="serial"
export STACK_WORKSPACE_DIR="${WORKSPACE_DIR}"
export STACK_MODE_LABEL="ROS2 serial driver"
export STACK_ENTRY_SCRIPT_NAME="auto_serial.sh"

usage() {
  cat <<EOF
Usage: ./auto_serial.sh [--attach|--detach|--stop|--status|--help]
EOF
}

while (($# > 0)); do
  case "$1" in
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

export STACK_TASK_NAMES="serial"
export STACK_TASK_CMDS="ros2 launch rm_serial_driver serial_driver.launch.py"
export STACK_TASK_START_DELAYS="0"

exec "${WORKSPACE_DIR}/scripts/start_stack.sh" "${STACK_MODE_ARG}"
