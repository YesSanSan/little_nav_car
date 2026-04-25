#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"
STACK_MODE_ARG="--detach"
WEB_HOST="${LC_WEB_HOST:-0.0.0.0}"
WEB_PORT="${LC_WEB_PORT:-8080}"
VISION_SETTINGS_FILE="${WORKSPACE_DIR}/build/lc_web_control/web_vision_settings.json"
VISION_YAML_FILE="${WORKSPACE_DIR}/src/lc_vision/config/vision.yaml"

export STACK_SESSION_NAME="web"
export STACK_WORKSPACE_DIR="${WORKSPACE_DIR}"
export STACK_MODE_LABEL="ROS2 web control"
export STACK_ENTRY_SCRIPT_NAME="web.sh"

usage() {
  cat <<EOF
Usage: ./web.sh [--attach|--detach|--stop|--status|--help] [--host HOST] [--port PORT]

Environment:
  LC_WEB_HOST  Default bind host (current: ${WEB_HOST})
  LC_WEB_PORT  Default HTTP port (current: ${WEB_PORT})
EOF
}

while (($# > 0)); do
  case "$1" in
    --attach|--detach|--stop|--status)
      STACK_MODE_ARG="$1"
      shift
      ;;
    --host)
      WEB_HOST="${2:?Missing value for --host}"
      shift 2
      ;;
    --port)
      WEB_PORT="${2:?Missing value for --port}"
      shift 2
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

python3 "${WORKSPACE_DIR}/scripts/sync_web_vision_settings.py" \
  --vision-yaml "${VISION_YAML_FILE}" \
  --settings-file "${VISION_SETTINGS_FILE}"

export STACK_TASK_NAMES="web_control"
export STACK_TASK_CMDS="ros2 launch lc_web_control lc_web_control.launch.py workspace_dir:=${WORKSPACE_DIR} bind_host:=${WEB_HOST} port:=${WEB_PORT} settings_file:=${VISION_SETTINGS_FILE}"
export STACK_TASK_START_DELAYS="0"
export STACK_IMU_TOPIC="/base/imu0"
export STACK_IMU_STABLE_DURATION_SEC="2.0"
export STACK_IMU_MAX_GAP_SEC="0.30"
export STACK_IMU_WAIT_SCRIPT="${WORKSPACE_DIR}/scripts/wait_for_imu_stable.py"

exec "${WORKSPACE_DIR}/scripts/start_stack.sh" "${STACK_MODE_ARG}"
