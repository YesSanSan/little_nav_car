#!/bin/bash

set -euo pipefail

SESSION_NAME="${STACK_SESSION_NAME:?STACK_SESSION_NAME is required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="${STACK_WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
RUNTIME_LOG_ROOT="${STACK_RUNTIME_LOG_ROOT:-${WORKSPACE_DIR}/runtime_logs}"
CURRENT_LOG_FILE="${RUNTIME_LOG_ROOT}/.${SESSION_NAME}_current_logdir"
MAX_LOG_DIRS="${STACK_MAX_LOG_DIRS:-5}"

IFS='|' read -r -a TASK_NAMES <<< "${STACK_TASK_NAMES:?STACK_TASK_NAMES is required}"
IFS='|' read -r -a TASK_CMDS <<< "${STACK_TASK_CMDS:?STACK_TASK_CMDS is required}"
IFS='|' read -r -a TASK_START_DELAYS <<< "${STACK_TASK_START_DELAYS:?STACK_TASK_START_DELAYS is required}"

IMU_TOPIC="${STACK_IMU_TOPIC:-/base/imu0}"
IMU_STABLE_DURATION_SEC="${STACK_IMU_STABLE_DURATION_SEC:-2.0}"
IMU_MAX_GAP_SEC="${STACK_IMU_MAX_GAP_SEC:-0.30}"
IMU_WAIT_SCRIPT="${STACK_IMU_WAIT_SCRIPT:-${WORKSPACE_DIR}/scripts/wait_for_imu_stable.py}"
MODE_LABEL="${STACK_MODE_LABEL:-ROS2 stack}"
ENTRY_SCRIPT_NAME="${STACK_ENTRY_SCRIPT_NAME:-start_stack.sh}"

if (( ${#TASK_NAMES[@]} != ${#TASK_CMDS[@]} || ${#TASK_NAMES[@]} != ${#TASK_START_DELAYS[@]} )); then
  echo "TASK_NAMES, TASK_CMDS, and TASK_START_DELAYS must have the same length." >&2
  exit 1
fi

usage() {
  cat <<EOF
Usage: ./${ENTRY_SCRIPT_NAME} [--attach|--detach|--stop|--status|--help]

  ./${ENTRY_SCRIPT_NAME}           Create a tmux session in detached mode and start all ${MODE_LABEL} tasks.
  ./${ENTRY_SCRIPT_NAME} --attach  Create the session and attach to it. If it already exists, attach directly.
  ./${ENTRY_SCRIPT_NAME} --detach  Same as the default behavior.
  ./${ENTRY_SCRIPT_NAME} --stop    Stop the tmux session.
  ./${ENTRY_SCRIPT_NAME} --status  Show session status, windows, and log directory.
  ./${ENTRY_SCRIPT_NAME} --help    Show this help message.

Tips:
  tmux attach -t ${SESSION_NAME}
  byobu attach -t ${SESSION_NAME}
EOF
}

session_exists() {
  tmux has-session -t "${SESSION_NAME}" 2>/dev/null
}

current_log_dir() {
  if [[ -f "${CURRENT_LOG_FILE}" ]]; then
    cat "${CURRENT_LOG_FILE}"
  fi
}

print_status() {
  local latest_link latest_target
  latest_link="${RUNTIME_LOG_ROOT}/latest"

  if session_exists; then
    echo "Session '${SESSION_NAME}' is running."
    echo "Windows:"
    tmux list-windows -t "${SESSION_NAME}" -F "  - #{window_index}: #{window_name}"
  else
    echo "Session '${SESSION_NAME}' is not running."
  fi

  if [[ -f "${CURRENT_LOG_FILE}" ]]; then
    echo "Current log dir: $(current_log_dir)"
  fi

  if [[ -L "${latest_link}" ]]; then
    latest_target="$(readlink -f "${latest_link}")"
    echo "Latest log dir: ${latest_target}"
  fi
}

prune_old_logs() {
  local dir_count=0
  local log_dir
  mapfile -t log_dirs < <(find "${RUNTIME_LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -rn | awk '{print $2}')

  for log_dir in "${log_dirs[@]}"; do
    ((dir_count += 1))
    if ((dir_count > MAX_LOG_DIRS)); then
      rm -rf "${log_dir}"
    fi
  done
}

build_wait_for_imu_command() {
  printf 'python3 %q --topic %q --stable-duration %q --max-gap %q && ' \
    "${IMU_WAIT_SCRIPT}" "${IMU_TOPIC}" "${IMU_STABLE_DURATION_SEC}" "${IMU_MAX_GAP_SEC}"
}

build_window_command() {
  local task_name="$1"
  local task_cmd="$2"
  local start_delay="$3"
  local log_file shell_cmd delay_cmd pre_cmd env_setup_cmd

  log_file="${LOG_DIR}/${task_name}.log"

  if (( start_delay > 0 )); then
    printf -v delay_cmd 'echo "[%s] Waiting %ss before launch..." && sleep %q && ' \
      "${task_name}" "${start_delay}" "${start_delay}"
  else
    delay_cmd=""
  fi

  printf -v env_setup_cmd '%s' \
    'unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH PYTHONPATH ROS_PACKAGE_PATH ROS_DISTRO ROS_VERSION ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION && source /opt/ros/jazzy/setup.bash && source install/setup.bash && '

  pre_cmd="${env_setup_cmd}${delay_cmd}"
  if [[ "${task_name}" == "localization" ]]; then
    printf -v pre_cmd '%s%s' "${pre_cmd}" "$(build_wait_for_imu_command)"
  fi

  printf -v shell_cmd \
    'cd %q && mkdir -p %q && exec > >(tee -a %q) 2>&1 && echo "[%s] Logging to %s" && exec stdbuf -oL -eL bash -lc %q' \
    "${WORKSPACE_DIR}" "${LOG_DIR}" "${log_file}" "${task_name}" "${log_file}" "${pre_cmd}${task_cmd}"

  printf '%s' "${shell_cmd}"
}

start_session() {
  local timestamp latest_link first_cmd first_name i target_index

  if session_exists; then
    echo "Session '${SESSION_NAME}' already exists. Use --status to inspect or --stop to terminate it." >&2
    return 1
  fi

  mkdir -p "${RUNTIME_LOG_ROOT}"
  timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
  LOG_DIR="${RUNTIME_LOG_ROOT}/${timestamp}"
  mkdir -p "${LOG_DIR}"

  printf '%s\n' "${LOG_DIR}" > "${CURRENT_LOG_FILE}"

  latest_link="${RUNTIME_LOG_ROOT}/latest"
  ln -sfn "${LOG_DIR}" "${latest_link}"
  prune_old_logs

  first_name="${TASK_NAMES[0]}"
  first_cmd="$(build_window_command "${first_name}" "${TASK_CMDS[0]}" "${TASK_START_DELAYS[0]}")"
  tmux new-session -d -s "${SESSION_NAME}" -n "${first_name}" "bash -lc $(printf '%q' "${first_cmd}")"
  tmux set-option -t "${SESSION_NAME}" remain-on-exit on >/dev/null

  for ((i = 1; i < ${#TASK_NAMES[@]}; ++i)); do
    target_index="${i}"
    tmux new-window -d -t "${SESSION_NAME}:${target_index}" -n "${TASK_NAMES[i]}" \
      "bash -lc $(printf '%q' "$(build_window_command "${TASK_NAMES[i]}" "${TASK_CMDS[i]}" "${TASK_START_DELAYS[i]}")")"
    sleep 0.2
  done

  echo "Started tmux session '${SESSION_NAME}'."
  echo "Log directory: ${LOG_DIR}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  echo "Or via byobu: byobu attach -t ${SESSION_NAME}"
}

stop_session() {
  if session_exists; then
    tmux kill-session -t "${SESSION_NAME}"
    echo "Stopped tmux session '${SESSION_NAME}'."
  else
    echo "Session '${SESSION_NAME}' is not running."
  fi

  rm -f "${CURRENT_LOG_FILE}"
}

main() {
  local mode="${1:---detach}"

  case "${mode}" in
    --detach)
      start_session
      ;;
    --attach)
      if session_exists; then
        echo "Session '${SESSION_NAME}' already exists. Attaching to it."
      else
        start_session
      fi
      exec tmux attach -t "${SESSION_NAME}"
      ;;
    --stop)
      stop_session
      ;;
    --status)
      print_status
      ;;
    --help|-h)
      usage
      ;;
    *)
      echo "Unknown option: ${mode}" >&2
      usage >&2
      return 1
      ;;
  esac
}

main "$@"
