#!/usr/bin/env bash
set -euo pipefail

# Webapp-local launcher for the SpaceMouse + recorder tmux stack.
# It mirrors scripts/tmux_spacemouse_record.sh, but uses the webapp recorder
# copy so browser controls can talk to a stdin command loop and status writer.

SESSION_NAME="${SESSION_NAME:-pine_spacemouse_record}"
ATTACH_ON_START="${ATTACH_ON_START:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PINE_DIR="${PINE_DIR:-${DEFAULT_PINE_DIR}}"
WEBAPP_DIR="${WEBAPP_DIR:-${SCRIPT_DIR}}"
LEGACY_TELEOP_DIR="${LEGACY_TELEOP_DIR:-${PINE_DIR}/gello_software}"

TELEOP_ENV="${TELEOP_ENV:-}"
if [[ -z "$TELEOP_ENV" ]]; then
  if [[ -d "${PINE_DIR}/data_record_env" ]]; then
    TELEOP_ENV="${PINE_DIR}/data_record_env"
  elif [[ -d "${LEGACY_TELEOP_DIR}/.venv" ]]; then
    TELEOP_ENV="${LEGACY_TELEOP_DIR}/.venv"
  elif [[ -d "${PINE_DIR}/.venv" ]]; then
    TELEOP_ENV="${PINE_DIR}/.venv"
  else
    TELEOP_ENV="${PINE_DIR}/data_record_env"
  fi
fi

CAMERA_ENV="${CAMERA_ENV:-${PINE_DIR}/data_record_env}"

QUEUE_TELEOP_SUPERVISOR="${QUEUE_TELEOP_SUPERVISOR:-${PINE_DIR}/queue_teleop_supervisor.py}"
SPACEMOUSE_QUEUE_PUBLISHER_SCRIPT="${SPACEMOUSE_QUEUE_PUBLISHER_SCRIPT:-${PINE_DIR}/../spacemouse_queue_publisher.py}"
XARM_QUEUE_TELEOP_SCRIPT="${XARM_QUEUE_TELEOP_SCRIPT:-${PINE_DIR}/../xarm_queue_teleop.py}"
XARM_SDK_PATH="${XARM_SDK_PATH:-${PINE_DIR}/../xArm-Python-SDK}"
CAMERA_SCRIPT="${CAMERA_SCRIPT:-${WEBAPP_DIR}/record_multi_camera_npy_web.py}"

ROBOT_BACKEND="${ROBOT_BACKEND:-xarm}"
XARM_CONTROLLER_PATH="${XARM_CONTROLLER_PATH:-${PINE_DIR}/../test.py}"
XARM_TELEOP_SPEED="${XARM_TELEOP_SPEED:-300}"
XARM_TELEOP_ANGULAR_SPEED="${XARM_TELEOP_ANGULAR_SPEED:-45}"
XARM_MOVE_ACCELERATION="${XARM_MOVE_ACCELERATION:-2000}"
XARM_COMMAND_PERIOD_S="${XARM_COMMAND_PERIOD_S:-0.01}"
XARM_TELEOP_CONTROL_MODE="${XARM_TELEOP_CONTROL_MODE:-servo}"
XARM_COMMAND_TRANSLATION_MAP="${XARM_COMMAND_TRANSLATION_MAP:--y,x,z}"
XARM_COMMAND_ROTATION_MAP="${XARM_COMMAND_ROTATION_MAP:--x,y,z}"
XARM_USE_BASE_RELATIVE_AA="${XARM_USE_BASE_RELATIVE_AA:-1}"
XARM_USE_TOOL_TWIST_AA="${XARM_USE_TOOL_TWIST_AA:-0}"
XARM_TOOL_TWIST_AXIS="${XARM_TOOL_TWIST_AXIS:--z}"
SPACEMOUSE_QUEUE_HOST="${SPACEMOUSE_QUEUE_HOST:-127.0.0.1}"
SPACEMOUSE_QUEUE_PORT="${SPACEMOUSE_QUEUE_PORT:-8765}"
SPACEMOUSE_QUEUE_AUTHKEY="${SPACEMOUSE_QUEUE_AUTHKEY:-spacemouse}"
SPACEMOUSE_QUEUE_PUBLISH_HZ="${SPACEMOUSE_QUEUE_PUBLISH_HZ:-200}"
SPACEMOUSE_RESPONSE_EXPONENT="${SPACEMOUSE_RESPONSE_EXPONENT:-1.5}"
XARM_QUEUE_POLL_HZ="${XARM_QUEUE_POLL_HZ:-250}"
if [[ "$ROBOT_BACKEND" == "xarm" ]]; then
  XARM_ROBOT_IP="${XARM_ROBOT_IP:-${ROBOT_IP:-192.168.1.10}}"
  ROBOT_IP="${ROBOT_IP:-${XARM_ROBOT_IP}}"
  UR_ROBOT_IP="${UR_ROBOT_IP:-${ROBOT_IP}}"
else
  UR_ROBOT_IP="${UR_ROBOT_IP:-${ROBOT_IP:-192.168.1.10}}"
  ROBOT_IP="${ROBOT_IP:-${UR_ROBOT_IP}}"
fi
RECORD_ROOT="${RECORD_ROOT:-${PINE_DIR}/recordings}"
FPS="${FPS:-15}"

HAND_SERIAL="${HAND_SERIAL:-}"
WRIST_SERIAL="${WRIST_SERIAL:-}"
EXTERNAL_SERIAL="${EXTERNAL_SERIAL:-}"
ALLOW_MISSING_HAND="${ALLOW_MISSING_HAND:-0}"
ALLOW_MISSING_WRIST="${ALLOW_MISSING_WRIST:-0}"
ALLOW_MISSING_EXTERNAL="${ALLOW_MISSING_EXTERNAL:-0}"

CAMERA_EXTRA_ARGS="${CAMERA_EXTRA_ARGS:-}"
KILL_STALE_CAMERA_PROCESS="${KILL_STALE_CAMERA_PROCESS:-1}"
TELEOP_STATUS_FILE="${TELEOP_STATUS_FILE:-${WEBAPP_DIR}/.runtime/${SESSION_NAME}_teleop_status.json}"

ENABLE_ROBOT_RECORDING="${ENABLE_ROBOT_RECORDING:-1}"
ROBOT_FPS="${ROBOT_FPS:-200}"
ENABLE_GRIPPER_STATE="${ENABLE_GRIPPER_STATE:-1}"
GRIPPER_PORT="${GRIPPER_PORT:-63352}"
ALLOW_MISSING_ROBOT="${ALLOW_MISSING_ROBOT:-0}"
ALLOW_MISSING_GRIPPER="${ALLOW_MISSING_GRIPPER:-1}"

export ROBOT_BACKEND
export XARM_CONTROLLER_PATH
export XARM_ROBOT_IP="${XARM_ROBOT_IP:-$ROBOT_IP}"
export ROBOT_IP
export UR_ROBOT_IP

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: command not found: $1"
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: directory not found: $1"
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: file not found: $1"
    exit 1
  fi
}

require_cmd tmux
require_dir "$PINE_DIR"
require_dir "$WEBAPP_DIR"
require_dir "$TELEOP_ENV"
require_dir "$CAMERA_ENV"
require_dir "$XARM_SDK_PATH"
require_file "$QUEUE_TELEOP_SUPERVISOR"
require_file "$SPACEMOUSE_QUEUE_PUBLISHER_SCRIPT"
require_file "$XARM_QUEUE_TELEOP_SCRIPT"
require_file "$CAMERA_SCRIPT"

printf -v robot_env_prefix \
  'export ROBOT_BACKEND=%q ROBOT_IP=%q XARM_IP=%q XARM_CONTROLLER_PATH=%q XARM_TELEOP_SPEED=%q XARM_TELEOP_ANGULAR_SPEED=%q XARM_MOVE_ACCELERATION=%q XARM_COMMAND_PERIOD_S=%q XARM_TELEOP_CONTROL_MODE=%q XARM_COMMAND_TRANSLATION_MAP=%q XARM_COMMAND_ROTATION_MAP=%q XARM_USE_BASE_RELATIVE_AA=%q XARM_USE_TOOL_TWIST_AA=%q XARM_TOOL_TWIST_AXIS=%q SPACEMOUSE_RESPONSE_EXPONENT=%q; ' \
  "$ROBOT_BACKEND" \
  "$ROBOT_IP" \
  "$ROBOT_IP" \
  "$XARM_CONTROLLER_PATH" \
  "$XARM_TELEOP_SPEED" \
  "$XARM_TELEOP_ANGULAR_SPEED" \
  "$XARM_MOVE_ACCELERATION" \
  "$XARM_COMMAND_PERIOD_S" \
  "$XARM_TELEOP_CONTROL_MODE" \
  "$XARM_COMMAND_TRANSLATION_MAP" \
  "$XARM_COMMAND_ROTATION_MAP" \
  "$XARM_USE_BASE_RELATIVE_AA" \
  "$XARM_USE_TOOL_TWIST_AA" \
  "$XARM_TOOL_TWIST_AXIS" \
  "$SPACEMOUSE_RESPONSE_EXPONENT"

teleop_cmd="${robot_env_prefix}source \"${TELEOP_ENV}/bin/activate\" && cd \"${PINE_DIR}\" && python \"${QUEUE_TELEOP_SUPERVISOR}\""
teleop_cmd+=" --publisher-script \"${SPACEMOUSE_QUEUE_PUBLISHER_SCRIPT}\""
teleop_cmd+=" --teleop-script \"${XARM_QUEUE_TELEOP_SCRIPT}\""
teleop_cmd+=" --status-file \"${TELEOP_STATUS_FILE}\""
teleop_cmd+=" --sdk-path \"${XARM_SDK_PATH}\""
teleop_cmd+=" --host \"${SPACEMOUSE_QUEUE_HOST}\" --port \"${SPACEMOUSE_QUEUE_PORT}\""
teleop_cmd+=" --authkey \"${SPACEMOUSE_QUEUE_AUTHKEY}\""
teleop_cmd+=" --publish-hz \"${SPACEMOUSE_QUEUE_PUBLISH_HZ}\" --poll-hz \"${XARM_QUEUE_POLL_HZ}\""

camera_cmd="${robot_env_prefix}source \"${CAMERA_ENV}/bin/activate\" && cd \"${WEBAPP_DIR}\" && python \"${CAMERA_SCRIPT}\" --root \"${RECORD_ROOT}\" --fps \"${FPS}\""
if [[ -n "$HAND_SERIAL" ]]; then
  camera_cmd+=" --hand-serial \"${HAND_SERIAL}\""
fi
if [[ -n "$WRIST_SERIAL" ]]; then
  camera_cmd+=" --wrist-serial \"${WRIST_SERIAL}\""
fi
if [[ -n "$EXTERNAL_SERIAL" ]]; then
  camera_cmd+=" --external-serial \"${EXTERNAL_SERIAL}\""
fi
if [[ "$ALLOW_MISSING_HAND" == "1" ]]; then
  camera_cmd+=" --allow-missing-hand"
fi
if [[ "$ALLOW_MISSING_WRIST" == "1" ]]; then
  camera_cmd+=" --allow-missing-wrist"
fi
if [[ "$ALLOW_MISSING_EXTERNAL" == "1" ]]; then
  camera_cmd+=" --allow-missing-external"
fi
if [[ -n "$CAMERA_EXTRA_ARGS" ]]; then
  camera_cmd+=" ${CAMERA_EXTRA_ARGS}"
fi

if [[ "$ENABLE_ROBOT_RECORDING" == "1" ]]; then
  if [[ -z "$ROBOT_IP" ]]; then
    echo "ERROR: ENABLE_ROBOT_RECORDING=1 but ROBOT_IP is empty."
    echo "Set ROBOT_IP=<ur_ip> (or UR_ROBOT_IP) before running."
    exit 1
  fi
  camera_cmd+=" --robot-ip \"${ROBOT_IP}\" --robot-fps \"${ROBOT_FPS}\""
  camera_cmd+=" --gripper-port \"${GRIPPER_PORT}\""
  if [[ "$ENABLE_GRIPPER_STATE" != "1" ]]; then
    camera_cmd+=" --no-enable-gripper-state"
  fi
  if [[ "$ALLOW_MISSING_ROBOT" == "1" ]]; then
    camera_cmd+=" --allow-missing-robot"
  fi
  if [[ "$ALLOW_MISSING_GRIPPER" != "1" ]]; then
    camera_cmd+=" --no-allow-missing-gripper"
  fi
fi

if [[ "$KILL_STALE_CAMERA_PROCESS" == "1" ]]; then
  pkill -f "${PINE_DIR}/data_recording/record_multi_camera_npy.py" 2>/dev/null || true
  pkill -f "${WEBAPP_DIR}/cam_test.py" 2>/dev/null || true
  pkill -f "${WEBAPP_DIR}/record_multi_camera_npy_web.py" 2>/dev/null || true
  sleep 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  if [[ "$ATTACH_ON_START" == "1" ]]; then
    exec tmux attach -t "$SESSION_NAME"
  fi
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -n main -c "$PINE_DIR"
tmux send-keys -t "$SESSION_NAME:main.0" "$teleop_cmd" C-m

tmux split-window -h -t "$SESSION_NAME:main.0" -c "$WEBAPP_DIR"
tmux send-keys -t "$SESSION_NAME:main.1" "$camera_cmd" C-m
tmux select-layout -t "$SESSION_NAME:main" even-horizontal
tmux set-option -t "$SESSION_NAME" mouse on
tmux select-pane -t "$SESSION_NAME:main.1"

echo "Started tmux session: $SESSION_NAME"
echo "  Pane 0 (left): SpaceMouse queue publisher + xArm queue teleop"
echo "  Pane 1 (right): webapp camera + robot state (focused)"

if [[ "$ATTACH_ON_START" == "1" ]]; then
  exec tmux attach -t "$SESSION_NAME"
fi
