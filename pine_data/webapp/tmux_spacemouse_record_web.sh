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

TELEOP_SCRIPT="${TELEOP_SCRIPT:-${PINE_DIR}/spacemouse_teleoperation_datafoundry/3DConnexion_UR5_Teleop_Gripper_pine_h5.py}"
CAMERA_SCRIPT="${CAMERA_SCRIPT:-${WEBAPP_DIR}/record_multi_camera_npy_web.py}"

ROBOT_BACKEND="${ROBOT_BACKEND:-xarm}"
XARM_CONTROLLER_PATH="${XARM_CONTROLLER_PATH:-${PINE_DIR}/../test.py}"
XARM_TELEOP_SPEED="${XARM_TELEOP_SPEED:-150}"
XARM_TELEOP_ROTATION_SPEED="${XARM_TELEOP_ROTATION_SPEED:-0.60}"
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

TELEOP_EXTRA_ARGS="${TELEOP_EXTRA_ARGS:-}"
CAMERA_EXTRA_ARGS="${CAMERA_EXTRA_ARGS:-}"
KILL_STALE_CAMERA_PROCESS="${KILL_STALE_CAMERA_PROCESS:-1}"

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
require_file "$TELEOP_SCRIPT"
require_file "$CAMERA_SCRIPT"

printf -v robot_env_prefix \
  'export ROBOT_BACKEND=%q XARM_CONTROLLER_PATH=%q XARM_TELEOP_SPEED=%q XARM_TELEOP_ROTATION_SPEED=%q; ' \
  "$ROBOT_BACKEND" \
  "$XARM_CONTROLLER_PATH" \
  "$XARM_TELEOP_SPEED" \
  "$XARM_TELEOP_ROTATION_SPEED"

teleop_cmd="${robot_env_prefix}source \"${TELEOP_ENV}/bin/activate\" && cd \"${PINE_DIR}\" && python \"${TELEOP_SCRIPT}\" --robot-ip \"${UR_ROBOT_IP}\" --root \"${RECORD_ROOT}\""
if [[ -n "$TELEOP_EXTRA_ARGS" ]]; then
  teleop_cmd+=" ${TELEOP_EXTRA_ARGS}"
fi

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
echo "  Pane 0 (left): spacemouse teleop"
echo "  Pane 1 (right): webapp camera + robot state (focused)"

if [[ "$ATTACH_ON_START" == "1" ]]; then
  exec tmux attach -t "$SESSION_NAME"
fi
