#!/usr/bin/env bash
set -euo pipefail

# Launch two tmux windows:
#   1) SpaceMouse UR teleop
#   2) Multi-camera recording
#
# Default env mapping:
#   - Teleop: /home/pine/pine_data/data_record_env (preferred)
#   - Camera: /home/pine/pine_data/data_record_env

SESSION_NAME="${SESSION_NAME:-pine_spacemouse_record}"

PINE_DIR="${PINE_DIR:-/home/pine/pine_data}"
LEGACY_TELEOP_DIR="${LEGACY_TELEOP_DIR:-/home/pine/gello_software}"

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

# Camera recorder must run in pine_data data_record_env (contains pynput/pyrealsense2 deps).
CAMERA_ENV="${CAMERA_ENV:-${PINE_DIR}/data_record_env}"

TELEOP_SCRIPT="${TELEOP_SCRIPT:-${PINE_DIR}/spacemouse_teleoperation_datafoundry/3DConnexion_UR5_Teleop_Gripper_pine_h5.py}"
CAMERA_SCRIPT="${CAMERA_SCRIPT:-${PINE_DIR}/data_recording/record_multi_camera_npy.py}"

UR_ROBOT_IP="${UR_ROBOT_IP:-192.168.1.10}"
RECORD_ROOT="${RECORD_ROOT:-${PINE_DIR}/recordings}"
FPS="${FPS:-15}"

# Camera serial overrides (optional)
HAND_SERIAL="${HAND_SERIAL:-}"
EXTERNAL_SERIAL="${EXTERNAL_SERIAL:-}"

# Useful when D405 is not connected
ALLOW_MISSING_HAND="${ALLOW_MISSING_HAND:-0}"
ALLOW_MISSING_EXTERNAL="${ALLOW_MISSING_EXTERNAL:-0}"

# Extra raw args if needed
TELEOP_EXTRA_ARGS="${TELEOP_EXTRA_ARGS:-}"
CAMERA_EXTRA_ARGS="${CAMERA_EXTRA_ARGS:-}"
KILL_STALE_CAMERA_PROCESS="${KILL_STALE_CAMERA_PROCESS:-1}"

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
require_dir "$TELEOP_ENV"
require_dir "$CAMERA_ENV"
require_file "$TELEOP_SCRIPT"
require_file "$CAMERA_SCRIPT"

teleop_cmd="source \"${TELEOP_ENV}/bin/activate\" && cd \"${PINE_DIR}\" && python \"${TELEOP_SCRIPT}\" --robot-ip \"${UR_ROBOT_IP}\" --root \"${RECORD_ROOT}\""
if [[ -n "$TELEOP_EXTRA_ARGS" ]]; then
  teleop_cmd+=" ${TELEOP_EXTRA_ARGS}"
fi

camera_cmd="source \"${CAMERA_ENV}/bin/activate\" && cd \"${PINE_DIR}\" && python \"${CAMERA_SCRIPT}\" --root \"${RECORD_ROOT}\" --fps \"${FPS}\""
if [[ -n "$HAND_SERIAL" ]]; then
  camera_cmd+=" --hand-serial \"${HAND_SERIAL}\""
fi
if [[ -n "$EXTERNAL_SERIAL" ]]; then
  camera_cmd+=" --external-serial \"${EXTERNAL_SERIAL}\""
fi
if [[ "$ALLOW_MISSING_HAND" == "1" ]]; then
  camera_cmd+=" --allow-missing-hand"
fi
if [[ "$ALLOW_MISSING_EXTERNAL" == "1" ]]; then
  camera_cmd+=" --allow-missing-external"
fi
if [[ -n "$CAMERA_EXTRA_ARGS" ]]; then
  camera_cmd+=" ${CAMERA_EXTRA_ARGS}"
fi

if [[ "$KILL_STALE_CAMERA_PROCESS" == "1" ]]; then
  # Avoid RealSense errno=16 caused by stale recorder processes.
  pkill -f "${PINE_DIR}/data_recording/record_multi_camera_npy.py" 2>/dev/null || true
  sleep 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exec tmux attach -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" -n main -c "$PINE_DIR"

tmux send-keys -t "$SESSION_NAME:main.0" "$teleop_cmd" C-m

tmux split-window -h -t "$SESSION_NAME:main" -c "$PINE_DIR"
tmux send-keys -t "$SESSION_NAME:main.1" "$camera_cmd" C-m
tmux select-layout -t "$SESSION_NAME:main" even-horizontal
tmux set-option -t "$SESSION_NAME" mouse on
tmux select-pane -t "$SESSION_NAME:main.1"

echo "Started tmux session: $SESSION_NAME"
echo "  Pane 0 (left): teleop"
echo "  Pane 1 (right): camera (focused)"
echo
echo "Tip: if you only have D435i connected, keep ALLOW_MISSING_HAND=1 and set EXTERNAL_SERIAL."

exec tmux attach -t "$SESSION_NAME"
