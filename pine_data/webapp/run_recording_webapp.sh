#!/usr/bin/env bash
set -euo pipefail

# Run the DataFoundry viewer plus web-controlled recorder from the camera env.
# The recorder is initialized from the browser, so this script only starts FastAPI.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PINE_DIR="${PINE_DIR:-${DEFAULT_PINE_DIR}}"
WEBAPP_DIR="${WEBAPP_DIR:-${SCRIPT_DIR}}"
WEBAPP_ENV="${WEBAPP_ENV:-${PINE_DIR}/data_record_env}"

RECORD_ROOT="${RECORD_ROOT:-${PINE_DIR}/recordings}"
DATA_DIR="${DATA_DIR:-${RECORD_ROOT}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVER_RELOAD="${SERVER_RELOAD:-0}"
KILL_STALE_CAMERA_PROCESS="${KILL_STALE_CAMERA_PROCESS:-1}"
RECORDING_BACKEND="${RECORDING_BACKEND:-tmux}"
ROBOT_BACKEND="${ROBOT_BACKEND:-xarm}"
XARM_CONTROLLER_PATH="${XARM_CONTROLLER_PATH:-${PINE_DIR}/../test.py}"
XARM_TELEOP_SPEED="${XARM_TELEOP_SPEED:-150}"
XARM_TELEOP_ROTATION_SPEED="${XARM_TELEOP_ROTATION_SPEED:-0.60}"
TMUX_RECORDING_PIPELINE="${TMUX_RECORDING_PIPELINE:-spacemouse}"
TMUX_RECORDING_SCRIPT="${TMUX_RECORDING_SCRIPT:-${WEBAPP_DIR}/tmux_spacemouse_record_web.sh}"
TMUX_RECORDING_SESSION="${TMUX_RECORDING_SESSION:-pine_spacemouse_record}"
TMUX_RECORDING_STATUS_FILE="${TMUX_RECORDING_STATUS_FILE:-${WEBAPP_DIR}/.runtime/${TMUX_RECORDING_SESSION}_status.json}"
HAND_SERIAL="${HAND_SERIAL:-218622270687}"
WRIST_SERIAL="${WRIST_SERIAL:-409122273232}"
EXTERNAL_SERIAL="${EXTERNAL_SERIAL:-409122274280}"
ALLOW_MISSING_HAND="${ALLOW_MISSING_HAND:-0}"
ALLOW_MISSING_WRIST="${ALLOW_MISSING_WRIST:-0}"
ALLOW_MISSING_EXTERNAL="${ALLOW_MISSING_EXTERNAL:-0}"
if [[ "$ROBOT_BACKEND" == "xarm" ]]; then
  XARM_ROBOT_IP="${XARM_ROBOT_IP:-${ROBOT_IP:-192.168.1.10}}"
  ROBOT_IP="${ROBOT_IP:-${XARM_ROBOT_IP}}"
  UR_ROBOT_IP="${UR_ROBOT_IP:-${ROBOT_IP}}"
else
  UR_ROBOT_IP="${UR_ROBOT_IP:-${ROBOT_IP:-192.168.1.10}}"
  ROBOT_IP="${ROBOT_IP:-${UR_ROBOT_IP}}"
fi
ROBOT_FPS="${ROBOT_FPS:-200}"
ENABLE_GRIPPER_STATE="${ENABLE_GRIPPER_STATE:-1}"
GRIPPER_PORT="${GRIPPER_PORT:-63352}"
ALLOW_MISSING_ROBOT="${ALLOW_MISSING_ROBOT:-0}"
ALLOW_MISSING_GRIPPER="${ALLOW_MISSING_GRIPPER:-1}"

if [[ ! -d "$WEBAPP_ENV" ]]; then
  echo "ERROR: environment not found: $WEBAPP_ENV"
  exit 1
fi

if [[ ! -d "$WEBAPP_DIR" ]]; then
  echo "ERROR: webapp directory not found: $WEBAPP_DIR"
  exit 1
fi

if [[ "$RECORDING_BACKEND" == "tmux" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is required when RECORDING_BACKEND=tmux"
    exit 1
  fi
  if [[ ! -f "$TMUX_RECORDING_SCRIPT" ]]; then
    echo "ERROR: tmux recording script not found: $TMUX_RECORDING_SCRIPT"
    exit 1
  fi
fi

if [[ "$KILL_STALE_CAMERA_PROCESS" == "1" ]]; then
  pkill -f "${PINE_DIR}/data_recording/record_multi_camera_npy.py" 2>/dev/null || true
  pkill -f "${WEBAPP_DIR}/cam_test.py" 2>/dev/null || true
  pkill -f "${WEBAPP_DIR}/record_multi_camera_npy_web.py" 2>/dev/null || true
  sleep 1
fi

if ! "${WEBAPP_ENV}/bin/python" -c "import fastapi, numpy, PIL" >/dev/null 2>&1; then
  echo "ERROR: WEBAPP_ENV is missing web dependencies: $WEBAPP_ENV"
  echo "Install webapp/requirements.txt into this environment, or set WEBAPP_ENV to an env that has FastAPI, NumPy, Pillow, and the recorder hardware deps."
  exit 1
fi

reload_arg="--no-server-reload"
if [[ "$SERVER_RELOAD" == "1" ]]; then
  reload_arg="--server-reload"
fi

export DATAFOUNDRY_RECORD_ROOT="$RECORD_ROOT"
export RECORDING_BACKEND
export TMUX_RECORDING_PIPELINE
export TMUX_RECORDING_SCRIPT
export TMUX_RECORDING_SESSION
export TMUX_RECORDING_STATUS_FILE
export HAND_SERIAL
export WRIST_SERIAL
export EXTERNAL_SERIAL
export ALLOW_MISSING_HAND
export ALLOW_MISSING_WRIST
export ALLOW_MISSING_EXTERNAL
export ROBOT_BACKEND
export XARM_CONTROLLER_PATH
export XARM_TELEOP_SPEED
export XARM_TELEOP_ROTATION_SPEED
export XARM_ROBOT_IP="${XARM_ROBOT_IP:-$ROBOT_IP}"
export UR_ROBOT_IP
export ROBOT_IP
export ROBOT_FPS
export ENABLE_GRIPPER_STATE
export GRIPPER_PORT
export ALLOW_MISSING_ROBOT
export ALLOW_MISSING_GRIPPER

source "${WEBAPP_ENV}/bin/activate"
cd "$WEBAPP_DIR"
python main.py --data-dir "$DATA_DIR" --host "$HOST" --port "$PORT" "$reload_arg"
