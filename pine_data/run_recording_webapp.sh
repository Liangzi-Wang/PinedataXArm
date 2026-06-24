#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PINE_DIR="${PINE_DIR:-${SCRIPT_DIR}}"
export WEBAPP_DIR="${WEBAPP_DIR:-${SCRIPT_DIR}/webapp}"
export ROBOT_BACKEND="${ROBOT_BACKEND:-xarm}"
export XARM_CONTROLLER_PATH="${XARM_CONTROLLER_PATH:-${SCRIPT_DIR}/../test.py}"

exec "${WEBAPP_DIR}/run_recording_webapp.sh" "$@"
