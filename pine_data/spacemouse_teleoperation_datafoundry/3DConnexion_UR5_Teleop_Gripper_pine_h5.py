import os
import sys
from pathlib import Path

# source /home/mainuser/UR5_Policy/data_record_env/bin/activate
# python /home/mainuser/UR5_Policy/spacemouse_teleoperation/3DConnexion_UR5_Teleop_Gripper_YinYu.py
# python /home/mainuser/UR5_Policy/data_recording/record_data.py

# k4aviewer
# realsense-viewer


ROBOT_BACKEND = os.environ.get("ROBOT_BACKEND", "ur").strip().lower()
if ROBOT_BACKEND == "xarm":
    PINE_DATA_DIR = Path(__file__).resolve().parents[1]
    if str(PINE_DATA_DIR) not in sys.path:
        sys.path.insert(0, str(PINE_DATA_DIR))
    from xarm_bridge import XArmControlInterface as RTDEControlInterface
    from xarm_bridge import XArmReceiveInterface as RTDEReceiveInterface
    from xarm_bridge import XArmSDKGripper
else:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    XArmSDKGripper = None

from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from threading import Thread, Event, Lock
from collections import defaultdict
from pynput import keyboard
import numpy as np
import time

import json
import re
import argparse
import random
import socket
from typing import Any, Optional

try:
    from gello.robots.robotiq_gripper import RobotiqGripper
except ImportError:
    try:
        from robotiq_gripper import RobotiqGripper
    except ImportError as local_import_error:
        RobotiqGripper = None
        _ROBOTIQ_IMPORT_ERROR = local_import_error
    else:
        _ROBOTIQ_IMPORT_ERROR = None
else:
    _ROBOTIQ_IMPORT_ERROR = None

#Instructions: Insert the plug into the fixed left socket and then pull it out.
#Instructions: Insert the plug into the fixed left socket and then pull it out with flexible plug location.
#Instructions: Insert the plug into the fixed left socket and then pull it out with flexible socket location.

#Instructions: Insert the 6-pin usb into the usb box.

# The only difference from 3DConnexion_UR5_Teleop.py is the ability to control the gripper position.

## ROBOT_HOST -> The robot's IP Address
## SpaceMouse response and speed scales -> To tune teleoperation velocity 
## The acceleration in rtde_c.speedL() -> If there is any latency in robot movement (increasing acceleration = increasing deceleration)

class Spacemouse(Thread):
    def __init__(self, max_value=500, deadzone=(0,0,0,0,0,0), dtype=np.float32):
        """
        Continuously listen to 3D connection space naviagtor events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
        
        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()
        
        super().__init__()
        self.stop_event = Event()
        print("[SpaceMouse] initialized")
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.motion_event = SpnavMotionEvent([0,0,0], [0,0,0], 0)
        self.last_motion_event_time = time.monotonic()
        self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

    def get_motion_state(self): #this method gets the movement of the mouse 
        me = self.motion_event
        state = np.array(me.translation + me.rotation, 
            dtype=self.dtype) / self.max_value
        effective_deadzone = np.maximum(self.deadzone, SPACEMOUSE_INPUT_DEADBAND)
        is_dead = (-effective_deadzone < state) & (state < effective_deadzone)
        state[is_dead] = 0
        active = ~is_dead
        if np.any(active):
            magnitude = np.abs(state[active])
            denom = np.maximum(1.0 - effective_deadzone[active], 1e-6)
            normalized = np.clip((magnitude - effective_deadzone[active]) / denom, 0.0, 1.0)
            state[active] = np.sign(state[active]) * np.power(normalized, SPACEMOUSE_RESPONSE_EXPONENT)
        return state
    
    def get_motion_state_transformed(self): #transforms get_motion_state 
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back
        """
        state = self.get_motion_state()

        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        tf_state[:3] *= TELEOP_TRANSLATION_SPEED_SCALE
        tf_state[3:] *= TELEOP_ROTATION_SPEED_SCALE
        if tf_state[2] < 0:
            tf_state[2] *= TELEOP_DOWNWARD_Z_EXTRA_SCALE

        return tf_state

    def motion_event_age_s(self):
        return max(0.0, time.monotonic() - self.last_motion_event_time)

    def clear_motion_state(self):
        self.motion_event = SpnavMotionEvent([0, 0, 0], [0, 0, 0], 0)
        self.last_motion_event_time = time.monotonic()

    def is_button_pressed(self, button_id):
        return self.button_state[button_id]

    def stop(self):
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        spnav_open()
        try:
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    self.motion_event = event
                    self.last_motion_event_time = time.monotonic()
                elif isinstance(event, SpnavButtonEvent):
                    self.button_state[event.bnum] = event.press
                else:
                    time.sleep(1/200)
        finally:
            spnav_close()

# Define robot parameters
ROBOT_HOST = os.environ.get("XARM_ROBOT_IP", os.environ.get("UR_ROBOT_IP", "192.168.1.10"))

# SpaceMouse analog response
SPACEMOUSE_INPUT_DEADBAND = 0.08
# Square-like mapping: more precision over most travel, fast near extremes.
SPACEMOUSE_RESPONSE_EXPONENT = float(os.environ.get("SPACEMOUSE_RESPONSE_EXPONENT", "2.0"))

# Speed configuration
# SpaceMouse outputs normalized values. Convert the configured xArm maximum
# linear speed from mm/s to the m/s expected by the existing RTDE-style loop.
XARM_TELEOP_SPEED_MM_S = float(os.environ.get("XARM_TELEOP_SPEED", "150"))
if not 0 < XARM_TELEOP_SPEED_MM_S <= 1000:
    raise ValueError("XARM_TELEOP_SPEED must be in the range (0, 1000] mm/s.")
TELEOP_TRANSLATION_SPEED_SCALE = XARM_TELEOP_SPEED_MM_S / 1000.0
TELEOP_ROTATION_SPEED_SCALE = float(os.environ.get("XARM_TELEOP_ROTATION_SPEED", "0.60"))
if not 0 < TELEOP_ROTATION_SPEED_SCALE <= 3.14:
    raise ValueError("XARM_TELEOP_ROTATION_SPEED must be in the range (0, 3.14] rad/s.")
TELEOP_DOWNWARD_Z_EXTRA_SCALE = 0.40
TELEOP_COMMAND_HORIZON_S = 0.01
NON_TELEOP_SPEED_SCALE = 1.00   # reset / non-teleop motions = 100%
SPACEMOUSE_ACTIVE_EPS = float(os.environ.get("SPACEMOUSE_ACTIVE_EPS", "1e-4"))
SPACEMOUSE_STALE_STOP_S = float(os.environ.get("SPACEMOUSE_STALE_STOP_S", "0.20"))
SPACEMOUSE_IDLE_SPEEDSTOP_INTERVAL_S = float(os.environ.get("SPACEMOUSE_IDLE_SPEEDSTOP_INTERVAL_S", "0.20"))
ROBOT_RESIDUAL_SPEED_STOP_EPS = float(os.environ.get("ROBOT_RESIDUAL_SPEED_STOP_EPS", "0.002"))
STOP_DRIFT_ZERO_SPEED_ACCEL = float(os.environ.get("STOP_DRIFT_ZERO_SPEED_ACCEL", "30.0"))
STOP_DRIFT_ZERO_SPEED_TIME_S = float(os.environ.get("STOP_DRIFT_ZERO_SPEED_TIME_S", "0.0"))
STOP_DRIFT_SUPPRESS_IDLE_ZERO_S = float(os.environ.get("STOP_DRIFT_SUPPRESS_IDLE_ZERO_S", "2.0"))
ENABLE_IDLE_ZERO_OVERRIDE = os.environ.get("ENABLE_IDLE_ZERO_OVERRIDE", "0").strip().lower() in {"1", "true", "yes", "on"}
TELEOP_STARTUP_COMMAND_GRACE_S = float(os.environ.get("TELEOP_STARTUP_COMMAND_GRACE_S", "2.0"))

# Non-teleop motion presets (UR defaults scaled by NON_TELEOP_SPEED_SCALE)
RESET_LIFT_SPEED = 0.25 * NON_TELEOP_SPEED_SCALE
RESET_LIFT_ACCEL = 0.50 * NON_TELEOP_SPEED_SCALE
RESET_MOVEJ_SPEED = 1.05 * NON_TELEOP_SPEED_SCALE
RESET_MOVEJ_ACCEL = 0.5 * NON_TELEOP_SPEED_SCALE
RESET_SUBTASK_MOVEL_SPEED = 0.25 * NON_TELEOP_SPEED_SCALE
RESET_SUBTASK_MOVEL_ACCEL = 0.50 * NON_TELEOP_SPEED_SCALE

# Key 0 is reserved for the normal full-task reset below.
# For 1+ reset modes, x/y/z get uniform noise and the gripper is set by "gripper".
# If "joint" is provided, reset first moveJ's to it, then moveL's to the TCP with XYZ noise.
SUBTASK_RESET_POSES: dict[int, dict[str, Any]] = {
    1: {
        "label": "Skill 1 - grasp RAM",
        "tcp": [-0.39, -0.4, 0.36, 0.03, -3.14, -0.06],
        "gripper": "closed",
    },
    2: {
        "label": "Skill 2 - insert RAM",
        "tcp": [-0.03, -0.4, 0.38, 2.24, -2.18, -0.06],
        "gripper": "closed",
    },
    3: {
        "label": "Skill 3 - insert CPU",
        "tcp": [-0.03, -0.35, 0.37, -0.04, -3.12, -0.14],
        "gripper": "closed",
    },
    4: {
        "label": "Skill 4 - insert GPU",
        "tcp": [-0.03, -0.44, 0.42, -0.04, -3.10, -0.12],
        "gripper": "closed",
    },
    5: {
        "label": "Skill 5 - insert DISK",
        "tcp": [0.11, -0.57, 0.49, 2.18, -2.22, -0.04],
        "gripper": "closed",
    },
}

ROBOTIQ_PORT = int(os.environ.get("ROBOTIQ_PORT", "63352"))
ROBOTIQ_OPEN_POS = int(os.environ.get("ROBOTIQ_OPEN_POS", "0"))
ROBOTIQ_CLOSE_POS = int(os.environ.get("ROBOTIQ_CLOSE_POS", "255"))
ROBOTIQ_SPEED = int(os.environ.get("ROBOTIQ_SPEED", "255"))
ROBOTIQ_FORCE = int(os.environ.get("ROBOTIQ_FORCE", "10"))

ROBOTIQ_MAX_CMD_VALUE = 255
ROBOTIQ_SPEED_LIMIT_PERCENT = 35
ROBOTIQ_SPEED_LIMIT = max(0, int(round(ROBOTIQ_MAX_CMD_VALUE * ROBOTIQ_SPEED_LIMIT_PERCENT / 100.0)))
ROBOTIQ_FORCE_LIMIT_PERCENT = 0.1
ROBOTIQ_FORCE_LIMIT = max(0, int(round(ROBOTIQ_MAX_CMD_VALUE * ROBOTIQ_FORCE_LIMIT_PERCENT / 100.0)))

GRIPPER_HOLD_SPEED_POS_PER_SEC = 30.0
GRIPPER_COMMAND_MIN_INTERVAL_S = 0.03
GRIPPER_CLOSE_STALL_EPS = 0.15
GRIPPER_CLOSE_STALL_BLOCK_S = 1.00
GRIPPER_CLOSE_STALL_GRACE_S = 0.80

# =========================
# Robotiq Gripper Helper
# =========================

class RobotiqSocketGripper:
    def __init__(
        self,
        hostname: str,
        port: int = ROBOTIQ_PORT,
        open_pos: int = ROBOTIQ_OPEN_POS,
        close_pos: int = ROBOTIQ_CLOSE_POS,
        speed: int = ROBOTIQ_SPEED,
        force: int = ROBOTIQ_FORCE,
        auto_activate: bool = False,
    ):
        if RobotiqGripper is None:
            raise ImportError(
                "Could not import RobotiqGripper implementation. "
                "Install gello package or ensure spacemouse_teleoperation/robotiq_gripper.py is available."
            ) from _ROBOTIQ_IMPORT_ERROR

        self._open_pos = int(open_pos)
        self._close_pos = int(close_pos)
        requested_speed = int(speed)
        requested_force = int(force)
        self._speed = max(0, min(requested_speed, ROBOTIQ_SPEED_LIMIT))
        if self._speed < requested_speed:
            print(
                f"[Robotiq] speed request {requested_speed} capped to {self._speed} "
                f"({ROBOTIQ_SPEED_LIMIT_PERCENT}% max)"
            )
        self._force = max(0, min(requested_force, ROBOTIQ_FORCE_LIMIT))
        if self._force < requested_force:
            print(
                f"[Robotiq] force request {requested_force} capped to {self._force} "
                f"({ROBOTIQ_FORCE_LIMIT_PERCENT}% max)"
            )
        self.state = "unknown"
        self._last_command_position: Optional[int] = None

        self._gripper = RobotiqGripper()
        self._gripper.connect(hostname=hostname, port=int(port))
        self._min_pos = int(self._gripper.get_min_position())
        self._max_pos = int(self._gripper.get_max_position())
        self._open_pos = max(self._min_pos, min(self._open_pos, self._max_pos))
        self._close_pos = max(self._min_pos, min(self._close_pos, self._max_pos))
        print(
            f"[Robotiq] connected at {hostname}:{port} "
            f"(range=[{self._min_pos}, {self._max_pos}], open={self._open_pos}, close={self._close_pos}, "
            f"speed={self._speed}, force={self._force})"
        )

        if auto_activate:
            self._gripper.activate(auto_calibrate=False)
            print("[Robotiq] activate() finished")

        self.open()

    def get_min_position(self) -> int:
        return self._min_pos

    def get_max_position(self) -> int:
        return self._max_pos

    def get_current_position(self) -> int:
        return int(self._gripper.get_current_position())

    def is_contact_detected(self) -> bool:
        try:
            return bool(self._gripper.is_gripping())
        except Exception:
            return False

    def command_position(self, position: int) -> int:
        clipped = max(self._min_pos, min(int(position), self._max_pos))
        if self._last_command_position == clipped:
            return clipped
        self._gripper.move(clipped, self._speed, self._force)
        self._last_command_position = clipped
        if clipped <= self._min_pos:
            self.state = "open"
        elif clipped >= self._max_pos:
            self.state = "closed"
        else:
            self.state = f"partial:{clipped}"
        return clipped

    def stop_motion(self) -> None:
        """Stop current go-to motion without issuing a new target position."""
        try:
            # Robotiq socket protocol: GTO=0 stops go-to execution.
            self._gripper._set_var(self._gripper.GTO, 0)
            self.state = "stopped"
            self._last_command_position = None
        except Exception as e:
            print("[Robotiq] stop warning:", e)

    def open(self):
        commanded = self.command_position(self._open_pos)
        print(f"[Robotiq] state -> open ({commanded})")

    def close(self):
        commanded = self.command_position(self._close_pos)
        print(f"[Robotiq] state -> closed ({commanded})")

    def disconnect(self):
        try:
            self._gripper.disconnect()
        except Exception as e:
            print("[Robotiq] disconnect warning:", e)

# =========================
# Force Mode Functions
# =========================

def zero_ft_sensor(rtde_c):
    # Zero the force/torque sensor readings to avoid initial deviation
    print("Zeroing FT Sensor...")
    rtde_c.zeroFtSensor()

def enter_force_mode(rtde_c, rtde_r):
    # Enter constant-force control along Z (downward), with Z the only compliant axis.
    print("Entering force mode...")
    task_frame = rtde_r.getActualTCPPose()  # Current TCP pose as reference frame
    zero_ft_sensor(rtde_c)
    selection = [0, 0, 1, 0, 0, 0]          # Only Z compliant
    Fz = 10.0                                # Target downward force (N)
    wrench = [0.0, 0.0, Fz, 0.0, 0.0, 0.0]
    type = 2                                 # 2 = force control mode
    limits = [float('inf'), float('inf'), 0.02,   # Z speed ≤ 2 cm/s (compliant axis)
              float('inf'), float('inf'), float('inf')]
    rtde_c.forceMode(task_frame, selection, wrench, type, limits)


def exit_force_mode(rtde_c):
    # Safer exit: stop motion first, short settle, then stop force-mode (avoids thread/state hiccups)
    print("Exiting force mode...")
    try:
        rtde_c.speedStop()
        time.sleep(0.02)
    except Exception as e:
        print("[warn] speedStop during exit:", e)
    try:
        rtde_c.forceModeStop()
    except Exception as e:
        print("[warn] forceModeStop:", e)


def force_stop_robot_control(rtde_c, force_mode_active: bool = False, label: str = "shutdown", stop_script: bool = False):
    """Best-effort hard stop for any active RTDE motion before exiting teleop."""
    print(f"\n[STOP] Force stopping robot control ({label})...")
    if rtde_c is None:
        return
    try:
        rtde_c.speedStop()
        time.sleep(0.03)
    except Exception as e:
        print(f"[warn] speedStop during {label}:", e)
    try:
        rtde_c.stopL(1.0)
        time.sleep(0.03)
    except Exception as e:
        print(f"[warn] stopL during {label}:", e)
    if force_mode_active:
        try:
            rtde_c.forceModeStop()
        except Exception as e:
            print(f"[warn] forceModeStop during {label}:", e)
    if stop_script:
        try:
            rtde_c.stopScript()
        except Exception as e:
            print(f"[warn] stopScript during {label}:", e)


def send_zero_speed_override(rtde_c, label: str = "stop-drift"):
    """Clear a stale speedL command without blocking for a full speedStop deceleration."""
    rtde_c.speedL(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        acceleration=STOP_DRIFT_ZERO_SPEED_ACCEL,
        time=STOP_DRIFT_ZERO_SPEED_TIME_S,
    )


def ensure_tcp_reachable(host: str, port: int, timeout_s: float, label: str):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return
    except OSError as e:
        raise RuntimeError(
            f"{label} unreachable at {host}:{port} within {timeout_s:.1f}s. "
            "Check robot IP/network and that the service is running."
        ) from e
        
        
def main():
    parser = argparse.ArgumentParser(description="UR5 teleoperation + trajectory recording.")
    parser.add_argument(
        "--root",
        default="./recordings",
        help="Root directory for saved data (default: ./recordings).",
    )
    parser.add_argument(
        "--robot-ip",
        default=ROBOT_HOST,
        help="Robot controller IP address.",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Disable Robotiq gripper control.",
    )
    parser.add_argument(
        "--gripper-port",
        type=int,
        default=ROBOTIQ_PORT,
        help="Robotiq socket port (default: 63352).",
    )
    parser.add_argument(
        "--gripper-open-pos",
        type=int,
        default=ROBOTIQ_OPEN_POS,
        help="Robotiq open position (0-255).",
    )
    parser.add_argument(
        "--gripper-close-pos",
        type=int,
        default=ROBOTIQ_CLOSE_POS,
        help="Robotiq close position (0-255).",
    )
    parser.add_argument(
        "--gripper-speed",
        type=int,
        default=ROBOTIQ_SPEED,
        help=f"Robotiq move speed (0-255), capped to {ROBOTIQ_SPEED_LIMIT_PERCENT}% ({ROBOTIQ_SPEED_LIMIT}).",
    )
    parser.add_argument(
        "--gripper-force",
        type=int,
        default=ROBOTIQ_FORCE,
        help=f"Robotiq move force (0-255), capped to {ROBOTIQ_FORCE_LIMIT_PERCENT}% ({ROBOTIQ_FORCE_LIMIT}).",
    )
    parser.add_argument(
        "--gripper-activate",
        action="store_true",
        help="Call gripper activate() on startup.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=float(os.environ.get("CONNECT_TIMEOUT", "5.0")),
        help="TCP reachability check timeout in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--instruction",
        default="",
        help="Optional task instruction. If omitted, prompt interactively.",
    )
    parser.add_argument(
        "--status-file",
        default="",
        help="Optional JSON status file path for web monitoring.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("keyboard", "stdin", "both"),
        default="keyboard",
        help="Control source for c/s/d/q/r commands (default: keyboard).",
    )
    parser.add_argument(
        "--subtask-segment-index",
        type=int,
        default=0,
        help="Subtask segment reset index. 0 uses the normal full-task reset; 1+ uses SUBTASK_RESET_POSES.",
    )
    parser.add_argument(
        "--subtask-reset-noise-xyz",
        type=float,
        default=0.01,
        help="Uniform XYZ reset noise magnitude in meters for subtask segment resets.",
    )
    args = parser.parse_args()

    # Reset position configuration
    RESET_TCP_POSE = [0.2506, -0.2463, 0.3242, 1.1388, -2.9149, -0.0240]
    RESET_JOINT_POSE = [0.9519, -1.7670, 1.9762, -1.7274, -1.5715, -0.5454]
    
    raw_instruction = args.instruction.strip()
    if not raw_instruction:
        raw_instruction = input("Enter the task instruction: ").strip()
    # Sanitize instruction for safe folder name
    safe_instruction = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_instruction).strip("._-")
    instruction = safe_instruction if safe_instruction else "untitled"
    date_str = time.strftime("%Y%m%d")
    base_data_folder = Path(args.root).expanduser().resolve()
    data_folder = str(base_data_folder / date_str / instruction / "camera_npy")
    status_file = Path(args.status_file).expanduser().resolve() if args.status_file else None
    status_lock = Lock()
    status_message = "Preparing SpaceMouse teleop..."
    status_last_error = ""
    status_last_saved_episode = ""
    latest_motion_state = np.zeros(6, dtype=np.float32)
    latest_tcp_pose = []
    latest_commanded_tcp_pose = []
    latest_joint_pose = []
    latest_force = []
    latest_gripper_position = None
    latest_button_0 = False
    latest_button_1 = False
    latest_timestamp = None
    teleop_ready = False
    shutdown_requested = False
    rtde_stop_requested = False
    listener = None
    gripper: Optional[RobotiqSocketGripper] = None
    gripper_min_position = ROBOTIQ_OPEN_POS
    gripper_max_position = ROBOTIQ_CLOSE_POS
    gripper_command_position: Optional[float] = None
    gripper_last_command_time = 0.0
    recording = False
    current_episode_time = None
    current_episode_folder = None
    active_subtask_segment_index = max(0, int(args.subtask_segment_index))
    active_subtask_reset_noise_xyz_m = max(0.0, float(args.subtask_reset_noise_xyz))
    reset_request_subtask_segment_index = active_subtask_segment_index
    reset_request_noise_xyz_m = active_subtask_reset_noise_xyz_m

    def update_instruction(raw_value: str) -> str:
        nonlocal instruction, data_folder
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value.strip()).strip("._-")
        instruction = safe if safe else "untitled"
        data_folder = str(base_data_folder / date_str / instruction / "camera_npy")
        os.makedirs(data_folder, exist_ok=True)
        return instruction

    def write_status(message: Optional[str] = None, last_error: Optional[str] = None) -> None:
        nonlocal status_message, status_last_error
        if status_file is None:
            return
        if message is not None:
            status_message = message
        if last_error is not None:
            status_last_error = last_error

        with status_lock:
            motion = latest_motion_state.tolist() if isinstance(latest_motion_state, np.ndarray) else list(latest_motion_state)
            payload = {
                "initialized": teleop_ready,
                "recording": recording,
                "message": status_message,
                "instruction": instruction,
                "data_folder": data_folder,
                "current_episode": current_episode_time,
                "last_saved_episode": status_last_saved_episode,
                "last_error": status_last_error,
                "spacemouse": {
                    "requested": True,
                    "connected": teleop_ready,
                    "available": teleop_ready and latest_timestamp is not None,
                    "status": "streaming" if (teleop_ready and latest_timestamp is not None) else "connected" if teleop_ready else "disconnected",
                    "latest_timestamp": latest_timestamp,
                    "motion_state": [round(float(value), 4) for value in motion],
                    "buttons": {
                        "left": bool(latest_button_0),
                        "right": bool(latest_button_1),
                    },
                },
                "robot": {
                    "requested": True,
                    "enabled": True,
                    "connected": teleop_ready,
                    "available": teleop_ready and latest_timestamp is not None,
                    "status": "streaming" if (teleop_ready and latest_timestamp is not None) else "connected" if teleop_ready else "disconnected",
                    "ip": args.robot_ip,
                    "latest": {
                        "timestamp": latest_timestamp,
                        "eef_pose": latest_tcp_pose,
                        "eef_pose_actual": latest_tcp_pose,
                        "eef_pose_commanded": latest_commanded_tcp_pose,
                        "eef_twist_command": [round(float(value), 4) for value in motion],
                        "joint_state": latest_joint_pose,
                        "tcp_wrench": latest_force,
                    },
                },
                "gripper": {
                    "requested": not args.no_gripper,
                    "enabled": gripper is not None,
                    "connected": gripper is not None,
                    "available": gripper is not None and latest_gripper_position is not None,
                    "status": "streaming" if (gripper is not None and latest_gripper_position is not None) else "connected" if gripper is not None else "disconnected",
                    "ip": args.robot_ip,
                    "port": args.gripper_port,
                    "latest_position": latest_gripper_position,
                },
                "subtask_reset": {
                    "active_segment_index": active_subtask_segment_index,
                    "pending_segment_index": reset_request_subtask_segment_index,
                    "noise_xyz_m": active_subtask_reset_noise_xyz_m,
                    "configured_segments": sorted(int(key) for key in SUBTASK_RESET_POSES.keys()),
                    "labels": {
                        str(int(key)): str(value.get("label", f"Subtask segment {key}"))
                        for key, value in SUBTASK_RESET_POSES.items()
                    },
                },
            }
        try:
            status_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = status_file.with_suffix(status_file.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            tmp_path.replace(status_file)
        except Exception:
            pass

    def parse_reset_command_args(raw_args: str) -> tuple[int, float]:
        parts = raw_args.split()
        segment_index = active_subtask_segment_index
        noise_xyz_m = active_subtask_reset_noise_xyz_m
        if len(parts) >= 1:
            try:
                segment_index = max(0, int(parts[0]))
            except ValueError:
                print(f"[RESET] Invalid subtask segment index: {parts[0]}")
        if len(parts) >= 2:
            try:
                noise_xyz_m = max(0.0, float(parts[1]))
            except ValueError:
                print(f"[RESET] Invalid reset XYZ noise: {parts[1]}")
        return segment_index, noise_xyz_m

    def request_reset(segment_index: Optional[int] = None, noise_xyz_m: Optional[float] = None) -> None:
        nonlocal reset_requested, active_subtask_segment_index, active_subtask_reset_noise_xyz_m
        nonlocal reset_request_subtask_segment_index, reset_request_noise_xyz_m
        if segment_index is not None:
            active_subtask_segment_index = max(0, int(segment_index))
        if noise_xyz_m is not None:
            active_subtask_reset_noise_xyz_m = max(0.0, float(noise_xyz_m))
        reset_request_subtask_segment_index = active_subtask_segment_index
        reset_request_noise_xyz_m = active_subtask_reset_noise_xyz_m
        reset_requested = True
        if reset_request_subtask_segment_index > 0:
            msg = (
                f"Reset to subtask segment {reset_request_subtask_segment_index} requested "
                f"(XYZ noise +/-{reset_request_noise_xyz_m:.3f}m)."
            )
        else:
            msg = "Reset to home position requested."
        print(f"\n[RESET] {msg}")
        write_status(message=msg, last_error=status_last_error)

    def resolve_reset_target(
        segment_index: int,
        noise_xyz_m: float,
    ) -> tuple[str, list[float], list[float] | None, list[float], str | None]:
        if int(segment_index) <= 0:
            return "home", list(RESET_TCP_POSE), list(RESET_JOINT_POSE), [0.0, 0.0, 0.0], None

        entry = SUBTASK_RESET_POSES.get(int(segment_index))
        if entry is None:
            raise RuntimeError(
                f"Subtask segment {segment_index} reset pose is not configured. "
                "Edit SUBTASK_RESET_POSES near the top of this teleop script and add its TCP pose."
            )
        target_tcp = np.asarray(entry.get("tcp", []), dtype=np.float64)
        if target_tcp.size != 6:
            raise RuntimeError(f"Subtask segment {segment_index} must define a 6D tcp pose.")

        noise = np.asarray([random.uniform(-noise_xyz_m, noise_xyz_m) for _ in range(3)], dtype=np.float64)
        target_tcp = target_tcp.copy()
        target_tcp[:3] += noise

        joint_value = entry.get("joint")
        target_joint = None
        if joint_value is not None:
            joint_array = np.asarray(joint_value, dtype=np.float64)
            if joint_array.size != 6:
                raise RuntimeError(f"Subtask segment {segment_index} joint pose must have 6 values when provided.")
            target_joint = joint_array.tolist()
        label = str(entry.get("label") or f"subtask-{segment_index}")
        gripper_mode = entry.get("gripper")
        return label, target_tcp.tolist(), target_joint, noise.tolist(), str(gripper_mode) if gripper_mode else None

    update_instruction(instruction)
    write_status(message=f"Preparing SpaceMouse teleop. Save path: {data_folder}", last_error="")

    if ROBOT_BACKEND == "xarm":
        print(f"[Init] using xArm SDK backend: {args.robot_ip}")
    else:
        print(f"[Init] checking UR RTDE reachability: {args.robot_ip}:30004")
        try:
            ensure_tcp_reachable(args.robot_ip, 30004, args.connect_timeout, "UR RTDE")
        except RuntimeError as e:
            print(f"[Init] {e}")
            print("[Init] Abort startup. Ensure robot is online and network is reachable.")
            write_status(message="UR RTDE is unreachable.", last_error=str(e))
            return

    sm = Spacemouse()
    sm.start()
    # Initialize RTDE interfaces
    print(f"[Init] connecting {ROBOT_BACKEND} robot interfaces: {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)
    print(f"[Init] {ROBOT_BACKEND} robot interfaces connected")
    teleop_command_enable_time = time.time() + TELEOP_STARTUP_COMMAND_GRACE_S

    if not args.no_gripper:
        try:
            if ROBOT_BACKEND == "xarm":
                print("[Init] connecting xArm SDK gripper")
                gripper = XArmSDKGripper(
                    hostname=args.robot_ip,
                    open_pos=int(os.environ.get("XARM_GRIPPER_OPEN_POS", "850")),
                    close_pos=int(os.environ.get("XARM_GRIPPER_CLOSE_POS", "0")),
                    speed=int(os.environ.get("XARM_GRIPPER_SPEED", "5000")),
                )
            else:
                print(f"[Init] checking Robotiq reachability: {args.robot_ip}:{args.gripper_port}")
                ensure_tcp_reachable(args.robot_ip, args.gripper_port, args.connect_timeout, "Robotiq")
                print("[Init] connecting Robotiq gripper")
                gripper = RobotiqSocketGripper(
                    hostname=args.robot_ip,
                    port=args.gripper_port,
                    open_pos=args.gripper_open_pos,
                    close_pos=args.gripper_close_pos,
                    speed=args.gripper_speed,
                    force=args.gripper_force,
                    auto_activate=args.gripper_activate,
                )
            gripper_min_position = gripper.get_min_position()
            gripper_max_position = gripper.get_max_position()
            latest_gripper_position = float(gripper.get_current_position())
            gripper_command_position = latest_gripper_position
            print(f"[{ROBOT_BACKEND} gripper] initial state:", gripper.state)
        except Exception as e:
            print(f"[{ROBOT_BACKEND} gripper] failed to initialize gripper:", e)
            print(f"[{ROBOT_BACKEND} gripper] Continue without gripper control.")
            status_last_error = str(e)
    else:
        print(f"[{ROBOT_BACKEND} gripper] disabled by --no-gripper")

    # Force-mode state
    force_mode_active = False
    
    # Reset flag
    reset_requested = False

    # --- Data recording buffers ---
    tcp_pose_array = []
    joint_pose_array = []
    force_array = []
    timestamp_array = []
        
    # Initialize recording state
    teleop_ready = True
    write_status(message=f"SpaceMouse teleop initialized. Save path: {data_folder}", last_error=status_last_error)

    def handle_start_recording(next_instruction: Optional[str] = None):
        nonlocal recording, current_episode_time, current_episode_folder
        nonlocal tcp_pose_array, joint_pose_array, force_array, timestamp_array
        if next_instruction is not None and next_instruction.strip():
            update_instruction(next_instruction)
        if not recording:
            tcp_pose_array = []
            joint_pose_array = []
            force_array = []
            timestamp_array = []
            current_episode_time = time.strftime("%Y%m%d%H%M%S")
            current_episode_folder = data_folder
            recording = True
            print(f"\n{'='*60}")
            print(f"[RECORDING] Started episode {current_episode_time}")
            print(f"[RECORDING] Saving to: {current_episode_folder}")
            print(f"{'='*60}\n")
            write_status(message=f"Recording trajectory episode {current_episode_time}...", last_error="")
        else:
            print("[RECORDING] Already recording! Press 's' to stop first.")
            write_status(message="SpaceMouse episode is already recording.", last_error=status_last_error)

    def handle_stop_recording():
        nonlocal recording, status_last_saved_episode
        if recording:
            print(f"\n[RECORDING] Stopping episode {current_episode_time}...")
            recording = False
            status_last_saved_episode = ""
            print(f"[RECORDING] Episode {current_episode_time} saved!")
            print(f"[RECORDING] Total frames: {len(tcp_pose_array)}")
            print(f"[RECORDING] Ready for next episode. Press 'c' to start.\n")
            write_status(
                message=(
                    f"Stopped SpaceMouse episode {current_episode_time}. "
                    "No separate teleop file is written; robot state is saved by the camera recorder."
                ),
                last_error="",
            )
        else:
            print("[RECORDING] Not recording. Press 'c' to start recording.")
            write_status(message="SpaceMouse trajectory recorder is already paused.", last_error=status_last_error)

    def handle_delete_last():
        nonlocal status_last_saved_episode
        if recording:
            print("[RECORDING] Cannot delete while recording.")
            write_status(message="Cannot delete SpaceMouse trajectory while recording.", last_error=status_last_error)
            return
        status_last_saved_episode = ""
        print("[RECORDING] Delete is handled by the camera recorder. No separate SpaceMouse file exists.")
        write_status(
            message="Delete is handled by the camera recorder. No separate SpaceMouse file exists.",
            last_error=status_last_error,
        )

    ctrl_pressed = False

    def request_shutdown_stop(source: str):
        nonlocal shutdown_requested
        shutdown_requested = True
        write_status(message="SpaceMouse teleop shutting down; force-stopping robot.", last_error=status_last_error)
        force_stop_robot_control(rtde_c, force_mode_active=force_mode_active, label=source, stop_script=False)

    def request_rtde_stop(source: str):
        nonlocal rtde_stop_requested
        rtde_stop_requested = True
        write_status(message=f"UR speedStop requested from {source}; current recording continues.", last_error=status_last_error)

    def on_key_press(key):
        nonlocal reset_requested, shutdown_requested, ctrl_pressed
        
        try:
            if key in {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}:
                ctrl_pressed = True
                return

            if ctrl_pressed and key.char == 'r':
                request_reset()
                
            elif key.char == 'c':
                handle_start_recording()
                    
            elif key.char == 's':
                handle_stop_recording()

            elif key.char == 'd':
                handle_delete_last()

            elif key.char == 'u':
                request_rtde_stop("keyboard-u")
                    
            elif key.char == 'q':
                if recording:
                    print("\n[RECORDING] Stopping current episode before quitting...")
                    handle_stop_recording()
                print("Quitting...")
                request_shutdown_stop("keyboard-q")
                return False  # Stop listener
                
        except AttributeError:
            if key in {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}:
                ctrl_pressed = True

    def on_key_release(key):
        nonlocal ctrl_pressed
        if key in {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}:
            ctrl_pressed = False

    def stdin_control_loop():
        nonlocal reset_requested, shutdown_requested
        for raw in sys.stdin:
            command = raw.strip()
            if not command:
                continue
            if command.startswith("c "):
                handle_start_recording(command[2:].strip())
            elif command == "c":
                handle_start_recording()
            elif command == "s":
                handle_stop_recording()
            elif command == "d":
                handle_delete_last()
            elif command == "u":
                request_rtde_stop("stdin-u")
            elif command == "r" or command.startswith("r "):
                segment_index, noise_xyz_m = parse_reset_command_args(command[1:].strip())
                request_reset(segment_index, noise_xyz_m)
            elif command.startswith("i "):
                next_instruction = command[2:].strip()
                update_instruction(next_instruction)
                write_status(message=f"Instruction set to {instruction}. Save path: {data_folder}", last_error="")
            elif command == "q":
                if recording:
                    print("\n[RECORDING] Stopping current episode before quitting...")
                    handle_stop_recording()
                request_shutdown_stop("stdin-q")
                break
            else:
                write_status(message=f"Unknown SpaceMouse command: {command}", last_error=status_last_error)

    if args.control_mode in {"keyboard", "both"}:
        listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        listener.start()
    if args.control_mode in {"stdin", "both"}:
        Thread(target=stdin_control_loop, daemon=True).start()

    # --- Position saving state ---
    position_file = "robot_trajectory.txt"
    frame_count = 0
    
    # Clear/create new file
    with open(position_file, 'w') as f:
        f.write("# Robot Trajectory Recording\n")
        f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Format: Frame, Timestamp, TCP_Pose(6), Joint_Angles(6), Force(6)\n\n")
    
    # --- New: two-button combo toggle (debounced) ---
    COMBO_WINDOW  = 0.20   # s: edges within this window → combo (wider = easier to trigger)
    COMBO_COOLDOWN = 0.35  # s: prevent repeated toggles while held (shorter = more responsive)
    SINGLE_SUPPRESS = 0.08 # s: briefly suppress single open/close after combo (shorter = quicker resume)

    prev_b0 = False
    prev_b1 = False
    b0_down_t = 0.0
    b1_down_t = 0.0
    last_loop_time = time.time()
    combo_block_until = 0.0
    combo_cooldown_until = 0.0
    close_blocked_by_contact = False
    close_stall_since: Optional[float] = None
    close_prev_position: Optional[float] = None
    close_hold_started_at: Optional[float] = None
    gripper_hold_mode = "idle"
    gripper_hold_progress_position: Optional[float] = None
    last_motion_was_active = False
    last_idle_speed_stop_time = 0.0
    suppress_idle_zero_until = 0.0

    print("\n" + "="*60)
    print("SPACEMOUSE TELEOPERATION WITH CONTINUOUS RECORDING")
    print("="*60)
    print("Controls:")
    print("  - Move SpaceMouse to control robot")
    print("  - SpaceMouse controls full 6-DoF TCP velocity")
    print("  - Hold Button 0: Slowly open gripper across full range")
    print("  - Hold Button 1: Slowly close gripper across full range")
    print("  - Close auto-stops on contact until Button 1 is released")
    print("  - Press 'Ctrl+R': Reset to home position")
    print("  - Press 'U': Clear local SpaceMouse command state without touching RTDE or recording")
    print("  - Ctrl+C: Quit and save trajectory")
    print(f"Maximum linear teleop speed: {XARM_TELEOP_SPEED_MM_S:.1f} mm/s")
    print(f"Maximum angular teleop speed: {TELEOP_ROTATION_SPEED_SCALE:.2f} rad/s")
    print(f"Downward Z extra scale: {TELEOP_DOWNWARD_Z_EXTRA_SCALE * 100:.0f}%")
    print(f"Startup command grace: {TELEOP_STARTUP_COMMAND_GRACE_S:.2f}s")
    print(f"SpaceMouse stale stop timeout: {SPACEMOUSE_STALE_STOP_S:.2f}s")
    print(f"Idle zero-speed override enabled: {ENABLE_IDLE_ZERO_OVERRIDE}")
    if ENABLE_IDLE_ZERO_OVERRIDE:
        print(f"Stop-drift zero speed override: accel={STOP_DRIFT_ZERO_SPEED_ACCEL:.1f}, time={STOP_DRIFT_ZERO_SPEED_TIME_S:.4f}s")
    print(f"Reset/non-teleop speed scale: {NON_TELEOP_SPEED_SCALE * 100:.0f}%")
    print(f"Recording to: {position_file}")
    print(f"Home Position: TCP={RESET_TCP_POSE[:3]}")
    print(f"Subtask reset segment index: {active_subtask_segment_index} (0 = normal home reset)")
    print(f"Subtask reset XYZ noise: +/-{active_subtask_reset_noise_xyz_m:.3f} m")
    print(f"Configured subtask reset segments: {sorted(SUBTASK_RESET_POSES.keys())}")
    print("="*60 + "\n")
    
    try:
        while True:
            if shutdown_requested:
                force_stop_robot_control(rtde_c, force_mode_active=force_mode_active, label="main-loop-shutdown", stop_script=False)
                break
            if rtde_stop_requested:
                rtde_stop_requested = False
                sm.clear_motion_state()
                latest_motion_state = np.zeros(6, dtype=np.float32)
                last_motion_was_active = False
                last_idle_speed_stop_time = time.time()
                suppress_idle_zero_until = last_idle_speed_stop_time + STOP_DRIFT_SUPPRESS_IDLE_ZERO_S
                write_status(message="Local SpaceMouse command cleared; recording continues.", last_error="")
                print("\n[SpaceMouse] Local motion command cleared by U; RTDE was not stopped.")
            try:
                robot_mode = rtde_r.getRobotMode()
            except Exception as e:
                write_status(message="UR RTDE receive error; use Initialize if RTDE needs a full reconnect.", last_error=str(e))
                print(f"\n[RTDE] receive error: {e}")
                time.sleep(0.1)
                continue
            if robot_mode == 7:
                loop_now = time.time()
                loop_dt = min(max(loop_now - last_loop_time, 0.0), 0.1)
                last_loop_time = loop_now

                # Check for reset request
                if reset_requested:
                    segment_index = int(reset_request_subtask_segment_index)
                    noise_xyz_m = float(reset_request_noise_xyz_m)
                    try:
                        reset_mode, target_tcp, target_joint, applied_noise, target_gripper = resolve_reset_target(
                            segment_index,
                            noise_xyz_m,
                        )
                    except RuntimeError as exc:
                        print(f"\n[RESET] {exc}")
                        write_status(message=str(exc), last_error=str(exc))
                        reset_requested = False
                    else:
                        print("\n" + "="*60)
                        print(f"[RESET] Moving to {reset_mode} position...")
                        print(f"Target TCP: {[round(x, 4) for x in target_tcp]}")
                        if segment_index > 0:
                            print(f"XYZ noise applied: {[round(x, 4) for x in applied_noise]}")
                        print(f"Target Joints: {target_joint if target_joint is not None else 'not configured'}")
                        print("="*60)

                        # Stop current motion
                        rtde_c.speedStop()
                        time.sleep(0.1)

                        # Lift TCP vertically by 2cm before reset
                        print("[RESET] Lifting TCP to plug out\n")
                        current_tcp = rtde_r.getActualTCPPose()
                        lift_tcp = current_tcp.copy()
                        lift_tcp[2] += 0.02  # Add 2cm (0.02m) to Z coordinate
                        rtde_c.moveL(lift_tcp, speed=RESET_LIFT_SPEED, acceleration=RESET_LIFT_ACCEL)
                        time.sleep(0.2)  # Wait for lift to complete

                        if segment_index > 0:
                            if target_joint is not None:
                                rtde_c.moveJ(target_joint, RESET_MOVEJ_SPEED, RESET_MOVEJ_ACCEL)
                                time.sleep(0.2)
                            rtde_c.moveL(target_tcp, speed=RESET_SUBTASK_MOVEL_SPEED, acceleration=RESET_SUBTASK_MOVEL_ACCEL)
                            if gripper is not None and target_gripper == "closed":
                                gripper.close()
                                try:
                                    latest_gripper_position = float(gripper.get_current_position())
                                except Exception:
                                    latest_gripper_position = None
                            elif gripper is not None and target_gripper == "open":
                                gripper.open()
                                try:
                                    latest_gripper_position = float(gripper.get_current_position())
                                except Exception:
                                    latest_gripper_position = None
                        else:
                            # Move to reset position using joint control (safer)
                            rtde_c.moveJ(RESET_JOINT_POSE, RESET_MOVEJ_SPEED, RESET_MOVEJ_ACCEL)

                        # Wait for movement to complete
                        time.sleep(0.5)

                        # Verify position
                        current_tcp = rtde_r.getActualTCPPose()
                        current_joints = rtde_r.getActualQ()
                        print(f"[RESET] Current TCP: {[round(x, 4) for x in current_tcp]}")
                        print(f"[RESET] Current Joints: {[round(j, 4) for j in current_joints]}")
                        print("[RESET] ✓ Reset complete!\n")
                        write_status(message=f"Reset complete: {reset_mode}.", last_error="")

                        reset_requested = False
                
                # Read teleop inputs and robot state

                motion_state = sm.get_motion_state_transformed()
                sm_age_s = sm.motion_event_age_s()
                sm_stale = sm_age_s > SPACEMOUSE_STALE_STOP_S
                if sm_stale:
                    motion_state = np.zeros_like(motion_state)
                motion_active = bool(np.max(np.abs(motion_state)) > SPACEMOUSE_ACTIVE_EPS)
                TCP_pose = rtde_r.getActualTCPPose()
                joint_pose = rtde_r.getActualQ()
                Force = rtde_r.getActualTCPForce()
                actual_velocity = rtde_r.getActualTCPSpeed()
                actual_velocity_array = np.asarray(actual_velocity, dtype=np.float64)
                residual_speed = float(np.max(np.abs(actual_velocity_array))) if actual_velocity_array.size else 0.0
                latest_motion_state = motion_state.copy()
                latest_tcp_pose = [float(x) for x in TCP_pose]
                latest_commanded_tcp_pose = [
                    float(x) for x in (
                        np.asarray(TCP_pose, dtype=np.float64)
                        + np.asarray(motion_state, dtype=np.float64) * TELEOP_COMMAND_HORIZON_S
                    )
                ]
                latest_joint_pose = [float(x) for x in joint_pose]
                latest_force = [float(x) for x in Force]
                latest_timestamp = time.time()
                latest_button_0 = bool(sm.is_button_pressed(0))
                latest_button_1 = bool(sm.is_button_pressed(1))
                if gripper is not None:
                    try:
                        latest_gripper_position = float(gripper.get_current_position())
                    except Exception:
                        latest_gripper_position = None
                if frame_count % 10 == 0:
                    write_status(last_error=status_last_error)
                
                # Save current position and force to file (every frame)
                frame_count += 1
                timestamp = time.time()
                with open(position_file, 'a') as f:
                    f.write(f"Frame_{frame_count:06d},{timestamp:.6f},")
                    f.write(f"{','.join(map(str, TCP_pose))},")
                    f.write(f"{','.join(map(str, joint_pose))},")
                    f.write(f"{','.join(map(str, Force))}\n")
                    
                # print(f"Frame {frame_count} | Force: {np.round(Force, 2)} | TCP: {np.round(TCP_pose, 2)}")
                # print("Motion state: ", motion_state)
                
                # Save current data to buffers (line by line recording)
                sm_active = int(motion_active)
                sm_text = (
                    f"SM:{np.round(motion_state, 3)} "
                    f"age:{sm_age_s:.2f}s active:{sm_active} btn:{int(latest_button_0)}{int(latest_button_1)}"
                )
                if recording:
                    tcp_pose_array.append(TCP_pose)
                    joint_pose_array.append(joint_pose)
                    force_array.append(Force)
                    timestamp_array.append(timestamp)
                    # Print recording status
                    print(
                        f"[REC] Frame {len(tcp_pose_array)} | Force: {np.round(Force, 2)} | "
                        f"TCP: {np.round(TCP_pose, 2)} | {sm_text}",
                        end='\r',
                    )
                else:
                    # Print normal status when not recording
                    print(f"Force: {np.round(Force, 2)} | TCP: {np.round(TCP_pose, 2)} | {sm_text}", end='\r')

                # Global slow teleop: keep the former near-work-area linear speed everywhere.
                # --- While in force-mode, don't fight Z force: block teleop Z only (rotations untouched) ---
                # if force_mode_active:
                #     motion_state[2] = 0.0

                # Send velocity only for active SpaceMouse input. When input returns to zero,
                # explicitly stop the active UR speed command so the robot cannot coast on it.
                now = time.time()
                teleop_commands_enabled = now >= teleop_command_enable_time
                if motion_active and teleop_commands_enabled:
                    rtde_c.speedL(motion_state, acceleration=5, time=0.01)
                    last_motion_was_active = True
                elif motion_active:
                    last_motion_was_active = False
                else:
                    should_speed_stop = (
                        teleop_commands_enabled
                        and ENABLE_IDLE_ZERO_OVERRIDE
                        and (
                            last_motion_was_active
                            or (
                                now >= suppress_idle_zero_until
                                and (sm_stale or residual_speed > ROBOT_RESIDUAL_SPEED_STOP_EPS)
                                and now - last_idle_speed_stop_time >= SPACEMOUSE_IDLE_SPEEDSTOP_INTERVAL_S
                            )
                        )
                    )
                    if should_speed_stop:
                        try:
                            send_zero_speed_override(rtde_c, label="idle")
                        except Exception as e:
                            print(f"\n[warn] zero-speed override during idle SpaceMouse input failed: {e}")
                        last_idle_speed_stop_time = now
                    last_motion_was_active = False

                # Optional velocity print (filtered)
                actual_velocity = [0 if abs(float(x)) < 0.01 else float(x) for x in actual_velocity_array.tolist()]
                # print("Current velocity vector: ", actual_velocity)

                # --------- Button handling ---------
                b0 = latest_button_0  # open
                b1 = latest_button_1  # close

                # Rising edges (for combo timing)
                if b0 and not prev_b0:
                    b0_down_t = now
                    print("[BTN] b0 down")
                if b1 and not prev_b1:
                    b1_down_t = now
                    close_hold_started_at = now
                    close_stall_since = None
                    close_prev_position = None
                    print("[BTN] b1 down")
                if (not b0) and prev_b0:
                    print("[BTN] b0 up")
                if (not b1) and prev_b1:
                    print("[BTN] b1 up")

                # Edge-triggered combo: both down within window and not in cooldown
                combo_edge = False
                if (b0 and b1) and (now >= combo_cooldown_until):
                    if (abs(b0_down_t - b1_down_t) <= COMBO_WINDOW) and ((not prev_b0) or (not prev_b1)):
                        combo_edge = True

                # if combo_edge:
                #     combo_block_until = now + SINGLE_SUPPRESS
                #     combo_cooldown_until = now + COMBO_COOLDOWN
                #     # Toggle Force-Mode with safe exit
                #     if force_mode_active:
                #         exit_force_mode(rtde_c)
                #         force_mode_active = False
                #         print("[FM] OFF")
                #     else:
                #         enter_force_mode(rtde_c, rtde_r)
                #         force_mode_active = True
                #         print("[FM] ON")

                if gripper is not None and latest_gripper_position is not None:
                    if not (b0 ^ b1):
                        gripper_command_position = float(latest_gripper_position)

                if (not b1) and close_blocked_by_contact:
                    close_blocked_by_contact = False

                if not (b1 and not b0):
                    close_hold_started_at = None
                    close_stall_since = None
                    close_prev_position = None

                if now >= combo_block_until and gripper is not None:
                    open_pressed_edge = b0 and not b1 and not prev_b0
                    close_pressed_edge = b1 and not b0 and not prev_b1
                    open_released_edge = (not b0) and prev_b0
                    close_released_edge = (not b1) and prev_b1

                    # Command continuous move once on button-down; gripper firmware handles smooth motion.
                    if open_pressed_edge:
                        print("[Robotiq] OPEN hold")
                        gripper_hold_mode = "open"
                        close_blocked_by_contact = False
                        gripper_hold_progress_position = (
                            float(latest_gripper_position) if latest_gripper_position is not None else None
                        )
                        gripper.command_position(gripper_min_position)
                        gripper_last_command_time = now

                    if close_pressed_edge:
                        print("[Robotiq] CLOSE hold")
                        gripper_hold_mode = "close"
                        close_hold_started_at = now
                        close_stall_since = now
                        close_prev_position = (
                            float(latest_gripper_position) if latest_gripper_position is not None else None
                        )
                        gripper_hold_progress_position = close_prev_position
                        if not close_blocked_by_contact:
                            gripper.command_position(gripper_max_position)
                            gripper_last_command_time = now

                    close_just_blocked = False
                    if gripper_hold_mode == "open" and latest_gripper_position is not None:
                        if gripper_hold_progress_position is None:
                            gripper_hold_progress_position = float(latest_gripper_position)
                        else:
                            gripper_hold_progress_position = min(
                                float(gripper_hold_progress_position),
                                float(latest_gripper_position),
                            )

                    if gripper_hold_mode == "close" and b1 and not b0 and not close_blocked_by_contact:
                        if latest_gripper_position is not None:
                            if gripper_hold_progress_position is None:
                                gripper_hold_progress_position = float(latest_gripper_position)
                            else:
                                gripper_hold_progress_position = max(
                                    float(gripper_hold_progress_position),
                                    float(latest_gripper_position),
                                )
                        if gripper.is_contact_detected():
                            close_blocked_by_contact = True
                            close_just_blocked = True
                        elif latest_gripper_position is not None:
                            if close_hold_started_at is None:
                                close_hold_started_at = now
                            if close_prev_position is None:
                                close_prev_position = float(latest_gripper_position)
                                close_stall_since = now
                            else:
                                pos_delta = float(latest_gripper_position) - float(close_prev_position)
                                if pos_delta >= GRIPPER_CLOSE_STALL_EPS:
                                    close_prev_position = float(latest_gripper_position)
                                    close_stall_since = now
                                elif (now - close_hold_started_at) < GRIPPER_CLOSE_STALL_GRACE_S:
                                    close_prev_position = float(latest_gripper_position)
                                    close_stall_since = now
                                elif close_stall_since is None:
                                    close_stall_since = now
                                elif (now - close_stall_since) >= GRIPPER_CLOSE_STALL_BLOCK_S:
                                    close_blocked_by_contact = True
                                    close_just_blocked = True

                        if close_blocked_by_contact and close_just_blocked:
                            gripper.stop_motion()
                            gripper_last_command_time = now
                            print("[Robotiq] close blocked until Button 1 release")

                    should_stop_hold = (
                        (gripper_hold_mode == "open" and (open_released_edge or not b0 or b1))
                        or (gripper_hold_mode == "close" and (close_released_edge or not b1 or b0 or close_blocked_by_contact))
                    )
                    if should_stop_hold:
                        gripper.stop_motion()
                        gripper_last_command_time = now
                        if latest_gripper_position is not None:
                            gripper_command_position = float(
                                np.clip(latest_gripper_position, gripper_min_position, gripper_max_position)
                            )
                        gripper_hold_mode = "idle"
                        gripper_hold_progress_position = None

                # Update prev button states
                if close_blocked_by_contact and b1 and not b0:
                    # Keep command at measured position so we don't keep requesting tighter closure.
                    if latest_gripper_position is not None:
                        gripper_command_position = float(latest_gripper_position)

                prev_b0, prev_b1 = b0, b1

                time.sleep(1/200)

            else:
                print("Robot is not ready.")
                write_status(message="Robot is not ready.", last_error=status_last_error)
                time.sleep(1)  # Wait longer if robot is not ready

    except KeyboardInterrupt:
        # Graceful shutdown
        print("\n\nStopping robot...")
        force_stop_robot_control(rtde_c, force_mode_active=force_mode_active, label="keyboard-interrupt", stop_script=True)
        if gripper is not None:
            gripper.disconnect()
        sm.stop()
        if listener is not None:
            listener.stop()
        
        # Save summary to file
        with open(position_file, 'a') as f:
            f.write(f"\n# Recording ended: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total frames recorded: {frame_count}\n")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total frames recorded: {frame_count}")
        print(f"Trajectory saved to: {position_file}")
        print(f"{'='*60}\n")
    finally:
        try:
            if gripper is not None:
                gripper.disconnect()
        except Exception:
            pass
        try:
            force_stop_robot_control(rtde_c, force_mode_active=force_mode_active, label="finally", stop_script=True)
        except Exception:
            pass
        try:
            if hasattr(rtde_c, "disconnect"):
                rtde_c.disconnect()
        except Exception:
            pass
        try:
            if hasattr(rtde_r, "disconnect"):
                rtde_r.disconnect()
        except Exception:
            pass
        try:
            sm.stop()
        except Exception:
            pass
        try:
            if listener is not None:
                listener.stop()
        except Exception:
            pass
        teleop_ready = False
        write_status(message="SpaceMouse teleop stopped.", last_error=status_last_error)

if __name__ == "__main__":
    main()
