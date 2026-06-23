# source /home/mainuser/UR5_Policy/data_record_env/bin/activate
# python /home/mainuser/UR5_Policy/spacemouse_teleoperation/3DConnexion_UR5_Teleop_Gripper_YinYu.py
# python /home/mainuser/UR5_Policy/data_recording/record_data.py

# k4aviewer
# realsense-viewer


from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from threading import Thread, Event, Lock
from collections import defaultdict
from pynput import keyboard
import numpy as np
import time

import h5py
import json
import os
import re
import argparse
import socket
import sys
from pathlib import Path
from typing import Optional

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
## TELEOP_SPEED_SCALE -> To increase/decrease teleoperation velocity 
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
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
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
        
        # Freeze yaw and pitch BEFORE transformation
        # Keep only rotation around Z axis in SpaceMouse frame (which becomes X after transform)
        state[3] = 0  # Freeze rotation around SpaceMouse X
        state[5] = 0  # Freeze rotation around SpaceMouse Y
        # state[5] is rotation around SpaceMouse Z - keep it (becomes roll around robot X)
        # 
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]

        # Set values lesser than 0.3 to 0 for better control
        tf_state[np.abs(tf_state) < 0.3] = 0
        tf_state = tf_state * TELEOP_SPEED_SCALE

        return tf_state

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
                elif isinstance(event, SpnavButtonEvent):
                    self.button_state[event.bnum] = event.press
                else:
                    time.sleep(1/200)
        finally:
            spnav_close()

# Define robot parameters
ROBOT_HOST = os.environ.get("UR_ROBOT_IP", "192.168.1.10")

# Speed configuration
TELEOP_SPEED_SCALE = 0.3        # 3D mouse teleoperation speed = 30%
NON_TELEOP_SPEED_SCALE = 1.00   # reset / non-teleop motions = 100%

# Non-teleop motion presets (UR defaults scaled by NON_TELEOP_SPEED_SCALE)
RESET_LIFT_SPEED = 0.25 * NON_TELEOP_SPEED_SCALE
RESET_LIFT_ACCEL = 0.50 * NON_TELEOP_SPEED_SCALE
RESET_MOVEJ_SPEED = 1.05 * NON_TELEOP_SPEED_SCALE
RESET_MOVEJ_ACCEL = 0.5 * NON_TELEOP_SPEED_SCALE

ROBOTIQ_PORT = int(os.environ.get("ROBOTIQ_PORT", "63352"))
ROBOTIQ_OPEN_POS = int(os.environ.get("ROBOTIQ_OPEN_POS", "100"))
ROBOTIQ_CLOSE_POS = int(os.environ.get("ROBOTIQ_CLOSE_POS", "255"))
ROBOTIQ_SPEED = int(os.environ.get("ROBOTIQ_SPEED", "255"))
ROBOTIQ_FORCE = int(os.environ.get("ROBOTIQ_FORCE", "10"))

# Continuous slowdown zone for safer teleop near contact regions.
SLOW_ZONE_X_MIN = -0.20
SLOW_ZONE_X_MAX = -0.15
SLOW_ZONE_Z_LEFT = 0.23
SLOW_ZONE_Z_RIGHT = 0.38
SLOW_ZONE_Z_RAMP = 0.06
SLOW_LINEAR_MIN_SCALE = 0.1
SLOW_Z_DOWN_MIN_SCALE = 0.40


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _smoothstep01(value: float) -> float:
    value = _clamp01(value)
    return value * value * (3.0 - 2.0 * value)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def compute_slowdown_scales(tcp_pose) -> tuple[float, float]:
    """Return smooth (linear_xyz_scale, downward_z_extra_scale) from TCP pose."""
    x = float(tcp_pose[0])
    z = float(tcp_pose[2])

    x_span = SLOW_ZONE_X_MAX - SLOW_ZONE_X_MIN
    if x_span <= 1e-9:
        x_weight = 0.0
    else:
        x_weight = _clamp01((x - SLOW_ZONE_X_MIN) / x_span)

    # Interpolate the Z threshold continuously between left/right workspace zones.
    z_threshold = _lerp(SLOW_ZONE_Z_LEFT, SLOW_ZONE_Z_RIGHT, x_weight)

    penetration = z_threshold - z
    if penetration <= 0.0:
        return 1.0, 1.0

    depth_weight = _smoothstep01(penetration / max(SLOW_ZONE_Z_RAMP, 1e-6))
    linear_scale = _lerp(1.0, SLOW_LINEAR_MIN_SCALE, depth_weight)
    down_scale = _lerp(1.0, SLOW_Z_DOWN_MIN_SCALE, depth_weight)
    return linear_scale, down_scale


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
        self._speed = int(speed)
        self._force = int(force)
        self.state = "unknown"

        self._gripper = RobotiqGripper()
        self._gripper.connect(hostname=hostname, port=int(port))
        print(
            f"[Robotiq] connected at {hostname}:{port} "
            f"(open={self._open_pos}, close={self._close_pos}, speed={self._speed}, force={self._force})"
        )

        if auto_activate:
            self._gripper.activate(auto_calibrate=False)
            print("[Robotiq] activate() finished")

        self.open()

    def _move(self, position: int, state_name: str):
        if self.state == state_name:
            return
        self._gripper.move(position, self._speed, self._force)
        self.state = state_name
        print(f"[Robotiq] state -> {self.state}")

    def open(self):
        self._move(self._open_pos, "open")

    def close(self):
        self._move(self._close_pos, "closed")

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


def ensure_tcp_reachable(host: str, port: int, timeout_s: float, label: str):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return
    except OSError as e:
        raise RuntimeError(
            f"{label} unreachable at {host}:{port} within {timeout_s:.1f}s. "
            "Check robot IP/network and that the service is running."
        ) from e
        
        
###
def save_trajectory_to_h5(tcp_poses, joint_poses, forces, timestamps, episode_time, data_folder):
    """Save robot trajectory data to HDF5 file"""
    os.makedirs(data_folder, exist_ok=True)
    h5_file = os.path.join(data_folder, f"trajectory_{episode_time}.h5")
    
    # Convert lists to numpy arrays
    tcp_poses = np.array(tcp_poses)
    joint_poses = np.array(joint_poses)
    forces = np.array(forces)
    timestamps = np.array(timestamps)
    
    print(f"\nSaving trajectory data to HDF5 format...")
    with h5py.File(h5_file, 'w') as f:
        # Create datasets with compression
        f.create_dataset('tcp_pose', data=tcp_poses, compression='gzip', compression_opts=4)
        f.create_dataset('joint_pose', data=joint_poses, compression='gzip', compression_opts=4)
        f.create_dataset('force', data=forces, compression='gzip', compression_opts=4)
        f.create_dataset('timestamps', data=timestamps, compression='gzip', compression_opts=4)
        
        # Save metadata as attributes
        # f.attrs['instruction'] = instruction
        # f.attrs['num_frames'] = len(tcp_poses)
        # f.attrs['timestamp'] = time.strftime("%Y%m%d%H%M%S")
        # f.attrs['robot'] = 'UR5'
        # f.attrs['control_frequency'] = 200  # Hz
        # f.attrs['start_time'] = float(timestamps[0]) if len(timestamps) > 0 else 0.0
        # f.attrs['end_time'] = float(timestamps[-1]) if len(timestamps) > 0 else 0.0
        # f.attrs['duration'] = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 0 else 0.0
    
    print(f"Saved {len(tcp_poses)} frames to {h5_file}")
    return h5_file
    
###

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
        help="UR controller IP address.",
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
        help="Robotiq move speed (0-255).",
    )
    parser.add_argument(
        "--gripper-force",
        type=int,
        default=ROBOTIQ_FORCE,
        help="Robotiq move force (0-255).",
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
    data_folder = str(base_data_folder / date_str / instruction / "trajs_h5")
    status_file = Path(args.status_file).expanduser().resolve() if args.status_file else None
    status_lock = Lock()
    status_message = "Preparing SpaceMouse teleop..."
    status_last_error = ""
    status_last_saved_episode = ""
    latest_motion_state = np.zeros(6, dtype=np.float32)
    latest_tcp_pose = []
    latest_joint_pose = []
    latest_force = []
    latest_gripper_position = None
    latest_button_0 = False
    latest_button_1 = False
    latest_timestamp = None
    teleop_ready = False
    shutdown_requested = False
    listener = None
    gripper: Optional[RobotiqSocketGripper] = None
    recording = False
    current_episode_time = None
    current_episode_folder = None

    def update_instruction(raw_value: str) -> str:
        nonlocal instruction, data_folder
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value.strip()).strip("._-")
        instruction = safe if safe else "untitled"
        data_folder = str(base_data_folder / date_str / instruction / "trajs_h5")
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
            }
        try:
            status_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = status_file.with_suffix(status_file.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            tmp_path.replace(status_file)
        except Exception:
            pass

    update_instruction(instruction)
    write_status(message=f"Preparing SpaceMouse teleop. Save path: {data_folder}", last_error="")

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
    print(f"[Init] connecting UR RTDE interfaces: {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)
    print("[Init] UR RTDE connected")

    if not args.no_gripper:
        try:
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
            print("[Robotiq] initial state:", gripper.state)
        except Exception as e:
            print("[Robotiq] failed to initialize gripper:", e)
            print("[Robotiq] Continue without gripper control.")
            status_last_error = str(e)
    else:
        print("[Robotiq] disabled by --no-gripper")

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
            h5_path = save_trajectory_to_h5(
                tcp_pose_array,
                joint_pose_array,
                force_array,
                timestamp_array,
                current_episode_time,
                data_folder,
            )
            recording = False
            status_last_saved_episode = h5_path
            print(f"[RECORDING] Episode {current_episode_time} saved!")
            print(f"[RECORDING] Total frames: {len(tcp_pose_array)}")
            print(f"[RECORDING] Ready for next episode. Press 'c' to start.\n")
            write_status(message=f"Saved trajectory episode {current_episode_time}.", last_error="")
        else:
            print("[RECORDING] Not recording. Press 'c' to start recording.")
            write_status(message="SpaceMouse trajectory recorder is already paused.", last_error=status_last_error)

    def handle_delete_last():
        nonlocal status_last_saved_episode
        if recording:
            print("[RECORDING] Cannot delete while recording.")
            write_status(message="Cannot delete SpaceMouse trajectory while recording.", last_error=status_last_error)
            return
        if not status_last_saved_episode:
            write_status(message="No saved SpaceMouse trajectory to delete.", last_error=status_last_error)
            return
        target = Path(status_last_saved_episode)
        if target.is_file():
            target.unlink()
            print(f"[RECORDING] Deleted {target.name}")
            write_status(message=f"Deleted trajectory {target.name}.", last_error="")
            status_last_saved_episode = ""
            return
        write_status(message="Saved SpaceMouse trajectory file was not found.", last_error=status_last_error)

    def on_key_press(key):
        nonlocal reset_requested, shutdown_requested
        
        try:
            if key.char == 'r':
                reset_requested = True
                print("\n[RESET] Reset to home position requested...")
                write_status(message="Reset to home position requested.", last_error=status_last_error)
                
            elif key.char == 'c':
                handle_start_recording()
                    
            elif key.char == 's':
                handle_stop_recording()

            elif key.char == 'd':
                handle_delete_last()
                    
            elif key.char == 'q':
                if recording:
                    print("\n[RECORDING] Stopping current episode before quitting...")
                    handle_stop_recording()
                print("Quitting...")
                shutdown_requested = True
                write_status(message="SpaceMouse teleop shutting down.", last_error=status_last_error)
                return False  # Stop listener
                
        except AttributeError:
            pass

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
            elif command == "r":
                reset_requested = True
                print("\n[RESET] Reset to home position requested...")
                write_status(message="Reset to home position requested.", last_error=status_last_error)
            elif command.startswith("i "):
                next_instruction = command[2:].strip()
                update_instruction(next_instruction)
                write_status(message=f"Instruction set to {instruction}. Save path: {data_folder}", last_error="")
            elif command == "q":
                if recording:
                    print("\n[RECORDING] Stopping current episode before quitting...")
                    handle_stop_recording()
                shutdown_requested = True
                write_status(message="SpaceMouse teleop shutting down.", last_error=status_last_error)
                break
            else:
                write_status(message=f"Unknown SpaceMouse command: {command}", last_error=status_last_error)

    if args.control_mode in {"keyboard", "both"}:
        listener = keyboard.Listener(on_press=on_key_press)
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
    combo_block_until = 0.0
    combo_cooldown_until = 0.0

    print("\n" + "="*60)
    print("SPACEMOUSE TELEOPERATION WITH CONTINUOUS RECORDING")
    print("="*60)
    print("Controls:")
    print("  - Move SpaceMouse to control robot")
    print("  - SpaceMouse rotation: ROLL only (pitch/yaw frozen)")
    print("  - Button 0: Open gripper")
    print("  - Button 1: Close gripper")
    print("  - Press 'r': Reset to home position")
    print("  - Ctrl+C: Quit and save trajectory")
    print(f"Teleop speed scale: {TELEOP_SPEED_SCALE * 100:.0f}%")
    print(f"Reset/non-teleop speed scale: {NON_TELEOP_SPEED_SCALE * 100:.0f}%")
    print(f"Recording to: {position_file}")
    print(f"Home Position: TCP={RESET_TCP_POSE[:3]}")
    print("="*60 + "\n")
    
    try:
        while True:
            if shutdown_requested:
                break
            if rtde_r.getRobotMode() == 7:
                # Check for reset request
                if reset_requested:
                    print("\n" + "="*60)
                    print("[RESET] Moving to home position...")
                    print(f"Target TCP: {RESET_TCP_POSE}")
                    print(f"Target Joints: {RESET_JOINT_POSE}")
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
                    
                    # Move to reset position using joint control (safer)
                    velocity = RESET_MOVEJ_SPEED  # rad/s
                    acceleration = RESET_MOVEJ_ACCEL  # rad/s^2
                    rtde_c.moveJ(RESET_JOINT_POSE, velocity, acceleration)
                    
                    # Wait for movement to complete
                    time.sleep(0.5)
                    
                    # Verify position
                    current_tcp = rtde_r.getActualTCPPose()
                    current_joints = rtde_r.getActualQ()
                    print(f"[RESET] Current TCP: {[round(x, 4) for x in current_tcp]}")
                    print(f"[RESET] Current Joints: {[round(j, 4) for j in current_joints]}")
                    print("[RESET] ✓ Reset complete!\n")
                    
                    reset_requested = False
                
                # Read teleop inputs and robot state

                motion_state = sm.get_motion_state_transformed()
                TCP_pose = rtde_r.getActualTCPPose()
                joint_pose = rtde_r.getActualQ()
                Force = rtde_r.getActualTCPForce()
                latest_motion_state = motion_state.copy()
                latest_tcp_pose = [float(x) for x in TCP_pose]
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
                if recording:
                    tcp_pose_array.append(TCP_pose)
                    joint_pose_array.append(joint_pose)
                    force_array.append(Force)
                    timestamp_array.append(timestamp)
                    # Print recording status
                    print(f"[REC] Frame {len(tcp_pose_array)} | Force: {np.round(Force, 2)} | TCP: {np.round(TCP_pose, 2)}", end='\r')
                else:
                    # Print normal status when not recording
                    print(f"Force: {np.round(Force, 2)} | TCP: {np.round(TCP_pose, 2)}", end='\r')

                # Continuous slowdown to avoid abrupt speed jumps at region boundaries.
                linear_scale, down_scale = compute_slowdown_scales(TCP_pose)
                if linear_scale < 0.999:
                    motion_state[:3] *= linear_scale
                    if motion_state[2] < 0:
                        motion_state[2] *= down_scale

                # --- While in force-mode, don't fight Z force: block teleop Z only (rotations untouched) ---
                # if force_mode_active:
                #     motion_state[2] = 0.0

                # Send velocity command
                rtde_c.speedL(motion_state, acceleration=5, time=0.01)

                # Optional velocity print (filtered)
                actual_velocity = rtde_r.getActualTCPSpeed()
                actual_velocity = [0 if abs(x) < 0.01 else x for x in actual_velocity]
                # print("Current velocity vector: ", actual_velocity)

                # --------- Button handling ---------
                now = time.time()
                b0 = latest_button_0  # open
                b1 = latest_button_1  # close

                # Rising edges (for combo timing)
                if b0 and not prev_b0:
                    b0_down_t = now
                    print("[BTN] b0 down")
                if b1 and not prev_b1:
                    b1_down_t = now
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

                if now >= combo_block_until and gripper is not None:
                    if b0 and not b1 and not prev_b0:
                        print("[Robotiq] OPEN")
                        gripper.open()
                    if b1 and not b0 and not prev_b1:
                        print("[Robotiq] CLOSE")
                        gripper.close()

                # Update prev button states
                prev_b0, prev_b1 = b0, b1

                time.sleep(1/200)

            else:
                print("Robot is not ready.")
                write_status(message="Robot is not ready.", last_error=status_last_error)
                time.sleep(1)  # Wait longer if robot is not ready

    except KeyboardInterrupt:
        # Graceful shutdown
        print("\n\nStopping robot...")
        try:
            if force_mode_active:
                exit_force_mode(rtde_c)
        except:
            pass
        if gripper is not None:
            gripper.disconnect()
        rtde_c.stopScript()
        sm.stop()
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
            rtde_c.stopScript()
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
