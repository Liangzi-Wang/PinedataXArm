import time
import json
import numpy as np
import pyrealsense2 as rs
import os
import sys
import argparse
import threading
from pynput import keyboard 
import shutil
import cv2
import re
from pathlib import Path
from typing import Optional, Set

try:
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    RTDEReceiveInterface = None

try:
    import robotiq_gripper
except ImportError:
    robotiq_gripper = None

## Script to collect data from two cameras (hand + external)
## D405 hand camera + D455 external camera
## Use this with spacemouse teleoperation script running separately

## Press 'c' to start recording new episode 
## Press 's' to stop recording and save current episode
## Press 'd' to delete most recent episode (only when not recording)
## Press 'q' to quit

status_file: Optional[Path] = None
status_message = "Recorder is not initialized."
status_last_error = ""
last_saved_episode = None
delete_requires_confirmation = True
record_root: Optional[Path] = None
FRAME_WAIT_TIMEOUT_MS = 100


def _sanitize_instruction(raw_instruction: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_instruction.strip()).strip("._-")
    return safe or "untitled"


def _set_instruction_and_data_folder(raw_instruction: str) -> str:
    global instruction, data_folder

    instruction = _sanitize_instruction(raw_instruction)
    root = globals().get("record_root")
    if isinstance(root, Path):
        date_str = time.strftime("%Y%m%d")
        data_folder = str(root / date_str / instruction / "camera_npy")
        os.makedirs(data_folder, exist_ok=True)
    return instruction


def _safe_buffer_len(name: str) -> int:
    value = globals().get(name, [])
    try:
        return len(value)
    except Exception:
        return 0


def _write_status(
    recorder: Optional["CameraRecorder"] = None,
    message: Optional[str] = None,
    last_error: Optional[str] = None,
):
    global status_message, status_last_error

    if message is not None:
        status_message = message
    if last_error is not None:
        status_last_error = last_error

    if status_file is None:
        return

    robot_enabled = bool(getattr(recorder, "robot_enabled", False))
    payload = {
        "initialized": recorder is not None,
        "recording": bool(globals().get("recording", False)),
        "saving": False,
        "message": status_message,
        "instruction": str(globals().get("instruction", "") or ""),
        "data_folder": str(globals().get("data_folder", "") or ""),
        "current_episode": globals().get("current_time"),
        "current_episode_folder": globals().get("current_episode_folder"),
        "last_saved_episode": globals().get("last_saved_episode"),
        "counts": {
            "hand_frames": _safe_buffer_len("color_hand_array"),
            "external_frames": _safe_buffer_len("color_ext_array"),
            "robot_frames": _safe_buffer_len("robot_timestamp_array"),
        },
        "hand_camera": {
            "name": getattr(recorder, "hand_camera_name", "disabled") if recorder is not None else "disabled",
            "serial": getattr(recorder, "hand_serial", "") if recorder is not None else "",
            "enabled": bool(getattr(recorder, "realsense_hand", None)) if recorder is not None else False,
        },
        "external_camera": {
            "name": getattr(recorder, "external_camera_name", "disabled") if recorder is not None else "disabled",
            "serial": getattr(recorder, "external_serial", "") if recorder is not None else "",
            "enabled": bool(getattr(recorder, "realsense_ext", None)) if recorder is not None else False,
        },
        "robot": {
            "enabled": robot_enabled,
            "ip": getattr(recorder, "robot_ip", "") if recorder is not None else "",
            "fps": getattr(recorder, "robot_capture_fps", 0) if recorder is not None else 0,
            "gripper_state_enabled": bool(getattr(recorder, "enable_gripper_state", False)) if recorder is not None else False,
            "latest": {},
        },
        "last_error": status_last_error,
        "backend": "tmux_camera_recorder",
        "live_preview_supported": False,
    }

    try:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = status_file.with_suffix(status_file.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(status_file)
    except Exception:
        pass

class CameraRecorder():
    def __init__(
        self,
        hand_serial_override: Optional[str] = None,
        external_serial_override: Optional[str] = None,
        allow_missing_hand: bool = False,
        allow_missing_external: bool = False,
        hand_product_ids: Optional[Set[str]] = None,
        external_product_ids: Optional[Set[str]] = None,
        capture_fps: int = 15,
        camera_start_retries: int = 20,
        camera_start_retry_delay: float = 0.5,
        camera_busy_reset: bool = True,
        camera_post_reset_wait: float = 2.0,
        robot_ip: Optional[str] = None,
        robot_capture_fps: int = 200,
        enable_gripper_state: bool = True,
        gripper_port: int = 63352,
        allow_missing_robot: bool = False,
        allow_missing_gripper: bool = True,
    ):
        # Params for D405 hand camera
        self.width_hand = 640
        self.height_hand = 480
        self.fps_hand = capture_fps
        self.depth_width_hand = self.width_hand
        self.depth_height_hand = self.height_hand

        # Params for D455 external camera
        self.width_ext = 640
        self.height_ext = 360
        self.fps_ext = capture_fps
        self.depth_width_ext = self.width_ext
        self.depth_height_ext = self.height_ext

        self.capture_fps = capture_fps
        self.hand_serial_override = hand_serial_override
        self.external_serial_override = external_serial_override
        self.allow_missing_hand = allow_missing_hand
        self.allow_missing_external = allow_missing_external
        self.hand_product_ids = {
            pid.strip().upper() for pid in (hand_product_ids or {"0B5B"}) if pid.strip()
        }
        self.external_product_ids = {
            pid.strip().upper() for pid in (external_product_ids or {"0B5C", "0B3A", "0B3D", "0B07"}) if pid.strip()
        }
        self.camera_start_retries = max(0, int(camera_start_retries))
        self.camera_start_retry_delay = max(0.0, float(camera_start_retry_delay))
        self.camera_busy_reset = bool(camera_busy_reset)
        self.camera_post_reset_wait = max(0.0, float(camera_post_reset_wait))

        self.robot_ip = robot_ip.strip() if robot_ip else None
        self.robot_capture_fps = max(1, int(robot_capture_fps))
        self.enable_gripper_state = bool(enable_gripper_state)
        self.gripper_port = int(gripper_port)
        self.allow_missing_robot = bool(allow_missing_robot)
        self.allow_missing_gripper = bool(allow_missing_gripper)
        self.rtde_r = None
        self.gripper = None
        self.robot_joint_torque_method = "unavailable"
        self.robot_enabled = self.robot_ip is not None
        self.buffer_lock = threading.Lock()
        self._last_status_flush = 0.0

        # Start cameras
        self.hand_serial = None
        self.external_serial = None
        self.hand_camera_name = "disabled"
        self.external_camera_name = "disabled"

        self.realsense_hand = self.initialize_hand_camera(enforce_required=False)
        self.realsense_ext = self.initialize_external_camera(enforce_required=False)

        if self.realsense_hand is None and self.realsense_ext is None:
            print("WARNING: No active RealSense camera. Live preview will stay blank until a camera is available.")

        if self.robot_enabled:
            self.initialize_robot()

        # Initialize data arrays
        self.clear_buffers()

    def initialize_robot(self):
        """Initialize robot state receiver and optional Robotiq gripper state reader."""
        if RTDEReceiveInterface is None:
            msg = (
                "rtde_receive is not installed, cannot record robot state. "
                "Install ur_rtde in data_record_env or disable robot recording."
            )
            if self.allow_missing_robot:
                print(f"WARNING: {msg}")
                self.robot_enabled = False
                return
            raise RuntimeError(msg)

        try:
            self.rtde_r = RTDEReceiveInterface(self.robot_ip)
            print(f"Robot state receiver ready: ip={self.robot_ip}, fps={self.robot_capture_fps}")
        except Exception as exc:
            msg = f"Failed to connect RTDEReceiveInterface to {self.robot_ip}: {exc}"
            if self.allow_missing_robot:
                print(f"WARNING: {msg}. Continue without robot recording.")
                self.robot_enabled = False
                self.rtde_r = None
                return
            raise RuntimeError(msg) from exc

        if not self.enable_gripper_state:
            return

        if robotiq_gripper is None:
            msg = (
                "robotiq_gripper module not found; gripper state will be disabled."
            )
            if self.allow_missing_gripper:
                print(f"WARNING: {msg}")
                self.enable_gripper_state = False
                return
            raise RuntimeError(msg)

        try:
            self.gripper = robotiq_gripper.RobotiqGripper()
            self.gripper.connect(self.robot_ip, self.gripper_port)
            print(f"Gripper state reader ready: {self.robot_ip}:{self.gripper_port}")
        except Exception as exc:
            msg = f"Failed to connect Robotiq gripper at {self.robot_ip}:{self.gripper_port}: {exc}"
            if self.allow_missing_gripper:
                print(f"WARNING: {msg}. Continue without gripper state.")
                self.gripper = None
                self.enable_gripper_state = False
                return
            raise RuntimeError(msg) from exc

    def _get_joint_torque(self):
        """Read joint torque using the first available RTDE method."""
        if self.rtde_r is None:
            raise RuntimeError("RTDE receiver is not initialized.")

        if hasattr(self.rtde_r, "getJointTorques"):
            self.robot_joint_torque_method = "getJointTorques"
            return np.asarray(self.rtde_r.getJointTorques(), dtype=np.float64)

        if hasattr(self.rtde_r, "getActualCurrentAsTorque"):
            self.robot_joint_torque_method = "getActualCurrentAsTorque"
            return np.asarray(self.rtde_r.getActualCurrentAsTorque(), dtype=np.float64)

        self.robot_joint_torque_method = "unavailable"
        return np.full((6,), np.nan, dtype=np.float64)

    def _get_gripper_position(self):
        """Read gripper opening value; return NaN if unavailable."""
        if not self.enable_gripper_state or self.gripper is None:
            return np.nan
        try:
            return float(self.gripper.get_current_position())
        except Exception:
            return np.nan

    def get_robot_observations(self):
        """Read one robot sample: joint, tcp pose, tcp wrench, joint torque, gripper state."""
        if self.rtde_r is None:
            raise RuntimeError("Robot receiver is not initialized.")

        joint_state = np.asarray(self.rtde_r.getActualQ(), dtype=np.float64)
        eef_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float64)
        tcp_wrench = np.asarray(self.rtde_r.getActualTCPForce(), dtype=np.float64)
        joint_torque = self._get_joint_torque()
        gripper_position = self._get_gripper_position()
        return joint_state, eef_pose, tcp_wrench, joint_torque, gripper_position

    def _build_pipeline_config(
        self,
        serial: str,
        color_width: int,
        color_height: int,
        depth_width: int,
        depth_height: int,
        fps: int,
    ):
        """Build a fresh RealSense pipeline+config pair."""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        return pipeline, config

    @staticmethod
    def _preferred_depth_resolution(camera_role: str, info: Optional[dict], color_width: int, color_height: int) -> tuple[int, int]:
        pid = str((info or {}).get("pid") or "").upper()
        if camera_role == "external" and pid == "0B07":
            if color_width > 848 or color_height > 480:
                return 848, 480
        return color_width, color_height

    def _start_pipeline_with_retry(
        self,
        camera_role: str,
        serial: str,
        allow_missing: bool,
        color_width: int,
        color_height: int,
        depth_width: int,
        depth_height: int,
        fps: int,
    ):
        """Start RealSense pipeline with retry when device is temporarily busy."""
        total_attempts = self.camera_start_retries + 1
        reset_attempted = False
        for attempt in range(1, total_attempts + 1):
            pipeline, config = self._build_pipeline_config(
                serial,
                color_width,
                color_height,
                depth_width,
                depth_height,
                fps,
            )
            try:
                pipeline.start(config)
                return pipeline
            except RuntimeError as exc:
                err_str = str(exc)
                busy = ("Device or resource busy" in err_str) or ("errno=16" in err_str)
                missing_video_node = (
                    "map_device_descriptor Cannot open '/dev/video" in err_str
                    and "No such file or directory" in err_str
                )

                should_retry = (busy or missing_video_node) and attempt < total_attempts
                if should_retry:
                    print(
                        f"WARNING: {camera_role} camera ({serial}) not ready, "
                        f"retry {attempt}/{self.camera_start_retries} in {self.camera_start_retry_delay:.2f}s..."
                    )
                    if self.camera_busy_reset and not reset_attempted:
                        reset_attempted = self._hardware_reset_camera(serial, camera_role)
                    time.sleep(self.camera_start_retry_delay)
                    continue

                msg = f"Failed to start {camera_role} camera ({serial}): {exc}"
                if busy:
                    msg += (
                        "\nHint: camera device is occupied. "
                        "Close realsense-viewer/other recording scripts or kill stale tmux sessions, then retry."
                    )
                    if not self.camera_busy_reset:
                        msg += " You can also enable --camera-busy-reset to auto-reset the device once."
                if missing_video_node:
                    msg += (
                        "\nHint: camera just re-enumerated and /dev/video node changed. "
                        "Increase --camera-post-reset-wait (e.g. 4.0) and retry."
                    )

                if allow_missing:
                    print(f"WARNING: {msg}. Continue without {camera_role} camera.")
                    return None

                raise RuntimeError(msg) from exc

    def _hardware_reset_camera(self, serial: str, camera_role: str) -> bool:
        """Try one hardware reset for a busy RealSense camera and wait for re-enumeration."""
        context = rs.context()
        for device in context.query_devices():
            try:
                device_serial = device.get_info(rs.camera_info.serial_number)
                if device_serial != serial:
                    continue
                print(
                    f"INFO: attempting hardware reset for {camera_role} camera ({serial}); "
                    f"waiting {self.camera_post_reset_wait:.2f}s for reconnect..."
                )
                device.hardware_reset()
                self._wait_for_device_by_serial(serial, timeout=self.camera_post_reset_wait)
                return True
            except Exception as exc:
                print(f"WARNING: hardware reset failed for {camera_role} camera ({serial}): {exc}")
                return False

        print(
            f"WARNING: could not find {camera_role} camera ({serial}) for hardware reset."
        )
        return False

    def _wait_for_device_by_serial(self, serial: str, timeout: float) -> bool:
        """Wait until a RealSense device with target serial is visible again."""
        deadline = time.time() + max(0.0, float(timeout))
        while time.time() <= deadline:
            if self._find_device_by_serial(serial) is not None:
                return True
            time.sleep(0.2)
        return self._find_device_by_serial(serial) is not None

    def clear_buffers(self):
        """Clear all data buffers"""
        global color_hand_array, depth_hand_array, timestamp_hand_array
        global color_ext_array, depth_ext_array, timestamp_ext_array
        global robot_timestamp_array, joint_state_array, eef_pose_array
        global tcp_wrench_array, joint_torque_array, gripper_position_array
        
        color_hand_array = []
        depth_hand_array = []
        timestamp_hand_array = []
        
        color_ext_array = []
        depth_ext_array = []
        timestamp_ext_array = []

        robot_timestamp_array = []
        joint_state_array = []
        eef_pose_array = []
        tcp_wrench_array = []
        joint_torque_array = []
        gripper_position_array = []

    def initialize_hand_camera(self, enforce_required: bool = False):
        """Initialize hand camera (default: D405; can be overridden by serial)."""
        selected_serial = None
        selected_info = None

        # Single-camera mode: when external serial is explicitly requested and
        # hand camera is optional, skip hand auto-detection unless user forces
        # --hand-serial.
        if (
            self.allow_missing_hand
            and self.hand_serial_override is None
            and self.external_serial_override is not None
        ):
            print(
                "INFO: hand camera disabled because --allow-missing-hand is set "
                "and --external-serial is provided without --hand-serial."
            )
            return None

        if self.hand_serial_override:
            selected_serial = self.hand_serial_override
            selected_info = self._find_device_by_serial(selected_serial)
            if selected_info is None:
                msg = (
                    f"Hand camera serial '{selected_serial}' not found. "
                    f"Connected RealSense devices: {self._describe_realsense_devices()}"
                )
                if self.allow_missing_hand or not enforce_required:
                    print(f"WARNING: {msg}. Continue without hand camera.")
                    return None
                raise RuntimeError(msg)
        else:
            selected_serial = self._find_device_by_product_ids(
                self.hand_product_ids,
                exclude_serial=self.external_serial_override,
            )
            if selected_serial is None:
                msg = (
                    f"Hand camera not found (expected PIDs: {sorted(self.hand_product_ids)}). "
                    f"Connected RealSense devices: {self._describe_realsense_devices()}"
                )
                if self.allow_missing_hand or not enforce_required:
                    print(f"WARNING: {msg}. Continue without hand camera.")
                    return None
                raise RuntimeError(msg)
            selected_info = self._find_device_by_serial(selected_serial)

        self.hand_serial = selected_serial
        self.depth_width_hand, self.depth_height_hand = self._preferred_depth_resolution(
            "hand",
            selected_info,
            self.width_hand,
            self.height_hand,
        )
        realsense = self._start_pipeline_with_retry(
            camera_role="hand",
            serial=selected_serial,
            allow_missing=(self.allow_missing_hand or not enforce_required),
            color_width=self.width_hand,
            color_height=self.height_hand,
            depth_width=self.depth_width_hand,
            depth_height=self.depth_height_hand,
            fps=self.fps_hand,
        )
        if realsense is None:
            return None
        
        # Warm up camera
        print(f"Warming up hand camera ({selected_serial})...")
        for _ in range(30):
            realsense.wait_for_frames()
        if selected_info is None:
            self.hand_camera_name = "hand_camera"
        else:
            self.hand_camera_name = f"{selected_info['name']} (pid={selected_info['pid']})"
        print(
            f"Hand camera ready: {self.hand_camera_name}, serial={selected_serial}, "
            f"color={self.width_hand}x{self.height_hand}@{self.fps_hand}, "
            f"depth={self.depth_width_hand}x{self.depth_height_hand}@{self.fps_hand}"
        )
        
        return realsense

    def initialize_external_camera(self, enforce_required: bool = False):
        """Initialize external camera (default: D455; can be overridden by serial)."""
        selected_serial = None
        selected_info = None

        if self.external_serial_override:
            selected_serial = self.external_serial_override
            selected_info = self._find_device_by_serial(selected_serial)
            if selected_info is None:
                msg = (
                    f"External camera serial '{selected_serial}' not found. "
                    f"Connected RealSense devices: {self._describe_realsense_devices()}"
                )
                if self.allow_missing_external or not enforce_required:
                    print(f"WARNING: {msg}. Continue without external camera.")
                    return None
                raise RuntimeError(msg)
        else:
            selected_serial = self._find_device_by_product_ids(
                self.external_product_ids,
                exclude_serial=self.hand_serial,
            )
            if selected_serial is None:
                msg = (
                    f"External camera not found (expected PIDs: {sorted(self.external_product_ids)}). "
                    f"Connected RealSense devices: {self._describe_realsense_devices()}"
                )
                if self.allow_missing_external or not enforce_required:
                    print(f"WARNING: {msg}. Continue without external camera.")
                    return None
                raise RuntimeError(msg)
            selected_info = self._find_device_by_serial(selected_serial)

        self.external_serial = selected_serial
        self.depth_width_ext, self.depth_height_ext = self._preferred_depth_resolution(
            "external",
            selected_info,
            self.width_ext,
            self.height_ext,
        )
        realsense = self._start_pipeline_with_retry(
            camera_role="external",
            serial=selected_serial,
            allow_missing=(self.allow_missing_external or not enforce_required),
            color_width=self.width_ext,
            color_height=self.height_ext,
            depth_width=self.depth_width_ext,
            depth_height=self.depth_height_ext,
            fps=self.fps_ext,
        )
        if realsense is None:
            return None
        
        # Warm up camera
        print(f"Warming up external camera ({selected_serial})...")
        for _ in range(30):
            realsense.wait_for_frames()
        if selected_info is None:
            self.external_camera_name = "external_camera"
        else:
            self.external_camera_name = f"{selected_info['name']} (pid={selected_info['pid']})"
        print(
            f"External camera ready: {self.external_camera_name}, serial={selected_serial}, "
            f"color={self.width_ext}x{self.height_ext}@{self.fps_ext}, "
            f"depth={self.depth_width_ext}x{self.depth_height_ext}@{self.fps_ext}"
        )
        
        return realsense

    def ensure_recording_cameras_ready(self):
        """Allow partial init, but require configured cameras before recording starts."""
        issues = []

        if self.realsense_hand is None:
            try:
                self.realsense_hand = self.initialize_hand_camera(enforce_required=not self.allow_missing_hand)
            except Exception as exc:
                issues.append(str(exc))
        if self.realsense_ext is None:
            try:
                self.realsense_ext = self.initialize_external_camera(enforce_required=not self.allow_missing_external)
            except Exception as exc:
                issues.append(str(exc))

        if not self.allow_missing_hand and self.realsense_hand is None:
            issues.append("Hand camera is required to start recording.")
        if not self.allow_missing_external and self.realsense_ext is None:
            issues.append("External camera is required to start recording.")

        if issues:
            detail = " ".join(dict.fromkeys(issues))
            self.last_error = detail
            raise RuntimeError(detail)

    def _find_device_by_product_ids(self, product_ids, exclude_serial=None):
        """Find first RealSense serial whose product_id is in product_ids."""
        context = rs.context()
        for device in context.query_devices():
            try:
                pid = device.get_info(rs.camera_info.product_id).upper()
                serial = device.get_info(rs.camera_info.serial_number)
            except Exception:
                continue
            if exclude_serial and serial == exclude_serial:
                continue
            if pid in product_ids:
                return serial
        return None

    def _find_device_by_serial(self, serial_query):
        """Find RealSense device info by serial number."""
        context = rs.context()
        for device in context.query_devices():
            try:
                serial = device.get_info(rs.camera_info.serial_number)
                if serial != serial_query:
                    continue
                return {
                    "serial": serial,
                    "pid": device.get_info(rs.camera_info.product_id).upper(),
                    "name": device.get_info(rs.camera_info.name),
                }
            except Exception:
                continue
        return None

    def _describe_realsense_devices(self):
        """Return a short description string of connected RealSense devices."""
        context = rs.context()
        desc = []
        for device in context.query_devices():
            try:
                name = device.get_info(rs.camera_info.name)
                pid = device.get_info(rs.camera_info.product_id)
                serial = device.get_info(rs.camera_info.serial_number)
                desc.append(f"{name}(pid={pid},serial={serial})")
            except Exception:
                continue
        return ", ".join(desc) if desc else "none"

    def downsample_with_fps(self, points, num_samples=1024):
        """Simple random sampling for point cloud downsampling"""
        if len(points) > num_samples:
            indices = np.random.choice(len(points), num_samples, replace=False)
            points = points[indices]
        return points

    def get_point_cloud_realsense(self, capture):
        """
        Extract a point cloud from a RealSense frameset.
        The output format is (N, 6): first three columns are XYZ coordinates, last three columns are RGB colors.
        """
        if capture is None:
            raise ValueError("RealSense capture is None.")

        # Extract depth and color frames from the frameset
        depth_frame = capture.get_depth_frame()
        color_frame = capture.get_color_frame()

        if not depth_frame or not color_frame:
            raise ValueError("RealSense frame is invalid.")

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Get camera intrinsics for depth frame
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        fx = intrinsics.fx  # focal length x
        fy = intrinsics.fy  # focal length y
        ppx = intrinsics.ppx  # principal point x
        ppy = intrinsics.ppy  # principal point y

        height, width = depth_image.shape

        # Generate pixel grid
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = x.flatten()
        y = y.flatten()
        depth = depth_image.flatten() * depth_frame.get_units()  # Convert depth to meters

        depth = 1000 * depth
        # Project pixels to camera coordinate system
        X = (x - ppx) * depth / fx
        Y = (y - ppy) * depth / fy
        Z = depth + 700

        # Build (N, 3) point array
        points = np.stack((X, Y, Z), axis=1)

        # Extract and normalize color (convert from BGR to RGB)
        colors = color_image.reshape(-1, 3)[:, ::-1] / 255.0

        # Filter valid depth range
        valid = (points[:, 2] > 720) & (points[:, 2] < 1300)
        points = points[valid]
        colors = colors[valid]

        points = np.concatenate((points, colors), axis=1)

        # Downsample to 1024 points
        points = self.downsample_with_fps(points, num_samples=1024)
        return points

    @staticmethod
    def _is_transient_frame_error(exc: Exception) -> bool:
        err = str(exc)
        return "frame_ref" in err or "null pointer passed for argument" in err

    def _extract_rgbd_images(self, capture):
        if capture is None:
            return None, None
        try:
            color_frame = capture.get_color_frame()
            depth_frame = capture.get_depth_frame()
            if color_frame is None or depth_frame is None:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
        except RuntimeError as exc:
            if self._is_transient_frame_error(exc):
                return None, None
            raise

        if color_image.size == 0 or depth_image.size == 0:
            return None, None

        return cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_image

    # Function to get visual observations (RGB-D only) 
    def get_visual_obs_hand(self):
        """Get visual observations from D405 hand camera"""
        if self.realsense_hand is None:
            return None, None
        try:
            capture_hand = self.realsense_hand.wait_for_frames(timeout_ms=FRAME_WAIT_TIMEOUT_MS)
        except RuntimeError as exc:
            if self._is_transient_frame_error(exc):
                return None, None
            raise
        return self._extract_rgbd_images(capture_hand)

    def get_visual_obs_external(self):
        """Get visual observations from D455 external camera"""
        if self.realsense_ext is None:
            return None, None
        try:
            capture_ext = self.realsense_ext.wait_for_frames(timeout_ms=FRAME_WAIT_TIMEOUT_MS)
        except RuntimeError as exc:
            if self._is_transient_frame_error(exc):
                return None, None
            raise
        return self._extract_rgbd_images(capture_ext)

    def _is_runtime_reconnect_error(self, exc: Exception) -> bool:
        """Return True if runtime error indicates pipeline/device should be restarted."""
        err = str(exc)
        reconnect_signals = [
            "Device disconnected",
            "Device or resource busy",
            "errno=16",
            "wait_for_frames cannot be called before start()",
            "No such file or directory",
            "map_device_descriptor Cannot open '/dev/video",
        ]
        return any(token in err for token in reconnect_signals)

    def _recover_camera_runtime(self, camera_role: str) -> bool:
        """Try to recover a camera pipeline while capture thread is running."""
        if camera_role == "hand":
            if self.hand_serial is None:
                return False
            try:
                if self.realsense_hand is not None:
                    self.realsense_hand.stop()
            except Exception:
                pass

            try:
                pipeline = self._start_pipeline_with_retry(
                    camera_role="hand",
                    serial=self.hand_serial,
                    allow_missing=self.allow_missing_hand,
                    color_width=self.width_hand,
                    color_height=self.height_hand,
                    depth_width=self.depth_width_hand,
                    depth_height=self.depth_height_hand,
                    fps=self.fps_hand,
                )
                self.realsense_hand = pipeline
                return pipeline is not None
            except Exception as exc:
                print(f"WARNING: hand camera reconnect exception: {exc}")
                return False

        if camera_role == "external":
            if self.external_serial is None:
                return False
            try:
                if self.realsense_ext is not None:
                    self.realsense_ext.stop()
            except Exception:
                pass

            try:
                pipeline = self._start_pipeline_with_retry(
                    camera_role="external",
                    serial=self.external_serial,
                    allow_missing=self.allow_missing_external,
                    color_width=self.width_ext,
                    color_height=self.height_ext,
                    depth_width=self.depth_width_ext,
                    depth_height=self.depth_height_ext,
                    fps=self.fps_ext,
                )
                self.realsense_ext = pipeline
                return pipeline is not None
            except Exception as exc:
                print(f"WARNING: external camera reconnect exception: {exc}")
                return False

        return False

    def _handle_command(self, command: str):
        global recording, current_time, current_episode_folder, last_saved_episode

        raw = (command or "").strip()
        cmd = raw.lower()[:1]
        cmd_arg = raw[1:].strip() if len(raw) > 1 else ""
        if not cmd:
            return True

        if cmd == 'i':
            if not cmd_arg:
                _write_status(self, message="Instruction is empty.")
                return True
            next_instruction = _set_instruction_and_data_folder(cmd_arg)
            print(f"Instruction set to: {next_instruction}")
            print(f"Save path: {data_folder}")
            _write_status(self, message=f"Instruction set to {next_instruction}. Save path: {data_folder}", last_error="")
            return True

        if cmd == 'c':
            if not recording:
                if cmd_arg:
                    next_instruction = _set_instruction_and_data_folder(cmd_arg)
                    print(f"Instruction set to: {next_instruction}")

                try:
                    self.ensure_recording_cameras_ready()
                except RuntimeError as exc:
                    print(f"Cannot start recording: {exc}")
                    _write_status(self, message=str(exc), last_error=str(exc))
                    return True

                self.clear_buffers()
                current_time = time.strftime("%Y%m%d%H%M%S")
                current_episode_folder = os.path.join(data_folder, current_time)
                os.makedirs(current_episode_folder, exist_ok=True)

                recording = True
                print(f"\n{'='*50}")
                print(f"Started recording episode {current_time}")
                print(f"Instruction: {instruction}")
                print(f"Save path: {data_folder}")
                print(f"{'='*50}\n")
                _write_status(
                    self,
                    message=f"Started recording episode {current_time}. Instruction: {instruction}.",
                    last_error="",
                )
            else:
                _write_status(self, message="Already recording.")
            return True

        if cmd == 's':
            if recording:
                print(f"\nStopping recording episode {current_time}...")
                recording = False
                _write_status(self, message=f"Stopping recording episode {current_time}...", last_error="")
                time.sleep(0.05)
                self.save_data(current_episode_folder)
                last_saved_episode = current_episode_folder

                print(f"Episode {current_time} saved successfully!")
                print(f"Hand camera frames: {len(color_hand_array)}")
                print(f"External camera frames: {len(color_ext_array)}")
                print(f"Robot state frames: {len(robot_timestamp_array)}")
                print(f"\nReady for next episode. Press 'c' to start recording.\n")
                _write_status(self, message=f"Saved episode {current_time}.", last_error="")
                time.sleep(0.5)
            else:
                _write_status(self, message="Recorder is already paused.")
            return True

        if cmd == 'q':
            if recording:
                print("\nStopping current recording before quitting...")
                recording = False
                _write_status(self, message="Stopping current recording before quitting...")
                time.sleep(0.05)
                self.save_data(current_episode_folder)
                last_saved_episode = current_episode_folder
            print("Quitting session...")
            _write_status(self, message="Recorder shut down.", last_error="")
            return False

        if cmd == 'd':
            if recording:
                print("Cannot delete while recording. Press 's' to stop recording first.")
                _write_status(self, message="Cannot delete while recording.")
                return True

            target_folder = last_saved_episode or current_episode_folder
            target_episode_name = Path(target_folder).name if target_folder else (current_time or "")
            should_delete = True
            if delete_requires_confirmation:
                confirmation = input(f"Are you sure you want to delete episode {target_episode_name}? (y/n): ")
                should_delete = 'y' in confirmation.lower()

            if should_delete and target_folder:
                if os.path.exists(target_folder):
                    shutil.rmtree(target_folder)
                    if last_saved_episode == target_folder:
                        last_saved_episode = None
                    print(f"Deleted episode {target_episode_name}")
                    _write_status(self, message=f"Deleted episode {target_episode_name}.", last_error="")
                else:
                    print("Episode folder not found.")
                    _write_status(self, message="Episode folder not found.")
            else:
                print("Deletion canceled.")
                _write_status(self, message="Deletion canceled.")
            return True

        return True

    # Function to handle global key presses
    def on_press(self, key):
        try:
            return self._handle_command(key.char)
        except AttributeError:
            return None

    def stdin_command_loop(self):
        print("Control mode: stdin (c=start, s=stop/save, d=delete, q=quit)")
        while True:
            line = sys.stdin.readline()
            if line == "":
                break
            should_continue = self._handle_command(line)
            if should_continue is False:
                break

    # Synchronization mechanism
    def synchronized_capture(self, frequency=None):
        """Capture camera data at specified frequency"""
        global recording

        if frequency is None:
            frequency = self.capture_fps
        interval = 1.0 / frequency

        while True:
            start_time = time.time()

            if recording:
                try:
                    # Get visual observations from available cameras
                    color_hand_image, depth_hand_image, timestamp_hand = None, None, None
                    color_ext_image, depth_ext_image, timestamp_ext = None, None, None
                    had_recovery_attempt = False

                    if self.realsense_hand is not None:
                        try:
                            color_hand_image, depth_hand_image = self.get_visual_obs_hand()
                            timestamp_hand = time.time()
                        except Exception as hand_exc:
                            if self._is_runtime_reconnect_error(hand_exc):
                                print(f"\nWARNING: hand camera runtime error: {hand_exc}")
                                recovered = self._recover_camera_runtime("hand")
                                had_recovery_attempt = True
                                if recovered:
                                    print("INFO: hand camera recovered.")
                                else:
                                    print("WARNING: hand camera recovery failed.")
                            else:
                                raise

                    if self.realsense_ext is not None:
                        try:
                            color_ext_image, depth_ext_image = self.get_visual_obs_external()
                            timestamp_ext = time.time()
                        except Exception as ext_exc:
                            if self._is_runtime_reconnect_error(ext_exc):
                                print(f"\nWARNING: external camera runtime error: {ext_exc}")
                                recovered = self._recover_camera_runtime("external")
                                had_recovery_attempt = True
                                if recovered:
                                    print("INFO: external camera recovered.")
                                else:
                                    print("WARNING: external camera recovery failed.")
                            else:
                                raise

                    if color_hand_image is None and color_ext_image is None:
                        if had_recovery_attempt:
                            continue
                        raise RuntimeError("No valid frame from any camera.")

                    # Save data to buffers
                    self.synchronize_data(color_hand_image, depth_hand_image, timestamp_hand,
                                        color_ext_image, depth_ext_image, timestamp_ext)
                    status_parts = []
                    if self.realsense_hand is not None:
                        status_parts.append(f"Hand: {len(color_hand_array)}")
                    if self.realsense_ext is not None:
                        status_parts.append(f"External: {len(color_ext_array)}")
                    print(" | ".join(status_parts), end='\r')

                    now = time.time()
                    if status_file is not None and (now - self._last_status_flush) >= 0.5:
                        _write_status(self)
                        self._last_status_flush = now

                except Exception as e:
                    print(f"\nError during capture: {e}")
                    _write_status(self, message=f"Error during capture: {e}", last_error=str(e))

            elapsed_time = time.time() - start_time
            time_to_sleep = interval - elapsed_time

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def synchronized_robot_capture(self, frequency=None):
        """Capture robot state at higher frequency than cameras."""
        global recording

        if not self.robot_enabled or self.rtde_r is None:
            return

        if frequency is None:
            frequency = self.robot_capture_fps
        interval = 1.0 / max(1, frequency)

        while True:
            start_time = time.time()

            if recording:
                try:
                    joint_state, eef_pose, tcp_wrench, joint_torque, gripper_position = self.get_robot_observations()
                    timestamp_robot = time.time()
                    self.synchronize_robot_data(
                        timestamp_robot,
                        joint_state,
                        eef_pose,
                        tcp_wrench,
                        joint_torque,
                        gripper_position,
                    )
                    now = time.time()
                    if status_file is not None and (now - self._last_status_flush) >= 0.5:
                        _write_status(self)
                        self._last_status_flush = now
                except Exception as exc:
                    print(f"\nError during robot capture: {exc}")
                    _write_status(self, message=f"Error during robot capture: {exc}", last_error=str(exc))

            elapsed_time = time.time() - start_time
            time_to_sleep = interval - elapsed_time

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def synchronize_data(self, color_hand_image, depth_hand_image, timestamp_hand,
                        color_ext_image, depth_ext_image, timestamp_ext):
        """Add data to buffers"""
        with self.buffer_lock:
            # Hand camera observations
            if color_hand_image is not None and depth_hand_image is not None and timestamp_hand is not None:
                color_hand_array.append(color_hand_image)
                depth_hand_array.append(depth_hand_image)
                timestamp_hand_array.append(timestamp_hand)

            # External camera observations
            if color_ext_image is not None and depth_ext_image is not None and timestamp_ext is not None:
                color_ext_array.append(color_ext_image)
                depth_ext_array.append(depth_ext_image)
                timestamp_ext_array.append(timestamp_ext)

    def synchronize_robot_data(
        self,
        timestamp_robot,
        joint_state,
        eef_pose,
        tcp_wrench,
        joint_torque,
        gripper_position,
    ):
        """Add robot state sample to buffers."""
        with self.buffer_lock:
            robot_timestamp_array.append(timestamp_robot)
            joint_state_array.append(joint_state)
            eef_pose_array.append(eef_pose)
            tcp_wrench_array.append(tcp_wrench)
            joint_torque_array.append(joint_torque)
            gripper_position_array.append(gripper_position)

    def save_data(self, episode_folder):
        """Save all buffered data to disk in NPY format"""
        # Convert lists to arrays under lock to avoid race with capture threads
        with self.buffer_lock:
            color_hand_image = np.array(color_hand_array)
            depth_hand_image = np.array(depth_hand_array)
            timestamps_hand = np.array(timestamp_hand_array)

            color_ext_image = np.array(color_ext_array)
            depth_ext_image = np.array(depth_ext_array)
            timestamps_ext = np.array(timestamp_ext_array)

            timestamps_robot = np.array(robot_timestamp_array)
            joint_state = np.array(joint_state_array)
            eef_pose = np.array(eef_pose_array)
            tcp_wrench = np.array(tcp_wrench_array)
            joint_torque = np.array(joint_torque_array)
            gripper_position = np.array(gripper_position_array)

        # Save data to NPY format
        print(f"\nSaving data to NPY format...")
        
        # Hand camera files
        rgb_hand_file = os.path.join(episode_folder, "rgb_hand.npy")
        depth_hand_file = os.path.join(episode_folder, "depth_hand.npy")
        timestamp_hand_file = os.path.join(episode_folder, "timestamps_hand.npy")
        
        # External camera files
        rgb_ext_file = os.path.join(episode_folder, "rgb_external.npy")
        depth_ext_file = os.path.join(episode_folder, "depth_external.npy")
        timestamp_ext_file = os.path.join(episode_folder, "timestamps_external.npy")

        # Robot files
        robot_timestamp_file = os.path.join(episode_folder, "timestamps_robot.npy")
        joint_state_file = os.path.join(episode_folder, "joint_state.npy")
        eef_pose_file = os.path.join(episode_folder, "eef_pose.npy")
        tcp_wrench_file = os.path.join(episode_folder, "tcp_wrench.npy")
        joint_torque_file = os.path.join(episode_folder, "joint_torque.npy")
        gripper_position_file = os.path.join(episode_folder, "gripper_position.npy")
        
        if len(color_hand_array) > 0:
            np.save(rgb_hand_file, color_hand_image)
            np.save(depth_hand_file, depth_hand_image)
            np.save(timestamp_hand_file, timestamps_hand)

        if len(color_ext_array) > 0:
            np.save(rgb_ext_file, color_ext_image)
            np.save(depth_ext_file, depth_ext_image)
            np.save(timestamp_ext_file, timestamps_ext)

        if len(robot_timestamp_array) > 0:
            np.save(robot_timestamp_file, timestamps_robot)
            np.save(joint_state_file, joint_state)
            np.save(eef_pose_file, eef_pose)
            np.save(tcp_wrench_file, tcp_wrench)
            np.save(joint_torque_file, joint_torque)
            np.save(gripper_position_file, gripper_position)
        
        # Save metadata as JSON
        metadata = {
            'instruction': instruction,
            'timestamp': current_time,
            'hand_camera': self.hand_camera_name if self.realsense_hand is not None else 'disabled',
            'external_camera': self.external_camera_name if self.realsense_ext is not None else 'disabled',
            'hand_serial': self.hand_serial if self.hand_serial is not None else '',
            'external_serial': self.external_serial if self.external_serial is not None else '',
            'frequency': self.capture_fps,
            'camera_mode': (
                'dual'
                if (self.realsense_hand is not None and self.realsense_ext is not None)
                else 'single'
            ),
            'hand_frames': len(color_hand_array),
            'external_frames': len(color_ext_array),
            'hand_start_time': float(timestamps_hand[0]) if len(timestamps_hand) > 0 else 0.0,
            'hand_end_time': float(timestamps_hand[-1]) if len(timestamps_hand) > 0 else 0.0,
            'hand_duration': float(timestamps_hand[-1] - timestamps_hand[0]) if len(timestamps_hand) > 0 else 0.0,
            'external_start_time': float(timestamps_ext[0]) if len(timestamps_ext) > 0 else 0.0,
            'external_end_time': float(timestamps_ext[-1]) if len(timestamps_ext) > 0 else 0.0,
            'external_duration': float(timestamps_ext[-1] - timestamps_ext[0]) if len(timestamps_ext) > 0 else 0.0,
            'robot_enabled': self.robot_enabled,
            'robot_ip': self.robot_ip if self.robot_ip is not None else '',
            'robot_frequency': self.robot_capture_fps if self.robot_enabled else 0,
            'robot_frames': len(robot_timestamp_array),
            'robot_start_time': float(timestamps_robot[0]) if len(timestamps_robot) > 0 else 0.0,
            'robot_end_time': float(timestamps_robot[-1]) if len(timestamps_robot) > 0 else 0.0,
            'robot_duration': float(timestamps_robot[-1] - timestamps_robot[0]) if len(timestamps_robot) > 0 else 0.0,
            'gripper_state_enabled': self.enable_gripper_state,
            'joint_torque_method': self.robot_joint_torque_method,
        }
        
        metadata_file = os.path.join(episode_folder, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        if len(color_hand_array) > 0:
            print(f"Saved hand camera frames: {len(color_hand_array)}")
            print(f"  RGB: {rgb_hand_file}")
            print(f"  Depth: {depth_hand_file}")
            print(f"  Timestamps: {timestamp_hand_file}")
        else:
            print("Hand camera disabled or no frames captured.")

        if len(color_ext_array) > 0:
            print(f"\nSaved external camera frames: {len(color_ext_array)}")
            print(f"  RGB: {rgb_ext_file}")
            print(f"  Depth: {depth_ext_file}")
            print(f"  Timestamps: {timestamp_ext_file}")
        else:
            print("\nExternal camera disabled or no frames captured.")

        if len(robot_timestamp_array) > 0:
            print(f"\nSaved robot state frames: {len(robot_timestamp_array)}")
            print(f"  Joint: {joint_state_file}")
            print(f"  EEF Pose: {eef_pose_file}")
            print(f"  TCP Wrench: {tcp_wrench_file}")
            print(f"  Joint Torque: {joint_torque_file}")
            print(f"  Gripper Position: {gripper_position_file}")
            print(f"  Timestamps: {robot_timestamp_file}")
        else:
            print("\nRobot state disabled or no frames captured.")
        print(f"\nMetadata: {metadata_file}")


def parse_pid_csv(pid_csv: str) -> Set[str]:
    return {pid.strip().upper() for pid in pid_csv.split(',') if pid.strip()}


def main():
    # Global variables for controlling the recording state
    global data_folder, recording, instruction, current_episode_folder, current_time
    global color_hand_array, depth_hand_array, timestamp_hand_array
    global color_ext_array, depth_ext_array, timestamp_ext_array
    global status_file, status_message, status_last_error, last_saved_episode, delete_requires_confirmation, record_root
    
    parser = argparse.ArgumentParser(description="Record synchronized D405 + D455 RGBD episodes.")
    parser.add_argument(
        "--root",
        default="./recordings",
        help="Root directory for saved data (default: ./recordings).",
    )
    parser.add_argument(
        "--hand-serial",
        default=None,
        help="Force hand camera serial (overwrite auto-detection).",
    )
    parser.add_argument(
        "--external-serial",
        default=None,
        help="Force external camera serial (overwrite auto-detection).",
    )
    parser.add_argument(
        "--allow-missing-hand",
        action="store_true",
        help="Allow running without hand camera (useful when D405 is not connected).",
    )
    parser.add_argument(
        "--allow-missing-external",
        action="store_true",
        help="Allow running without external camera.",
    )
    parser.add_argument(
        "--hand-product-ids",
        default="0B5B",
        help="Comma-separated hand camera product IDs for auto-detection.",
    )
    parser.add_argument(
        "--external-product-ids",
        default="0B5C,0B3A,0B3D,0B07",
        help="Comma-separated external camera product IDs for auto-detection (includes D455/D435i).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Capture frequency for both cameras.",
    )
    parser.add_argument(
        "--camera-start-retries",
        type=int,
        default=20,
        help="Retry count when camera is busy at startup (errno=16).",
    )
    parser.add_argument(
        "--camera-start-retry-delay",
        type=float,
        default=0.5,
        help="Delay seconds between camera startup retries.",
    )
    parser.add_argument(
        "--camera-busy-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto hardware-reset a busy camera once during startup retries (default: enabled).",
    )
    parser.add_argument(
        "--camera-post-reset-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after camera hardware reset before next retry.",
    )
    parser.add_argument(
        "--robot-ip",
        default=os.environ.get("UR_ROBOT_IP", ""),
        help="UR controller IP for high-frequency robot state recording (empty disables robot recording).",
    )
    parser.add_argument(
        "--robot-fps",
        type=int,
        default=200,
        help="Robot state capture frequency (joint/eef/wrench/torque/gripper).",
    )
    parser.add_argument(
        "--enable-gripper-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record Robotiq gripper position via socket (default: enabled).",
    )
    parser.add_argument(
        "--gripper-port",
        type=int,
        default=63352,
        help="Robotiq socket port.",
    )
    parser.add_argument(
        "--allow-missing-robot",
        action="store_true",
        help="Continue with camera-only recording if robot connection fails.",
    )
    parser.add_argument(
        "--allow-missing-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue if gripper state is unavailable (default: enabled).",
    )
    parser.add_argument(
        "--instruction",
        default="",
        help="Task instruction used in folder naming. If empty, prompt from stdin.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("keyboard", "stdin"),
        default="keyboard",
        help="Control source for c/s/d/q commands.",
    )
    parser.add_argument(
        "--status-file",
        default="",
        help="Optional JSON file path for writing recorder runtime status.",
    )
    parser.add_argument(
        "--no-delete-confirm",
        action="store_true",
        help="Delete latest episode without y/n confirmation prompt.",
    )
    args = parser.parse_args()

    status_file = Path(args.status_file).expanduser().resolve() if args.status_file else None
    status_message = "Recorder process starting..."
    status_last_error = ""
    last_saved_episode = None
    delete_requires_confirmation = not args.no_delete_confirm
    _write_status(None, message=status_message, last_error="")

    recording = False
    record_root = Path(args.root).expanduser().resolve()
    raw_instruction = args.instruction.strip()
    if not raw_instruction:
        if args.control_mode == "stdin":
            raw_instruction = "untitled"
        else:
            try:
                raw_instruction = input("Enter the task instruction: ").strip()
            except EOFError:
                raw_instruction = ""
    _set_instruction_and_data_folder(raw_instruction)
    current_episode_folder = None
    current_time = None
    _write_status(None, message=f"Preparing recorder. Save path: {data_folder}", last_error="")

    # Initialize camera recording system
    recorder = CameraRecorder(
        hand_serial_override=args.hand_serial,
        external_serial_override=args.external_serial,
        allow_missing_hand=args.allow_missing_hand,
        allow_missing_external=args.allow_missing_external,
        hand_product_ids=parse_pid_csv(args.hand_product_ids),
        external_product_ids=parse_pid_csv(args.external_product_ids),
        capture_fps=args.fps,
        camera_start_retries=args.camera_start_retries,
        camera_start_retry_delay=args.camera_start_retry_delay,
        camera_busy_reset=args.camera_busy_reset,
        camera_post_reset_wait=args.camera_post_reset_wait,
        robot_ip=args.robot_ip,
        robot_capture_fps=args.robot_fps,
        enable_gripper_state=args.enable_gripper_state,
        gripper_port=args.gripper_port,
        allow_missing_robot=args.allow_missing_robot,
        allow_missing_gripper=args.allow_missing_gripper,
    )
    _write_status(recorder, message=f"Recorder initialized. Save path: {data_folder}", last_error="")

    print("\n" + "="*50)
    print("MULTI CAMERA RECORDING SYSTEM")
    print(f"Hand camera: {recorder.hand_camera_name} | serial={recorder.hand_serial}")
    print(f"External camera: {recorder.external_camera_name} | serial={recorder.external_serial}")
    if recorder.robot_enabled:
        print(
            "Robot: "
            f"ip={recorder.robot_ip}, fps={recorder.robot_capture_fps}, "
            f"gripper_state={'on' if recorder.enable_gripper_state else 'off'}"
        )
    else:
        print("Robot: disabled")
    print("="*50)
    print("\nControls:")
    print("  'c' - Start recording new episode")
    print("  'c <instruction>' - Start and set instruction for this episode")
    print("  'i <instruction>' - Update instruction for the next episode")
    print("  's' - Stop recording and save")
    print("  'd' - Delete most recent episode")
    print("  'q' - Quit")
    print(f"Current instruction: {instruction}")
    print(f"\nSave path: {data_folder}")
    print("\n" + "="*50)
    print("\nNOTE: Use spacemouse teleoperation script separately")
    print("      to control the robot during recording.")
    print("\nReady! Press 'c' to start recording.\n")

    # Start the synchronized capture in a separate thread
    capture_thread = threading.Thread(target=recorder.synchronized_capture, daemon=True)
    capture_thread.start()

    if recorder.robot_enabled:
        robot_thread = threading.Thread(target=recorder.synchronized_robot_capture, daemon=True)
        robot_thread.start()

    if args.control_mode == "stdin":
        recorder.stdin_command_loop()
    else:
        listener = keyboard.Listener(on_press=recorder.on_press)
        listener.start()
        listener.join()

    _write_status(recorder, message="Recorder stopped.")
    print("\nStopped dual camera recording system.")


if __name__ == '__main__':
    main()
