import time
import json
import numpy as np
import pyrealsense2 as rs
import os
import argparse
import sys
import threading
import shutil
import subprocess
import cv2
import re
from pathlib import Path
from typing import Optional, Set

PINE_DATA_DIR = Path(__file__).resolve().parents[1]
ROBOT_BACKEND = os.getenv("ROBOT_BACKEND", "ur").strip().lower()

try:
    from pynput import keyboard
except Exception:
    keyboard = None

if ROBOT_BACKEND == "xarm":
    if str(PINE_DATA_DIR) not in sys.path:
        sys.path.insert(0, str(PINE_DATA_DIR))
    try:
        from xarm_bridge import XArmReceiveInterface as RTDEReceiveInterface
        from xarm_bridge import XArmSDKGripper
    except ImportError:
        RTDEReceiveInterface = None
        XArmSDKGripper = None
else:
    XArmSDKGripper = None
    try:
        from rtde_receive import RTDEReceiveInterface
    except ImportError:
        RTDEReceiveInterface = None

try:
    import robotiq_gripper
except ImportError:
    data_recording_dir = PINE_DATA_DIR / "data_recording"
    if str(data_recording_dir) not in sys.path:
        sys.path.insert(0, str(data_recording_dir))
    try:
        import robotiq_gripper
    except ImportError:
        robotiq_gripper = None

## Script to collect data from two cameras (hand + external)
## D405 hand camera + D405 external camera
## Use this with spacemouse teleoperation script running separately

## Press 'c' to start recording new episode 
## Press 's' to stop recording and save current episode
## Press 'd' to delete most recent episode (only when not recording)
## Press 'q' to quit

status_file: Optional[Path] = None
preview_dir: Optional[Path] = None
status_message = "Recorder is not initialized."
status_last_error = ""
status_saving = False
last_saved_episode = None
delete_requires_confirmation = True
record_root: Optional[Path] = None
FRAME_WAIT_TIMEOUT_MS = 100
STREAM_STALE_AFTER_S = 1.0
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
SAVED_FRAME_WIDTH = 640
SAVED_FRAME_HEIGHT = 480
RGB_CANDIDATES = [
    (848, 480, 15),
    (848, 480, 30),
    (848, 480, 5),
    (1280, 720, 15),
    (1280, 720, 30),
    (640, 480, 15),
    (640, 480, 30),
    (640, 480, 5),
    (480, 270, 15),
    (480, 270, 5),
    (424, 240, 15),
    (424, 240, 5),
]
RGB_FPS_FALLBACKS = (30, 15, 5)


def _sanitize_instruction(raw_instruction: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_instruction.strip()).strip("._-")
    if not safe:
        return "untitled"

    # Guard against malformed stacked artifacts like:
    # "untitled_c_untitled_c_untitled_c" -> "untitled"
    parts = safe.split("_c_")
    if len(parts) >= 3 and parts[-1] == "c":
        repeated = [item for item in parts[:-1] if item]
        if repeated and len(set(repeated)) == 1:
            return repeated[0]

    return safe


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


def _resolve_ffmpeg_binary() -> str:
    ffmpeg_path = shutil.which(FFMPEG_BIN)
    if ffmpeg_path:
        return ffmpeg_path
    raise RuntimeError(
        f"Required executable '{FFMPEG_BIN}' was not found in PATH. "
        "Install ffmpeg or set FFMPEG_BIN to the correct binary."
    )


def _vision_entry(
    *,
    path: str,
    storage: str,
    codec: str,
    shape: tuple[int, ...],
    dtype: str,
    fps: int,
    channel_order: Optional[str] = None,
    verified_lossless: bool = False,
) -> dict:
    entry = {
        "path": path,
        "storage": storage,
        "codec": codec,
        "shape": [int(dim) for dim in shape],
        "dtype": dtype,
        "fps": int(fps),
        "verified_lossless": bool(verified_lossless),
    }
    if channel_order is not None:
        entry["channel_order"] = channel_order.upper()
        entry["channel_convention"] = "webapp_rgb_no_swap"
    return entry


class FFmpegVideoWriter:
    def __init__(
        self,
        *,
        ffmpeg_bin: str,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        input_pixel_format: str,
        label: str,
        codec: str = "ffv1",
        output_pixel_format: Optional[str] = None,
        output_args: Optional[list[str]] = None,
    ):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.input_pixel_format = input_pixel_format
        self.label = label
        self.codec = codec
        self.output_pixel_format = output_pixel_format or input_pixel_format
        self.output_args = list(output_args or [])
        self.frame_count = 0
        self._proc: Optional[subprocess.Popen] = None
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            input_pixel_format,
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            codec,
            "-pix_fmt",
            self.output_pixel_format,
            *self.output_args,
            str(output_path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError(f"{self.label} writer is not running.")
        if self.input_pixel_format == "rgb24":
            if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
                raise ValueError(
                    f"{self.label} expected uint8 frame {(self.height, self.width, 3)}, got {frame.dtype} {frame.shape}"
                )
        elif self.input_pixel_format == "gray16le":
            if frame.dtype != np.uint16 or frame.shape != (self.height, self.width):
                raise ValueError(
                    f"{self.label} expected uint16 frame {(self.height, self.width)}, got {frame.dtype} {frame.shape}"
                )
        else:
            raise ValueError(f"Unsupported pixel format: {self.input_pixel_format}")

        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            self.frame_count += 1
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(self._error_message(f"write failed: {exc}")) from exc

    def close(self) -> None:
        if self._proc is None:
            return
        stderr_text = ""
        return_code = 0
        try:
            if self._proc.stdin is not None and not self._proc.stdin.closed:
                self._proc.stdin.close()
            if self._proc.stderr is not None:
                stderr_text = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
            return_code = self._proc.wait()
        finally:
            self._proc = None
        if return_code != 0:
            raise RuntimeError(self._error_message(stderr_text or f"ffmpeg exited with code {return_code}"))

    def abort(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.kill()
            self._proc.wait()
        except Exception:
            pass
        finally:
            self._proc = None

    def _error_message(self, detail: str) -> str:
        return f"{self.label} ffmpeg writer error: {detail}"


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
    input_status = recorder.get_input_status_payload() if recorder is not None else {}
    latest_robot = input_status.get("robot", {}).get("latest", {}) if recorder is not None else {}
    payload = {
        "initialized": recorder is not None,
        "recording": bool(globals().get("recording", False)),
        "saving": bool(globals().get("status_saving", False)),
        "message": status_message,
        "instruction": str(globals().get("instruction", "") or ""),
        "data_folder": str(globals().get("data_folder", "") or ""),
        "current_episode": globals().get("current_time"),
        "current_episode_folder": globals().get("current_episode_folder"),
        "last_saved_episode": globals().get("last_saved_episode"),
        "counts": {
            "hand_frames": _safe_buffer_len("timestamp_hand_array"),
            "wrist_frames": _safe_buffer_len("timestamp_wrist_array"),
            "external_frames": _safe_buffer_len("timestamp_ext_array"),
            "robot_frames": _safe_buffer_len("robot_timestamp_array"),
            "subtask_labels": _safe_buffer_len("subtask_timestamp_array"),
        },
        "hand_camera": input_status.get(
            "hand_camera",
            {"requested": True, "enabled": False, "connected": False, "available": False, "status": "disconnected"},
        ),
        "wrist_camera": input_status.get(
            "wrist_camera",
            {"requested": True, "enabled": False, "connected": False, "available": False, "status": "disconnected"},
        ),
        "external_camera": input_status.get(
            "external_camera",
            {"requested": True, "enabled": False, "connected": False, "available": False, "status": "disconnected"},
        ),
        "robot": input_status.get(
            "robot",
            {
                "requested": False,
                "enabled": robot_enabled,
                "connected": False,
                "available": False,
                "status": "disconnected",
                "ip": getattr(recorder, "robot_ip", "") if recorder is not None else "",
                "fps": getattr(recorder, "robot_capture_fps", 0) if recorder is not None else 0,
                "gripper_state_enabled": bool(getattr(recorder, "enable_gripper_state", False)) if recorder is not None else False,
                "latest": latest_robot,
            },
        ),
        "gripper": input_status.get(
            "gripper",
            {
                "requested": False,
                "enabled": False,
                "connected": False,
                "available": False,
                "status": "disconnected",
                "ip": getattr(recorder, "robot_ip", "") if recorder is not None else "",
                "port": getattr(recorder, "gripper_port", 0) if recorder is not None else 0,
                "latest_position": None,
            },
        ),
        "last_error": status_last_error,
        "backend": "tmux_web_recorder",
        "live_preview_supported": preview_dir is not None,
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
        wrist_serial_override: Optional[str] = None,
        external_serial_override: Optional[str] = None,
        allow_missing_hand: bool = False,
        allow_missing_wrist: bool = False,
        allow_missing_external: bool = False,
        hand_product_ids: Optional[Set[str]] = None,
        wrist_product_ids: Optional[Set[str]] = None,
        external_product_ids: Optional[Set[str]] = None,
        capture_fps: int = 15,
        hand_width: int = 640,
        hand_height: int = 480,
        wrist_width: int = 640,
        wrist_height: int = 480,
        external_width: int = 848,
        external_height: int = 480,
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
        teleop_status_file: Optional[str] = None,
    ):
        # Params for D405 hand camera
        self.width_hand = max(160, int(hand_width))
        self.height_hand = max(120, int(hand_height))
        self.fps_hand = capture_fps
        self.depth_width_hand = self.width_hand
        self.depth_height_hand = self.height_hand

        # Params for D405 wrist camera
        self.width_wrist = max(160, int(wrist_width))
        self.height_wrist = max(120, int(wrist_height))
        self.fps_wrist = capture_fps
        self.depth_width_wrist = self.width_wrist
        self.depth_height_wrist = self.height_wrist

        # Params for D405 external camera
        self.width_ext = max(160, int(external_width))
        self.height_ext = max(120, int(external_height))
        self.fps_ext = capture_fps
        self.depth_width_ext = self.width_ext
        self.depth_height_ext = self.height_ext

        self.saved_frame_width = SAVED_FRAME_WIDTH
        self.saved_frame_height = SAVED_FRAME_HEIGHT
        self.capture_fps = capture_fps
        self.hand_serial_override = hand_serial_override
        self.wrist_serial_override = wrist_serial_override
        self.external_serial_override = external_serial_override
        self.allow_missing_hand = allow_missing_hand
        self.allow_missing_wrist = allow_missing_wrist
        self.allow_missing_external = allow_missing_external
        self.hand_product_ids = {
            pid.strip().upper() for pid in (hand_product_ids or {"0B5B"}) if pid.strip()
        }
        self.wrist_product_ids = {
            pid.strip().upper() for pid in (wrist_product_ids or {"0B5B"}) if pid.strip()
        }
        self.external_product_ids = {
            pid.strip().upper() for pid in (external_product_ids or {"0B5B"}) if pid.strip()
        }
        self.camera_start_retries = max(0, int(camera_start_retries))
        self.camera_start_retry_delay = max(0.0, float(camera_start_retry_delay))
        self.camera_busy_reset = bool(camera_busy_reset)
        self.camera_post_reset_wait = max(0.0, float(camera_post_reset_wait))

        self.robot_ip = robot_ip.strip() if robot_ip else None
        self.robot_capture_fps = max(1, int(robot_capture_fps))
        self.enable_gripper_state = bool(enable_gripper_state)
        self.gripper_requested = bool(enable_gripper_state and self.robot_ip)
        self.gripper_port = int(gripper_port)
        self.allow_missing_robot = bool(allow_missing_robot)
        self.allow_missing_gripper = bool(allow_missing_gripper)
        self.teleop_status_file = Path(teleop_status_file).expanduser().resolve() if teleop_status_file else None
        self.gripper_state_source = "teleop_status" if self.teleop_status_file is not None else "socket"
        self.rtde_r = None
        self.gripper = None
        self.robot_joint_torque_method = "unavailable"
        self.robot_enabled = self.robot_ip is not None
        self.robot_requested = self.robot_enabled
        self.buffer_lock = threading.Lock()
        self.latest_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_error = ""
        self.latest_camera = {
            "hand": {"rgb": None, "depth": None, "timestamp": None},
            "wrist": {"rgb": None, "depth": None, "timestamp": None},
            "external": {"rgb": None, "depth": None, "timestamp": None},
        }
        self.latest_robot = {
            "timestamp": None,
            "joint_state": None,
            "eef_pose": None,
            "eef_pose_commanded": None,
            "eef_twist_command": None,
            "tcp_wrench": None,
            "joint_torque": None,
            "gripper_position": None,
        }
        self._last_status_flush = 0.0
        self._last_preview_flush = {}
        self._episode_video_writers: dict[str, FFmpegVideoWriter] = {}
        self._episode_ffmpeg_bin: Optional[str] = None

        # Start cameras
        self.hand_serial = None
        self.wrist_serial = None
        self.external_serial = None
        self.hand_camera_name = "disabled"
        self.wrist_camera_name = "disabled"
        self.external_camera_name = "disabled"

        # Use explicit serials when provided. Start the two on-wrist/hand USB2
        # devices before the external camera to keep roles deterministic.
        self.realsense_hand = self.initialize_hand_camera(enforce_required=False)
        self.realsense_wrist = self.initialize_wrist_camera(enforce_required=False)
        self.realsense_ext = self.initialize_external_camera(enforce_required=False)

        if self.realsense_hand is None and self.realsense_wrist is None and self.realsense_ext is None:
            print("WARNING: No active RealSense camera. Live preview will stay blank until a camera is available.")

        if self.robot_enabled:
            self.initialize_robot()

        # Initialize data arrays
        self.clear_buffers()

    def _start_episode_video_writers(self, episode_folder: str) -> None:
        ffmpeg_bin = _resolve_ffmpeg_binary()
        episode_path = Path(episode_folder).resolve()
        writer_specs = []
        if self.realsense_hand is not None:
            writer_specs.extend(
                [
                    {
                        "key": "rgb_hand",
                        "filename": "rgb_hand.mp4",
                        "width": self.saved_frame_width,
                        "height": self.saved_frame_height,
                        "fps": self.fps_hand,
                        "input_pixel_format": "rgb24",
                        "codec": "libx264",
                        "output_pixel_format": "yuv420p",
                        "output_args": ["-preset", "ultrafast", "-crf", "20"],
                    },
                    {
                        "key": "depth_hand",
                        "filename": "depth_hand_raw.mkv",
                        "width": self.saved_frame_width,
                        "height": self.saved_frame_height,
                        "fps": self.fps_hand,
                        "input_pixel_format": "gray16le",
                        "codec": "ffv1",
                        "output_pixel_format": "gray16le",
                    },
                ]
            )
        if self.realsense_wrist is not None:
            writer_specs.extend(
                [
                    {
                        "key": "rgb_wrist",
                        "filename": "rgb_wrist.mp4",
                        "width": self.saved_frame_width,
                        "height": self.saved_frame_height,
                        "fps": self.fps_wrist,
                        "input_pixel_format": "rgb24",
                        "codec": "libx264",
                        "output_pixel_format": "yuv420p",
                        "output_args": ["-preset", "ultrafast", "-crf", "20"],
                    },
                    {
                        "key": "depth_wrist",
                        "filename": "depth_wrist_raw.mkv",
                        "width": self.saved_frame_width,
                        "height": self.saved_frame_height,
                        "fps": self.fps_wrist,
                        "input_pixel_format": "gray16le",
                        "codec": "ffv1",
                        "output_pixel_format": "gray16le",
                    },
                ]
            )
        if self.realsense_ext is not None:
            writer_specs.extend(
                [
                    {
                        "key": "rgb_external",
                        "filename": "rgb_external.mp4",
                        "width": self.saved_frame_width,
                        "height": self.saved_frame_height,
                        "fps": self.fps_ext,
                        "input_pixel_format": "rgb24",
                        "codec": "libx264",
                        "output_pixel_format": "yuv420p",
                        "output_args": ["-preset", "ultrafast", "-crf", "20"],
                    },
                    {
                        "key": "depth_external",
                        "filename": "depth_external_raw.mkv",
                        "width": self.saved_frame_width,
                        "height": self.saved_frame_height,
                        "fps": self.fps_ext,
                        "input_pixel_format": "gray16le",
                        "codec": "ffv1",
                        "output_pixel_format": "gray16le",
                    },
                ]
            )

        started: dict[str, FFmpegVideoWriter] = {}
        try:
            for spec in writer_specs:
                writer = FFmpegVideoWriter(
                    ffmpeg_bin=ffmpeg_bin,
                    output_path=episode_path / str(spec["filename"]),
                    width=int(spec["width"]),
                    height=int(spec["height"]),
                    fps=int(spec["fps"]),
                    input_pixel_format=str(spec["input_pixel_format"]),
                    label=str(spec["key"]),
                    codec=str(spec.get("codec", "ffv1")),
                    output_pixel_format=str(spec.get("output_pixel_format") or spec["input_pixel_format"]),
                    output_args=list(spec.get("output_args") or []),
                )
                started[str(spec["key"])] = writer
        except Exception:
            for writer in started.values():
                writer.abort()
            raise

        self._episode_ffmpeg_bin = ffmpeg_bin
        self._episode_video_writers = started

    def _close_episode_video_writers(self, abort: bool = False) -> None:
        writers = self._episode_video_writers
        self._episode_video_writers = {}
        self._episode_ffmpeg_bin = None
        errors = []
        for key in sorted(writers):
            writer = writers[key]
            try:
                if abort:
                    writer.abort()
                else:
                    writer.close()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    @staticmethod
    def _resize_frame_for_writer(frame: np.ndarray, writer: FFmpegVideoWriter) -> np.ndarray:
        expected_shape = (
            (writer.height, writer.width, 3)
            if writer.input_pixel_format == "rgb24"
            else (writer.height, writer.width)
        )
        if frame.shape == expected_shape:
            return frame
        interpolation = cv2.INTER_NEAREST if writer.input_pixel_format == "gray16le" else cv2.INTER_AREA
        resized = cv2.resize(frame, (writer.width, writer.height), interpolation=interpolation)
        return resized.astype(frame.dtype, copy=False)

    def _write_frame_to_writer(self, key: str, frame: np.ndarray) -> None:
        writer = self._episode_video_writers.get(key)
        if writer is not None:
            writer.write(self._resize_frame_for_writer(frame, writer))

    def _write_episode_video_frames(
        self,
        color_hand_image,
        depth_hand_image,
        color_wrist_image,
        depth_wrist_image,
        color_ext_image,
        depth_ext_image,
    ) -> None:
        if color_hand_image is not None and depth_hand_image is not None:
            self._write_frame_to_writer("rgb_hand", color_hand_image)
            self._write_frame_to_writer("depth_hand", depth_hand_image)
        if color_wrist_image is not None and depth_wrist_image is not None:
            self._write_frame_to_writer("rgb_wrist", color_wrist_image)
            self._write_frame_to_writer("depth_wrist", depth_wrist_image)
        if color_ext_image is not None and depth_ext_image is not None:
            self._write_frame_to_writer("rgb_external", color_ext_image)
            self._write_frame_to_writer("depth_external", depth_ext_image)

    def initialize_robot(self):
        """Initialize robot state receiver and optional gripper state reader."""
        if RTDEReceiveInterface is None:
            msg = (
                "Robot receiver backend is unavailable, cannot record robot state. "
                "Install ur_rtde for ROBOT_BACKEND=ur, or set XARM_CONTROLLER_PATH "
                "for ROBOT_BACKEND=xarm."
            )
            if self.allow_missing_robot:
                print(f"WARNING: {msg}")
                self.robot_enabled = False
                return
            raise RuntimeError(msg)

        try:
            self.rtde_r = RTDEReceiveInterface(self.robot_ip)
            print(
                f"Robot state receiver ready: backend={ROBOT_BACKEND}, "
                f"ip={self.robot_ip}, fps={self.robot_capture_fps}"
            )
        except Exception as exc:
            msg = f"Failed to connect robot receiver ({ROBOT_BACKEND}) to {self.robot_ip}: {exc}"
            if self.allow_missing_robot:
                print(f"WARNING: {msg}. Continue without robot recording.")
                self.robot_enabled = False
                self.rtde_r = None
                return
            raise RuntimeError(msg) from exc

        if not self.enable_gripper_state:
            return

        if self.gripper_state_source == "teleop_status":
            print(
                "Gripper state reader using SpaceMouse teleop status file: "
                f"{self.teleop_status_file}"
            )
            return

        if ROBOT_BACKEND == "xarm":
            if XArmSDKGripper is None:
                msg = "xArm gripper backend is unavailable; gripper state will be disabled."
                if self.allow_missing_gripper:
                    print(f"WARNING: {msg}")
                    self.enable_gripper_state = False
                    return
                raise RuntimeError(msg)
            try:
                self.gripper = XArmSDKGripper(self.robot_ip)
                print(f"xArm gripper state reader ready: {self.robot_ip}")
            except Exception as exc:
                msg = f"Failed to initialize xArm gripper at {self.robot_ip}: {exc}"
                if self.allow_missing_gripper:
                    print(f"WARNING: {msg}. Continue without gripper state.")
                    self.gripper = None
                    self.enable_gripper_state = False
                    return
                raise RuntimeError(msg) from exc
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
        if not self.enable_gripper_state:
            return np.nan
        if self.gripper_state_source == "teleop_status":
            status = self._load_teleop_status()
            gripper = status.get("gripper") if isinstance(status.get("gripper"), dict) else {}
            latest_position = gripper.get("latest_position")
            try:
                return float(latest_position)
            except (TypeError, ValueError):
                return np.nan
        if self.gripper is None:
            return np.nan
        try:
            return float(self.gripper.get_current_position())
        except Exception:
            return np.nan

    def _load_teleop_status(self) -> dict:
        if self.teleop_status_file is None or not self.teleop_status_file.is_file():
            return {}
        try:
            payload = json.loads(self.teleop_status_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _command_vector(data, expected_len: int = 6) -> np.ndarray:
        if not isinstance(data, (list, tuple)) or len(data) != expected_len:
            return np.full((expected_len,), np.nan, dtype=np.float64)
        values = []
        for item in data:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                values.append(np.nan)
        return np.asarray(values, dtype=np.float64)

    def _get_teleop_command_observations(self) -> tuple[np.ndarray, np.ndarray]:
        status = self._load_teleop_status()
        robot_latest = status.get("robot", {}).get("latest", {}) if isinstance(status.get("robot"), dict) else {}
        spacemouse = status.get("spacemouse", {}) if isinstance(status.get("spacemouse"), dict) else {}
        eef_pose_commanded = self._command_vector(robot_latest.get("eef_pose_commanded"))
        eef_twist_command = self._command_vector(
            robot_latest.get("eef_twist_command", spacemouse.get("motion_state"))
        )
        return eef_pose_commanded, eef_twist_command

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

    def stop(self):
        """Stop background loops and release hardware handles."""
        self.stop_event.set()
        self._close_episode_video_writers(abort=True)
        for pipeline in (self.realsense_hand, self.realsense_wrist, self.realsense_ext):
            if pipeline is None:
                continue
            try:
                pipeline.stop()
            except Exception:
                pass
        if self.gripper is not None and hasattr(self.gripper, "disconnect"):
            try:
                self.gripper.disconnect()
            except Exception:
                pass

    def _set_last_error(self, message: str):
        self.last_error = message

    @staticmethod
    def _depth_preview(depth_image):
        depth = np.asarray(depth_image)
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size == 0:
            return np.zeros(depth.shape[:2], dtype=np.uint8)
        lo, hi = np.percentile(valid, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    def _write_preview_image(self, path: Path, image) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        if not cv2.imwrite(str(tmp_path), image):
            return
        tmp_path.replace(path)

    def _write_latest_preview(self, camera_role, rgb_image, depth_image, timestamp):
        if preview_dir is None:
            return
        last_flush = self._last_preview_flush.get(camera_role, 0.0)
        if timestamp is not None and (timestamp - last_flush) < 0.1:
            return
        now = time.time()
        self._last_preview_flush[camera_role] = timestamp if timestamp is not None else now
        try:
            if rgb_image is not None:
                self._write_preview_image(
                    preview_dir / f"{camera_role}_rgb.jpg",
                    cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
                )
            if depth_image is not None:
                self._write_preview_image(
                    preview_dir / f"{camera_role}_depth.png",
                    self._depth_preview(depth_image),
                )
        except Exception as exc:
            self._set_last_error(f"Failed to write live preview: {exc}")

    def _clear_live_preview(self, camera_role):
        with self.latest_lock:
            self.latest_camera[camera_role] = {"rgb": None, "depth": None, "timestamp": None}
        if preview_dir is None:
            return
        for kind, suffix in (("rgb", "jpg"), ("depth", "png")):
            path = preview_dir / f"{camera_role}_{kind}.{suffix}"
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except Exception:
                pass

    def _update_latest_camera(self, camera_role, rgb_image, depth_image, timestamp):
        with self.latest_lock:
            self.latest_camera[camera_role] = {
                "rgb": rgb_image.copy() if rgb_image is not None else None,
                "depth": depth_image.copy() if depth_image is not None else None,
                "timestamp": timestamp,
            }
        self._write_latest_preview(camera_role, rgb_image, depth_image, timestamp)

    def _update_latest_robot(
        self,
        timestamp_robot,
        joint_state,
        eef_pose,
        eef_pose_commanded,
        eef_twist_command,
        tcp_wrench,
        joint_torque,
        gripper_position,
    ):
        with self.latest_lock:
            self.latest_robot = {
                "timestamp": timestamp_robot,
                "joint_state": joint_state.copy(),
                "eef_pose": eef_pose.copy(),
                "eef_pose_commanded": eef_pose_commanded.copy(),
                "eef_twist_command": eef_twist_command.copy(),
                "tcp_wrench": tcp_wrench.copy(),
                "joint_torque": joint_torque.copy(),
                "gripper_position": float(gripper_position),
            }

    def get_latest_frame(self, camera_role: str, frame_kind: str):
        """Return a copy of the latest live frame for web preview."""
        with self.latest_lock:
            camera = self.latest_camera.get(camera_role, {})
            frame = camera.get(frame_kind)
            timestamp = camera.get("timestamp")
            return (frame.copy() if frame is not None else None), timestamp

    def get_latest_robot_snapshot(self):
        """Return JSON-serializable latest robot state for web status."""
        def normalize(value):
            if isinstance(value, np.ndarray):
                return [normalize(item) for item in value.tolist()]
            if isinstance(value, (float, np.floating)):
                return float(value) if np.isfinite(value) else None
            return value

        with self.latest_lock:
            return {
                key: normalize(value)
                for key, value in self.latest_robot.items()
            }

    def _camera_status(self, role: str, pipeline, name: str, serial: Optional[str]):
        latest = self.latest_camera.get(role, {})
        latest_timestamp = latest.get("timestamp")
        available = (
            latest_timestamp is not None
            and (time.time() - float(latest_timestamp)) <= STREAM_STALE_AFTER_S
        )
        connected = pipeline is not None
        status = "streaming" if available else "connected" if connected else "disconnected"
        if role == "hand":
            width = int(self.width_hand)
            height = int(self.height_hand)
            fps = int(self.fps_hand)
        elif role == "wrist":
            width = int(self.width_wrist)
            height = int(self.height_wrist)
            fps = int(self.fps_wrist)
        else:
            width = int(self.width_ext)
            height = int(self.height_ext)
            fps = int(self.fps_ext)
        return {
            "name": name if connected else "missing",
            "serial": serial or "",
            "requested": True,
            "enabled": connected,
            "connected": connected,
            "available": available,
            "status": status,
            "latest_timestamp": latest_timestamp,
            "width": width,
            "height": height,
            "fps": fps,
        }

    def _robot_status(self):
        latest = self.get_latest_robot_snapshot()
        connected = bool(self.robot_enabled and self.rtde_r is not None)
        latest_timestamp = latest.get("timestamp")
        available = (
            latest_timestamp is not None
            and (time.time() - float(latest_timestamp)) <= STREAM_STALE_AFTER_S
        )
        status = "streaming" if available else "connected" if connected else "disconnected"
        return {
            "requested": self.robot_requested,
            "enabled": connected,
            "connected": connected,
            "available": available,
            "status": status,
            "ip": self.robot_ip or "",
            "fps": self.robot_capture_fps if self.robot_requested else 0,
            "gripper_state_enabled": bool(self.enable_gripper_state),
            "latest": latest,
        }

    def _gripper_status(self):
        latest = self.get_latest_robot_snapshot()
        gripper_position = latest.get("gripper_position")
        latest_timestamp = latest.get("timestamp")
        connected = self.gripper is not None or self.gripper_state_source == "teleop_status"
        try:
            gripper_position_value = float(gripper_position)
            gripper_position_valid = np.isfinite(gripper_position_value)
        except (TypeError, ValueError):
            gripper_position_valid = False
        available = (
            gripper_position_valid
            and latest_timestamp is not None
            and (time.time() - float(latest_timestamp)) <= STREAM_STALE_AFTER_S
        )
        status = "streaming" if available else "connected" if connected else "disconnected"
        return {
            "requested": self.gripper_requested,
            "enabled": connected,
            "connected": connected,
            "available": available,
            "status": status,
            "ip": self.robot_ip or "",
            "port": self.gripper_port,
            "source": self.gripper_state_source,
            "latest_position": gripper_position,
        }

    def get_input_status_payload(self):
        return {
            "hand_camera": self._camera_status("hand", self.realsense_hand, self.hand_camera_name, self.hand_serial),
            "wrist_camera": self._camera_status("wrist", self.realsense_wrist, self.wrist_camera_name, self.wrist_serial),
            "external_camera": self._camera_status("external", self.realsense_ext, self.external_camera_name, self.external_serial),
            "robot": self._robot_status(),
            "gripper": self._gripper_status(),
        }

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

    @staticmethod
    def _camera_profile_candidates(width: int, height: int, fps: int) -> list[tuple[int, int, int]]:
        requested = (int(width), int(height), int(fps))
        candidates = [requested]
        for fallback_fps in RGB_FPS_FALLBACKS:
            candidates.append((requested[0], requested[1], fallback_fps))
        candidates.extend(RGB_CANDIDATES)
        unique: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for item in candidates:
            if item[0] <= 0 or item[1] <= 0 or item[2] <= 0:
                continue
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def _supported_rgb_modes(self, serial: str) -> set[tuple[int, int, int]]:
        modes: set[tuple[int, int, int]] = set()
        context = rs.context()
        target = None
        for device in context.query_devices():
            try:
                if device.get_info(rs.camera_info.serial_number) == serial:
                    target = device
                    break
            except Exception:
                continue
        if target is None:
            return modes

        for sensor in target.query_sensors():
            for profile in sensor.get_stream_profiles():
                try:
                    if str(profile.stream_type()) != "stream.color":
                        continue
                    video = profile.as_video_stream_profile()
                    modes.add((int(video.width()), int(video.height()), int(profile.fps())))
                except Exception:
                    continue
        return modes

    def _start_pipeline_with_retry(
        self,
        camera_role: str,
        serial: str,
        allow_missing: bool,
        color_width: int,
        color_height: int,
        fps: int,
        device_info: Optional[dict] = None,
    ):
        """Start RealSense pipeline with retry and profile fallback."""
        supported = self._supported_rgb_modes(serial)
        requested = (int(color_width), int(color_height), int(fps))
        candidates = self._camera_profile_candidates(color_width, color_height, fps)
        if supported:
            candidates = [item for item in candidates if item in supported]
            if requested not in candidates:
                same_resolution = sorted(
                    item for item in candidates if item[0] == requested[0] and item[1] == requested[1]
                )
                if same_resolution:
                    print(
                        f"WARNING: {camera_role} camera ({serial}) does not advertise requested "
                        f"profile {requested[0]}x{requested[1]}@{requested[2]}; trying same resolution "
                        f"with available fps: {same_resolution}"
                    )
                else:
                    print(
                        f"WARNING: {camera_role} camera ({serial}) does not advertise requested "
                        f"resolution {requested[0]}x{requested[1]}@{requested[2]}; falling back to {candidates[:5]}"
                    )

        if not candidates:
            msg = (
                f"No supported RGB profile found for {camera_role} camera ({serial}). "
                f"Requested {color_width}x{color_height}@{fps}."
            )
            if allow_missing:
                print(f"WARNING: {msg} Continue without {camera_role} camera.")
                return None, None
            raise RuntimeError(msg)

        failures: list[str] = []
        total_attempts = self.camera_start_retries + 1

        for profile_width, profile_height, profile_fps in candidates:
            profile_name = f"{profile_width}x{profile_height}@{profile_fps}"
            depth_width, depth_height = self._preferred_depth_resolution(
                camera_role,
                device_info,
                profile_width,
                profile_height,
            )
            reset_attempted = False

            for attempt in range(1, total_attempts + 1):
                pipeline, config = self._build_pipeline_config(
                    serial,
                    profile_width,
                    profile_height,
                    depth_width,
                    depth_height,
                    profile_fps,
                )
                try:
                    pipeline.start(config)
                    return pipeline, {
                        "color_width": profile_width,
                        "color_height": profile_height,
                        "depth_width": depth_width,
                        "depth_height": depth_height,
                        "fps": profile_fps,
                    }
                except RuntimeError as exc:
                    err_str = str(exc)
                    busy = ("Device or resource busy" in err_str) or ("errno=16" in err_str)
                    missing_video_node = (
                        "map_device_descriptor Cannot open '/dev/video" in err_str
                        and "No such file or directory" in err_str
                    )

                    should_retry = (busy or missing_video_node) and attempt < total_attempts
                    if should_retry:
                        short_err = err_str.strip().replace("\n", " ")
                        if len(short_err) > 160:
                            short_err = short_err[:157] + "..."
                        print(
                            f"WARNING: {camera_role} camera ({serial}) profile {profile_name} not ready, "
                            f"retry {attempt}/{self.camera_start_retries} in {self.camera_start_retry_delay:.2f}s... "
                            f"reason: {short_err}"
                        )
                        if self.camera_busy_reset and not reset_attempted:
                            reset_attempted = self._hardware_reset_camera(serial, camera_role)
                        time.sleep(self.camera_start_retry_delay)
                        continue

                    msg = f"Failed to start {camera_role} camera ({serial}) with profile {profile_name}: {exc}"
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

                    failures.append(msg)
                    if busy:
                        # Busy is not profile-specific; trying additional profiles is unlikely to help.
                        detail = "\n  - " + "\n  - ".join(failures[-5:])
                        final_msg = (
                            f"Failed to start {camera_role} camera ({serial}) because device is busy.{detail}"
                        )
                        if allow_missing:
                            print(f"WARNING: {final_msg} Continue without {camera_role} camera.")
                            return None, None
                        raise RuntimeError(final_msg)
                    break

        detail = "\n  - " + "\n  - ".join(failures[-5:])
        msg = f"Failed to start {camera_role} camera ({serial}) after profile fallbacks.{detail}"
        if allow_missing:
            print(f"WARNING: {msg} Continue without {camera_role} camera.")
            return None, None
        raise RuntimeError(msg)

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
        global color_wrist_array, depth_wrist_array, timestamp_wrist_array
        global color_ext_array, depth_ext_array, timestamp_ext_array
        global robot_timestamp_array, joint_state_array, eef_pose_array
        global eef_pose_commanded_array, eef_twist_command_array
        global tcp_wrench_array, joint_torque_array, gripper_position_array
        global subtask_timestamp_array
        
        color_hand_array = []
        depth_hand_array = []
        timestamp_hand_array = []

        color_wrist_array = []
        depth_wrist_array = []
        timestamp_wrist_array = []
        
        color_ext_array = []
        depth_ext_array = []
        timestamp_ext_array = []

        robot_timestamp_array = []
        joint_state_array = []
        eef_pose_array = []
        eef_pose_commanded_array = []
        eef_twist_command_array = []
        tcp_wrench_array = []
        joint_torque_array = []
        gripper_position_array = []
        subtask_timestamp_array = []

    def record_subtask_label(self) -> float:
        """Record a subtask label timestamp for the current episode."""
        global subtask_timestamp_array

        label_timestamp = time.time()
        with self.buffer_lock:
            subtask_timestamp_array.append(label_timestamp)
        return label_timestamp

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
        realsense, selected_profile = self._start_pipeline_with_retry(
            camera_role="hand",
            serial=selected_serial,
            allow_missing=(self.allow_missing_hand or not enforce_required),
            color_width=self.width_hand,
            color_height=self.height_hand,
            fps=self.fps_hand,
            device_info=selected_info,
        )
        if realsense is None:
            return None

        if selected_profile is not None:
            self.width_hand = selected_profile["color_width"]
            self.height_hand = selected_profile["color_height"]
            self.depth_width_hand = selected_profile["depth_width"]
            self.depth_height_hand = selected_profile["depth_height"]
            self.fps_hand = selected_profile["fps"]

        if selected_info is None:
            self.hand_camera_name = "hand_camera"
        else:
            self.hand_camera_name = f"{selected_info['name']} (pid={selected_info['pid']})"
        print(
            f"Hand camera connected: {self.hand_camera_name}, serial={selected_serial}, "
            f"color={self.width_hand}x{self.height_hand}@{self.fps_hand}, "
            f"depth={self.depth_width_hand}x{self.depth_height_hand}@{self.fps_hand}"
        )
        
        return realsense

    def initialize_wrist_camera(self, enforce_required: bool = False):
        """Initialize wrist camera (default: D405; can be overridden by serial)."""
        selected_serial = None
        selected_info = None

        if self.wrist_serial_override:
            selected_serial = self.wrist_serial_override
            selected_info = self._find_device_by_serial(selected_serial)
            if selected_info is None:
                msg = (
                    f"Wrist camera serial '{selected_serial}' not found. "
                    f"Connected RealSense devices: {self._describe_realsense_devices()}"
                )
                if self.allow_missing_wrist or not enforce_required:
                    print(f"WARNING: {msg}. Continue without wrist camera.")
                    return None
                raise RuntimeError(msg)
        else:
            selected_serial = self._find_device_by_product_ids(
                self.wrist_product_ids,
                exclude_serial=self.hand_serial,
            )
            if selected_serial is None:
                msg = (
                    f"Wrist camera not found (expected PIDs: {sorted(self.wrist_product_ids)}). "
                    f"Connected RealSense devices: {self._describe_realsense_devices()}"
                )
                if self.allow_missing_wrist or not enforce_required:
                    print(f"WARNING: {msg}. Continue without wrist camera.")
                    return None
                raise RuntimeError(msg)
            selected_info = self._find_device_by_serial(selected_serial)

        self.wrist_serial = selected_serial
        realsense, selected_profile = self._start_pipeline_with_retry(
            camera_role="wrist",
            serial=selected_serial,
            allow_missing=(self.allow_missing_wrist or not enforce_required),
            color_width=self.width_wrist,
            color_height=self.height_wrist,
            fps=self.fps_wrist,
            device_info=selected_info,
        )
        if realsense is None:
            return None

        if selected_profile is not None:
            self.width_wrist = selected_profile["color_width"]
            self.height_wrist = selected_profile["color_height"]
            self.depth_width_wrist = selected_profile["depth_width"]
            self.depth_height_wrist = selected_profile["depth_height"]
            self.fps_wrist = selected_profile["fps"]

        if selected_info is None:
            self.wrist_camera_name = "wrist_camera"
        else:
            self.wrist_camera_name = f"{selected_info['name']} (pid={selected_info['pid']})"
        print(
            f"Wrist camera connected: {self.wrist_camera_name}, serial={selected_serial}, "
            f"color={self.width_wrist}x{self.height_wrist}@{self.fps_wrist}, "
            f"depth={self.depth_width_wrist}x{self.depth_height_wrist}@{self.fps_wrist}"
        )

        return realsense

    def initialize_external_camera(self, enforce_required: bool = False):
        """Initialize external camera (default: D405; can be overridden by serial)."""
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
        realsense, selected_profile = self._start_pipeline_with_retry(
            camera_role="external",
            serial=selected_serial,
            allow_missing=(self.allow_missing_external or not enforce_required),
            color_width=self.width_ext,
            color_height=self.height_ext,
            fps=self.fps_ext,
            device_info=selected_info,
        )
        if realsense is None:
            return None

        if selected_profile is not None:
            self.width_ext = selected_profile["color_width"]
            self.height_ext = selected_profile["color_height"]
            self.depth_width_ext = selected_profile["depth_width"]
            self.depth_height_ext = selected_profile["depth_height"]
            self.fps_ext = selected_profile["fps"]

        if selected_info is None:
            self.external_camera_name = "external_camera"
        else:
            self.external_camera_name = f"{selected_info['name']} (pid={selected_info['pid']})"
        print(
            f"External camera connected: {self.external_camera_name}, serial={selected_serial}, "
            f"color={self.width_ext}x{self.height_ext}@{self.fps_ext}, "
            f"depth={self.depth_width_ext}x{self.depth_height_ext}@{self.fps_ext}"
        )
        
        return realsense

    def ensure_recording_cameras_ready(self):
        """Allow partial init, but require configured cameras before recording starts."""
        issues = []

        if self.realsense_ext is None:
            try:
                self.realsense_ext = self.initialize_external_camera(enforce_required=not self.allow_missing_external)
            except Exception as exc:
                issues.append(str(exc))
        if self.realsense_hand is None:
            try:
                self.realsense_hand = self.initialize_hand_camera(enforce_required=not self.allow_missing_hand)
            except Exception as exc:
                issues.append(str(exc))
        if self.realsense_wrist is None:
            try:
                self.realsense_wrist = self.initialize_wrist_camera(enforce_required=not self.allow_missing_wrist)
            except Exception as exc:
                issues.append(str(exc))

        if not self.allow_missing_hand and self.realsense_hand is None:
            issues.append("Hand camera is required to start recording.")
        if not self.allow_missing_wrist and self.realsense_wrist is None:
            issues.append("Wrist camera is required to start recording.")
        if not self.allow_missing_external and self.realsense_ext is None:
            issues.append("External camera is required to start recording.")

        if issues:
            detail = " ".join(dict.fromkeys(issues))
            self._set_last_error(detail)
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

    def _get_frames_nonblocking(self, pipeline):
        """Read frames without stalling the whole recorder loop on a slow camera."""
        capture = pipeline.poll_for_frames()
        if capture is not None:
            return capture
        try:
            return pipeline.wait_for_frames(timeout_ms=FRAME_WAIT_TIMEOUT_MS)
        except RuntimeError as exc:
            if self._is_transient_frame_error(exc) or "Frame didn't arrive within" in str(exc):
                return None
            raise

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
            if color_frame is None:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).copy() if depth_frame is not None else None
        except RuntimeError as exc:
            if self._is_transient_frame_error(exc):
                return None, None
            raise

        if color_image.size == 0:
            return None, None
        if depth_image is not None and depth_image.size == 0:
            depth_image = None

        return cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_image

    # Function to get visual observations (RGB-D only) 
    def get_visual_obs_hand(self):
        """Get visual observations from D405 hand camera"""
        if self.realsense_hand is None:
            return None, None
        capture_hand = self._get_frames_nonblocking(self.realsense_hand)
        return self._extract_rgbd_images(capture_hand)

    def get_visual_obs_wrist(self):
        """Get visual observations from D405 wrist camera"""
        if self.realsense_wrist is None:
            return None, None
        capture_wrist = self._get_frames_nonblocking(self.realsense_wrist)
        return self._extract_rgbd_images(capture_wrist)

    def get_visual_obs_external(self):
        """Get visual observations from D405 external camera"""
        if self.realsense_ext is None:
            return None, None
        capture_ext = self._get_frames_nonblocking(self.realsense_ext)
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
                info = self._find_device_by_serial(self.hand_serial)
                pipeline, selected_profile = self._start_pipeline_with_retry(
                    camera_role="hand",
                    serial=self.hand_serial,
                    allow_missing=self.allow_missing_hand,
                    color_width=self.width_hand,
                    color_height=self.height_hand,
                    fps=self.fps_hand,
                    device_info=info,
                )
                self.realsense_hand = pipeline
                if selected_profile is not None:
                    self.width_hand = selected_profile["color_width"]
                    self.height_hand = selected_profile["color_height"]
                    self.depth_width_hand = selected_profile["depth_width"]
                    self.depth_height_hand = selected_profile["depth_height"]
                    self.fps_hand = selected_profile["fps"]
                return pipeline is not None
            except Exception as exc:
                print(f"WARNING: hand camera reconnect exception: {exc}")
                self.realsense_hand = None
                return False

        if camera_role == "wrist":
            if self.wrist_serial is None:
                return False
            try:
                if self.realsense_wrist is not None:
                    self.realsense_wrist.stop()
            except Exception:
                pass

            try:
                info = self._find_device_by_serial(self.wrist_serial)
                pipeline, selected_profile = self._start_pipeline_with_retry(
                    camera_role="wrist",
                    serial=self.wrist_serial,
                    allow_missing=self.allow_missing_wrist,
                    color_width=self.width_wrist,
                    color_height=self.height_wrist,
                    fps=self.fps_wrist,
                    device_info=info,
                )
                self.realsense_wrist = pipeline
                if selected_profile is not None:
                    self.width_wrist = selected_profile["color_width"]
                    self.height_wrist = selected_profile["color_height"]
                    self.depth_width_wrist = selected_profile["depth_width"]
                    self.depth_height_wrist = selected_profile["depth_height"]
                    self.fps_wrist = selected_profile["fps"]
                return pipeline is not None
            except Exception as exc:
                print(f"WARNING: wrist camera reconnect exception: {exc}")
                self.realsense_wrist = None
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
                info = self._find_device_by_serial(self.external_serial)
                pipeline, selected_profile = self._start_pipeline_with_retry(
                    camera_role="external",
                    serial=self.external_serial,
                    allow_missing=self.allow_missing_external,
                    color_width=self.width_ext,
                    color_height=self.height_ext,
                    fps=self.fps_ext,
                    device_info=info,
                )
                self.realsense_ext = pipeline
                if selected_profile is not None:
                    self.width_ext = selected_profile["color_width"]
                    self.height_ext = selected_profile["color_height"]
                    self.depth_width_ext = selected_profile["depth_width"]
                    self.depth_height_ext = selected_profile["depth_height"]
                    self.fps_ext = selected_profile["fps"]
                return pipeline is not None
            except Exception as exc:
                err = str(exc)
                if (
                    "Device or resource busy" in err
                    and (self.realsense_hand is not None or self.realsense_wrist is not None)
                ):
                    # Workaround for sporadic cross-camera metadata node contention:
                    # restart wrist/hand after external reconnect succeeds.
                    try:
                        if self.realsense_hand is not None:
                            self.realsense_hand.stop()
                    except Exception:
                        pass
                    try:
                        if self.realsense_wrist is not None:
                            self.realsense_wrist.stop()
                    except Exception:
                        pass
                    self.realsense_hand = None
                    self.realsense_wrist = None
                    try:
                        info = self._find_device_by_serial(self.external_serial)
                        pipeline, selected_profile = self._start_pipeline_with_retry(
                            camera_role="external",
                            serial=self.external_serial,
                            allow_missing=self.allow_missing_external,
                            color_width=self.width_ext,
                            color_height=self.height_ext,
                            fps=self.fps_ext,
                            device_info=info,
                        )
                        self.realsense_ext = pipeline
                        if selected_profile is not None:
                            self.width_ext = selected_profile["color_width"]
                            self.height_ext = selected_profile["color_height"]
                            self.depth_width_ext = selected_profile["depth_width"]
                            self.depth_height_ext = selected_profile["depth_height"]
                            self.fps_ext = selected_profile["fps"]
                        if self.realsense_hand is None:
                            self.realsense_hand = self.initialize_hand_camera(
                                enforce_required=not self.allow_missing_hand
                            )
                        if self.realsense_wrist is None:
                            self.realsense_wrist = self.initialize_wrist_camera(
                                enforce_required=not self.allow_missing_wrist
                            )
                        return pipeline is not None
                    except Exception as retry_exc:
                        print(
                            "WARNING: external reconnect retry after hand restart failed: "
                            f"{retry_exc}"
                        )
                print(f"WARNING: external camera reconnect exception: {exc}")
                self.realsense_ext = None
                return False

        return False

    def _handle_command(self, command: str):
        global recording, current_time, current_episode_folder, last_saved_episode, status_saving

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
                try:
                    self._start_episode_video_writers(current_episode_folder)
                except Exception as exc:
                    shutil.rmtree(current_episode_folder, ignore_errors=True)
                    current_episode_folder = None
                    current_time = None
                    print(f"Cannot start video writers: {exc}")
                    _write_status(self, message=str(exc), last_error=str(exc))
                    return True

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

        if cmd == 'l':
            if not recording:
                _write_status(self, message="Start recording before labeling a subtask.")
                return True

            label_timestamp = self.record_subtask_label()
            label_index = len(subtask_timestamp_array)
            print(f"Recorded subtask label {label_index} at {label_timestamp:.3f}")
            _write_status(
                self,
                message=f"Recorded subtask label {label_index} at {label_timestamp:.3f}.",
                last_error="",
            )
            return True

        if cmd == 's':
            if recording:
                print(f"\nStopping recording episode {current_time}...")
                recording = False
                status_saving = True
                _write_status(self, message=f"Stopping recording episode {current_time}...", last_error="")
                time.sleep(0.05)
                try:
                    self.save_data(current_episode_folder)
                    last_saved_episode = current_episode_folder
                finally:
                    status_saving = False

                print(f"Episode {current_time} saved successfully!")
                print(f"Hand camera frames: {len(timestamp_hand_array)}")
                print(f"Wrist camera frames: {len(timestamp_wrist_array)}")
                print(f"External camera frames: {len(timestamp_ext_array)}")
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
                status_saving = True
                _write_status(self, message="Stopping current recording before quitting...")
                time.sleep(0.05)
                try:
                    self.save_data(current_episode_folder)
                    last_saved_episode = current_episode_folder
                finally:
                    status_saving = False
            print("Quitting session...")
            _write_status(self, message="Recorder shut down.", last_error="")
            self.stop_event.set()
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

        print(f"Unknown command: {raw}")
        _write_status(self, message=f"Unknown command: {raw}")
        return True

    # Function to handle key presses
    def on_press(self, key):
        try:
            return self._handle_command(key.char)
        except AttributeError:
            return None

    def stdin_command_loop(self):
        print("Control mode: stdin (c=start, l=label subtask, s=stop/save, d=delete, q=quit)")
        while not self.stop_event.is_set():
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

        while not self.stop_event.is_set():
            start_time = time.time()

            try:
                # Get visual observations from available cameras. This runs even
                # while not recording so the web UI can show live inputs.
                color_hand_image, depth_hand_image, timestamp_hand = None, None, None
                color_wrist_image, depth_wrist_image, timestamp_wrist = None, None, None
                color_ext_image, depth_ext_image, timestamp_ext = None, None, None
                had_recovery_attempt = False

                if self.realsense_hand is not None:
                    try:
                        color_hand_image, depth_hand_image = self.get_visual_obs_hand()
                        if color_hand_image is not None:
                            timestamp_hand = time.time()
                            self._update_latest_camera("hand", color_hand_image, depth_hand_image, timestamp_hand)
                    except Exception as hand_exc:
                        if self._is_runtime_reconnect_error(hand_exc):
                            print(f"\nWARNING: hand camera runtime error: {hand_exc}")
                            self._set_last_error(str(hand_exc))
                            recovered = self._recover_camera_runtime("hand")
                            had_recovery_attempt = True
                            if recovered:
                                print("INFO: hand camera recovered.")
                            else:
                                print("WARNING: hand camera recovery failed.")
                                self._clear_live_preview("hand")
                        else:
                            raise
                else:
                    self._clear_live_preview("hand")

                if self.realsense_wrist is not None:
                    try:
                        color_wrist_image, depth_wrist_image = self.get_visual_obs_wrist()
                        if color_wrist_image is not None:
                            timestamp_wrist = time.time()
                            self._update_latest_camera("wrist", color_wrist_image, depth_wrist_image, timestamp_wrist)
                    except Exception as wrist_exc:
                        if self._is_runtime_reconnect_error(wrist_exc):
                            print(f"\nWARNING: wrist camera runtime error: {wrist_exc}")
                            self._set_last_error(str(wrist_exc))
                            recovered = self._recover_camera_runtime("wrist")
                            had_recovery_attempt = True
                            if recovered:
                                print("INFO: wrist camera recovered.")
                            else:
                                print("WARNING: wrist camera recovery failed.")
                                self._clear_live_preview("wrist")
                        else:
                            raise
                else:
                    self._clear_live_preview("wrist")

                if self.realsense_ext is not None:
                    try:
                        color_ext_image, depth_ext_image = self.get_visual_obs_external()
                        if color_ext_image is not None:
                            timestamp_ext = time.time()
                            self._update_latest_camera("external", color_ext_image, depth_ext_image, timestamp_ext)
                    except Exception as ext_exc:
                        if self._is_runtime_reconnect_error(ext_exc):
                            print(f"\nWARNING: external camera runtime error: {ext_exc}")
                            self._set_last_error(str(ext_exc))
                            recovered = self._recover_camera_runtime("external")
                            had_recovery_attempt = True
                            if recovered:
                                print("INFO: external camera recovered.")
                            else:
                                print("WARNING: external camera recovery failed.")
                                self._clear_live_preview("external")
                        else:
                            raise
                else:
                    self._clear_live_preview("external")

                if color_hand_image is None and color_wrist_image is None and color_ext_image is None:
                    if had_recovery_attempt:
                        continue
                    now = time.time()
                    if status_file is not None and (now - self._last_status_flush) >= 0.5:
                        _write_status(self)
                        self._last_status_flush = now
                    elapsed_time = time.time() - start_time
                    time_to_sleep = interval - elapsed_time
                    if time_to_sleep > 0:
                        time.sleep(time_to_sleep)
                    continue

                if recording:
                    self._write_episode_video_frames(
                        color_hand_image,
                        depth_hand_image,
                        color_wrist_image,
                        depth_wrist_image,
                        color_ext_image,
                        depth_ext_image,
                    )
                    self.synchronize_data(
                        color_hand_image, depth_hand_image, timestamp_hand,
                        color_wrist_image, depth_wrist_image, timestamp_wrist,
                        color_ext_image, depth_ext_image, timestamp_ext,
                    )
                    status_parts = []
                    if self.realsense_hand is not None:
                        status_parts.append(f"Hand: {len(timestamp_hand_array)}")
                    if self.realsense_wrist is not None:
                        status_parts.append(f"Wrist: {len(timestamp_wrist_array)}")
                    if self.realsense_ext is not None:
                        status_parts.append(f"External: {len(timestamp_ext_array)}")
                    print(" | ".join(status_parts), end='\r')

                now = time.time()
                if status_file is not None and (now - self._last_status_flush) >= 0.5:
                    _write_status(self)
                    self._last_status_flush = now

            except Exception as e:
                self._set_last_error(str(e))
                _write_status(self, message=f"Error during capture: {e}", last_error=str(e))
                if recording:
                    print(f"\nError during capture: {e}")

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

        while not self.stop_event.is_set():
            start_time = time.time()

            try:
                joint_state, eef_pose, tcp_wrench, joint_torque, gripper_position = self.get_robot_observations()
                eef_pose_commanded, eef_twist_command = self._get_teleop_command_observations()
                timestamp_robot = time.time()
                self._update_latest_robot(
                    timestamp_robot,
                    joint_state,
                    eef_pose,
                    eef_pose_commanded,
                    eef_twist_command,
                    tcp_wrench,
                    joint_torque,
                    gripper_position,
                )
                if recording:
                    self.synchronize_robot_data(
                        timestamp_robot,
                        joint_state,
                        eef_pose,
                        eef_pose_commanded,
                        eef_twist_command,
                        tcp_wrench,
                        joint_torque,
                        gripper_position,
                    )
                now = time.time()
                if status_file is not None and (now - self._last_status_flush) >= 0.5:
                    _write_status(self)
                    self._last_status_flush = now
            except Exception as exc:
                self._set_last_error(str(exc))
                _write_status(self, message=f"Error during robot capture: {exc}", last_error=str(exc))
                if recording:
                    print(f"\nError during robot capture: {exc}")

            elapsed_time = time.time() - start_time
            time_to_sleep = interval - elapsed_time

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def synchronize_data(
        self,
        color_hand_image, depth_hand_image, timestamp_hand,
        color_wrist_image, depth_wrist_image, timestamp_wrist,
        color_ext_image, depth_ext_image, timestamp_ext,
    ):
        """Add data to buffers"""
        with self.buffer_lock:
            if color_hand_image is not None and depth_hand_image is not None and timestamp_hand is not None:
                timestamp_hand_array.append(timestamp_hand)

            if color_wrist_image is not None and depth_wrist_image is not None and timestamp_wrist is not None:
                timestamp_wrist_array.append(timestamp_wrist)

            if color_ext_image is not None and depth_ext_image is not None and timestamp_ext is not None:
                timestamp_ext_array.append(timestamp_ext)

    def synchronize_robot_data(
        self,
        timestamp_robot,
        joint_state,
        eef_pose,
        eef_pose_commanded,
        eef_twist_command,
        tcp_wrench,
        joint_torque,
        gripper_position,
    ):
        """Add robot state sample to buffers."""
        with self.buffer_lock:
            robot_timestamp_array.append(timestamp_robot)
            joint_state_array.append(joint_state)
            eef_pose_array.append(eef_pose)
            eef_pose_commanded_array.append(eef_pose_commanded)
            eef_twist_command_array.append(eef_twist_command)
            tcp_wrench_array.append(tcp_wrench)
            joint_torque_array.append(joint_torque)
            gripper_position_array.append(gripper_position)

    def save_data(self, episode_folder):
        """Save timestamps and metadata after closing episode video writers."""
        self._close_episode_video_writers()
        with self.buffer_lock:
            timestamps_hand = np.array(timestamp_hand_array)
            timestamps_wrist = np.array(timestamp_wrist_array)

            timestamps_ext = np.array(timestamp_ext_array)

            timestamps_robot = np.array(robot_timestamp_array)
            joint_state = np.array(joint_state_array)
            eef_pose = np.array(eef_pose_array)
            eef_pose_commanded = np.array(eef_pose_commanded_array)
            eef_twist_command = np.array(eef_twist_command_array)
            tcp_wrench = np.array(tcp_wrench_array)
            joint_torque = np.array(joint_torque_array)
            gripper_position = np.array(gripper_position_array)
            subtask_timestamps = np.array(subtask_timestamp_array, dtype=float)

        print(f"\nSaving episode metadata and non-vision arrays...")

        timestamp_hand_file = os.path.join(episode_folder, "timestamps_hand.npy")
        timestamp_wrist_file = os.path.join(episode_folder, "timestamps_wrist.npy")
        timestamp_ext_file = os.path.join(episode_folder, "timestamps_external.npy")

        # Robot files
        robot_timestamp_file = os.path.join(episode_folder, "timestamps_robot.npy")
        joint_state_file = os.path.join(episode_folder, "joint_state.npy")
        eef_pose_file = os.path.join(episode_folder, "eef_pose.npy")
        eef_pose_commanded_file = os.path.join(episode_folder, "eef_pose_commanded.npy")
        eef_twist_command_file = os.path.join(episode_folder, "eef_twist_command.npy")
        tcp_wrench_file = os.path.join(episode_folder, "tcp_wrench.npy")
        joint_torque_file = os.path.join(episode_folder, "joint_torque.npy")
        gripper_position_file = os.path.join(episode_folder, "gripper_position.npy")
        subtask_timestamp_file = os.path.join(episode_folder, "subtask_timestamps.npy")
        has_teleop_command_source = self.teleop_status_file is not None

        if len(timestamps_hand) > 0:
            np.save(timestamp_hand_file, timestamps_hand)

        if len(timestamps_wrist) > 0:
            np.save(timestamp_wrist_file, timestamps_wrist)

        if len(timestamps_ext) > 0:
            np.save(timestamp_ext_file, timestamps_ext)

        if len(robot_timestamp_array) > 0:
            np.save(robot_timestamp_file, timestamps_robot)
            np.save(joint_state_file, joint_state)
            np.save(eef_pose_file, eef_pose)
            if has_teleop_command_source:
                np.save(eef_pose_commanded_file, eef_pose_commanded)
                np.save(eef_twist_command_file, eef_twist_command)
            np.save(tcp_wrench_file, tcp_wrench)
            np.save(joint_torque_file, joint_torque)
            np.save(gripper_position_file, gripper_position)

        np.save(subtask_timestamp_file, subtask_timestamps)
        
        # Save metadata as JSON
        metadata = {
            'instruction': instruction,
            'timestamp': current_time,
            'hand_camera': self.hand_camera_name if self.realsense_hand is not None else 'disabled',
            'wrist_camera': self.wrist_camera_name if self.realsense_wrist is not None else 'disabled',
            'external_camera': self.external_camera_name if self.realsense_ext is not None else 'disabled',
            'hand_serial': self.hand_serial if self.hand_serial is not None else '',
            'wrist_serial': self.wrist_serial if self.wrist_serial is not None else '',
            'external_serial': self.external_serial if self.external_serial is not None else '',
            'frequency': self.capture_fps,
            'camera_mode': (
                'triple'
                if sum(cam is not None for cam in (self.realsense_hand, self.realsense_wrist, self.realsense_ext)) >= 3
                else 'dual'
                if sum(cam is not None for cam in (self.realsense_hand, self.realsense_wrist, self.realsense_ext)) == 2
                else 'single'
            ),
            'color_order': 'rgb',
            'hand_frames': len(timestamps_hand),
            'wrist_frames': len(timestamps_wrist),
            'external_frames': len(timestamps_ext),
            'hand_start_time': float(timestamps_hand[0]) if len(timestamps_hand) > 0 else 0.0,
            'hand_end_time': float(timestamps_hand[-1]) if len(timestamps_hand) > 0 else 0.0,
            'hand_duration': float(timestamps_hand[-1] - timestamps_hand[0]) if len(timestamps_hand) > 0 else 0.0,
            'wrist_start_time': float(timestamps_wrist[0]) if len(timestamps_wrist) > 0 else 0.0,
            'wrist_end_time': float(timestamps_wrist[-1]) if len(timestamps_wrist) > 0 else 0.0,
            'wrist_duration': float(timestamps_wrist[-1] - timestamps_wrist[0]) if len(timestamps_wrist) > 0 else 0.0,
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
            'gripper_state_source': self.gripper_state_source if self.enable_gripper_state else '',
            'joint_torque_method': self.robot_joint_torque_method,
            'teleop_command_source': str(self.teleop_status_file) if has_teleop_command_source else '',
            'subtask_timestamps': subtask_timestamps.tolist(),
            'subtask_labels': int(len(subtask_timestamps)),
            'vision_storage': 'video',
        }
        vision_files = {}
        if len(timestamps_hand) > 0:
            vision_files["rgb_hand"] = _vision_entry(
                path="rgb_hand.mp4",
                storage="mp4",
                codec="h264",
                shape=(len(timestamps_hand), self.saved_frame_height, self.saved_frame_width, 3),
                dtype="uint8",
                fps=self.fps_hand,
                channel_order="rgb",
            )
            vision_files["depth_hand"] = _vision_entry(
                path="depth_hand_raw.mkv",
                storage="mkv",
                codec="ffv1",
                shape=(len(timestamps_hand), self.saved_frame_height, self.saved_frame_width),
                dtype="uint16",
                fps=self.fps_hand,
                verified_lossless=True,
            )
        if len(timestamps_wrist) > 0:
            vision_files["rgb_wrist"] = _vision_entry(
                path="rgb_wrist.mp4",
                storage="mp4",
                codec="h264",
                shape=(len(timestamps_wrist), self.saved_frame_height, self.saved_frame_width, 3),
                dtype="uint8",
                fps=self.fps_wrist,
                channel_order="rgb",
            )
            vision_files["depth_wrist"] = _vision_entry(
                path="depth_wrist_raw.mkv",
                storage="mkv",
                codec="ffv1",
                shape=(len(timestamps_wrist), self.saved_frame_height, self.saved_frame_width),
                dtype="uint16",
                fps=self.fps_wrist,
                verified_lossless=True,
            )
        if len(timestamps_ext) > 0:
            vision_files["rgb_external"] = _vision_entry(
                path="rgb_external.mp4",
                storage="mp4",
                codec="h264",
                shape=(len(timestamps_ext), self.saved_frame_height, self.saved_frame_width, 3),
                dtype="uint8",
                fps=self.fps_ext,
                channel_order="rgb",
            )
            vision_files["depth_external"] = _vision_entry(
                path="depth_external_raw.mkv",
                storage="mkv",
                codec="ffv1",
                shape=(len(timestamps_ext), self.saved_frame_height, self.saved_frame_width),
                dtype="uint16",
                fps=self.fps_ext,
                verified_lossless=True,
            )
        metadata['vision_files'] = vision_files
        metadata['vision_codec_tool'] = "webapp/record_multi_camera_npy_web.py"
        
        metadata_file = os.path.join(episode_folder, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        if len(timestamps_hand) > 0:
            print(f"Saved hand camera frames: {len(timestamps_hand)}")
            print(f"  RGB MP4: {os.path.join(episode_folder, 'rgb_hand.mp4')}")
            print(f"  Depth: {os.path.join(episode_folder, 'depth_hand_raw.mkv')}")
            print(f"  Timestamps: {timestamp_hand_file}")
        else:
            print("Hand camera disabled or no frames captured.")

        if len(timestamps_wrist) > 0:
            print(f"\nSaved wrist camera frames: {len(timestamps_wrist)}")
            print(f"  RGB MP4: {os.path.join(episode_folder, 'rgb_wrist.mp4')}")
            print(f"  Depth: {os.path.join(episode_folder, 'depth_wrist_raw.mkv')}")
            print(f"  Timestamps: {timestamp_wrist_file}")
        else:
            print("\nWrist camera disabled or no frames captured.")

        if len(timestamps_ext) > 0:
            print(f"\nSaved external camera frames: {len(timestamps_ext)}")
            print(f"  RGB MP4: {os.path.join(episode_folder, 'rgb_external.mp4')}")
            print(f"  Depth: {os.path.join(episode_folder, 'depth_external_raw.mkv')}")
            print(f"  Timestamps: {timestamp_ext_file}")
        else:
            print("\nExternal camera disabled or no frames captured.")

        if len(robot_timestamp_array) > 0:
            print(f"\nSaved robot state frames: {len(robot_timestamp_array)}")
            print(f"  Joint: {joint_state_file}")
            print(f"  EEF Pose (actual): {eef_pose_file}")
            if has_teleop_command_source:
                print(f"  EEF Pose (commanded): {eef_pose_commanded_file}")
                print(f"  EEF Twist Command: {eef_twist_command_file}")
            print(f"  TCP Wrench: {tcp_wrench_file}")
            print(f"  Joint Torque: {joint_torque_file}")
            print(f"  Gripper Position: {gripper_position_file}")
            print(f"  Timestamps: {robot_timestamp_file}")
        else:
            print("\nRobot state disabled or no frames captured.")
        print(f"\nSaved subtask labels: {len(subtask_timestamps)}")
        print(f"  Timestamps: {subtask_timestamp_file}")
        print(f"\nMetadata: {metadata_file}")


def parse_pid_csv(pid_csv: str) -> Set[str]:
    return {pid.strip().upper() for pid in pid_csv.split(',') if pid.strip()}


def main():
    # Global variables for controlling the recording state
    global data_folder, recording, instruction, current_episode_folder, current_time
    global color_hand_array, depth_hand_array, timestamp_hand_array
    global color_wrist_array, depth_wrist_array, timestamp_wrist_array
    global color_ext_array, depth_ext_array, timestamp_ext_array
    global robot_timestamp_array, joint_state_array, eef_pose_array
    global eef_pose_commanded_array, eef_twist_command_array
    global tcp_wrench_array, joint_torque_array, gripper_position_array
    global subtask_timestamp_array
    global status_file, preview_dir, status_message, status_last_error, status_saving
    global last_saved_episode, delete_requires_confirmation, record_root
    
    parser = argparse.ArgumentParser(description="Record synchronized D405 + D405 RGBD episodes.")
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
        "--wrist-serial",
        default=None,
        help="Force wrist camera serial (overwrite auto-detection).",
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
        "--allow-missing-wrist",
        action="store_true",
        help="Allow running without wrist camera.",
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
        "--wrist-product-ids",
        default="0B5B",
        help="Comma-separated wrist camera product IDs for auto-detection.",
    )
    parser.add_argument(
        "--external-product-ids",
        default="0B5B",
        help="Comma-separated external camera product IDs for auto-detection.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Capture frequency for both cameras.",
    )
    parser.add_argument(
        "--hand-width",
        type=int,
        default=640,
        help="Hand camera RGB/depth width.",
    )
    parser.add_argument(
        "--hand-height",
        type=int,
        default=480,
        help="Hand camera RGB/depth height.",
    )
    parser.add_argument(
        "--wrist-width",
        type=int,
        default=640,
        help="Wrist camera RGB/depth width.",
    )
    parser.add_argument(
        "--wrist-height",
        type=int,
        default=480,
        help="Wrist camera RGB/depth height.",
    )
    parser.add_argument(
        "--external-width",
        type=int,
        default=848,
        help="External camera RGB/depth width.",
    )
    parser.add_argument(
        "--external-height",
        type=int,
        default=480,
        help="External camera RGB/depth height.",
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
        default=os.environ.get("XARM_ROBOT_IP", os.environ.get("UR_ROBOT_IP", "")),
        help="Robot controller IP for high-frequency robot state recording (empty disables robot recording).",
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
        "--teleop-status-file",
        default="",
        help="Optional SpaceMouse teleop status JSON path for commanded pose/twist recording.",
    )
    parser.add_argument(
        "--no-delete-confirm",
        action="store_true",
        help="Delete latest episode without y/n confirmation prompt.",
    )
    args = parser.parse_args()

    status_file = Path(args.status_file).expanduser().resolve() if args.status_file else None
    preview_dir = status_file.with_name(f"{status_file.stem}_previews") if status_file is not None else None
    status_message = "Recorder process starting..."
    status_last_error = ""
    status_saving = False
    last_saved_episode = None
    delete_requires_confirmation = not args.no_delete_confirm
    _write_status(None, message=status_message, last_error="")

    recording = False
    color_hand_array = []
    depth_hand_array = []
    timestamp_hand_array = []
    color_wrist_array = []
    depth_wrist_array = []
    timestamp_wrist_array = []
    color_ext_array = []
    depth_ext_array = []
    timestamp_ext_array = []
    robot_timestamp_array = []
    joint_state_array = []
    eef_pose_array = []
    eef_pose_commanded_array = []
    eef_twist_command_array = []
    tcp_wrench_array = []
    joint_torque_array = []
    gripper_position_array = []
    subtask_timestamp_array = []
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
        wrist_serial_override=args.wrist_serial,
        external_serial_override=args.external_serial,
        allow_missing_hand=args.allow_missing_hand,
        allow_missing_wrist=args.allow_missing_wrist,
        allow_missing_external=args.allow_missing_external,
        hand_product_ids=parse_pid_csv(args.hand_product_ids),
        wrist_product_ids=parse_pid_csv(args.wrist_product_ids),
        external_product_ids=parse_pid_csv(args.external_product_ids),
        capture_fps=args.fps,
        hand_width=args.hand_width,
        hand_height=args.hand_height,
        wrist_width=args.wrist_width,
        wrist_height=args.wrist_height,
        external_width=args.external_width,
        external_height=args.external_height,
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
        teleop_status_file=args.teleop_status_file,
    )
    _write_status(recorder, message=f"Recorder initialized. Save path: {data_folder}", last_error="")

    print("\n" + "="*50)
    print("MULTI CAMERA RECORDING SYSTEM")
    print(f"Hand camera: {recorder.hand_camera_name} | serial={recorder.hand_serial}")
    print(f"Wrist camera: {recorder.wrist_camera_name} | serial={recorder.wrist_serial}")
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
    print("  'l' - Label subtask timestamp in the current episode")
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
        if keyboard is None:
            raise RuntimeError("pynput keyboard listener is unavailable; use --control-mode stdin instead.")
        listener = keyboard.Listener(on_press=recorder.on_press)
        listener.start()
        listener.join()

    recorder.stop()
    _write_status(recorder, message="Recorder stopped.")
    print("\nStopped dual camera recording system.")


if __name__ == '__main__':
    main()
