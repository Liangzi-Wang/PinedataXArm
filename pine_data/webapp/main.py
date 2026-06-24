from __future__ import annotations

import argparse
import concurrent.futures
import functools
import importlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from episode_codec import require_ffmpeg as _require_ffmpeg_binary


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RECORD_ROOT = Path(os.getenv("DATAFOUNDRY_RECORD_ROOT", BASE_DIR.parent / "recordings")).resolve()
DATA_DIR = Path(os.getenv("DATAFOUNDRY_DATA_DIR", DEFAULT_RECORD_ROOT)).resolve()
DEFAULT_DATASET = os.getenv("DATAFOUNDRY_DEFAULT_DATASET", "")
STATIC_DIR = BASE_DIR / "static"
LOGO_DIR = BASE_DIR / "logo"

app = FastAPI(title="DataFoundry Viewer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/logo", StaticFiles(directory=LOGO_DIR), name="logo")

_SUMMARY_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}
PSEUDO_DATASET_NAME = "recordings"
RECORDING_BACKEND_MODE = os.getenv("RECORDING_BACKEND", "embedded").strip().lower()
if RECORDING_BACKEND_MODE not in {"embedded", "tmux"}:
    RECORDING_BACKEND_MODE = "embedded"

CAMERA_STREAM_STALE_AFTER_S = 1.0
ROBOT_STREAM_STALE_AFTER_S = 1.0
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
REPLAY_VIDEO_CHUNK_SIZE = max(1, int(os.getenv("DATAFOUNDRY_REPLAY_VIDEO_CHUNK_SIZE", "64")))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def _normalize_instruction(raw_instruction: str) -> str:
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


REPLAY_PREFETCH_ENABLED = _env_bool("DATAFOUNDRY_REPLAY_PREFETCH", True)
REPLAY_PREFETCH_WORKERS = _env_int("DATAFOUNDRY_REPLAY_PREFETCH_WORKERS", 1, minimum=1)
_REPLAY_PREFETCH_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = (
    concurrent.futures.ThreadPoolExecutor(
        max_workers=REPLAY_PREFETCH_WORKERS,
        thread_name_prefix="replay-prefetch",
    )
    if REPLAY_PREFETCH_ENABLED
    else None
)
_REPLAY_PREFETCH_PENDING: set[tuple[str, int, int, str, str, int]] = set()
_REPLAY_PREFETCH_LOCK = threading.Lock()


class RecordingInitRequest(BaseModel):
    pipeline: str = Field(default="spacemouse", pattern="^spacemouse$")
    root: str = Field(default=str(DEFAULT_RECORD_ROOT))
    instruction: str = Field(default="untitled")
    hand_serial: str = os.getenv("HAND_SERIAL", "218622270687")
    wrist_serial: str = os.getenv("WRIST_SERIAL", "409122273232")
    external_serial: str = os.getenv("EXTERNAL_SERIAL", "409122274280")
    allow_missing_hand: bool = _env_bool("ALLOW_MISSING_HAND", False)
    allow_missing_wrist: bool = _env_bool("ALLOW_MISSING_WRIST", False)
    allow_missing_external: bool = _env_bool("ALLOW_MISSING_EXTERNAL", False)
    hand_product_ids: str = "0B5B"
    wrist_product_ids: str = "0B5B"
    external_product_ids: str = "0B5B"
    fps: int = Field(default=15, ge=1, le=60)
    hand_width: int = Field(default=640, ge=160, le=4096)
    hand_height: int = Field(default=480, ge=120, le=4096)
    wrist_width: int = Field(default=640, ge=160, le=4096)
    wrist_height: int = Field(default=480, ge=120, le=4096)
    external_width: int = Field(default=848, ge=160, le=4096)
    external_height: int = Field(default=480, ge=120, le=4096)
    camera_start_retries: int = Field(default=20, ge=0, le=200)
    camera_start_retry_delay: float = Field(default=0.5, ge=0.0, le=30.0)
    camera_busy_reset: bool = True
    camera_post_reset_wait: float = Field(default=2.0, ge=0.0, le=60.0)
    robot_ip: str = os.getenv("XARM_ROBOT_IP", os.getenv("UR_ROBOT_IP", os.getenv("ROBOT_IP", "")))
    robot_fps: int = Field(default=200, ge=1, le=1000)
    enable_gripper_state: bool = _env_bool("ENABLE_GRIPPER_STATE", True)
    gripper_port: int = Field(default=63352, ge=1, le=65535)
    allow_missing_robot: bool = _env_bool("ALLOW_MISSING_ROBOT", False)
    allow_missing_gripper: bool = _env_bool("ALLOW_MISSING_GRIPPER", True)
    subtask_segment_index: int = Field(default=0, ge=0, le=999)
    subtask_reset_noise_xyz_m: float = Field(default=0.01, ge=0.0, le=0.10)


class RecordingStartRequest(BaseModel):
    instruction: str = Field(default="")


class RecordingResetRequest(BaseModel):
    subtask_segment_index: int = Field(default=0, ge=0, le=999)
    subtask_reset_noise_xyz_m: float = Field(default=0.01, ge=0.0, le=0.10)


class EpisodeDeleteRequest(BaseModel):
    dataset: str = Field(default=PSEUDO_DATASET_NAME)
    episode: str = Field(default="")


class RecordingManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.module = None
        self.recorder = None
        self.camera_thread: threading.Thread | None = None
        self.robot_thread: threading.Thread | None = None
        self.config: RecordingInitRequest | None = None
        self.data_folder: Path | None = None
        self.last_saved_episode: str | None = None
        self.last_message = "Recorder is not initialized."
        self.is_saving = False

    def _load_module(self):
        if self.module is None:
            if str(BASE_DIR) not in sys.path:
                sys.path.insert(0, str(BASE_DIR))
            self.module = importlib.import_module("record_multi_camera_npy_web")
        return self.module

    @staticmethod
    def _safe_instruction(raw_instruction: str) -> str:
        return _normalize_instruction(raw_instruction)

    def initialize(self, config: RecordingInitRequest) -> dict[str, Any]:
        with self.lock:
            if self.recorder is not None:
                self.shutdown(save_current=False)

            module = self._load_module()
            instruction = self._safe_instruction(config.instruction)
            data_folder = Path(config.root).expanduser().resolve() / time.strftime("%Y%m%d") / instruction / "camera_npy"
            data_folder.mkdir(parents=True, exist_ok=True)

            module.data_folder = str(data_folder)
            module.recording = False
            module.instruction = instruction
            module.current_episode_folder = None
            module.current_time = None

            recorder = module.CameraRecorder(
                hand_serial_override=config.hand_serial.strip() or None,
                wrist_serial_override=config.wrist_serial.strip() or None,
                external_serial_override=config.external_serial.strip() or None,
                allow_missing_hand=config.allow_missing_hand,
                allow_missing_wrist=config.allow_missing_wrist,
                allow_missing_external=config.allow_missing_external,
                hand_product_ids=module.parse_pid_csv(config.hand_product_ids),
                wrist_product_ids=module.parse_pid_csv(config.wrist_product_ids),
                external_product_ids=module.parse_pid_csv(config.external_product_ids),
                capture_fps=config.fps,
                hand_width=config.hand_width,
                hand_height=config.hand_height,
                wrist_width=config.wrist_width,
                wrist_height=config.wrist_height,
                external_width=config.external_width,
                external_height=config.external_height,
                camera_start_retries=config.camera_start_retries,
                camera_start_retry_delay=config.camera_start_retry_delay,
                camera_busy_reset=config.camera_busy_reset,
                camera_post_reset_wait=config.camera_post_reset_wait,
                robot_ip=config.robot_ip.strip() or None,
                robot_capture_fps=config.robot_fps,
                enable_gripper_state=config.enable_gripper_state,
                gripper_port=config.gripper_port,
                allow_missing_robot=config.allow_missing_robot,
                allow_missing_gripper=config.allow_missing_gripper,
            )

            self.recorder = recorder
            self.config = config
            self.data_folder = data_folder
            self.last_saved_episode = None
            self.last_message = f"Recorder initialized. Saving to {data_folder}"

            self.camera_thread = threading.Thread(target=recorder.synchronized_capture, daemon=True)
            self.camera_thread.start()
            if recorder.robot_enabled:
                self.robot_thread = threading.Thread(target=recorder.synchronized_robot_capture, daemon=True)
                self.robot_thread.start()
            else:
                self.robot_thread = None

            return self.status()

    def require_ready(self):
        if self.recorder is None or self.module is None:
            raise HTTPException(status_code=409, detail="Recorder is not initialized.")
        return self.module, self.recorder

    def _update_instruction(self, module: Any, raw_instruction: str) -> str:
        instruction = self._safe_instruction(raw_instruction)
        if self.config is not None:
            self.config.instruction = instruction
            root = Path(self.config.root).expanduser().resolve()
        else:
            root = DEFAULT_RECORD_ROOT

        data_folder = root / time.strftime("%Y%m%d") / instruction / "camera_npy"
        data_folder.mkdir(parents=True, exist_ok=True)

        module.instruction = instruction
        module.data_folder = str(data_folder)
        self.data_folder = data_folder
        return instruction

    def start_episode(self, instruction: str | None = None) -> dict[str, Any]:
        with self.lock:
            module, recorder = self.require_ready()
            if getattr(module, "recording", False):
                self.last_message = "Already recording."
                return self.status()

            requested_instruction = (instruction or "").strip()
            if requested_instruction:
                active_instruction = self._update_instruction(module, requested_instruction)
            else:
                active_instruction = str(getattr(module, "instruction", "untitled"))

            try:
                recorder.ensure_recording_cameras_ready()
            except RuntimeError as exc:
                detail = str(exc)
                self.last_message = detail
                raise HTTPException(status_code=409, detail=detail) from exc

            recorder.clear_buffers()
            module.current_time = time.strftime("%Y%m%d%H%M%S")
            episode_folder = Path(module.data_folder) / module.current_time
            episode_folder.mkdir(parents=True, exist_ok=True)
            module.current_episode_folder = str(episode_folder)
            module.recording = True
            self.last_message = f"Started recording episode {module.current_time}. Instruction: {active_instruction}."
            return self.status()

    def stop_episode(self) -> dict[str, Any]:
        with self.lock:
            module, recorder = self.require_ready()
            if not getattr(module, "recording", False):
                self.last_message = "Recorder is already paused."
                return self.status()

            module.recording = False
            time.sleep(0.05)
            episode_folder = getattr(module, "current_episode_folder", None)
            if episode_folder is None:
                self.last_message = "Recording stopped, but no episode folder was active."
                return self.status()

            self.is_saving = True
            try:
                recorder.save_data(episode_folder)
                self.last_saved_episode = episode_folder
                self.last_message = f"Saved episode {getattr(module, 'current_time', '')}."
            finally:
                self.is_saving = False
            return self.status()

    def reset_pose(self, reset_config: RecordingResetRequest | None = None) -> dict[str, Any]:
        raise HTTPException(
            status_code=409,
            detail="Reset is only supported when using the SpaceMouse tmux pipeline.",
        )

    def label_subtask(self) -> dict[str, Any]:
        with self.lock:
            module, recorder = self.require_ready()
            if not getattr(module, "recording", False):
                self.last_message = "Start recording before labeling a subtask."
                return self.status()

            label_timestamp = recorder.record_subtask_label()
            labels = getattr(module, "subtask_timestamp_array", [])
            label_index = len(labels) if hasattr(labels, "__len__") else 1
            self.last_message = f"Recorded subtask label {label_index} at {label_timestamp:.3f}."
            return self.status()

    def delete_last_episode(self) -> dict[str, Any]:
        with self.lock:
            module, _recorder = self.require_ready()
            if getattr(module, "recording", False):
                raise HTTPException(status_code=409, detail="Stop recording before deleting an episode.")

            target = self.last_saved_episode or getattr(module, "current_episode_folder", None)
            if not target:
                self.last_message = "No episode is available to delete."
                return self.status()

            target_path = Path(target)
            if target_path.exists():
                shutil.rmtree(target_path)
                self.last_message = f"Deleted episode folder {target_path.name}."
            else:
                self.last_message = "Episode folder was already missing."
            if self.last_saved_episode == target:
                self.last_saved_episode = None
            return self.status()

    def shutdown(self, save_current: bool = True) -> dict[str, Any]:
        with self.lock:
            if self.recorder is None or self.module is None:
                self.last_message = "Recorder is not initialized."
                return self.status()

            module = self.module
            if save_current and getattr(module, "recording", False):
                self.stop_episode()
            else:
                module.recording = False

            self.recorder.stop()
            for worker in (self.camera_thread, self.robot_thread):
                if worker is not None:
                    worker.join(timeout=2.0)

            self.recorder = None
            self.camera_thread = None
            self.robot_thread = None
            self.last_message = "Recorder shut down."
            return self.status()

    def status(self) -> dict[str, Any]:
        with self.lock:
            module = self.module
            recorder = self.recorder
            initialized = recorder is not None and module is not None
            counts = {
                "hand_frames": len(getattr(module, "timestamp_hand_array", [])) if module else 0,
                "wrist_frames": len(getattr(module, "timestamp_wrist_array", [])) if module else 0,
                "external_frames": len(getattr(module, "timestamp_ext_array", [])) if module else 0,
                "robot_frames": len(getattr(module, "robot_timestamp_array", [])) if module else 0,
            }
            input_status = recorder.get_input_status_payload() if recorder is not None else {}
            return {
                "initialized": initialized,
                "recording": bool(getattr(module, "recording", False)) if module else False,
                "saving": self.is_saving,
                "message": self.last_message,
                "data_folder": str(self.data_folder) if self.data_folder else "",
                "current_episode": getattr(module, "current_time", None) if module else None,
                "current_episode_folder": getattr(module, "current_episode_folder", None) if module else None,
                "last_saved_episode": self.last_saved_episode,
                "counts": counts,
                "hand_camera": input_status.get(
                    "hand_camera",
                    {
                        "requested": True,
                        "enabled": False,
                        "connected": False,
                        "available": False,
                        "status": "disconnected",
                        "width": self.config.hand_width if self.config is not None else 640,
                        "height": self.config.hand_height if self.config is not None else 480,
                    },
                ),
                "wrist_camera": input_status.get(
                    "wrist_camera",
                    {
                        "requested": True,
                        "enabled": False,
                        "connected": False,
                        "available": False,
                        "status": "disconnected",
                        "width": self.config.wrist_width if self.config is not None else 640,
                        "height": self.config.wrist_height if self.config is not None else 480,
                    },
                ),
                "external_camera": input_status.get(
                    "external_camera",
                    {
                        "requested": True,
                        "enabled": False,
                        "connected": False,
                        "available": False,
                        "status": "disconnected",
                        "width": self.config.external_width if self.config is not None else 848,
                        "height": self.config.external_height if self.config is not None else 480,
                    },
                ),
                "robot": input_status.get(
                    "robot",
                    {"requested": False, "enabled": False, "connected": False, "available": False, "status": "disconnected", "latest": {}},
                ),
                "gripper": input_status.get(
                    "gripper",
                    {"requested": False, "enabled": False, "connected": False, "available": False, "status": "disconnected"},
                ),
                "spacemouse": {
                    "requested": False,
                    "connected": False,
                    "status": "disconnected",
                    "text": "SpaceMouse pipeline is not active.",
                    "lines": [],
                },
                "last_error": recorder.last_error if recorder is not None else "",
                "config": self.config.model_dump() if self.config is not None else RecordingInitRequest().model_dump(),
                "backend": "embedded",
                "live_preview_supported": True,
            }

    def latest_frame(self, camera: str, kind: str) -> tuple[np.ndarray, float | None]:
        with self.lock:
            _module, recorder = self.require_ready()
            frame, timestamp = recorder.get_latest_frame(camera, kind)
            if frame is None:
                raise HTTPException(status_code=404, detail=f"No live {camera} {kind} frame available yet.")
            return frame, timestamp


class TmuxRecordingManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.config: RecordingInitRequest | None = None
        self.last_message = "Tmux recorder is not initialized."
        self.last_error = ""
        self.last_saved_episode: str | None = None
        self.window_name = os.getenv("TMUX_RECORDING_WINDOW", "main")
        self.status_lines = self._env_int("TMUX_RECORDING_STATUS_LINES", 8, minimum=1)
        default_camera_env = os.getenv("CAMERA_ENV") or os.getenv("WEBAPP_ENV") or str(BASE_DIR.parent / "data_record_env")
        self.camera_env = Path(default_camera_env).expanduser().resolve()
        self.camera_script = Path(
            os.getenv("TMUX_RECORDING_CAMERA_SCRIPT", str(BASE_DIR / "record_multi_camera_npy_web.py"))
        ).expanduser().resolve()
        self.pipeline_profiles = self._build_pipeline_profiles()
        self.active_pipeline = "spacemouse"
        self.shutdown_save_in_progress = False
        self.shutdown_save_thread: threading.Thread | None = None
        self._set_active_pipeline(self.active_pipeline)

    def _build_pipeline_profiles(self) -> dict[str, dict[str, Any]]:
        return {
            "spacemouse": {
                "session_name": os.getenv(
                    "TMUX_SPACEMOUSE_RECORDING_SESSION",
                    os.getenv("TMUX_RECORDING_SESSION", "pine_spacemouse_record"),
                ),
                "script_path": Path(
                    os.getenv(
                        "TMUX_SPACEMOUSE_RECORDING_SCRIPT",
                        os.getenv("TMUX_RECORDING_SCRIPT", str(BASE_DIR / "tmux_spacemouse_record_web.sh")),
                    )
                ).expanduser().resolve(),
                "camera_pane_index": os.getenv("TMUX_SPACEMOUSE_RECORDING_CAMERA_PANE", "1"),
                "teleop_pane_index": os.getenv("TMUX_SPACEMOUSE_RECORDING_TELEOP_PANE", "0"),
                "pane_labels": {
                    os.getenv("TMUX_SPACEMOUSE_RECORDING_TELEOP_PANE", "0"): "spacemouse",
                    os.getenv("TMUX_SPACEMOUSE_RECORDING_CAMERA_PANE", "1"): "camera",
                },
            },
        }

    def _set_active_pipeline(self, pipeline: str) -> None:
        profile = self.pipeline_profiles[pipeline]
        self.active_pipeline = pipeline
        self.session_name = str(profile["session_name"])
        self.camera_pane_index = str(profile["camera_pane_index"])
        self.teleop_pane_index = str(profile.get("teleop_pane_index", ""))
        self.pane_labels = dict(profile["pane_labels"])
        self.script_path = Path(profile["script_path"]).expanduser().resolve()
        default_status = BASE_DIR / ".runtime" / f"{self.session_name}_status.json"
        self.status_file = Path(str(profile.get("status_file", default_status))).expanduser().resolve()
        default_teleop_status = BASE_DIR / ".runtime" / f"{self.session_name}_teleop_status.json"
        self.teleop_status_file = Path(str(profile.get("teleop_status_file", default_teleop_status))).expanduser().resolve()
        self.preview_dir = self.status_file.with_name(f"{self.status_file.stem}_previews")
        if not hasattr(self, "shutdown_save_in_progress"):
            self.shutdown_save_in_progress = False
        if not hasattr(self, "shutdown_save_thread"):
            self.shutdown_save_thread = None

    def _detect_running_pipeline(self) -> str | None:
        for pipeline in [self.active_pipeline, *[name for name in self.pipeline_profiles if name != self.active_pipeline]]:
            profile = self.pipeline_profiles[pipeline]
            proc = self._run_tmux("has-session", "-t", str(profile["session_name"]))
            if proc.returncode == 0:
                return pipeline
        return None

    def _other_running_pipeline(self, target_pipeline: str) -> str | None:
        for pipeline, profile in self.pipeline_profiles.items():
            if pipeline == target_pipeline:
                continue
            proc = self._run_tmux("has-session", "-t", str(profile["session_name"]))
            if proc.returncode == 0:
                return pipeline
        return None

    def _camera_target(self) -> str:
        return f"{self.session_name}:{self.window_name}.{self.camera_pane_index}"

    def _teleop_target(self) -> str | None:
        if not self.teleop_pane_index:
            return None
        return f"{self.session_name}:{self.window_name}.{self.teleop_pane_index}"

    @staticmethod
    def _env_int(name: str, default: int, minimum: int | None = None) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return max(minimum, value) if minimum is not None else value

    @staticmethod
    def _status_value(data: dict[str, Any], key: str, default: Any) -> Any:
        value = data.get(key)
        return default if value is None else value

    @staticmethod
    def _command_error(proc: subprocess.CompletedProcess[str], fallback: str) -> str:
        return proc.stderr.strip() or proc.stdout.strip() or fallback

    def _run_tmux(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["tmux", *args], capture_output=True, text=True)

    def _session_exists(self) -> bool:
        return self._run_tmux("has-session", "-t", self.session_name).returncode == 0

    def _pane_current_command(self, pane_index: str) -> str:
        proc = self._run_tmux(
            "display-message",
            "-p",
            "-t",
            f"{self.session_name}:{self.window_name}.{pane_index}",
            "#{pane_current_command}",
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()

    def _camera_recorder_is_running(self) -> bool:
        command = self._pane_current_command(self.camera_pane_index)
        return bool(command) and command not in {"bash", "zsh", "sh", "fish"}

    def _build_camera_launch_command(self, config: RecordingInitRequest) -> str:
        args = [
            str(self.camera_script),
            "--root",
            str(Path(config.root).expanduser().resolve()),
            "--fps",
            str(config.fps),
            "--hand-width",
            str(config.hand_width),
            "--hand-height",
            str(config.hand_height),
            "--wrist-width",
            str(config.wrist_width),
            "--wrist-height",
            str(config.wrist_height),
            "--external-width",
            str(config.external_width),
            "--external-height",
            str(config.external_height),
            "--instruction",
            self._safe_instruction(config.instruction),
            "--control-mode",
            "stdin",
            "--status-file",
            str(self.status_file),
            "--teleop-status-file",
            str(self.teleop_status_file),
            "--no-delete-confirm",
            "--hand-product-ids",
            config.hand_product_ids,
            "--wrist-product-ids",
            config.wrist_product_ids,
            "--external-product-ids",
            config.external_product_ids,
            "--camera-start-retries",
            str(config.camera_start_retries),
            "--camera-start-retry-delay",
            str(config.camera_start_retry_delay),
            "--camera-post-reset-wait",
            str(config.camera_post_reset_wait),
            "--camera-busy-reset" if config.camera_busy_reset else "--no-camera-busy-reset",
        ]
        if config.hand_serial.strip():
            args.extend(["--hand-serial", config.hand_serial.strip()])
        if config.wrist_serial.strip():
            args.extend(["--wrist-serial", config.wrist_serial.strip()])
        if config.external_serial.strip():
            args.extend(["--external-serial", config.external_serial.strip()])
        if config.allow_missing_hand:
            args.append("--allow-missing-hand")
        if config.allow_missing_wrist:
            args.append("--allow-missing-wrist")
        if config.allow_missing_external:
            args.append("--allow-missing-external")
        robot_ip = config.robot_ip.strip()
        if robot_ip:
            args.extend([
                "--robot-ip",
                robot_ip,
                "--robot-fps",
                str(config.robot_fps),
                "--gripper-port",
                str(config.gripper_port),
            ])
            if not config.enable_gripper_state:
                args.append("--no-enable-gripper-state")
            if config.allow_missing_robot:
                args.append("--allow-missing-robot")
            if not config.allow_missing_gripper:
                args.append("--no-allow-missing-gripper")

        return (
            f"source {shlex.quote(str(self.camera_env / 'bin' / 'activate'))} && "
            f"cd {shlex.quote(str(BASE_DIR))} && "
            f"python {shlex.join(args)}"
        )

    def _restart_camera_recorder(self, config: RecordingInitRequest) -> None:
        if not self._session_exists():
            raise HTTPException(status_code=409, detail="Tmux recording session is not initialized.")
        if not self.camera_script.is_file():
            raise HTTPException(status_code=500, detail=f"camera recorder script not found: {self.camera_script}")
        if not (self.camera_env / "bin" / "activate").is_file():
            raise HTTPException(status_code=500, detail=f"camera environment not found: {self.camera_env}")

        if self.status_file.exists():
            try:
                self.status_file.unlink()
            except OSError:
                pass
        if self.preview_dir.exists():
            shutil.rmtree(self.preview_dir, ignore_errors=True)

        target = self._camera_target()
        self._run_tmux("send-keys", "-t", target, "C-c")
        time.sleep(0.1)
        proc = self._run_tmux("send-keys", "-t", target, self._build_camera_launch_command(config), "C-m")
        if proc.returncode != 0:
            detail = self._command_error(proc, "Could not start camera recorder in tmux pane.")
            self.last_error = detail
            raise HTTPException(status_code=500, detail=detail)
        self.last_message = "Started camera recorder in existing tmux session."
        self._wait_for_recorder_ready()

    def _capture_pane_lines(self, pane_index: str) -> list[str]:
        proc = self._run_tmux(
            "capture-pane",
            "-p",
            "-S",
            f"-{self.status_lines}",
            "-t",
            f"{self.session_name}:{self.window_name}.{pane_index}",
        )
        if proc.returncode != 0:
            return []
        return [line.rstrip() for line in proc.stdout.splitlines() if line.strip()]

    def _camera_pane_tail(self) -> str:
        return "\n".join(self._capture_pane_lines(self.camera_pane_index)[-self.status_lines:])

    def _wait_for_recorder_ready(self, timeout_s: float = 20.0) -> None:
        deadline = time.time() + timeout_s
        last_status: dict[str, Any] = {}
        while time.time() < deadline:
            status_payload = self._load_status_file()
            if status_payload:
                last_status = status_payload
                if bool(status_payload.get("initialized")) and self._camera_recorder_is_running():
                    self.last_error = ""
                    return
                message = str(status_payload.get("message") or "")
                if "error" in message.lower() or status_payload.get("last_error"):
                    break
            if not self._camera_recorder_is_running():
                break
            time.sleep(0.25)

        detail = str(last_status.get("last_error") or last_status.get("message") or "").strip()
        if not detail:
            detail = self._camera_pane_tail() or "Camera recorder did not finish initializing."
        self.last_error = detail
        self.last_message = "Camera recorder failed to initialize."
        raise HTTPException(status_code=409, detail=detail)

    def _wait_for_recording_start(self, previous_error: str, timeout_s: float = 2.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status_payload = self._load_status_file()
            if status_payload:
                if bool(status_payload.get("recording")):
                    self.last_error = ""
                    return
                last_error = str(status_payload.get("last_error") or "").strip()
                if last_error and last_error != previous_error:
                    self.last_error = last_error
                    self.last_message = str(status_payload.get("message") or last_error)
                    raise HTTPException(status_code=409, detail=last_error)
            time.sleep(0.1)

    def _wait_for_preview_warmup(self, timeout_s: float = 3.0) -> None:
        """Let initialize return a post-warmup snapshot when a camera is connected."""
        deadline = time.time() + timeout_s
        saw_connected_camera = False

        while time.time() < deadline:
            status_payload = self._load_status_file()
            if not status_payload:
                time.sleep(0.1)
                continue

            connected_roles: list[str] = []
            for role, key in (("hand", "hand_camera"), ("wrist", "wrist_camera"), ("external", "external_camera")):
                source = status_payload.get(key)
                if isinstance(source, dict) and bool(source.get("connected") or source.get("enabled")):
                    connected_roles.append(role)

            if not connected_roles:
                return

            saw_connected_camera = True
            for role in connected_roles:
                source = status_payload.get(f"{role}_camera")
                latest_timestamp = source.get("latest_timestamp") if isinstance(source, dict) else None
                if self._is_fresh_timestamp(latest_timestamp, CAMERA_STREAM_STALE_AFTER_S) or self._preview_is_fresh(role, "rgb", max_age_s=timeout_s):
                    return

            time.sleep(0.1)

        if saw_connected_camera:
            self.last_message = "Recorder initialized. Waiting for live preview frames..."

    def _tmux_status(self, initialized: bool) -> dict[str, Any]:
        tmux_payload: dict[str, Any] = {
            "session": self.session_name,
            "window": self.window_name,
            "panes": [],
        }
        if not initialized:
            return tmux_payload

        proc = self._run_tmux(
            "list-panes",
            "-t",
            f"{self.session_name}:{self.window_name}",
            "-F",
            "#{pane_index} #{pane_current_command}",
        )
        if proc.returncode != 0:
            return tmux_payload

        for raw_line in proc.stdout.splitlines():
            if not raw_line.strip():
                continue
            parts = raw_line.split(maxsplit=1)
            pane_index = parts[0].strip()
            if not pane_index:
                continue
            command = parts[1].strip() if len(parts) > 1 else ""
            label = self.pane_labels.get(pane_index, f"pane {pane_index}")
            tmux_payload["panes"].append({
                "index": pane_index,
                "label": label,
                "command": command,
                "lines": self._capture_pane_lines(pane_index),
            })
        return tmux_payload

    def _load_status_file(self) -> dict[str, Any]:
        if not self.status_file.is_file():
            return {}
        try:
            payload = json.loads(self.status_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_teleop_status_file(self) -> dict[str, Any]:
        if not self.teleop_status_file.is_file():
            return {}
        try:
            payload = json.loads(self.teleop_status_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _is_fresh_timestamp(value: Any, max_age_s: float) -> bool:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return False
        return (time.time() - timestamp) <= max_age_s

    def _preview_is_fresh(self, camera: str, kind: str, max_age_s: float = CAMERA_STREAM_STALE_AFTER_S) -> bool:
        suffix = "jpg" if kind == "rgb" else "png"
        preview_path = self.preview_dir / f"{camera}_{kind}.{suffix}"
        if not preview_path.is_file():
            return False
        try:
            age = time.time() - preview_path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age <= max_age_s

    def _normalized_camera_source(
        self,
        source: dict[str, Any] | None,
        *,
        session_initialized: bool,
        camera_role: str,
    ) -> dict[str, Any]:
        base = {
            "requested": True,
            "enabled": False,
            "connected": False,
            "available": False,
            "status": "disconnected",
            "name": "missing",
            "serial": "",
            "latest_timestamp": None,
            "width": None,
            "height": None,
            "fps": 0,
        }
        if not isinstance(source, dict):
            return base
        merged = {**base, **source}
        if not session_initialized:
            return base

        connected = bool(merged.get("connected") or merged.get("enabled"))
        timestamp_fresh = self._is_fresh_timestamp(merged.get("latest_timestamp"), CAMERA_STREAM_STALE_AFTER_S)
        preview_fresh = self._preview_is_fresh(camera_role, "rgb")
        available = connected and timestamp_fresh and preview_fresh
        merged["connected"] = connected
        merged["enabled"] = connected
        merged["available"] = available
        merged["status"] = "streaming" if available else "connected" if connected else "disconnected"
        return merged

    def _normalized_robot_source(self, source: dict[str, Any] | None, *, session_initialized: bool) -> dict[str, Any]:
        base = {
            "requested": False,
            "enabled": False,
            "connected": False,
            "available": False,
            "status": "disconnected",
            "ip": "",
            "fps": 0,
            "gripper_state_enabled": False,
            "latest": {},
        }
        if not isinstance(source, dict):
            return base
        merged = {**base, **source}
        if not session_initialized:
            return base

        connected = bool(merged.get("connected") or merged.get("enabled"))
        latest = merged.get("latest") if isinstance(merged.get("latest"), dict) else {}
        available = connected and self._is_fresh_timestamp(latest.get("timestamp"), ROBOT_STREAM_STALE_AFTER_S)
        merged["connected"] = connected
        merged["enabled"] = connected
        merged["available"] = available
        merged["status"] = "streaming" if available else "connected" if connected else "disconnected"
        merged["latest"] = latest
        return merged

    def _normalized_gripper_source(self, source: dict[str, Any] | None, *, session_initialized: bool) -> dict[str, Any]:
        base = {
            "requested": False,
            "enabled": False,
            "connected": False,
            "available": False,
            "status": "disconnected",
            "ip": "",
            "port": 0,
            "latest_position": None,
        }
        if not isinstance(source, dict):
            return base
        merged = {**base, **source}
        if not session_initialized:
            return base

        connected = bool(merged.get("connected") or merged.get("enabled"))
        available = connected and merged.get("latest_position") is not None
        merged["connected"] = connected
        merged["enabled"] = connected
        merged["available"] = available
        merged["status"] = "streaming" if available else "connected" if connected else "disconnected"
        return merged

    def _spacemouse_payload(
        self,
        tmux_payload: dict[str, Any],
        *,
        session_initialized: bool,
        teleop_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = {
            "requested": True,
            "connected": False,
            "available": False,
            "status": "disconnected",
            "text": "Waiting for SpaceMouse session...",
            "lines": [],
            "motion_state": [],
            "buttons": {"left": False, "right": False},
        }
        if not session_initialized:
            return base

        if isinstance(teleop_status, dict):
            live = teleop_status.get("spacemouse")
            if isinstance(live, dict):
                payload = {**base, **live}
                payload["requested"] = True
                payload["connected"] = bool(payload.get("connected"))
                payload["available"] = bool(payload.get("available")) and self._is_fresh_timestamp(
                    payload.get("latest_timestamp"),
                    ROBOT_STREAM_STALE_AFTER_S,
                )
                payload["status"] = "streaming" if payload["available"] else "connected" if payload["connected"] else "disconnected"
                motion = payload.get("motion_state")
                buttons = payload.get("buttons") if isinstance(payload.get("buttons"), dict) else {}
                motion_text = "-"
                if isinstance(motion, list) and motion:
                    motion_text = np.array2string(np.asarray(motion, dtype=float), precision=3, separator=", ")
                payload["text"] = "\n".join([
                    f"Status: {payload['status']}",
                    f"Motion: {motion_text}",
                    (
                        f"Buttons: left={'pressed' if buttons.get('left') else 'released'}, "
                        f"right={'pressed' if buttons.get('right') else 'released'}"
                    ),
                ])
                return payload

        pane = None
        for item in tmux_payload.get("panes", []):
            if item.get("label") == "spacemouse":
                pane = item
                break

        if pane is None:
            base["text"] = "SpaceMouse pane not found."
            return base

        command = str(pane.get("command") or "").strip()
        lines = list(pane.get("lines") or [])
        connected = bool(command) and command not in {"bash", "zsh", "sh", "fish"}
        return {
            "requested": True,
            "connected": connected,
            "available": connected,
            "status": "streaming" if connected else "connected",
            "text": "\n".join(lines) if lines else ("Waiting for SpaceMouse input..." if connected else "SpaceMouse process is not running."),
            "lines": lines,
            "command": command,
            "motion_state": [],
            "buttons": {"left": False, "right": False},
        }

    @staticmethod
    def _safe_instruction(raw_instruction: str) -> str:
        return _normalize_instruction(raw_instruction)

    def _build_script_env(self, config: RecordingInitRequest) -> dict[str, str]:
        instruction = self._safe_instruction(config.instruction)
        robot_ip = config.robot_ip.strip()
        camera_extra_args = [
            "--instruction",
            instruction,
            "--hand-width",
            str(config.hand_width),
            "--hand-height",
            str(config.hand_height),
            "--wrist-width",
            str(config.wrist_width),
            "--wrist-height",
            str(config.wrist_height),
            "--external-width",
            str(config.external_width),
            "--external-height",
            str(config.external_height),
            "--control-mode",
            "stdin",
            "--status-file",
            str(self.status_file),
            "--no-delete-confirm",
            "--hand-product-ids",
            config.hand_product_ids,
            "--wrist-product-ids",
            config.wrist_product_ids,
            "--external-product-ids",
            config.external_product_ids,
            "--camera-start-retries",
            str(config.camera_start_retries),
            "--camera-start-retry-delay",
            str(config.camera_start_retry_delay),
            "--camera-post-reset-wait",
            str(config.camera_post_reset_wait),
            "--camera-busy-reset" if config.camera_busy_reset else "--no-camera-busy-reset",
        ]

        teleop_extra_args: list[str] = []
        camera_extra_args.extend([
            "--teleop-status-file",
            str(self.teleop_status_file),
        ])
        teleop_extra_args.extend([
            "--instruction",
            instruction,
            "--control-mode",
            "both",
            "--status-file",
            str(self.teleop_status_file),
            "--subtask-segment-index",
            str(config.subtask_segment_index),
            "--subtask-reset-noise-xyz",
            str(config.subtask_reset_noise_xyz_m),
        ])
        env = os.environ.copy()
        env.update({
            "SESSION_NAME": self.session_name,
            "ATTACH_ON_START": "0",
            "ROBOT_BACKEND": os.getenv("ROBOT_BACKEND", "xarm"),
            "XARM_CONTROLLER_PATH": os.getenv(
                "XARM_CONTROLLER_PATH",
                str((BASE_DIR.parent / "../test.py").resolve()),
            ),
            "RECORD_ROOT": str(Path(config.root).expanduser().resolve()),
            "FPS": str(config.fps),
            "HAND_SERIAL": config.hand_serial.strip(),
            "WRIST_SERIAL": config.wrist_serial.strip(),
            "EXTERNAL_SERIAL": config.external_serial.strip(),
            "ALLOW_MISSING_HAND": "1" if config.allow_missing_hand else "0",
            "ALLOW_MISSING_WRIST": "1" if config.allow_missing_wrist else "0",
            "ALLOW_MISSING_EXTERNAL": "1" if config.allow_missing_external else "0",
            "ROBOT_FPS": str(config.robot_fps),
            "ENABLE_GRIPPER_STATE": "1" if config.enable_gripper_state else "0",
            "GRIPPER_PORT": str(config.gripper_port),
            "ALLOW_MISSING_ROBOT": "1" if config.allow_missing_robot else "0",
            "ALLOW_MISSING_GRIPPER": "1" if config.allow_missing_gripper else "0",
            "CAMERA_EXTRA_ARGS": shlex.join(camera_extra_args),
        })
        env["TELEOP_EXTRA_ARGS"] = shlex.join(teleop_extra_args)

        if robot_ip:
            env["ROBOT_IP"] = robot_ip
            env["UR_ROBOT_IP"] = robot_ip
            env["XARM_ROBOT_IP"] = robot_ip
            env["ENABLE_ROBOT_RECORDING"] = "1"
        else:
            env["ROBOT_IP"] = ""
            env["UR_ROBOT_IP"] = ""
            env["XARM_ROBOT_IP"] = ""
            env["ENABLE_ROBOT_RECORDING"] = "0"

        return env

    def _send_camera_command(self, command: str, action_message: str) -> None:
        if not self._session_exists():
            raise HTTPException(status_code=409, detail="Tmux recording session is not initialized.")
        if not self._camera_recorder_is_running():
            detail = self._camera_pane_tail() or "Camera recorder is not running."
            self.last_error = detail
            raise HTTPException(status_code=409, detail=detail)
        proc = self._run_tmux("send-keys", "-t", self._camera_target(), command, "C-m")
        if proc.returncode != 0:
            detail = self._command_error(proc, "Could not send command to camera pane.")
            self.last_error = detail
            raise HTTPException(status_code=500, detail=detail)
        self.last_message = action_message

    def _teleop_process_is_running(self) -> bool:
        target = self._teleop_target()
        if target is None:
            return False
        command = self._pane_current_command(self.teleop_pane_index)
        return bool(command) and command not in {"bash", "zsh", "sh", "fish"}

    def _wait_for_pipeline_processes_to_idle(self, timeout_s: float = 2.5) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            teleop_running = self._teleop_process_is_running()
            camera_running = self._camera_recorder_is_running()
            if not teleop_running and not camera_running:
                return
            time.sleep(0.1)

    def _wait_for_camera_save_complete(self, timeout_s: float = 120.0) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        last_payload: dict[str, Any] = {}
        while time.time() < deadline:
            payload = self._load_status_file()
            if payload:
                last_payload = payload
                recording = bool(payload.get("recording", False))
                saving = bool(payload.get("saving", False))
                last_saved = str(payload.get("last_saved_episode") or "").strip()
                message = str(payload.get("message") or "")
                if not recording and not saving and (last_saved or message.startswith("Saved episode")):
                    return payload
                last_error = str(payload.get("last_error") or "").strip()
                if last_error and not recording and not saving:
                    raise HTTPException(status_code=409, detail=last_error)
            if not self._camera_recorder_is_running():
                break
            time.sleep(0.2)

        detail = str(last_payload.get("last_error") or last_payload.get("message") or "").strip()
        if not detail:
            detail = self._camera_pane_tail() or "Timed out waiting for episode save before shutdown."
        self.last_error = detail
        raise HTTPException(status_code=409, detail=detail)

    def _send_teleop_command(self, command: str) -> None:
        target = self._teleop_target()
        if target is None or not self._teleop_process_is_running():
            return
        self._run_tmux("send-keys", "-t", target, command, "C-m")

    def _request_teleop_force_stop(self) -> None:
        """Ask teleop to stop the robot immediately, with Ctrl+C fallback."""
        self._send_teleop_command("q")
        deadline = time.time() + 0.3
        while time.time() < deadline:
            if not self._teleop_process_is_running():
                return
            time.sleep(0.05)
        if self._teleop_process_is_running():
            target = self._teleop_target()
            if target is not None:
                self._run_tmux("send-keys", "-t", target, "C-c")

    def _close_tmux_session_after_camera_done(self) -> None:
        try:
            saved_payload = self._wait_for_camera_save_complete(timeout_s=120.0)
            saved_episode = str(saved_payload.get("last_saved_episode") or "").strip()
            if saved_episode:
                self.last_saved_episode = saved_episode

            if self._camera_recorder_is_running():
                self._run_tmux("send-keys", "-t", self._camera_target(), "q", "C-m")
            self._wait_for_pipeline_processes_to_idle(timeout_s=2.5)
            if self._session_exists():
                proc = self._run_tmux("kill-session", "-t", self.session_name)
                if proc.returncode != 0:
                    detail = self._command_error(proc, "Failed to kill tmux recording session after save.")
                    self.last_error = detail
                    self.last_message = detail
                    return
            self.last_error = ""
            if saved_episode:
                self.last_message = f"Saved episode {Path(saved_episode).name} and stopped the robot. You can Initialize again."
            else:
                self.last_message = "Saved current episode and stopped the robot. You can Initialize again."
        except HTTPException as exc:
            detail = str(exc.detail)
            self.last_error = detail
            self.last_message = f"Robot was stopped, but episode save did not finish: {detail}"
        except Exception as exc:
            detail = str(exc)
            self.last_error = detail
            self.last_message = f"Robot was stopped, but episode save did not finish: {detail}"
        finally:
            self.shutdown_save_in_progress = False

    def initialize(self, config: RecordingInitRequest) -> dict[str, Any]:
        with self.lock:
            self._set_active_pipeline(config.pipeline)
            self.config = config
            self.last_error = ""

            other_pipeline = self._other_running_pipeline(config.pipeline)
            if other_pipeline is not None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"The {other_pipeline} pipeline session is already running. "
                        "Use Shutdown before initializing a different pipeline."
                    ),
                )

            if not self.script_path.is_file():
                raise HTTPException(status_code=500, detail=f"tmux script not found: {self.script_path}")

            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            if self._session_exists():
                self._restart_camera_recorder(config)
                self._wait_for_preview_warmup()
                time.sleep(0.5)
                self.last_message = f"Reconfigured camera recorder in tmux session: {self.session_name}"
                return self.status()

            if self.status_file.exists():
                try:
                    self.status_file.unlink()
                except OSError:
                    pass
            if self.teleop_status_file.exists():
                try:
                    self.teleop_status_file.unlink()
                except OSError:
                    pass
            if self.preview_dir.exists():
                shutil.rmtree(self.preview_dir, ignore_errors=True)

            proc = subprocess.run(
                ["bash", str(self.script_path)],
                cwd=str(BASE_DIR.parent),
                env=self._build_script_env(config),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                detail = self._command_error(proc, "Failed to start tmux recording backend.")
                self.last_error = detail
                self.last_message = "Failed to start tmux recording backend."
                raise HTTPException(status_code=400, detail=detail)

            self.last_message = f"Started tmux recording session: {self.session_name}"
            self._wait_for_recorder_ready()
            self._wait_for_preview_warmup()
            return self.status()

    def start_episode(self, instruction: str | None = None) -> dict[str, Any]:
        with self.lock:
            if self.config is not None and not self._camera_recorder_is_running():
                self._restart_camera_recorder(self.config)
            self._wait_for_recorder_ready(timeout_s=2.0)
            status_before_start = self._load_status_file()
            previous_error = str(status_before_start.get("last_error") or "").strip()

            instruction_text = (instruction or "").strip()
            if instruction_text:
                instruction_text = " ".join(instruction_text.splitlines()).strip()
                if instruction_text:
                    safe_instruction = self._safe_instruction(instruction_text)
                    if self.config is not None:
                        self.config.instruction = safe_instruction
                    # Set instruction explicitly first so the subsequent start command
                    # is always a plain "c" and cannot pollute folder naming.
                    self._send_camera_command(f"i {safe_instruction}", "Instruction updated for next episode.")
                    self._send_teleop_command(f"i {safe_instruction}")

            self._send_camera_command("c", "Start command sent to camera recorder.")
            self._wait_for_recording_start(previous_error)
            self._send_teleop_command("c")
            return self.status()

    def stop_episode(self) -> dict[str, Any]:
        with self.lock:
            self._send_camera_command("s", "Stop/save command sent to camera recorder.")
            self._send_teleop_command("s")
            return self.status()

    def reset_pose(self, reset_config: RecordingResetRequest | None = None) -> dict[str, Any]:
        with self.lock:
            if not self._session_exists():
                raise HTTPException(status_code=409, detail="Tmux recording session is not initialized.")
            if not self._teleop_process_is_running():
                detail = "\n".join(self._capture_pane_lines(self.teleop_pane_index)[-self.status_lines:]) or "SpaceMouse teleop process is not running."
                self.last_error = detail
                raise HTTPException(status_code=409, detail=detail)
            if reset_config is not None and self.config is not None:
                self.config.subtask_segment_index = int(reset_config.subtask_segment_index)
                self.config.subtask_reset_noise_xyz_m = float(reset_config.subtask_reset_noise_xyz_m)
            segment_index = 0 if reset_config is None else int(reset_config.subtask_segment_index)
            noise_xyz_m = 0.01 if reset_config is None else float(reset_config.subtask_reset_noise_xyz_m)
            self._send_teleop_command(f"r {segment_index} {noise_xyz_m:.6f}")
            if segment_index > 0:
                self.last_message = f"Reset command sent to SpaceMouse teleop for subtask segment {segment_index}."
            else:
                self.last_message = "Reset command sent to SpaceMouse teleop."
            return self.status()

    def stop_rtde_motion(self) -> dict[str, Any]:
        with self.lock:
            if not self._session_exists():
                raise HTTPException(status_code=409, detail="Tmux recording session is not initialized.")
            if not self._teleop_process_is_running():
                detail = "\n".join(self._capture_pane_lines(self.teleop_pane_index)[-self.status_lines:]) or "SpaceMouse teleop process is not running."
                self.last_error = detail
                raise HTTPException(status_code=409, detail=detail)
            self._send_teleop_command("u")
            self.last_message = "Local SpaceMouse clear command sent; current episode recording continues."
            return self.status()

    def reconnect_rtde(self) -> dict[str, Any]:
        # Backward-compatible endpoint name: U is now a light speedStop, not a full RTDE reconnect.
        return self.stop_rtde_motion()

    def label_subtask(self) -> dict[str, Any]:
        with self.lock:
            self._send_camera_command("l", "Subtask label command sent to camera recorder.")
            return self.status()

    def delete_last_episode(self) -> dict[str, Any]:
        with self.lock:
            self._send_camera_command("d", "Delete command sent to camera recorder.")
            self._send_teleop_command("d")
            return self.status()

    def shutdown_after_save_async(self) -> dict[str, Any]:
        with self.lock:
            if not self._session_exists():
                self.last_message = "Tmux recorder is not initialized."
                return self.status()
            if self.shutdown_save_in_progress:
                self.last_message = "Robot stop requested; episode save is still running in background."
                return self.status()

            snapshot = self.status()
            if not bool(snapshot.get("recording", False)):
                return self.shutdown(save_current=False)

            self.shutdown_save_in_progress = True
            self.last_error = ""
            self.last_message = "Force-stopping robot now; saving active episode in background..."

            self._request_teleop_force_stop()
            try:
                self._send_camera_command("s", "Robot stopped; saving active episode in background...")
            except Exception:
                self.shutdown_save_in_progress = False
                raise

            self.shutdown_save_thread = threading.Thread(
                target=self._close_tmux_session_after_camera_done,
                daemon=True,
                name="shutdown-save-waiter",
            )
            self.shutdown_save_thread.start()
            return self.status()

    def shutdown(self, save_current: bool = True) -> dict[str, Any]:
        with self.lock:
            if not self._session_exists():
                self.last_message = "Tmux recorder is not initialized."
                return self.status()

            if save_current:
                snapshot = self.status()
                if bool(snapshot.get("recording", False)):
                    self._request_teleop_force_stop()
                    self._send_camera_command("s", "Robot stopped; saving active episode before shutdown...")
                    saved_payload = self._wait_for_camera_save_complete(timeout_s=120.0)
                    saved_episode = str(saved_payload.get("last_saved_episode") or "").strip()
                    if saved_episode:
                        self.last_saved_episode = saved_episode

            self._send_teleop_command("q")
            self._run_tmux("send-keys", "-t", self._camera_target(), "q", "C-m")
            self._wait_for_pipeline_processes_to_idle(timeout_s=2.5)
            if self._teleop_process_is_running():
                target = self._teleop_target()
                if target is not None:
                    self._run_tmux("send-keys", "-t", target, "C-c")
                self._wait_for_pipeline_processes_to_idle(timeout_s=1.0)
            proc = self._run_tmux("kill-session", "-t", self.session_name)
            if proc.returncode != 0:
                detail = self._command_error(proc, "Failed to kill tmux recording session.")
                self.last_error = detail
                raise HTTPException(status_code=500, detail=detail)

            self.last_message = f"Stopped tmux recording session: {self.session_name}"
            return self.status()

    def status(self) -> dict[str, Any]:
        with self.lock:
            effective_config = self.config if self.config is not None else RecordingInitRequest(pipeline=self.active_pipeline)

            session_initialized = self._session_exists()
            payload: dict[str, Any] = {
                "pipeline": self.active_pipeline,
                "session_initialized": session_initialized,
                "initialized": False,
                "recording": False,
                "saving": False,
                "message": self.last_message if not session_initialized else "Tmux session is running.",
                "data_folder": "",
                "current_episode": None,
                "current_episode_folder": None,
                "last_saved_episode": self.last_saved_episode,
                "counts": {
                    "hand_frames": 0,
                    "wrist_frames": 0,
                    "external_frames": 0,
                    "robot_frames": 0,
                },
                "hand_camera": {
                    "requested": True,
                    "enabled": False,
                    "connected": False,
                    "available": False,
                    "status": "disconnected",
                    "width": effective_config.hand_width,
                    "height": effective_config.hand_height,
                    "fps": effective_config.fps,
                },
                "wrist_camera": {
                    "requested": True,
                    "enabled": False,
                    "connected": False,
                    "available": False,
                    "status": "disconnected",
                    "width": effective_config.wrist_width,
                    "height": effective_config.wrist_height,
                    "fps": effective_config.fps,
                },
                "external_camera": {
                    "requested": True,
                    "enabled": False,
                    "connected": False,
                    "available": False,
                    "status": "disconnected",
                    "width": effective_config.external_width,
                    "height": effective_config.external_height,
                    "fps": effective_config.fps,
                },
                "robot": {
                    "requested": False,
                    "enabled": False,
                    "connected": False,
                    "available": False,
                    "status": "disconnected",
                    "ip": "",
                    "fps": 0,
                    "gripper_state_enabled": False,
                    "latest": {},
                },
                "gripper": {
                    "requested": False,
                    "enabled": False,
                    "connected": False,
                    "available": False,
                    "status": "disconnected",
                    "ip": "",
                    "port": 0,
                    "latest_position": None,
                },
                "spacemouse": {
                    "requested": True,
                    "connected": False,
                    "status": "disconnected",
                    "text": "Waiting for SpaceMouse session...",
                    "lines": [],
                },
                "subtask_reset": {
                    "active_segment_index": effective_config.subtask_segment_index,
                    "noise_xyz_m": effective_config.subtask_reset_noise_xyz_m,
                    "configured_segments": [],
                },
                "last_error": self.last_error,
                "config": (
                    effective_config.model_dump()
                ),
                "backend": "tmux",
                "live_preview_supported": True,
                "tmux": self._tmux_status(session_initialized),
            }
            teleop_status = self._load_teleop_status_file() if session_initialized else {}
            payload["spacemouse"] = self._spacemouse_payload(
                payload["tmux"],
                session_initialized=session_initialized,
                teleop_status=teleop_status,
            )

            status_payload = self._load_status_file() if session_initialized else {}
            if status_payload:
                payload["initialized"] = bool(self._status_value(status_payload, "initialized", False)) if session_initialized else False
                payload["recording"] = bool(self._status_value(status_payload, "recording", False)) if payload["initialized"] else False
                payload["saving"] = bool(self._status_value(status_payload, "saving", False)) if payload["initialized"] else False
                payload["message"] = str(self._status_value(status_payload, "message", payload["message"]))
                payload["data_folder"] = str(self._status_value(status_payload, "data_folder", ""))
                payload["current_episode"] = self._status_value(status_payload, "current_episode", None)
                payload["current_episode_folder"] = self._status_value(status_payload, "current_episode_folder", None)
                payload["last_saved_episode"] = self._status_value(status_payload, "last_saved_episode", payload["last_saved_episode"])
                if isinstance(status_payload.get("counts"), dict):
                    payload["counts"] = status_payload["counts"]
                if isinstance(status_payload.get("hand_camera"), dict):
                    payload["hand_camera"] = status_payload["hand_camera"]
                if isinstance(status_payload.get("wrist_camera"), dict):
                    payload["wrist_camera"] = status_payload["wrist_camera"]
                if isinstance(status_payload.get("external_camera"), dict):
                    payload["external_camera"] = status_payload["external_camera"]
                if isinstance(status_payload.get("robot"), dict):
                    payload["robot"] = status_payload["robot"]
                if isinstance(status_payload.get("gripper"), dict):
                    payload["gripper"] = status_payload["gripper"]
                payload["live_preview_supported"] = bool(
                    self._status_value(status_payload, "live_preview_supported", payload["live_preview_supported"])
                )
                payload["last_error"] = str(self._status_value(status_payload, "last_error", payload["last_error"]))
                instruction_value = str(self._status_value(status_payload, "instruction", "") or "")
                if self.config is not None and instruction_value:
                    self.config.instruction = self._safe_instruction(instruction_value)

            if teleop_status:
                if isinstance(teleop_status.get("robot"), dict):
                    payload["robot"] = teleop_status["robot"]
                if isinstance(teleop_status.get("gripper"), dict):
                    payload["gripper"] = teleop_status["gripper"]
                if isinstance(teleop_status.get("subtask_reset"), dict):
                    payload["subtask_reset"] = teleop_status["subtask_reset"]
                teleop_message = str(teleop_status.get("message") or "").strip()
                if teleop_message and not payload["recording"]:
                    payload["message"] = teleop_message
                teleop_error = str(teleop_status.get("last_error") or "").strip()
                if teleop_error and not payload["last_error"]:
                    payload["last_error"] = teleop_error

            payload["hand_camera"] = self._normalized_camera_source(
                payload.get("hand_camera"),
                session_initialized=session_initialized,
                camera_role="hand",
            )
            payload["wrist_camera"] = self._normalized_camera_source(
                payload.get("wrist_camera"),
                session_initialized=session_initialized,
                camera_role="wrist",
            )
            payload["external_camera"] = self._normalized_camera_source(
                payload.get("external_camera"),
                session_initialized=session_initialized,
                camera_role="external",
            )
            payload["robot"] = self._normalized_robot_source(
                payload.get("robot"),
                session_initialized=session_initialized,
            )
            payload["gripper"] = self._normalized_gripper_source(
                payload.get("gripper"),
                session_initialized=session_initialized,
            )

            if self.shutdown_save_in_progress:
                payload["saving"] = True
                payload["message"] = self.last_message or "Robot stopped; saving active episode in background..."

            if payload["last_saved_episode"]:
                self.last_saved_episode = str(payload["last_saved_episode"])
            if not session_initialized:
                payload["recording"] = False
                payload["saving"] = False
                payload["data_folder"] = ""
                payload["current_episode"] = None
                payload["current_episode_folder"] = None
                payload["counts"] = {
                    "hand_frames": 0,
                    "wrist_frames": 0,
                    "external_frames": 0,
                    "robot_frames": 0,
                    "subtask_labels": 0,
                }
                payload["live_preview_supported"] = False
                payload["message"] = self.last_message
            elif not payload["initialized"] and not status_payload:
                payload["message"] = self.last_message
            return payload

    def latest_frame(self, camera: str, kind: str) -> tuple[np.ndarray, float | None]:
        suffix = "jpg" if kind == "rgb" else "png"
        preview_path = self.preview_dir / f"{camera}_{kind}.{suffix}"
        if not preview_path.is_file():
            raise HTTPException(status_code=404, detail=f"No live {camera} {kind} preview available yet.")
        if not self._preview_is_fresh(camera, kind):
            raise HTTPException(status_code=404, detail=f"Live {camera} {kind} preview is stale.")
        try:
            with Image.open(preview_path) as image:
                image = image.convert("RGB" if kind == "rgb" else "L")
                frame = np.asarray(image).copy()
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Could not read live preview: {exc}") from exc
        return frame, preview_path.stat().st_mtime


if RECORDING_BACKEND_MODE == "tmux":
    _RECORDING_MANAGER: RecordingManager | TmuxRecordingManager = TmuxRecordingManager()
else:
    _RECORDING_MANAGER = RecordingManager()


def _set_runtime_paths(data_dir: Path, default_dataset: str = "") -> None:
    global DATA_DIR, DEFAULT_DATASET
    DATA_DIR = data_dir.resolve()
    DEFAULT_DATASET = default_dataset
    _SUMMARY_CACHE.clear()


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_scalar(value.item())
        if value.size <= 32:
            return [_normalize_scalar(item) for item in value.tolist()]
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): _normalize_scalar(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return path.name


def _recordings_stamp(root: Path) -> tuple[int, int]:
    latest_mtime = 0
    total_size = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        latest_mtime = max(latest_mtime, stat.st_mtime_ns)
        total_size += stat.st_size
    return latest_mtime, total_size


def _resolve_recordings_dataset(dataset: str) -> Path:
    if dataset not in {PSEUDO_DATASET_NAME, "", str(DATA_DIR)}:
        raise HTTPException(status_code=404, detail=f"Recording set not found: {dataset}")
    if not DATA_DIR.is_dir():
        raise HTTPException(status_code=404, detail=f"Recordings directory not found: {DATA_DIR}")
    return DATA_DIR


def _iter_dataset_files() -> list[Path]:
    if not DATA_DIR.is_dir():
        return []
    return [DATA_DIR]


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return _normalize_scalar(json.load(handle))
    except Exception as exc:
        return {"metadata_error": str(exc)}


def _stream_suffix(stream_name: str) -> str:
    suffix = stream_name
    for prefix in ("rgb_", "depth_"):
        if suffix.startswith(prefix):
            suffix = suffix[len(prefix):]
            break
    for ending in (".npy", ".mkv", ".mp4"):
        if suffix.endswith(ending):
            suffix = suffix[: -len(ending)]
            break
    if suffix.endswith("_raw"):
        suffix = suffix[:-4]
    return suffix


def _iter_episode_dirs(root: Path) -> list[Path]:
    episode_dirs = set()
    for metadata_file in root.rglob("metadata.json"):
        if metadata_file.parent.name.isdigit() and metadata_file.parent.parent.name == "camera_npy":
            episode_dirs.add(metadata_file.parent)
    return sorted(episode_dirs, key=lambda item: _safe_relative_path(item), reverse=True)


def _array_info(path: Path) -> dict[str, Any] | None:
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as exc:
        return {
            "path": path.name,
            "shape": [],
            "dtype": "unreadable",
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "error": str(exc),
        }
    return {
        "path": path.name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size_bytes": path.stat().st_size,
    }


def _collect_npy_summaries(episode_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for npy_file in sorted(episode_dir.glob("*.npy")):
        info = _array_info(npy_file)
        if info is not None:
            summaries.append(info)
    return summaries


def _collect_encoded_summaries(episode_dir: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    vision_files = metadata.get("vision_files") or {}
    for key, entry in vision_files.items():
        if not isinstance(entry, dict):
            continue
        path_name = str(entry.get("path") or "")
        if not path_name:
            continue
        path = episode_dir / path_name
        if not path.is_file():
            continue
        summaries.append({
            "path": path_name,
            "shape": list(entry.get("shape") or []),
            "dtype": str(entry.get("dtype") or ""),
            "size_bytes": path.stat().st_size,
            "storage": str(entry.get("storage") or ""),
            "codec": str(entry.get("codec") or ""),
        })
    return summaries


def _collect_episode_datasets(episode_dir: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = _collect_npy_summaries(episode_dir)
    existing_paths = {item["path"] for item in datasets}
    for item in _collect_encoded_summaries(episode_dir, metadata):
        if item["path"] not in existing_paths:
            datasets.append(item)
    return sorted(datasets, key=lambda item: item["path"])


def _collect_npy_streams(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    streams = []
    for item in datasets:
        name = item["path"]
        shape = item.get("shape") or []
        display_name = name
        if len(shape) == 4 and shape[-1] in {1, 3, 4} and name.startswith("rgb_"):
            streams.append({
                "path": name,
                "display_path": display_name,
                "kind": "rgb",
                "shape": shape,
                "dtype": item.get("dtype", ""),
                "height": int(shape[1]),
                "width": int(shape[2]),
            })
        elif len(shape) == 3 and name.startswith("depth_"):
            if name.endswith("_raw.mkv"):
                display_name = name.replace("_raw.mkv", ".mkv")
            streams.append({
                "path": name,
                "display_path": display_name,
                "kind": "depth",
                "shape": shape,
                "dtype": item.get("dtype", ""),
                "height": int(shape[1]),
                "width": int(shape[2]),
            })
    return sorted(streams, key=lambda stream: (stream["kind"] != "rgb", stream["path"]))


def _load_timestamps(episode_dir: Path, stream_name: str | None = None) -> np.ndarray:
    timestamp_files = []
    if stream_name:
        suffix = _stream_suffix(stream_name)
        timestamp_files.append(episode_dir / f"timestamps_{suffix}.npy")
    timestamp_files.extend(sorted(episode_dir.glob("timestamps_*.npy")))
    for path in timestamp_files:
        if not path.is_file():
            continue
        try:
            timestamps = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
        except Exception:
            continue
        if timestamps.ndim == 1 and len(timestamps) > 0:
            return timestamps
    return np.array([], dtype=np.float64)


def _estimate_fps(timestamps: np.ndarray) -> float | None:
    if len(timestamps) < 2:
        return None
    deltas = np.diff(timestamps.astype(np.float64))
    positive = deltas[deltas > 0]
    if len(positive) == 0:
        return None
    return float(1.0 / np.median(positive))


def _episode_start_from_name(timestamp_name: str) -> float | None:
    try:
        return time.mktime(time.strptime(timestamp_name, "%Y%m%d%H%M%S"))
    except Exception:
        return None


def _format_timestamp(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _build_npy_episode_payload(root: Path, episode_dir: Path) -> dict[str, Any]:
    metadata = _load_json_file(episode_dir / "metadata.json")
    datasets = _collect_episode_datasets(episode_dir, metadata)
    streams = _collect_npy_streams(datasets)
    primary_stream = streams[0]["path"] if streams else None
    timestamps = _load_timestamps(episode_dir, primary_stream)
    episode_size_bytes = sum(int(item.get("size_bytes", 0) or 0) for item in datasets)

    steps = max([int(stream["shape"][0]) for stream in streams] or [0])
    start_ts = float(timestamps[0]) if len(timestamps) > 0 else None
    end_ts = float(timestamps[-1]) if len(timestamps) > 0 else None
    if start_ts is None:
        start_ts = _normalize_scalar(metadata.get("external_start_time") or metadata.get("hand_start_time") or metadata.get("robot_start_time") or 0.0) or None
        if start_ts == 0.0:
            start_ts = _episode_start_from_name(episode_dir.name)
    if end_ts is None:
        end_ts = _normalize_scalar(metadata.get("external_end_time") or metadata.get("hand_end_time") or metadata.get("robot_end_time") or 0.0) or None
        if end_ts == 0.0:
            end_ts = start_ts

    duration_s = float(end_ts - start_ts) if start_ts is not None and end_ts is not None and end_ts >= start_ts else 0.0
    rel_path = _safe_relative_path(episode_dir)
    instruction = metadata.get("instruction") or (episode_dir.parents[1].name if len(episode_dir.parents) > 1 else "")
    label = f"{episode_dir.name} / {instruction}" if instruction else episode_dir.name
    return {
        "id": rel_path,
        "label": label,
        "group_path": rel_path,
        "steps": steps,
        "duration_s": duration_s,
        "fps_estimate": _estimate_fps(timestamps) or metadata.get("frequency"),
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "start_time_utc": _format_timestamp(start_ts),
        "end_time_utc": _format_timestamp(end_ts),
        "stream_count": len(streams),
        "episode_size_bytes": episode_size_bytes,
        "streams": streams,
        "group_attrs": metadata,
        "file_attrs": {"recording_root": str(root), "episode_dir": str(episode_dir)},
        "datasets": datasets,
        "source_dataset": PSEUDO_DATASET_NAME,
        "is_converting": False,
    }


def _episode_sort_key(episode: dict[str, Any]) -> tuple[float, str]:
    raw_start = episode.get("start_timestamp")
    raw_end = episode.get("end_timestamp")
    timestamp = raw_start if raw_start is not None else raw_end
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        ts = 0.0
    if not np.isfinite(ts):
        ts = 0.0
    return (ts, str(episode.get("id", "")))


def _cached_dataset_summary(path: Path) -> dict[str, Any]:
    latest_mtime, total_size = _recordings_stamp(path)
    cache_key = (str(path), latest_mtime, total_size)
    cached = _SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    episode_dirs = _iter_episode_dirs(path)
    episodes = [_build_npy_episode_payload(path, episode_dir) for episode_dir in episode_dirs]
    episodes.sort(key=_episode_sort_key, reverse=True)

    summary = {
        "dataset": PSEUDO_DATASET_NAME,
        "dataset_path": str(path),
        "episodes": episodes,
        "episode_count": len(episodes),
        "file_size_bytes": total_size,
        "updated_at_ns": latest_mtime,
    }

    _SUMMARY_CACHE.clear()
    _SUMMARY_CACHE[cache_key] = summary
    return summary


def _resolve_episode_dir(episode_id: str) -> Path:
    episode_dir = (DATA_DIR / episode_id).resolve()
    if DATA_DIR != episode_dir and DATA_DIR not in episode_dir.parents:
        raise HTTPException(status_code=400, detail="Episode path must stay inside the data directory.")
    if not episode_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")
    return episode_dir


def _normalize_depth_frame(frame: np.ndarray) -> np.ndarray:
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return np.zeros(frame.shape, dtype=np.uint8)

    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.uint8)

    scaled = np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _frame_to_image(frame: np.ndarray) -> tuple[Image.Image, str]:
    if frame.ndim == 2:
        return Image.fromarray(_normalize_depth_frame(frame), mode="L"), "png"
    if frame.ndim == 3 and frame.shape[2] == 1:
        return Image.fromarray(_normalize_depth_frame(frame[:, :, 0]), mode="L"), "png"
    if frame.ndim == 3 and frame.shape[2] in {3, 4}:
        return Image.fromarray(frame.astype(np.uint8)), "jpeg"
    raise HTTPException(status_code=400, detail=f"Unsupported frame shape: {frame.shape}")


def _resize_image(image: Image.Image, max_width: int | None) -> Image.Image:
    if max_width is None or max_width <= 0 or image.width <= max_width:
        return image

    height = max(1, round(image.height * max_width / image.width))
    return image.resize((max_width, height), Image.Resampling.BILINEAR)


def _read_exact_stdout(stream, expected_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < expected_bytes:
        chunk = stream.read(min(1024 * 1024, expected_bytes - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total != expected_bytes:
        raise HTTPException(status_code=400, detail=f"Expected {expected_bytes} bytes from ffmpeg, received {total}.")
    return b"".join(chunks)


def _find_vision_entry(metadata: dict[str, Any], stream_name: str) -> dict[str, Any] | None:
    vision_files = metadata.get("vision_files") or {}
    for entry in vision_files.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("path") or "") == stream_name:
            return entry
    return None


def _decode_video_frame(
    stream_path: Path,
    *,
    frame_index: int,
    shape: list[int],
    pix_fmt: str,
    dtype: np.dtype,
) -> np.ndarray:
    ffmpeg_bin = _require_ffmpeg_binary(FFMPEG_BIN)
    height = int(shape[1])
    width = int(shape[2])
    channels = int(shape[3]) if len(shape) == 4 else 1
    expected_bytes = height * width * channels * np.dtype(dtype).itemsize
    cmd = [
        ffmpeg_bin,
        "-loglevel",
        "error",
        "-i",
        str(stream_path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-vsync",
        "0",
        "-",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdout is not None
        raw_bytes = _read_exact_stdout(proc.stdout, expected_bytes)
        stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
        return_code = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise

    if return_code != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode frame from {stream_path.name}: {stderr_text.strip() or 'unknown ffmpeg error'}",
        )

    array = np.frombuffer(raw_bytes, dtype=dtype)
    return array.reshape((height, width, channels) if channels > 1 else (height, width)).copy()


@functools.lru_cache(maxsize=64)
def _decode_video_chunk(
    dataset_path_str: str,
    dataset_mtime_ns: int,
    dataset_size: int,
    episode: str,
    stream: str,
    chunk_start: int,
) -> tuple[np.ndarray, ...]:
    del dataset_mtime_ns, dataset_size

    dataset_path = Path(dataset_path_str)
    episode_dir = (dataset_path / episode).resolve()
    if dataset_path != episode_dir and dataset_path not in episode_dir.parents:
        raise HTTPException(status_code=400, detail="Episode path must stay inside the data directory.")

    stream_path = (episode_dir / stream).resolve()
    if episode_dir != stream_path.parent:
        raise HTTPException(status_code=400, detail="Stream path must stay inside the episode folder.")
    if not stream_path.is_file():
        raise HTTPException(status_code=404, detail=f"Stream not found: {stream}")

    metadata = _load_json_file(episode_dir / "metadata.json")
    entry = _find_vision_entry(metadata, stream)
    if not entry:
        raise HTTPException(status_code=404, detail=f"No metadata found for stream {stream}")

    shape = list(entry.get("shape") or [])
    if len(shape) < 3:
        raise HTTPException(status_code=400, detail=f"Invalid shape metadata for stream {stream}")

    total_frames = int(shape[0])
    if chunk_start >= total_frames:
        raise HTTPException(status_code=400, detail="frame_index is out of range.")

    frame_count = min(REPLAY_VIDEO_CHUNK_SIZE, total_frames - chunk_start)
    dtype_name = str(entry.get("dtype") or "")
    if not dtype_name:
        raise HTTPException(status_code=400, detail=f"Missing dtype metadata for stream {stream}")

    dtype = np.uint8 if (len(shape) == 4 and int(shape[3]) in {3, 4}) else np.uint16
    pix_fmt = "gray16le"
    channels = 1
    if dtype is np.uint8:
        channel_order = str(entry.get("channel_order", "RGB")).lower()
        pix_fmt = "rgb24" if channel_order == "rgb" else "bgr24"
        channels = int(shape[3])

    ffmpeg_bin = _require_ffmpeg_binary(FFMPEG_BIN)
    height = int(shape[1])
    width = int(shape[2])
    expected_bytes = frame_count * height * width * channels * np.dtype(dtype).itemsize
    cmd = [
        ffmpeg_bin,
        "-loglevel",
        "error",
        "-i",
        str(stream_path),
        "-vf",
        f"select=between(n\\,{chunk_start}\\,{chunk_start + frame_count - 1})",
        "-frames:v",
        str(frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-vsync",
        "0",
        "-",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdout is not None
        raw_bytes = _read_exact_stdout(proc.stdout, expected_bytes)
        stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
        return_code = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise

    if return_code != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode replay chunk from {stream_path.name}: {stderr_text.strip() or 'unknown ffmpeg error'}",
        )

    flat = np.frombuffer(raw_bytes, dtype=dtype)
    if channels > 1:
        frames = flat.reshape((frame_count, height, width, channels))
    else:
        frames = flat.reshape((frame_count, height, width))
    return tuple(np.asarray(frame).copy() for frame in frames)


def _schedule_chunk_prefetch(
    dataset_path_str: str,
    dataset_mtime_ns: int,
    dataset_size: int,
    episode: str,
    stream: str,
    *,
    next_chunk_start: int,
    total_frames: int,
) -> None:
    if _REPLAY_PREFETCH_EXECUTOR is None:
        return
    if next_chunk_start >= total_frames:
        return

    key = (
        dataset_path_str,
        dataset_mtime_ns,
        dataset_size,
        episode,
        stream,
        next_chunk_start,
    )
    with _REPLAY_PREFETCH_LOCK:
        if key in _REPLAY_PREFETCH_PENDING:
            return
        _REPLAY_PREFETCH_PENDING.add(key)

    def _prefetch_task() -> None:
        try:
            _decode_video_chunk(*key)
        except Exception:
            # Best-effort optimization; never fail request path because prefetch failed.
            pass
        finally:
            with _REPLAY_PREFETCH_LOCK:
                _REPLAY_PREFETCH_PENDING.discard(key)

    try:
        _REPLAY_PREFETCH_EXECUTOR.submit(_prefetch_task)
    except Exception:
        with _REPLAY_PREFETCH_LOCK:
            _REPLAY_PREFETCH_PENDING.discard(key)


@functools.lru_cache(maxsize=128)
def _encoded_frame_bytes(
    dataset_path_str: str,
    dataset_mtime_ns: int,
    dataset_size: int,
    episode: str,
    stream: str,
    frame_index: int,
    max_width: int | None,
) -> tuple[bytes, str]:
    dataset_path = Path(dataset_path_str)
    episode_dir = (dataset_path / episode).resolve()
    if dataset_path != episode_dir and dataset_path not in episode_dir.parents:
        raise HTTPException(status_code=400, detail="Episode path must stay inside the data directory.")

    stream_path = (episode_dir / stream).resolve()
    if episode_dir != stream_path.parent:
        raise HTTPException(status_code=400, detail="Stream path must stay inside the episode folder.")
    if not stream_path.is_file():
        raise HTTPException(status_code=404, detail=f"Stream not found: {stream}")

    if stream_path.suffix == ".npy":
        try:
            item = np.load(stream_path, mmap_mode="r")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read stream {stream}: {exc}") from exc
        if item.ndim < 3 or frame_index >= item.shape[0]:
            raise HTTPException(status_code=400, detail="frame_index is out of range.")
        frame = np.asarray(item[frame_index])
    elif stream_path.suffix in {".mkv", ".mp4"}:
        metadata = _load_json_file(episode_dir / "metadata.json")
        entry = _find_vision_entry(metadata, stream)
        if not entry:
            raise HTTPException(status_code=404, detail=f"No metadata found for stream {stream}")
        shape = list(entry.get("shape") or [])
        total_frames = int(shape[0]) if len(shape) >= 1 else 0
        if len(shape) < 3 or frame_index >= total_frames:
            raise HTTPException(status_code=400, detail="frame_index is out of range.")
        chunk_start = (frame_index // REPLAY_VIDEO_CHUNK_SIZE) * REPLAY_VIDEO_CHUNK_SIZE
        decoded_frames = _decode_video_chunk(
            dataset_path_str,
            dataset_mtime_ns,
            dataset_size,
            episode,
            stream,
            chunk_start,
        )
        if frame_index == chunk_start:
            _schedule_chunk_prefetch(
                dataset_path_str,
                dataset_mtime_ns,
                dataset_size,
                episode,
                stream,
                next_chunk_start=chunk_start + REPLAY_VIDEO_CHUNK_SIZE,
                total_frames=total_frames,
            )
        frame = np.asarray(decoded_frames[frame_index - chunk_start])
    else:
        raise HTTPException(status_code=404, detail=f"Unsupported stream type: {stream_path.suffix}")

    image, output_format = _frame_to_image(frame)
    image = _resize_image(image, max_width)

    buffer = io.BytesIO()
    if output_format == "jpeg":
        image.save(buffer, format="JPEG", quality=80, optimize=False)
        media_type = "image/jpeg"
    else:
        image.save(buffer, format="PNG")
        media_type = "image/png"
    return buffer.getvalue(), media_type


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def config() -> JSONResponse:
    status = _RECORDING_MANAGER.status()
    return JSONResponse({
        "data_dir": str(DATA_DIR),
        "default_dataset": DEFAULT_DATASET,
        "recording_defaults": RecordingInitRequest().model_dump(),
        "recording_backend": status.get("backend", RECORDING_BACKEND_MODE),
        "live_preview_supported": bool(status.get("live_preview_supported", True)),
    })


@app.get("/api/datasets")
def list_datasets() -> JSONResponse:
    latest_mtime, total_size = _recordings_stamp(DATA_DIR) if DATA_DIR.is_dir() else (0, 0)
    payload = {
        "data_dir": str(DATA_DIR),
        "datasets": [
            {
                "name": PSEUDO_DATASET_NAME,
                "path": str(DATA_DIR),
                "size_bytes": total_size,
                "updated_at_ns": latest_mtime,
            }
            for path in _iter_dataset_files()
        ],
    }
    return JSONResponse(payload)


@app.get("/api/summary")
def dataset_summary(dataset: str = Query(...)) -> JSONResponse:
    path = _resolve_recordings_dataset(dataset)
    return JSONResponse(_cached_dataset_summary(path))


@app.get("/api/episode")
def episode_detail(dataset: str = Query(...), episode: str = Query(...)) -> JSONResponse:
    path = _resolve_recordings_dataset(dataset)
    summary = _cached_dataset_summary(path)
    for item in summary["episodes"]:
        if item["id"] == episode:
            return JSONResponse(item)
    raise HTTPException(status_code=404, detail=f"Episode not found: {episode}")


@app.post("/api/episode/delete")
def episode_delete(payload: EpisodeDeleteRequest) -> JSONResponse:
    path = _resolve_recordings_dataset(payload.dataset)
    episode = payload.episode.strip()
    if not episode:
        raise HTTPException(status_code=400, detail="episode is required.")

    episode_dir = (path / episode).resolve()
    if path != episode_dir and path not in episode_dir.parents:
        raise HTTPException(status_code=400, detail="Episode path must stay inside the data directory.")
    if not episode_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode}")
    shutil.rmtree(episode_dir)
    _SUMMARY_CACHE.clear()
    _encoded_frame_bytes.cache_clear()
    _decode_video_chunk.cache_clear()
    return JSONResponse({"deleted": True, "episode": episode})


@app.get("/api/frame")
def frame(
    dataset: str = Query(...),
    episode: str = Query(...),
    stream: str = Query(...),
    frame_index: int = Query(..., ge=0),
    max_width: int | None = Query(default=960, ge=1, le=4096),
) -> Response:
    path = _resolve_recordings_dataset(dataset)
    latest_mtime, total_size = _recordings_stamp(path)
    image_bytes, media_type = _encoded_frame_bytes(
        str(path),
        latest_mtime,
        total_size,
        episode,
        stream,
        frame_index,
        max_width,
    )
    return Response(content=image_bytes, media_type=media_type)


@app.get("/api/recording/status")
def recording_status() -> JSONResponse:
    return JSONResponse(_RECORDING_MANAGER.status())


@app.post("/api/recording/initialize")
def recording_initialize(payload: RecordingInitRequest) -> JSONResponse:
    try:
        return JSONResponse(_RECORDING_MANAGER.initialize(payload))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/recording/start")
def recording_start(payload: RecordingStartRequest | None = None) -> JSONResponse:
    instruction = payload.instruction if payload is not None else ""
    return JSONResponse(_RECORDING_MANAGER.start_episode(instruction=instruction))


@app.post("/api/recording/stop")
def recording_stop() -> JSONResponse:
    return JSONResponse(_RECORDING_MANAGER.stop_episode())


@app.post("/api/recording/reset")
def recording_reset(payload: RecordingResetRequest | None = None) -> JSONResponse:
    return JSONResponse(_RECORDING_MANAGER.reset_pose(reset_config=payload))


@app.post("/api/recording/stop-rtde-motion")
def recording_stop_rtde_motion() -> JSONResponse:
    if not hasattr(_RECORDING_MANAGER, "stop_rtde_motion"):
        raise HTTPException(status_code=409, detail="UR speedStop is only available for the SpaceMouse tmux backend.")
    return JSONResponse(_RECORDING_MANAGER.stop_rtde_motion())


@app.post("/api/recording/reconnect-rtde")
def recording_reconnect_rtde() -> JSONResponse:
    if not hasattr(_RECORDING_MANAGER, "reconnect_rtde"):
        raise HTTPException(status_code=409, detail="UR speedStop is only available for the SpaceMouse tmux backend.")
    return JSONResponse(_RECORDING_MANAGER.reconnect_rtde())


@app.post("/api/recording/label")
def recording_label() -> JSONResponse:
    return JSONResponse(_RECORDING_MANAGER.label_subtask())


@app.post("/api/recording/delete")
def recording_delete() -> JSONResponse:
    return JSONResponse(_RECORDING_MANAGER.delete_last_episode())


@app.post("/api/recording/shutdown")
def recording_shutdown() -> JSONResponse:
    return JSONResponse(_RECORDING_MANAGER.shutdown(save_current=False))


@app.post("/api/recording/shutdown-save")
def recording_shutdown_save() -> JSONResponse:
    if hasattr(_RECORDING_MANAGER, "shutdown_after_save_async"):
        return JSONResponse(_RECORDING_MANAGER.shutdown_after_save_async())
    return JSONResponse(_RECORDING_MANAGER.shutdown(save_current=True))


@app.get("/api/recording/frame")
def recording_frame(
    camera: str = Query(..., pattern="^(hand|wrist|external)$"),
    kind: str = Query(..., pattern="^(rgb|depth)$"),
    max_width: int | None = Query(default=640, ge=1, le=4096),
) -> Response:
    frame_array, _timestamp = _RECORDING_MANAGER.latest_frame(camera, kind)
    image, output_format = _frame_to_image(frame_array)
    image = _resize_image(image, max_width)

    buffer = io.BytesIO()
    if output_format == "jpeg":
        image.save(buffer, format="JPEG", quality=80, optimize=False)
        media_type = "image/jpeg"
    else:
        image.save(buffer, format="PNG")
        media_type = "image/png"
    return Response(content=buffer.getvalue(), media_type=media_type)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local NPY recordings viewer.")
    parser.add_argument(
        "--data-dir",
        default="",
        help="Recordings root containing YYYYMMDD/<instruction>/camera_npy/<episode> folders.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Deprecated compatibility option; use --data-dir for NPY recordings.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the web server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the web server.")
    parser.set_defaults(reload=False)
    parser.add_argument(
        "--server-reload",
        dest="reload",
        action="store_true",
        help="Enable web server code auto-reload while developing.",
    )
    parser.add_argument(
        "--no-server-reload",
        dest="reload",
        action="store_false",
        help="Disable web server code auto-reload.",
    )
    return parser.parse_args()


def _configure_from_args(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else DATA_DIR
    default_dataset = DEFAULT_DATASET

    if args.dataset:
        dataset_path = Path(args.dataset).expanduser().resolve()
        if dataset_path.is_dir():
            data_dir = dataset_path
        else:
            raise SystemExit(f"Expected a recordings directory, got: {dataset_path}")
        default_dataset = PSEUDO_DATASET_NAME
    elif not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    os.environ["DATAFOUNDRY_DATA_DIR"] = str(data_dir)
    os.environ["DATAFOUNDRY_DEFAULT_DATASET"] = default_dataset
    _set_runtime_paths(data_dir, default_dataset)


if __name__ == "__main__":
    args = _parse_args()
    _configure_from_args(args)

    import uvicorn

    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)
