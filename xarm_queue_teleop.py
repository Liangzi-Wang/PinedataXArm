from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any

import numpy as np

try:
    from xarm.wrapper import XArmAPI
except ImportError as exc:
    XArmAPI = None
    XARM_IMPORT_ERROR: Exception | None = exc
else:
    XARM_IMPORT_ERROR = None


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def axis_mask(raw: str) -> np.ndarray:
    try:
        values = [float(item.strip()) for item in raw.split(",")]
    except ValueError:
        values = [1.0, 1.0, 1.0]
    values = (values + [1.0, 1.0, 1.0])[:3]
    return np.asarray(values, dtype=np.float64)


def parse_axis_expr(expr: str) -> tuple[float, str]:
    expr = expr.strip().lower()
    sign = 1.0
    if expr.startswith("+"):
        expr = expr[1:].strip()
    elif expr.startswith("-"):
        sign = -1.0
        expr = expr[1:].strip()
    if not expr:
        raise ValueError
    return sign, expr


def axis_map(raw: str | None, default: str) -> np.ndarray:
    spec = (raw or default).strip()
    source_index = {"x": 0, "y": 1, "z": 2}
    matrix = np.zeros((3, 3), dtype=np.float64)
    try:
        expressions = [item.strip().lower() for item in spec.split(",")]
        if len(expressions) != 3:
            raise ValueError
        for row, expr in enumerate(expressions):
            sign, axis = parse_axis_expr(expr)
            matrix[row, source_index[axis]] = sign
        if not np.allclose(np.abs(matrix).sum(axis=0), 1.0) or not np.allclose(np.abs(matrix).sum(axis=1), 1.0):
            raise ValueError
        return matrix
    except (KeyError, ValueError):
        if raw is not None:
            print(f"[xArm] ignoring invalid XARM_COMMAND_TRANSLATION_MAP={raw!r}; using {default!r}", flush=True)
        return axis_map(None, default)


def axis_vector(raw: str | None, default: str) -> np.ndarray:
    try:
        sign, axis = parse_axis_expr(raw or default)
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    except (KeyError, ValueError):
        if raw is not None:
            print(f"[xArm] ignoring invalid XARM_TOOL_TWIST_AXIS={raw!r}; using {default!r}", flush=True)
        return axis_vector(None, default)
    vector = np.zeros(3, dtype=np.float64)
    vector[axis_index] = sign
    return vector


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


class SpaceMouseQueueManager(BaseManager):
    pass


SpaceMouseQueueManager.register("get_status_queue")

# Match the UR teleop speedL SCALE_FACTOR=0.5 m/s used in temp/pine_data.
UR_TELEOP_TRANSLATION_SPEED_MM_S = 500.0
UR_TELEOP_ROTATION_SPEED_DEG_S = float(np.rad2deg(0.60))

# Fixed reset from UI.jpg plus the hidden 7th joint recovered from RAM.zip.
# Use joints for motion; the UI TCP pose is meters + rotvec and must not be
# passed directly to xArmAPI.set_position().
XARM_DEFAULT_RESET_JOINTS_RAD = np.asarray(
    [-0.28, -0.17, -0.02, 0.94, 0.05, 1.10, 2.921741],
    dtype=np.float64,
)


class QueueXArmTeleop:
    def __init__(self, status_file: Path | None = None) -> None:
        if XArmAPI is None:
            raise RuntimeError(f"xarm-python-sdk is not installed: {XARM_IMPORT_ERROR}")
        self.robot_ip = os.getenv("XARM_IP", os.getenv("ROBOT_IP", "192.168.1.206")).strip()
        self.speed_mm_s = max(1.0, env_float("XARM_TELEOP_SPEED", UR_TELEOP_TRANSLATION_SPEED_MM_S))
        self.angular_speed_deg_s = max(1.0, env_float("XARM_TELEOP_ANGULAR_SPEED", UR_TELEOP_ROTATION_SPEED_DEG_S))
        self.move_acc_mm_s2 = max(1.0, env_float("XARM_MOVE_ACCELERATION", 2000.0))
        self.command_period_s = max(0.005, env_float("XARM_COMMAND_PERIOD_S", 0.01))
        self.max_step_mm = max(0.1, env_float("XARM_MAX_STEP_MM", self.speed_mm_s * self.command_period_s))
        self.max_rotation_step_deg = max(
            0.1,
            env_float("XARM_MAX_ROTATION_STEP_DEG", self.angular_speed_deg_s * self.command_period_s),
        )
        self.spacemouse_timeout_s = max(0.0, env_float("SPACEMOUSE_CONTROL_TIMEOUT_S", 0.30))
        self.axis_mask = axis_mask(os.getenv("XARM_TRANSLATION_AXIS_MASK", "1,1,1"))
        self.rotation_axis_mask = axis_mask(os.getenv("XARM_ROTATION_AXIS_MASK", "1,1,1"))
        self.command_translation_map_spec = os.getenv("XARM_COMMAND_TRANSLATION_MAP", "-y,x,z").strip()
        self.command_translation_map = axis_map(self.command_translation_map_spec, "-y,x,z")
        self.state_translation_map = np.linalg.inv(self.command_translation_map)
        self.command_rotation_map_spec = os.getenv("XARM_COMMAND_ROTATION_MAP", "x,y,z").strip()
        self.command_rotation_map = axis_map(self.command_rotation_map_spec, "x,y,z")
        self.state_rotation_map = np.linalg.inv(self.command_rotation_map)
        self.use_base_relative_aa = env_bool("XARM_USE_BASE_RELATIVE_AA", True)
        self.use_tool_twist_aa = env_bool("XARM_USE_TOOL_TWIST_AA", False)
        self.tool_twist_axis_spec = os.getenv("XARM_TOOL_TWIST_AXIS", "-z").strip()
        self.tool_twist_axis = axis_vector(self.tool_twist_axis_spec, "-z")
        self.control_mode = os.getenv("XARM_TELEOP_CONTROL_MODE", "servo").strip().lower()
        if self.control_mode not in {"servo", "position"}:
            self.control_mode = "servo"
        self.enable_gripper_control = env_bool("ENABLE_SPACEMOUSE_GRIPPER_CONTROL", True)
        self.gripper_open_pos = env_int("XARM_GRIPPER_OPEN_POS", 850)
        self.gripper_close_pos = env_int("XARM_GRIPPER_CLOSE_POS", 0)
        self.gripper_step = max(1, env_int("XARM_GRIPPER_STEP", 50))
        self.gripper_speed = max(1, env_int("XARM_GRIPPER_SPEED", 500))
        self.gripper_repeat_period_s = max(0.01, env_float("XARM_GRIPPER_REPEAT_PERIOD_S", 0.10))
        self.reset_joint_speed_rad_s = max(0.01, env_float("XARM_RESET_JOINT_SPEED_RAD_S", 0.5))
        self.reset_joint_acc_rad_s2 = max(0.01, env_float("XARM_RESET_JOINT_ACCEL_RAD_S2", 1.0))
        self.reset_dry_run = env_bool("XARM_RESET_DRY_RUN", False)
        self.status_file = status_file
        self.status_period_s = 1.0 / max(1.0, env_float("XARM_STATUS_FILE_HZ", 50.0))
        self.stop_on_spacemouse_idle = env_bool("XARM_STOP_ON_SPACEMOUSE_IDLE", True)
        self.arm: Any = XArmAPI(self.robot_ip)
        if not self.arm.connected:
            raise RuntimeError(f"failed to connect to xArm at {self.robot_ip}")
        self._last_motion_error_time = 0.0
        self._last_servo_recover_time = 0.0
        self.arm.clean_error()
        self.arm.clean_warn()
        self.configure_motion_mode()
        self.reset_joints = self.read_configured_reset_joints()
        print(f"[xArm] connected to {self.robot_ip}")
        print(
            f"[xArm] control={self.control_mode}, speed={self.speed_mm_s:.1f} mm/s, "
            f"angular={self.angular_speed_deg_s:.1f} deg/s, "
            f"command_hz={1.0 / self.command_period_s:.1f}"
        )
        print(f"[xArm] command_translation_map={self.command_translation_map_spec}")
        print(
            f"[xArm] command_rotation_map={self.command_rotation_map_spec}, "
            f"base_relative_aa={int(self.use_base_relative_aa)}, "
            f"tool_twist_aa={int(self.use_tool_twist_aa)}, axis={self.tool_twist_axis_spec}"
        )

        self.latest_command = {
            "timestamp": None,
            "step_mm": [0.0, 0.0, 0.0],
            "xarm_step_mm": [0.0, 0.0, 0.0],
            "rotation_step_deg": [0.0, 0.0, 0.0],
            "xarm_rotation_step_deg": [0.0, 0.0, 0.0],
            "twist_command": [0.0] * 6,
            "target_pose": [],
            "xarm_target_pose": [],
            "orientation_command_mode": "",
            "source": "idle",
        }
        self.latest_gripper_position: float | None = None
        self.latest_gripper_normalized_position: float | None = None
        self._initialize_gripper_status()
        self._last_command_time = 0.0
        self._last_status_time = 0.0
        self._last_gripper_command_time = 0.0
        self._last_status_write_time = 0.0
        self._prev_left_button = False
        self._prev_right_button = False
        self._spacemouse_motion_active = False
        self._teleop_target_pose: np.ndarray | None = None
        self._last_motion_direction = np.zeros(6, dtype=np.float64)
        self.subtask_reset = {
            "active_segment_index": 0,
            "noise_xyz_m": 0.0,
            "last_reset_timestamp": None,
            "status": "ready" if self.reset_joints is not None else "unavailable",
        }

    def read_configured_reset_joints(self) -> np.ndarray | None:
        raw_joints = os.getenv("XARM_RESET_JOINTS", "").strip()
        if raw_joints:
            try:
                values = [float(item.strip()) for item in raw_joints.split(",")]
            except ValueError:
                values = []
            if len(values) >= len(XARM_DEFAULT_RESET_JOINTS_RAD):
                return np.asarray(values[: len(XARM_DEFAULT_RESET_JOINTS_RAD)], dtype=np.float64)
            print(
                f"[xArm] ignoring invalid XARM_RESET_JOINTS={raw_joints!r}; "
                f"expected {len(XARM_DEFAULT_RESET_JOINTS_RAD)} joint angles in radians",
                flush=True,
            )
        return XARM_DEFAULT_RESET_JOINTS_RAD.copy()

    def command_translation(self, semantic_translation: np.ndarray) -> np.ndarray:
        return self.command_translation_map @ semantic_translation

    def semantic_translation(self, xarm_translation: np.ndarray) -> np.ndarray:
        return self.state_translation_map @ xarm_translation

    def command_rotation(self, semantic_rotation: np.ndarray) -> np.ndarray:
        return self.command_rotation_map @ semantic_rotation

    def semantic_rotation(self, xarm_rotation: np.ndarray) -> np.ndarray:
        return self.state_rotation_map @ xarm_rotation

    def semantic_pose(self, xarm_pose: np.ndarray) -> np.ndarray:
        pose = xarm_pose.copy()
        pose[:3] = self.semantic_translation(pose[:3])
        return pose

    def semantic_pose_aa(self, xarm_pose_aa: np.ndarray) -> np.ndarray:
        pose = xarm_pose_aa.copy()
        pose[:3] = self.semantic_translation(pose[:3]) / 1000.0
        pose[3:6] = self.semantic_rotation(pose[3:6])
        return pose

    def configure_motion_mode(self) -> None:
        self.arm.motion_enable(enable=True)
        if self.control_mode == "servo":
            # Match the xArm SDK servo examples: enter a normal ready state
            # first, then switch into servo mode before streaming targets.
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.1)
            self.arm.set_mode(1)
            self.arm.set_state(0)
            time.sleep(0.1)
            self.seed_servo_pose()
        else:
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(1.0)

    def seed_servo_pose(self) -> None:
        pose, code = self.position()
        if pose is None:
            print(f"[xArm] failed to read position before servo seed, code={code}", flush=True)
            return
        code = self.arm.set_servo_cartesian(
            pose.tolist(),
            speed=self.speed_mm_s,
            mvacc=self.move_acc_mm_s2,
            is_radian=False,
        )
        if code != 0:
            print(f"[xArm] set_servo_cartesian seed failed, code={code}; {self.arm_diagnostics()}", flush=True)

    def arm_diagnostics(self) -> str:
        parts = []
        try:
            parts.append(f"state={self.arm.get_state()}")
        except Exception:
            pass
        try:
            parts.append(f"err_warn={self.arm.get_err_warn_code()}")
        except Exception:
            pass
        return ", ".join(parts) if parts else "diagnostics unavailable"

    def recover_servo_mode(self) -> None:
        if self.control_mode != "servo":
            return
        now = time.monotonic()
        if now - self._last_servo_recover_time < 1.0:
            return
        self._last_servo_recover_time = now
        try:
            self.arm.clean_error()
            self.arm.clean_warn()
            self.configure_motion_mode()
            print("[xArm] re-entered servo mode after command failure", flush=True)
        except Exception as exc:
            print(f"[xArm] servo recovery failed: {exc}", flush=True)

    def move_to_reset_pose(self, segment_index: int = 0, noise_xyz_m: float = 0.0) -> bool:
        effective_noise_xyz_m = float(noise_xyz_m)
        if not np.isfinite(effective_noise_xyz_m) or effective_noise_xyz_m < 0.0:
            effective_noise_xyz_m = 0.0
        applied_noise_xyz_m = [0.0, 0.0, 0.0]
        if self.reset_joints is None:
            print("[xArm] reset requested, but no reset joints are available", flush=True)
            self.subtask_reset.update({
                "active_segment_index": int(segment_index),
                "noise_xyz_m": effective_noise_xyz_m,
                "applied_noise_xyz_m": applied_noise_xyz_m,
                "last_reset_timestamp": time.time(),
                "status": "unavailable",
            })
            self.write_status(force=True)
            return False

        target = self.reset_joints.copy()
        self.stop_motion("reset")
        self._teleop_target_pose = None
        self._last_motion_direction = np.zeros(6, dtype=np.float64)
        self.subtask_reset.update({
            "active_segment_index": int(segment_index),
            "noise_xyz_m": effective_noise_xyz_m,
            "applied_noise_xyz_m": applied_noise_xyz_m,
            "last_reset_timestamp": time.time(),
            "status": "resetting",
        })
        self.write_status(force=True)

        try:
            print(
                f"[xArm] reset target joints(rad)={[round(float(v), 4) for v in target.tolist()]}, "
                f"speed={self.reset_joint_speed_rad_s:.3f} rad/s, "
                f"acc={self.reset_joint_acc_rad_s2:.3f} rad/s^2, "
                f"dry_run={int(self.reset_dry_run)}",
                flush=True,
            )
            if self.reset_dry_run:
                self.subtask_reset["status"] = "ready"
                self.write_status(force=True)
                return True
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.1)
            code = self.arm.set_servo_angle(
                angle=target.tolist(),
                speed=self.reset_joint_speed_rad_s,
                mvacc=self.reset_joint_acc_rad_s2,
                is_radian=True,
                wait=True,
                radius=None,
            )
            if code != 0:
                print(f"[xArm] reset set_servo_angle failed, code={code}; {self.arm_diagnostics()}", flush=True)
                self.subtask_reset["status"] = "failed"
                self.write_status(force=True)
                return False
            if int(segment_index) > 0 and effective_noise_xyz_m > 0.0:
                pose, pose_code = self.position()
                if pose is None:
                    print(f"[xArm] reset TCP noise skipped: failed to read TCP pose, code={pose_code}", flush=True)
                    self.subtask_reset["status"] = "failed"
                    self.write_status(force=True)
                    return False
                noise_m = np.random.uniform(-effective_noise_xyz_m, effective_noise_xyz_m, size=3)
                target_pose = pose.copy()
                # Reset noise is Cartesian-only: never perturb the fixed 7D joint target.
                target_pose[:3] += noise_m * 1000.0
                print(
                    f"[xArm] reset TCP noise(m)={[round(float(v), 6) for v in noise_m.tolist()]}, "
                    f"target_pose={[round(float(v), 4) for v in target_pose.tolist()]}",
                    flush=True,
                )
                code = self.arm.set_position(
                    *target_pose.tolist(),
                    speed=self.speed_mm_s,
                    mvacc=self.move_acc_mm_s2,
                    wait=True,
                )
                if code != 0:
                    print(f"[xArm] reset TCP noise set_position failed, code={code}; {self.arm_diagnostics()}", flush=True)
                    self.subtask_reset["status"] = "failed"
                    self.write_status(force=True)
                    return False
                applied_noise_xyz_m = [round(float(value), 6) for value in noise_m.tolist()]
                self.subtask_reset["applied_noise_xyz_m"] = applied_noise_xyz_m
            if self.control_mode == "servo":
                self.configure_motion_mode()
            else:
                self.arm.set_mode(0)
                self.arm.set_state(0)
            self.subtask_reset["status"] = "ready"
            self.set_zero_command("reset")
            self.write_status(force=True)
            print(f"[xArm] reset complete: joints(rad)={[round(float(v), 4) for v in target.tolist()]}", flush=True)
            return True
        except Exception as exc:
            self.subtask_reset["status"] = "failed"
            self.write_status(force=True)
            print(f"[xArm] reset failed: {exc}", flush=True)
            return False

    def close(self) -> None:
        try:
            self.stop_motion("shutdown")
        finally:
            try:
                self.arm.disconnect()
            except Exception:
                pass

    def position(self) -> tuple[np.ndarray | None, int]:
        code, pose = self.arm.get_position()
        if code != 0:
            return None, int(code)
        return np.asarray(pose, dtype=np.float64), 0

    def position_aa(self) -> tuple[np.ndarray | None, int]:
        if not hasattr(self.arm, "get_position_aa"):
            return None, -1
        code, pose = self.arm.get_position_aa(is_radian=True)
        if code != 0:
            return None, int(code)
        return np.asarray(pose, dtype=np.float64), 0

    def gripper_position(self) -> float:
        for method in ("get_gripper_position", "get_gripper_pos"):
            if not hasattr(self.arm, method):
                continue
            try:
                result = getattr(self.arm, method)()
                if isinstance(result, tuple) and len(result) >= 2 and result[0] == 0:
                    return float(result[1])
                return float(result)
            except Exception:
                pass
        return float("nan")

    def gripper_normalized_position(self, position: float) -> float:
        close_pos = float(self.gripper_close_pos)
        open_pos = float(self.gripper_open_pos)
        span = open_pos - close_pos
        if abs(span) <= 1e-9:
            return 0.0
        normalized = (float(position) - close_pos) / span
        return float(np.clip(normalized, 0.0, 1.0))

    def _initialize_gripper_status(self) -> None:
        if not self.enable_gripper_control:
            return
        position = self.gripper_position()
        if np.isfinite(position):
            self.latest_gripper_position = float(position)
            self.latest_gripper_normalized_position = self.gripper_normalized_position(position)

    def clip_gripper_position(self, position: float) -> int:
        low = min(self.gripper_close_pos, self.gripper_open_pos)
        high = max(self.gripper_close_pos, self.gripper_open_pos)
        return int(round(min(max(float(position), low), high)))

    def set_gripper(self, position: int) -> None:
        if not self.enable_gripper_control:
            return
        clipped = self.clip_gripper_position(position)
        self.arm.set_gripper_position(clipped, speed=self.gripper_speed, wait=False)
        self.latest_gripper_position = float(clipped)
        self.latest_gripper_normalized_position = self.gripper_normalized_position(clipped)

    def current_gripper_position_for_command(self) -> float:
        if self.latest_gripper_position is not None and np.isfinite(self.latest_gripper_position):
            return float(self.latest_gripper_position)
        position = self.gripper_position()
        if np.isfinite(position):
            return position
        return (float(self.gripper_open_pos) + float(self.gripper_close_pos)) / 2.0

    def nudge_gripper(self, direction: int) -> None:
        current = self.current_gripper_position_for_command()
        open_sign = 1 if self.gripper_open_pos >= self.gripper_close_pos else -1
        target = current + float(direction * open_sign * self.gripper_step)
        self.set_gripper(self.clip_gripper_position(target))

    def handle_gripper(self, status: dict[str, Any]) -> None:
        buttons = status.get("buttons") or {}
        left = bool(buttons.get("left", False))
        right = bool(buttons.get("right", False))
        now = time.monotonic()
        if left != right:
            direction = 1 if left else -1
            button_changed = left != self._prev_left_button or right != self._prev_right_button
            repeat_due = now - self._last_gripper_command_time >= self.gripper_repeat_period_s
            if button_changed or repeat_due:
                self.nudge_gripper(direction)
                self._last_gripper_command_time = now
        self._prev_left_button = left
        self._prev_right_button = right

    def gripper_status_payload(self) -> dict[str, Any]:
        connected = bool(self.enable_gripper_control)
        available = self.latest_gripper_position is not None
        return {
            "requested": bool(self.enable_gripper_control),
            "enabled": connected,
            "connected": connected,
            "available": available,
            "status": "streaming" if available else "connected" if connected else "disconnected",
            "ip": self.robot_ip,
            "latest_timestamp": time.time(),
            "latest_position": self.latest_gripper_position,
            "normalized_position": self.latest_gripper_normalized_position,
            "raw_position": self.latest_gripper_position,
            "open_position": self.gripper_open_pos,
            "close_position": self.gripper_close_pos,
            "source": "xarm_queue_teleop",
        }

    def robot_status_payload(self) -> dict[str, Any]:
        command_timestamp = self.latest_command.get("timestamp")
        return {
            "requested": True,
            "enabled": bool(self.arm.connected),
            "connected": bool(self.arm.connected),
            "available": command_timestamp is not None,
            "status": "streaming" if command_timestamp is not None else "connected",
            "ip": self.robot_ip,
            "fps": round(1.0 / self.command_period_s, 3),
            "latest": {
                "timestamp": command_timestamp or time.time(),
                "eef_pose_commanded": self.latest_command.get("target_pose", []),
                "eef_twist_command": self.latest_command.get("twist_command", [0.0] * 6),
            },
        }

    def write_status(self, *, force: bool = False) -> None:
        if self.status_file is None:
            return
        now = time.monotonic()
        if not force and now - self._last_status_write_time < self.status_period_s:
            return
        try:
            existing = (
                json.loads(self.status_file.read_text(encoding="utf-8"))
                if self.status_file.is_file()
                else {}
            )
        except Exception:
            existing = {}
        payload = existing if isinstance(existing, dict) else {}
        payload.update({
            "initialized": True,
            "message": "Queue xArm teleop running.",
            "last_error": "",
            "gripper": self.gripper_status_payload(),
            "robot": self.robot_status_payload(),
            "subtask_reset": self.subtask_reset,
        })
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.status_file.with_suffix(self.status_file.suffix + f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.status_file)
        self._last_status_write_time = now

    def set_zero_command(self, source: str) -> None:
        self.latest_command = {
            "timestamp": time.time(),
            "step_mm": [0.0, 0.0, 0.0],
            "xarm_step_mm": [0.0, 0.0, 0.0],
            "rotation_step_deg": [0.0, 0.0, 0.0],
            "xarm_rotation_step_deg": [0.0, 0.0, 0.0],
            "twist_command": [0.0] * 6,
            "target_pose": [],
            "xarm_target_pose": [],
            "orientation_command_mode": "",
            "source": source,
        }

    def stop_motion(self, source: str) -> None:
        self.set_zero_command(source)
        self._last_motion_direction = np.zeros(6, dtype=np.float64)
        if not self._spacemouse_motion_active:
            self._teleop_target_pose = None
            return
        self._spacemouse_motion_active = False
        self._teleop_target_pose = None
        if not self.stop_on_spacemouse_idle:
            return
        if self.control_mode == "servo":
            return
        try:
            self.arm.set_state(4)
            time.sleep(0.05)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(1 if self.control_mode == "servo" else 0)
            self.arm.set_state(0)
        except Exception as exc:
            print(f"[xArm] stop warning: {exc}")

    def send_relative_motion(self, motion_state: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_command_time < self.command_period_s:
            return
        self._last_command_time = now

        semantic_translation = np.asarray(motion_state[:3], dtype=np.float64) * self.axis_mask
        semantic_rotation = np.asarray(motion_state[3:], dtype=np.float64) * self.rotation_axis_mask
        semantic_translation_norm = float(np.linalg.norm(semantic_translation))
        semantic_rotation_norm = float(np.linalg.norm(semantic_rotation))
        if semantic_translation_norm <= 1e-9 and semantic_rotation_norm <= 1e-9:
            self.stop_motion("idle")
            return

        translation = self.command_translation(semantic_translation)
        rotation = self.command_rotation(semantic_rotation)
        translation_norm = float(np.linalg.norm(translation))
        rotation_norm = float(np.linalg.norm(rotation))

        motion_direction = np.zeros(6, dtype=np.float64)
        if translation_norm > 1e-9:
            motion_direction[:3] = translation / translation_norm
        if rotation_norm > 1e-9:
            motion_direction[3:] = rotation / rotation_norm
        if bool(np.any((motion_direction * self._last_motion_direction) < -0.2)):
            self._teleop_target_pose = None

        semantic_translation_step = np.zeros(3, dtype=np.float64)
        semantic_translation_velocity = np.zeros(3, dtype=np.float64)
        if semantic_translation_norm > 1e-9:
            translation_direction = semantic_translation / semantic_translation_norm
            translation_magnitude = min(semantic_translation_norm, 1.0)
            semantic_translation_velocity = translation_direction * translation_magnitude * (self.speed_mm_s / 1000.0)
            semantic_translation_step = semantic_translation_velocity * 1000.0 * self.command_period_s
            step_norm = float(np.linalg.norm(semantic_translation_step))
            if step_norm > self.max_step_mm:
                semantic_translation_step = semantic_translation_step / step_norm * self.max_step_mm
                semantic_translation_velocity = semantic_translation_step / (1000.0 * self.command_period_s)
        translation_step = self.command_translation(semantic_translation_step)

        semantic_rotation_step = np.zeros(3, dtype=np.float64)
        semantic_rotation_velocity = np.zeros(3, dtype=np.float64)
        if semantic_rotation_norm > 1e-9:
            semantic_rotation_velocity = np.clip(semantic_rotation, -1.0, 1.0) * np.deg2rad(self.angular_speed_deg_s)
            semantic_rotation_step = np.rad2deg(semantic_rotation_velocity * self.command_period_s)
            semantic_rotation_step = np.clip(
                semantic_rotation_step,
                -self.max_rotation_step_deg,
                self.max_rotation_step_deg,
            )
            semantic_rotation_velocity = np.deg2rad(semantic_rotation_step) / self.command_period_s
        rotation_step = self.command_rotation(semantic_rotation_step)
        use_tool_twist_aa = (
            not self.use_base_relative_aa
            and self.use_tool_twist_aa
            and self.control_mode == "servo"
            and hasattr(self.arm, "set_servo_cartesian_aa")
            and translation_norm <= 1e-9
            and abs(rotation_step[2]) > 1e-9
            and abs(rotation_step[2]) >= max(abs(rotation_step[0]), abs(rotation_step[1])) * 1.5
        )

        if self._teleop_target_pose is None:
            pose, code = self.position()
            if pose is None:
                print(f"[xArm] failed to read position before motion, code={code}")
                return
            target = pose.copy()
        else:
            target = self._teleop_target_pose.copy()
        target[:3] += translation_step
        target[3:6] = wrap_degrees(target[3:6] + rotation_step)
        semantic_target = self.semantic_pose(target)
        semantic_target[:3] /= 1000.0
        semantic_target[3:6] = np.deg2rad(semantic_target[3:6])
        pose_aa, _ = self.position_aa()
        if pose_aa is not None:
            target_aa = pose_aa.copy()
            target_aa[:3] += translation_step
            target_aa[3:6] += np.deg2rad(rotation_step)
            semantic_target = self.semantic_pose_aa(target_aa)

        used_base_relative_aa = False
        used_tool_twist_aa = False
        if (
            self.use_base_relative_aa
            and self.control_mode == "servo"
            and hasattr(self.arm, "set_servo_cartesian_aa")
        ):
            code = self.arm.set_servo_cartesian_aa(
                [*translation_step.tolist(), *rotation_step.tolist()],
                speed=self.speed_mm_s,
                mvacc=self.move_acc_mm_s2,
                is_radian=False,
                is_tool_coord=False,
                relative=True,
                wait=False,
            )
            command_name = "set_servo_cartesian_aa(base_relative)"
            used_base_relative_aa = code == 0
        elif use_tool_twist_aa:
            axis_angle = self.tool_twist_axis * rotation_step[2]
            code = self.arm.set_servo_cartesian_aa(
                [0.0, 0.0, 0.0, *axis_angle.tolist()],
                speed=self.angular_speed_deg_s,
                mvacc=self.move_acc_mm_s2,
                is_radian=False,
                is_tool_coord=True,
                relative=True,
                wait=False,
            )
            command_name = "set_servo_cartesian_aa(tool_twist)"
            used_tool_twist_aa = code == 0
        else:
            code = -1
            command_name = ""

        if not (used_base_relative_aa or used_tool_twist_aa) and self.control_mode == "servo" and hasattr(self.arm, "set_servo_cartesian"):
            code = self.arm.set_servo_cartesian(
                target.tolist(),
                speed=self.speed_mm_s,
                mvacc=self.move_acc_mm_s2,
                is_radian=False,
            )
            command_name = "set_servo_cartesian"
        elif not (used_base_relative_aa or used_tool_twist_aa):
            code = self.arm.set_position(
                *target.tolist(),
                speed=self.speed_mm_s,
                mvacc=self.move_acc_mm_s2,
                wait=False,
            )
            command_name = "set_position"
        if code != 0:
            self._spacemouse_motion_active = False
            self._teleop_target_pose = None
            now = time.monotonic()
            if now - self._last_motion_error_time >= 0.5:
                print(f"[xArm] {command_name} failed, code={code}; {self.arm_diagnostics()}", flush=True)
                self._last_motion_error_time = now
            self.recover_servo_mode()
            return

        self._spacemouse_motion_active = True
        self._teleop_target_pose = target.copy()
        self._last_motion_direction = motion_direction
        self.latest_command = {
            "timestamp": time.time(),
            "step_mm": [round(float(value), 6) for value in semantic_translation_step.tolist()],
            "xarm_step_mm": [round(float(value), 6) for value in translation_step.tolist()],
            "rotation_step_deg": [round(float(value), 6) for value in semantic_rotation_step.tolist()],
            "xarm_rotation_step_deg": [round(float(value), 6) for value in rotation_step.tolist()],
            "twist_command": [
                *[round(float(value), 6) for value in semantic_translation_velocity.tolist()],
                *[round(float(value), 6) for value in semantic_rotation_velocity.tolist()],
            ],
            "target_pose": [round(float(value), 6) for value in semantic_target.tolist()],
            "xarm_target_pose": [round(float(value), 6) for value in target.tolist()],
            "orientation_command_mode": (
                "base_axis_angle" if used_base_relative_aa else "tool_axis_angle" if used_tool_twist_aa else "rpy_target"
            ),
            "source": "spacemouse_queue",
        }

    def handle_status(self, status: dict[str, Any]) -> None:
        self._last_status_time = time.monotonic()
        self.handle_gripper(status)
        if not status.get("available", False) or status.get("stale", False):
            self.stop_motion(str(status.get("status") or "unavailable"))
            return
        motion = np.asarray(status.get("motion_state") or [], dtype=np.float64)
        if motion.shape != (6,):
            self.stop_motion("invalid_motion_state")
            return
        if bool(status.get("active", False)):
            self.send_relative_motion(motion)
        else:
            self.stop_motion("idle")

    def check_timeout(self) -> None:
        if self.spacemouse_timeout_s <= 0.0:
            return
        if time.monotonic() - self._last_status_time > self.spacemouse_timeout_s:
            self.stop_motion("queue_timeout")


def drain_latest(status_queue: Any) -> dict[str, Any] | None:
    latest = None
    while True:
        try:
            latest = status_queue.get_nowait()
        except queue.Empty:
            return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Control xArm TCP from a SpaceMouse status queue.")
    parser.add_argument("--host", default=os.getenv("SPACEMOUSE_QUEUE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=env_int("SPACEMOUSE_QUEUE_PORT", 8765))
    parser.add_argument("--authkey", default=os.getenv("SPACEMOUSE_QUEUE_AUTHKEY", "spacemouse"))
    parser.add_argument("--poll-hz", type=float, default=max(1.0, env_float("XARM_QUEUE_POLL_HZ", 250.0)))
    parser.add_argument("--status-file", default=os.getenv("TELEOP_STATUS_FILE", ""))
    args = parser.parse_args()

    manager = SpaceMouseQueueManager(address=(args.host, args.port), authkey=args.authkey.encode("utf-8"))
    manager.connect()
    status_queue = manager.get_status_queue()
    print(f"[xArm] connected to SpaceMouse queue at {args.host}:{args.port}, poll={args.poll_hz:.1f} Hz")

    status_path = Path(args.status_file).expanduser().resolve() if args.status_file else None
    teleop = QueueXArmTeleop(status_file=status_path)
    teleop.write_status(force=True)
    command_queue: queue.Queue[str] = queue.Queue()

    def read_commands() -> None:
        for raw_line in sys.stdin:
            command_queue.put(raw_line.strip())

    def handle_command(raw_command: str) -> bool:
        parts = raw_command.strip().split()
        if not parts:
            return True
        command = parts[0].lower()
        if command == "q":
            return False
        if command == "r":
            segment_index = 0
            noise_xyz_m = 0.0
            if len(parts) >= 2:
                try:
                    segment_index = int(parts[1])
                except ValueError:
                    segment_index = 0
            if len(parts) >= 3:
                try:
                    noise_xyz_m = float(parts[2])
                except ValueError:
                    noise_xyz_m = 0.0
            teleop.move_to_reset_pose(segment_index=segment_index, noise_xyz_m=noise_xyz_m)
            return True
        if command == "u":
            teleop.stop_motion("clear")
            teleop.write_status(force=True)
            print("[xArm] cleared current SpaceMouse motion target", flush=True)
        return True

    threading.Thread(target=read_commands, daemon=True, name="xarm-teleop-stdin").start()
    period_s = 1.0 / float(args.poll_hz)
    try:
        while True:
            while True:
                try:
                    raw_command = command_queue.get_nowait()
                except queue.Empty:
                    break
                if not handle_command(raw_command):
                    raise KeyboardInterrupt

            status = drain_latest(status_queue)
            if status is not None:
                teleop.handle_status(status)
                teleop.write_status()
                command = teleop.latest_command
                if command.get("source") == "spacemouse_queue":
                    print(
                        "[xArm] "
                        f"step={command['step_mm']} rot={command['rotation_step_deg']} "
                        f"gesture={status.get('gesture_translation')}/{status.get('gesture_rotation')}",
                        end="\r",
                        flush=True,
                    )
            else:
                teleop.check_timeout()
                teleop.write_status()
            time.sleep(period_s)
    except KeyboardInterrupt:
        print("\n[xArm] stopping.")
    finally:
        teleop.write_status(force=True)
        teleop.close()


if __name__ == "__main__":
    main()
