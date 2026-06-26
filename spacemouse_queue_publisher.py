from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import queue
import threading
import time
from collections import defaultdict
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any

import numpy as np

try:
    from spnav import spnav_close, spnav_open, spnav_poll_event, SpnavButtonEvent, SpnavMotionEvent
except Exception as exc:
    spnav_close = None
    spnav_open = None
    spnav_poll_event = None
    SpnavButtonEvent = None
    SpnavMotionEvent = None
    SPNAV_IMPORT_ERROR: Exception | None = exc
else:
    SPNAV_IMPORT_ERROR = None


class _CSpnavMotion(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("z", ctypes.c_int),
        ("rx", ctypes.c_int),
        ("ry", ctypes.c_int),
        ("rz", ctypes.c_int),
        ("period", ctypes.c_uint),
        ("data", ctypes.c_uint),
    ]


class _CSpnavButton(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("press", ctypes.c_int),
        ("bnum", ctypes.c_int),
    ]


class _CSpnavEvent(ctypes.Union):
    _fields_ = [
        ("type", ctypes.c_int),
        ("motion", _CSpnavMotion),
        ("button", _CSpnavButton),
    ]


class CtypesSpnavMotionEvent:
    def __init__(self, event: _CSpnavEvent) -> None:
        motion = event.motion
        self.translation = (motion.x, motion.y, motion.z)
        self.rotation = (motion.rx, motion.ry, motion.rz)


class CtypesSpnavButtonEvent:
    def __init__(self, event: _CSpnavEvent) -> None:
        button = event.button
        self.press = bool(button.press)
        self.bnum = int(button.bnum)


def install_ctypes_spnav_backend() -> str:
    global spnav_close, spnav_open, spnav_poll_event, SpnavButtonEvent, SpnavMotionEvent

    candidates = [
        ctypes.util.find_library("spnav"),
        "libspnav.so.0",
        "libspnav.so",
    ]
    lib = None
    load_errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
            break
        except OSError as exc:
            load_errors.append(f"{candidate}: {exc}")
    if lib is None:
        raise RuntimeError("Could not load libspnav. Tried: " + "; ".join(load_errors))

    lib.spnav_open.argtypes = []
    lib.spnav_open.restype = ctypes.c_int
    lib.spnav_close.argtypes = []
    lib.spnav_close.restype = ctypes.c_int
    lib.spnav_poll_event.argtypes = [ctypes.POINTER(_CSpnavEvent)]
    lib.spnav_poll_event.restype = ctypes.c_int

    def ctypes_spnav_open() -> None:
        if lib.spnav_open() == -1:
            raise RuntimeError("spnav_open failed. Is spacenavd running and can this user access it?")

    def ctypes_spnav_close() -> None:
        lib.spnav_close()

    def ctypes_spnav_poll_event() -> Any:
        event = _CSpnavEvent()
        if lib.spnav_poll_event(ctypes.byref(event)) == 0:
            return None
        if int(event.type) == 1:
            return CtypesSpnavMotionEvent(event)
        if int(event.type) == 2:
            return CtypesSpnavButtonEvent(event)
        return None

    spnav_open = ctypes_spnav_open
    spnav_close = ctypes_spnav_close
    spnav_poll_event = ctypes_spnav_poll_event
    SpnavMotionEvent = CtypesSpnavMotionEvent
    SpnavButtonEvent = CtypesSpnavButtonEvent
    return "ctypes-libspnav"


if SPNAV_IMPORT_ERROR is not None:
    try:
        install_ctypes_spnav_backend()
        SPNAV_IMPORT_ERROR = None
    except Exception as exc:
        SPNAV_IMPORT_ERROR = exc


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


def parse_axis_map(raw: str | None, default: str, source_axes: tuple[str, ...], target_axes: tuple[str, ...]) -> np.ndarray:
    spec = (raw or default).strip()
    source_index = {name: idx for idx, name in enumerate(source_axes)}
    target_index = {name: idx for idx, name in enumerate(target_axes)}
    matrix = np.zeros((3, 3), dtype=np.float32)
    try:
        if "=" in spec:
            assignments = [item.strip() for item in spec.split(",") if item.strip()]
            for assignment in assignments:
                target, expr = [part.strip().lower() for part in assignment.split("=", 1)]
                row = target_index[target]
                sign, axis = parse_axis_expr(expr)
                matrix[row, source_index[axis]] = sign
        else:
            expressions = [item.strip().lower() for item in spec.split(",")]
            if len(expressions) != 3:
                raise ValueError
            for row, expr in enumerate(expressions):
                sign, axis = parse_axis_expr(expr)
                matrix[row, source_index[axis]] = sign
        return matrix
    except (KeyError, ValueError):
        if raw is not None:
            return parse_axis_map(None, default, source_axes, target_axes)
        raise


def parse_axis_values(raw: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if raw is None:
        return np.asarray(default, dtype=np.float32)
    try:
        values = [float(item.strip()) for item in raw.split(",")]
    except ValueError:
        return np.asarray(default, dtype=np.float32)
    values = (values + list(default))[:3]
    return np.asarray(values, dtype=np.float32)


def rounded_list(values: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values.tolist()]


def dominant_axis(values: np.ndarray, axis_names: tuple[str, ...]) -> str:
    if values.size == 0 or not np.any(values):
        return ""
    return axis_names[int(np.argmax(np.abs(values)))]


def translation_gesture(motion: np.ndarray, eps: float) -> str:
    x, y, z = [float(value) for value in motion[:3]]
    ax, ay, az = abs(x), abs(y), abs(z)
    if max(ax, ay, az) <= eps:
        return "idle"
    if az > max(ax, ay) * 1.25:
        return "z_up" if z > 0 else "z_down"
    x_active = ax > eps
    y_active = ay > eps
    if x_active and y_active:
        return f"{'right' if x > 0 else 'left'}_{'up' if y > 0 else 'down'}"
    if x_active:
        return "right" if x > 0 else "left"
    if y_active:
        return "up" if y > 0 else "down"
    return "idle"


def rotation_gesture(motion: np.ndarray, eps: float) -> str:
    roll, pitch, yaw = [float(value) for value in motion[3:6]]
    values = np.asarray([roll, pitch, yaw], dtype=np.float32)
    if float(np.max(np.abs(values))) <= eps:
        return "idle"
    axis = int(np.argmax(np.abs(values)))
    if axis == 2:
        return "twist_ccw" if yaw > 0 else "twist_cw"
    if axis == 1:
        return "tilt_right" if pitch > 0 else "tilt_left"
    return "tilt_backward" if roll > 0 else "tilt_forward"


class SpaceMouseReader:
    def __init__(self) -> None:
        self.enabled = env_bool("ENABLE_SPACEMOUSE_TELEOP", True)
        self.max_value = max(1, env_int("SPACEMOUSE_MAX_VALUE", 500))
        self.input_deadband = max(0.0, env_float("SPACEMOUSE_INPUT_DEADBAND", 0.08))
        self.response_exponent = max(0.1, env_float("SPACEMOUSE_RESPONSE_EXPONENT", 1.5))
        self.translation_scale = env_float("TELEOP_TRANSLATION_SPEED_SCALE", 1.0)
        self.rotation_scale = env_float("TELEOP_ROTATION_SPEED_SCALE", 1.0)
        self.downward_z_scale = env_float("TELEOP_DOWNWARD_Z_EXTRA_SCALE", 1.0)
        self.active_eps = max(0.0, env_float("SPACEMOUSE_ACTIVE_EPS", 1e-4))
        self.stale_stop_s = max(0.0, env_float("SPACEMOUSE_STALE_STOP_S", 0.20))
        self.translation_map_spec = os.getenv("SPACEMOUSE_TRANSLATION_MAP", "-z,x,y").strip()
        self.rotation_map_spec = os.getenv("SPACEMOUSE_ROTATION_MAP", "roll=-rx,pitch=-rz,yaw=ry").strip()
        self.rotation_sign_spec = os.getenv("SPACEMOUSE_ROTATION_SIGN", "1,1,1").strip()
        self.translation_map = parse_axis_map(
            self.translation_map_spec,
            "-z,x,y",
            ("x", "y", "z"),
            ("x", "y", "z"),
        )
        self.rotation_map = parse_axis_map(
            self.rotation_map_spec,
            "roll=-rx,pitch=-rz,yaw=ry",
            ("rx", "ry", "rz"),
            ("roll", "pitch", "yaw"),
        )
        self.rotation_sign = parse_axis_values(self.rotation_sign_spec, (1.0, 1.0, 1.0))
        self.button_state: defaultdict[int, bool] = defaultdict(bool)
        self.raw_state = np.zeros(6, dtype=np.float32)
        self.filtered_state = np.zeros(6, dtype=np.float32)
        self.last_motion_event_time = time.monotonic()
        self.last_error = ""
        self.connected = False
        self.open()

    def open(self) -> None:
        if not self.enabled:
            self.last_error = "SpaceMouse teleop is disabled."
            return
        if SPNAV_IMPORT_ERROR is not None or spnav_open is None:
            self.last_error = f"spnav is not installed: {SPNAV_IMPORT_ERROR}"
            return
        try:
            spnav_open()
            self.connected = True
            self.last_error = ""
        except Exception as exc:
            self.connected = False
            self.last_error = f"Failed to open SpaceMouse via spnav: {exc}"

    def close(self) -> None:
        if self.connected and spnav_close is not None:
            try:
                spnav_close()
            except Exception:
                pass
        self.connected = False

    def poll_events(self) -> None:
        if not self.connected or spnav_poll_event is None:
            return
        motion_state: np.ndarray | None = None
        for _ in range(32):
            event = spnav_poll_event()
            if event is None:
                break
            if SpnavMotionEvent is not None and isinstance(event, SpnavMotionEvent):
                event_state = np.asarray(event.translation + event.rotation, dtype=np.float32)
                if motion_state is None:
                    motion_state = event_state
                else:
                    use_event_axis = np.abs(event_state) >= np.abs(motion_state)
                    motion_state[use_event_axis] = event_state[use_event_axis]
                self.last_motion_event_time = time.monotonic()
            elif SpnavButtonEvent is not None and isinstance(event, SpnavButtonEvent):
                self.button_state[int(event.bnum)] = bool(event.press)
        if motion_state is not None:
            self.raw_state = motion_state

    def filtered_raw_state(self) -> np.ndarray:
        state = self.raw_state.astype(np.float32) / float(self.max_value)
        deadband = np.full(6, self.input_deadband, dtype=np.float32)
        is_dead = (-deadband < state) & (state < deadband)
        state[is_dead] = 0.0
        active = ~is_dead
        if np.any(active):
            magnitude = np.abs(state[active])
            denom = np.maximum(1.0 - deadband[active], 1e-6)
            normalized = np.clip((magnitude - deadband[active]) / denom, 0.0, 1.0)
            state[active] = np.sign(state[active]) * np.power(normalized, self.response_exponent)
        return state

    def motion_from_filtered(self, state: np.ndarray) -> np.ndarray:
        motion = np.zeros(6, dtype=np.float32)
        motion[:3] = self.translation_map @ state[:3]
        motion[3:] = self.rotation_sign * (self.rotation_map @ state[3:])
        motion[:3] *= self.translation_scale
        motion[3:] *= self.rotation_scale
        if motion[2] < 0:
            motion[2] *= self.downward_z_scale
        return motion

    def status(self) -> dict[str, Any]:
        self.poll_events()
        now = time.monotonic()
        age_s = max(0.0, now - self.last_motion_event_time)
        stale = age_s > self.stale_stop_s
        self.filtered_state = self.filtered_raw_state()
        motion = np.zeros(6, dtype=np.float32) if stale else self.motion_from_filtered(self.filtered_state)
        active = bool(np.max(np.abs(motion)) > self.active_eps)
        available = bool(self.connected and not self.last_error)
        status = "active" if active else "idle" if available else "disconnected"
        if stale and available:
            status = "stale"
        return {
            "requested": bool(self.enabled),
            "enabled": bool(self.enabled),
            "connected": bool(self.connected),
            "available": available,
            "status": status,
            "latest_timestamp": time.time(),
            "raw_state": [int(value) for value in self.raw_state.tolist()],
            "filtered_state": rounded_list(self.filtered_state),
            "translation_raw": [int(value) for value in self.raw_state[:3].tolist()],
            "rotation_raw": [int(value) for value in self.raw_state[3:].tolist()],
            "translation_raw_dominant_axis": dominant_axis(self.raw_state[:3], ("x", "y", "z")),
            "rotation_raw_dominant_axis": dominant_axis(self.raw_state[3:], ("rx", "ry", "rz")),
            "motion_state": rounded_list(motion),
            "gesture_translation": translation_gesture(motion, self.active_eps),
            "gesture_rotation": rotation_gesture(motion, self.active_eps),
            "translation_map": self.translation_map_spec,
            "rotation_map": self.rotation_map_spec,
            "rotation_sign": self.rotation_sign_spec,
            "active": active,
            "stale": stale,
            "age_s": age_s,
            "buttons": {
                "left": bool(self.button_state[0]),
                "right": bool(self.button_state[1]),
            },
            "last_error": self.last_error,
        }


_STATUS_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)


class SpaceMouseQueueManager(BaseManager):
    pass


def get_status_queue() -> queue.Queue[dict[str, Any]]:
    return _STATUS_QUEUE


SpaceMouseQueueManager.register("get_status_queue", callable=get_status_queue)


def replace_latest_status(status: dict[str, Any]) -> None:
    while True:
        try:
            _STATUS_QUEUE.get_nowait()
        except queue.Empty:
            break
    _STATUS_QUEUE.put(status)


def write_status_file(status_file: Path, status: dict[str, Any], message: str) -> None:
    payload = {
        "initialized": bool(status.get("connected")),
        "message": message,
        "last_error": str(status.get("last_error") or ""),
        "spacemouse": status,
    }
    status_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_file.with_suffix(status_file.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, status_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish SpaceMouse TCP commands through a process-shared queue.")
    parser.add_argument("--host", default=os.getenv("SPACEMOUSE_QUEUE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=env_int("SPACEMOUSE_QUEUE_PORT", 8765))
    parser.add_argument("--authkey", default=os.getenv("SPACEMOUSE_QUEUE_AUTHKEY", "spacemouse"))
    parser.add_argument("--publish-hz", type=float, default=max(1.0, env_float("SPACEMOUSE_QUEUE_PUBLISH_HZ", 200.0)))
    parser.add_argument("--status-file", default=os.getenv("SPACEMOUSE_STATUS_FILE", ""))
    parser.add_argument("--status-hz", type=float, default=max(1.0, env_float("SPACEMOUSE_STATUS_FILE_HZ", 20.0)))
    args = parser.parse_args()

    manager = SpaceMouseQueueManager(address=(args.host, args.port), authkey=args.authkey.encode("utf-8"))
    server = manager.get_server()
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[SpaceMouse] queue server listening on {args.host}:{args.port} at {args.publish_hz:.1f} Hz", flush=True)

    spacemouse = SpaceMouseReader()
    period_s = 1.0 / float(args.publish_hz)
    status_path = Path(args.status_file).expanduser().resolve() if args.status_file else None
    status_period_s = 1.0 / max(1.0, float(args.status_hz))
    last_status_write = 0.0
    latest_status: dict[str, Any] = {}
    try:
        while True:
            status = spacemouse.status()
            latest_status = status
            replace_latest_status(status)
            now = time.monotonic()
            if status_path is not None and now - last_status_write >= status_period_s:
                write_status_file(status_path, status, "Queue SpaceMouse publisher running.")
                last_status_write = now
            if status["active"]:
                print(
                    "[SpaceMouse] "
                    f"{status['gesture_translation']}/{status['gesture_rotation']} "
                    f"motion={status['motion_state']}",
                    end="\r",
                    flush=True,
                )
            elif status["status"] != "idle":
                print(f"[SpaceMouse] {status['status']}: {status['last_error']}", end="\r", flush=True)
            time.sleep(period_s)
    except KeyboardInterrupt:
        print("\n[SpaceMouse] stopping.")
    finally:
        if status_path is not None:
            stopped_status = dict(latest_status) if latest_status else {
                "requested": True,
                "enabled": True,
                "latest_timestamp": time.time(),
                "motion_state": [0.0] * 6,
                "buttons": {"left": False, "right": False},
                "last_error": "",
            }
            stopped_status.update({
                "connected": False,
                "available": False,
                "active": False,
                "status": "disconnected",
                "motion_state": [0.0] * 6,
            })
            write_status_file(status_path, stopped_status, "Queue SpaceMouse publisher stopped.")
        spacemouse.close()


if __name__ == "__main__":
    main()
