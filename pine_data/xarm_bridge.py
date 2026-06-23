from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Iterable


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _sdk_path() -> Path:
    configured = os.getenv("XARM_SDK_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "xArm-Python-SDK"


SDK_PATH = _sdk_path()
if str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

XARM_SDK_HOME = Path(os.getenv("XARM_SDK_HOME", Path(__file__).resolve().parent / ".runtime" / "xarm_home")).expanduser().resolve()
XARM_SDK_HOME.mkdir(parents=True, exist_ok=True)

_ORIGINAL_HOME = os.environ.get("HOME")
os.environ["HOME"] = str(XARM_SDK_HOME)
try:
    from xarm.wrapper import XArmAPI
except ImportError as exc:
    raise ImportError(
        f"Could not import xArm SDK from {SDK_PATH}. "
        "Set XARM_SDK_PATH or install xarm-python-sdk."
    ) from exc
finally:
    if _ORIGINAL_HOME is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _ORIGINAL_HOME


class XArmBridgeError(RuntimeError):
    pass


def _as_float_list(values: Iterable[float], length: int | None = None) -> list[float]:
    items = [float(value) for value in values]
    if length is not None and len(items) < length:
        items.extend([0.0] * (length - len(items)))
    return items[:length] if length is not None else items


def _require_code_ok(code: int, action: str) -> None:
    if int(code) != 0:
        raise XArmBridgeError(f"{action} failed with xArm code {code}")


def _result_value(result, action: str):
    if not isinstance(result, tuple) or len(result) < 2:
        raise XArmBridgeError(f"{action} returned an unexpected result: {result!r}")
    code = int(result[0])
    _require_code_ok(code, action)
    return result[1]


class _SharedXArmClient:
    def __init__(self, host: str) -> None:
        self.host = host
        self.lock = threading.RLock()
        self.arm = XArmAPI(host, is_radian=True)
        self.mode: int | None = None
        self.last_tcp_speed = [0.0] * 6

    def set_mode(self, mode: int) -> None:
        with self.lock:
            _require_code_ok(self.arm.motion_enable(True), "motion_enable")
            if self.mode != mode:
                _require_code_ok(self.arm.set_mode(mode), f"set_mode({mode})")
                self.mode = mode
            _require_code_ok(self.arm.set_state(0), "set_state(0)")

    def stop_velocity(self) -> None:
        with self.lock:
            try:
                self.set_mode(5)
                _require_code_ok(
                    self.arm.vc_set_cartesian_velocity(
                        [0.0] * 6,
                        is_radian=True,
                        duration=float(os.getenv("XARM_STOP_DURATION_S", "0.1")),
                    ),
                    "vc_set_cartesian_velocity(stop)",
                )
            finally:
                self.last_tcp_speed = [0.0] * 6

    def disconnect(self) -> None:
        with self.lock:
            self.arm.disconnect()


_CLIENTS: dict[str, _SharedXArmClient] = {}
_CLIENTS_LOCK = threading.Lock()


def _client_for(host: str) -> _SharedXArmClient:
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(host)
        if client is None:
            client = _SharedXArmClient(host)
            _CLIENTS[host] = client
        return client


class XArmControlInterface:
    def __init__(self, host: str, *_, **__) -> None:
        self.client = _client_for(host)

    def speedL(self, speed, acceleration=None, time=0.01):
        del acceleration
        speeds = _as_float_list(speed, 6)
        # Existing UR teleop commands are m/s for XYZ and rad/s for rotation.
        xarm_speeds = [speeds[0] * 1000.0, speeds[1] * 1000.0, speeds[2] * 1000.0, *speeds[3:]]
        duration = float(time) if time and float(time) > 0 else float(os.getenv("XARM_SPEED_COMMAND_DURATION_S", "0.05"))
        with self.client.lock:
            self.client.set_mode(5)
            _require_code_ok(
                self.client.arm.vc_set_cartesian_velocity(
                    xarm_speeds,
                    is_radian=True,
                    duration=duration,
                ),
                "vc_set_cartesian_velocity",
            )
            self.client.last_tcp_speed = speeds

    def speedStop(self):
        self.client.stop_velocity()

    def stopL(self, acceleration=1.0):
        del acceleration
        self.client.stop_velocity()

    def stopScript(self):
        self.client.stop_velocity()

    def moveL(self, pose, speed=0.1, acceleration=0.1, asynchronous=False):
        del asynchronous
        if not _env_bool("XARM_ENABLE_RESET_MOTIONS", False):
            raise XArmBridgeError(
                "xArm moveL is disabled because the existing reset poses are UR-specific. "
                "Set XARM_ENABLE_RESET_MOTIONS=1 only after replacing them with xArm-safe poses."
            )
        pose_values = _as_float_list(pose, 6)
        with self.client.lock:
            self.client.set_mode(0)
            _require_code_ok(
                self.client.arm.set_position(
                    x=pose_values[0] * 1000.0,
                    y=pose_values[1] * 1000.0,
                    z=pose_values[2] * 1000.0,
                    roll=pose_values[3],
                    pitch=pose_values[4],
                    yaw=pose_values[5],
                    speed=float(speed) * 1000.0,
                    mvacc=float(acceleration) * 1000.0,
                    is_radian=True,
                    wait=True,
                ),
                "set_position",
            )

    def moveJ(self, joint_pose, velocity=0.5, acceleration=0.5, asynchronous=False):
        del asynchronous
        if not _env_bool("XARM_ENABLE_RESET_MOTIONS", False):
            raise XArmBridgeError(
                "xArm moveJ is disabled because the existing reset poses are UR-specific. "
                "Set XARM_ENABLE_RESET_MOTIONS=1 only after replacing them with xArm-safe poses."
            )
        with self.client.lock:
            self.client.set_mode(0)
            _require_code_ok(
                self.client.arm.set_servo_angle(
                    angle=_as_float_list(joint_pose),
                    speed=float(velocity),
                    mvacc=float(acceleration),
                    is_radian=True,
                    wait=True,
                ),
                "set_servo_angle",
            )

    def zeroFtSensor(self):
        with self.client.lock:
            if hasattr(self.client.arm, "set_ft_sensor_zero"):
                _require_code_ok(self.client.arm.set_ft_sensor_zero(), "set_ft_sensor_zero")

    def forceMode(self, *_, **__):
        raise XArmBridgeError("xArm forceMode bridge is not implemented.")

    def forceModeStop(self):
        self.client.stop_velocity()

    def disconnect(self):
        self.client.disconnect()


class XArmReceiveInterface:
    def __init__(self, host: str, *_, **__) -> None:
        self.client = _client_for(host)

    def getRobotMode(self):
        with self.client.lock:
            code, state = self.client.arm.get_state()
        if int(code) != 0:
            return 0
        return 7 if int(state) in {0, 1} else 0

    def getActualQ(self):
        with self.client.lock:
            angles = _result_value(self.client.arm.get_servo_angle(is_radian=True), "get_servo_angle")
        return _as_float_list(angles)

    def getActualTCPPose(self):
        with self.client.lock:
            pose = _result_value(self.client.arm.get_position(is_radian=True), "get_position")
        values = _as_float_list(pose, 6)
        return [values[0] / 1000.0, values[1] / 1000.0, values[2] / 1000.0, *values[3:]]

    def getActualTCPForce(self):
        with self.client.lock:
            try:
                values = _result_value(self.client.arm.get_ft_sensor_data(), "get_ft_sensor_data")
            except Exception:
                return [float("nan")] * 6
        return _as_float_list(values, 6)

    def getActualTCPSpeed(self):
        return list(self.client.last_tcp_speed)

    def getJointTorques(self):
        with self.client.lock:
            try:
                values = _result_value(self.client.arm.get_joints_torque(), "get_joints_torque")
            except Exception:
                values = None
            if values is None:
                states = _result_value(
                    self.client.arm.get_joint_states(is_radian=True, num=3),
                    "get_joint_states",
                )
                values = states[2] if len(states) >= 3 else []
        return _as_float_list(values)

    def getActualCurrentAsTorque(self):
        return self.getJointTorques()

    def disconnect(self):
        self.client.disconnect()


class XArmSDKGripper:
    def __init__(
        self,
        hostname: str,
        port: int | None = None,
        open_pos: int | None = None,
        close_pos: int | None = None,
        speed: int | None = None,
        force: int | None = None,
        auto_activate: bool = False,
    ) -> None:
        del port, force, auto_activate
        self.client = _client_for(hostname)
        self._open_pos = int(open_pos if open_pos is not None else os.getenv("XARM_GRIPPER_OPEN_POS", "850"))
        self._close_pos = int(close_pos if close_pos is not None else os.getenv("XARM_GRIPPER_CLOSE_POS", "0"))
        self._speed = int(speed if speed is not None else os.getenv("XARM_GRIPPER_SPEED", "5000"))
        self._virtual_max_pos = abs(self._close_pos - self._open_pos)
        self._close_direction = 1 if self._close_pos >= self._open_pos else -1
        self.state = "unknown"
        self._last_command_position: int | None = None
        with self.client.lock:
            _require_code_ok(self.client.arm.set_gripper_mode(0), "set_gripper_mode")
            _require_code_ok(self.client.arm.set_gripper_enable(True), "set_gripper_enable")
            _require_code_ok(self.client.arm.set_gripper_speed(self._speed), "set_gripper_speed")
        self.open()

    def get_min_position(self) -> int:
        return 0

    def get_max_position(self) -> int:
        return self._virtual_max_pos

    def get_current_position(self) -> int:
        with self.client.lock:
            physical = int(_result_value(self.client.arm.get_gripper_position(), "get_gripper_position"))
        return max(0, min((physical - self._open_pos) * self._close_direction, self._virtual_max_pos))

    def is_contact_detected(self) -> bool:
        with self.client.lock:
            try:
                status = int(_result_value(self.client.arm.get_gripper_status(), "get_gripper_status"))
            except Exception:
                return False
        return (status & 0x03) == 2

    def command_position(self, position: int) -> int:
        clipped = max(0, min(int(position), self._virtual_max_pos))
        if self._last_command_position == clipped:
            return clipped
        physical_position = self._open_pos + clipped * self._close_direction
        with self.client.lock:
            _require_code_ok(
                self.client.arm.set_gripper_position(
                    physical_position,
                    wait=False,
                    speed=self._speed,
                    auto_enable=True,
                ),
                "set_gripper_position",
            )
        self._last_command_position = clipped
        if clipped == 0:
            self.state = "open"
        elif clipped == self._virtual_max_pos:
            self.state = "closed"
        else:
            self.state = f"partial:{clipped}"
        return clipped

    def stop_motion(self) -> None:
        try:
            self.command_position(self.get_current_position())
            self.state = "stopped"
            self._last_command_position = None
        except Exception as exc:
            print("[xArm gripper] stop warning:", exc)

    def open(self):
        commanded = self.command_position(0)
        print(f"[xArm gripper] state -> open ({commanded})")

    def close(self):
        commanded = self.command_position(self._virtual_max_pos)
        print(f"[xArm gripper] state -> closed ({commanded})")

    def disconnect(self):
        pass
