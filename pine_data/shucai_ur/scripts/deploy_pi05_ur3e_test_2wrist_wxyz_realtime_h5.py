"""
PI0 Robot Controller + real-time H5 logging
"""

import os
import time

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from openpi_client import websocket_client_policy

OUTPUT_H5_DIR = "/home/pine/openpi/verify"


def _find_realsense_serial_by_name(target_substr: str) -> str:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError("pyrealsense2 is required for RealSense cameras.") from exc

    context = rs.context()
    devices = context.query_devices()
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        if target_substr.lower() in name.lower():
            return device.get_info(rs.camera_info.serial_number)

    available = [dev.get_info(rs.camera_info.name) for dev in devices]
    raise ConnectionError(f"RealSense device containing '{target_substr}' not found. Available: {available}")


def make_unique_file_path(save_dir: str, base_name: str, ext: str = "h5") -> str:
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    while True:
        path = os.path.join(save_dir, f"{base_name}_{idx}.{ext}")
        if not os.path.exists(path):
            return path
        idx += 1


class RealSenseCamera:
    def __init__(self, serial_number: str, width: int = 640, height: int = 480, fps: int = 30):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError("pyrealsense2 is required for RealSense cameras.") from exc

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(config)

    def read(self) -> Image.Image:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("Failed to read color frame from RealSense camera")
        bgr = np.asanyarray(color.get_data())
        return Image.fromarray(bgr[:, :, ::-1])

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()


class UR3eRTDEController:
    def __init__(
        self,
        hostname: str = "192.168.201.101",
        robot_port: int = 30004,
        speed: float = 0.25,
        acceleration: float = 0.5,
        async_move: bool = True,
    ):
        self.hostname = hostname
        self.robot_port = robot_port
        self.speed = speed
        self.acceleration = acceleration
        self.async_move = async_move
        self._control = None
        self._receive = None
        self._warned_gripper = False
        self.connect()

    @property
    def is_connected(self) -> bool:
        return self._control is not None and self._receive is not None

    def connect(self) -> None:
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError:
            from ur_rtde import rtde_control as _rtde_control
            from ur_rtde import rtde_receive as _rtde_receive

            RTDEControlInterface = _rtde_control.RTDEControlInterface
            RTDEReceiveInterface = _rtde_receive.RTDEReceiveInterface

        self._control = RTDEControlInterface(self.hostname, self.robot_port)
        self._receive = RTDEReceiveInterface(self.hostname, self.robot_port)

    def get_actual_tcp_pose(self):
        if not self.is_connected:
            return None
        if hasattr(self._receive, "getActualTCPPose"):
            return self._receive.getActualTCPPose()
        if hasattr(self._receive, "getActualTCP"):
            return self._receive.getActualTCP()
        return None

    def disconnect(self) -> None:
        if self._control is not None:
            if hasattr(self._control, "stopScript"):
                self._control.stopScript()
            if hasattr(self._control, "disconnect"):
                self._control.disconnect()
        if self._receive is not None and hasattr(self._receive, "disconnect"):
            self._receive.disconnect()
        self._control = None
        self._receive = None

    def execute_eef(self, pose, task_name=None):
        if not self.is_connected:
            raise ConnectionError("UR3e RTDE not connected")

        if len(pose) == 8:
            x, y, z, qw, qx, qy, qz, grip = pose
            rpy = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)
        elif len(pose) == 7:
            x, y, z, roll, pitch, yaw, grip = pose
            rpy = np.array([roll, pitch, yaw], dtype=np.float32)
        else:
            raise ValueError("Pose must be length 7 (rpy) or 8 (quaternion).")

        if grip is not None and not self._warned_gripper:
            print("[Client] Gripper command received but UR3e gripper control not implemented.")
            self._warned_gripper = True

        rotvec = R.from_euler("xyz", rpy, degrees=False).as_rotvec()
        tcp_pose = [float(x), float(y), float(z), float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]
        print("tcp_pose:", tcp_pose)
        self._control.moveL(tcp_pose, self.speed, self.acceleration, self.async_move)
        return tcp_pose


class RealtimeH5Logger:
    def __init__(self, path: str):
        import h5py

        self.path = path
        self._h5 = h5py.File(path, "w")
        self._count = 0

        self._step_ds = self._h5.create_dataset("step", shape=(0,), maxshape=(None,), dtype="int32")
        self._chunk_ds = self._h5.create_dataset("chunk_idx", shape=(0,), maxshape=(None,), dtype="int32")
        self._timestamp_ds = self._h5.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype="float64")
        self._model_action_ds = self._h5.create_dataset(
            "model_action", shape=(0, 7), maxshape=(None, 7), dtype="float32"
        )
        self._executed_tcp_ds = self._h5.create_dataset(
            "executed_tcp_pose", shape=(0, 6), maxshape=(None, 6), dtype="float32"
        )
        self._actual_tcp_ds = self._h5.create_dataset(
            "actual_tcp_pose", shape=(0, 6), maxshape=(None, 6), dtype="float32"
        )

    def append(self, step: int, chunk_idx: int, timestamp: float, model_action, executed_tcp_pose, actual_tcp_pose):
        idx = self._count
        self._step_ds.resize((idx + 1,))
        self._chunk_ds.resize((idx + 1,))
        self._timestamp_ds.resize((idx + 1,))
        self._model_action_ds.resize((idx + 1, 7))
        self._executed_tcp_ds.resize((idx + 1, 6))
        self._actual_tcp_ds.resize((idx + 1, 6))

        self._step_ds[idx] = int(step)
        self._chunk_ds[idx] = int(chunk_idx)
        self._timestamp_ds[idx] = float(timestamp)
        self._model_action_ds[idx] = np.asarray(model_action, dtype=np.float32).reshape(7,)
        self._executed_tcp_ds[idx] = np.asarray(executed_tcp_pose, dtype=np.float32).reshape(6,)
        if actual_tcp_pose is None:
            self._actual_tcp_ds[idx] = np.full((6,), np.nan, dtype=np.float32)
        else:
            self._actual_tcp_ds[idx] = np.asarray(actual_tcp_pose, dtype=np.float32).reshape(6,)

        self._count += 1
        self._h5.flush()

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None


class PI0RobotController:
    POSITION_PRESETS = {
        "insertion": (0.2506, -0.2463, 0.3242, 0.0060212706, 0.3638795806, -0.9313949679, -0.0076686951, 0.0),
    }
    reset_pose = (0.2506, -0.2463, 0.3242, 0.0060212706, 0.3638795806, -0.9313949679, -0.0076686951, 0.0)
    TASK_PROMPTS = {"insertion": "insert"}

    def __init__(self, controller, websocket_host="0.0.0.0", websocket_port=8000, h5_path=None):
        self.controller = controller
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.main_camera = None
        self.wrist_camera = None
        np.set_printoptions(precision=4, suppress=True)

        self.client = websocket_client_policy.WebsocketClientPolicy(websocket_host, websocket_port)
        print(f"Initialized PI0 client with WebSocket at {websocket_host}:{websocket_port}")

        self.h5_path = h5_path or make_unique_file_path(OUTPUT_H5_DIR, "pi05_rtde_realtime", "h5")
        self.logger = RealtimeH5Logger(self.h5_path)
        print(f"[Client] Realtime H5 logging enabled: {self.h5_path}")

    def initialize_camera(self, warm_up_frames=10):
        main_serial = os.getenv("MAIN_RS_SERIAL") or _find_realsense_serial_by_name("D455")
        wrist_serial = os.getenv("WRIST_RS_SERIAL") or _find_realsense_serial_by_name("D405")
        self.main_camera = RealSenseCamera(main_serial)
        self.wrist_camera = RealSenseCamera(wrist_serial)
        print(f"Camera pipeline initialized. Main(D455)={main_serial}, Wrist(D405)={wrist_serial}")
        for i in range(warm_up_frames):
            _ = self.main_camera.read()
            _ = self.wrist_camera.read()
            print(f"Warm-up frame {i + 1}/{warm_up_frames} successful")

    def capture_images(self, step=None, save_dir="/home/pine/openpi/visualization"):
        try:
            main_image = self.main_camera.read()
            wrist_image = self.wrist_camera.read()
            if main_image is None or wrist_image is None:
                return None, None, None
            main_path = None
            if step is not None:
                os.makedirs(save_dir, exist_ok=True)
                main_path = os.path.join(save_dir, f"captured_image_{step}.png")
                wrist_path = os.path.join(save_dir, f"captured_image_wrist_{step}.png")
                main_image.save(main_path)
                wrist_image.save(wrist_path)
            return main_image, wrist_image, main_path
        except Exception as exc:
            print(f"Failed to capture images: {exc}")
            return None, None, None

    def quaternion_to_rpy(self, quaternion):
        qw, qx, qy, qz = quaternion
        return R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)

    def preset_position(self, task_name):
        self.controller.execute_eef(self.POSITION_PRESETS.get(task_name, self.reset_pose), task_name)
        return True

    def prepare_inference_data(self, main_image, wrist_image, current_state, prompt):
        return {
            "observation/image": np.asarray(main_image, dtype=np.uint8),
            "observation/wrist_image": np.asarray(wrist_image, dtype=np.uint8),
            "observation/state": current_state,
            "prompt": prompt,
        }

    def process_action(self, action_step, current_position, current_rpy):
        delta_position = action_step[:3]
        new_position = [round(current_position[i] + delta_position[i], 5) for i in range(3)]
        delta_rpy = action_step[3:6]
        new_rpy = [round(current_rpy[i] + delta_rpy[i], 5) for i in range(3)]
        gripper_position = round(action_step[6], 5)
        return new_position, new_rpy, gripper_position

    def execute_action_chunk(self, all_actions, chunk_size, merge_step, step, init_pose, init_rpy, init_gripper, task_name):
        n_steps = min(len(all_actions), chunk_size)
        current_pose, current_rpy, current_gripper = init_pose, init_rpy, init_gripper
        for step_idx in range(0, n_steps, merge_step):
            self.capture_images(step + step_idx)
            merged_chunk = all_actions[step_idx : step_idx + merge_step]
            merged_action_prefix = np.sum(merged_chunk[:, :6], axis=0)
            gripper_command = merged_chunk[-1][6]
            action_step = np.concatenate([merged_action_prefix, [gripper_command]])

            new_pos, new_rpy, grip = self.process_action(action_step, current_pose, current_rpy)
            grip = 1.0 if grip > 0.5 else 0.0
            final_action = new_pos + new_rpy + [grip]
            print(f"Executing: {final_action}")
            cmd_tcp = self.controller.execute_eef(final_action, task_name)
            actual_tcp = self.controller.get_actual_tcp_pose()
            self.logger.append(step, step_idx, time.time(), action_step, cmd_tcp, actual_tcp)
            current_pose, current_rpy, current_gripper = new_pos, new_rpy, grip
        return current_pose, current_rpy, current_gripper

    def run_control_loop(self, task_name="insertion", n_iterations=200, chunk_size=8, merge_step=2, loop_interval=0.1):
        if self.main_camera is None or self.wrist_camera is None:
            self.initialize_camera()
        if not self.preset_position(task_name):
            return

        prompt = self.TASK_PROMPTS.get(task_name, self.TASK_PROMPTS["insertion"])
        execute_action = self.reset_pose
        init_pose = list(execute_action[:3])
        init_rpy_exe = self.quaternion_to_rpy(execute_action[3:7])
        init_gripper = execute_action[7]

        step = 0
        try:
            while step < n_iterations:
                start_time = time.time()
                main_image, wrist_image, _ = self.capture_images(step)
                if main_image is None or wrist_image is None:
                    time.sleep(loop_interval)
                    continue

                current_state = np.concatenate((init_pose, init_rpy_exe, [init_gripper]))
                element = self.prepare_inference_data(wrist_image, wrist_image, current_state, prompt)
                action = self.client.infer(element)
                all_actions = np.asarray(action["actions"], dtype=np.float32)
                init_pose, init_rpy_exe, init_gripper = self.execute_action_chunk(
                    all_actions, chunk_size, merge_step, step, init_pose, init_rpy_exe, init_gripper, task_name
                )
                sleep_time = loop_interval - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                step += 1
        except KeyboardInterrupt:
            print("\nControl loop interrupted by user")
        finally:
            print("Control loop ended")
            self.controller.execute_eef(execute_action, "reset")
            self.logger.close()
            if self.main_camera is not None:
                self.main_camera.stop()
            if self.wrist_camera is not None:
                self.wrist_camera.stop()
            self.controller.disconnect()


def main():
    controller = UR3eRTDEController()
    robot_system = PI0RobotController(controller)
    robot_system.run_control_loop(task_name="insertion", n_iterations=1000, chunk_size=10, merge_step=1, loop_interval=0.1)


if __name__ == "__main__":
    main()
