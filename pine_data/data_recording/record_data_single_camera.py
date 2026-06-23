import time
import json
import numpy as np
import pyrealsense2 as rs
import os
import threading
from pynput import keyboard 
from rtde_receive import RTDEReceiveInterface 
from rtde_control import RTDEControlInterface
from rtde_io import RTDEIOInterface as RTDEIO
import shutil
import cv2
from PIL import Image

## Script to collect demonstrations of task for robot (Single Camera - D405 Hand Camera Only)

## Press 'r' to reset to home position
## Press 'c' to start recording new episode 
## Press 's' to stop recording and save current episode
## Press 'd' to delete most recent episode (only when not recording)
## Press 'q' to quit

# Festo gripper configuration
FESTO_OPEN_DO = 0   # Digital output to open gripper
FESTO_CLOSE_DO = 2  # Digital output to close gripper
FESTO_USE_TOOL_DO = False  # Set to True if using tool connector

class FestoGripper:
    """Festo pneumatic gripper controller via RTDE digital outputs"""
    def __init__(self, rtde_io, open_do=FESTO_OPEN_DO, close_do=FESTO_CLOSE_DO):
        self.rtde_io = rtde_io
        self.open_do = open_do
        self.close_do = close_do
        self.state = None
        self._position = 0  # 0 = open, 255 = closed
        print(f"[Festo] connected via RTDEIO (open_do={open_do}, close_do={close_do})")
        self.release()  # Start in open position

    def _set_outputs(self, open_on: bool, close_on: bool):
        """Set digital outputs to control gripper"""
        self.rtde_io.setStandardDigitalOut(self.open_do, open_on)
        self.rtde_io.setStandardDigitalOut(self.close_do, close_on)
        if FESTO_USE_TOOL_DO:
            try:
                self.rtde_io.setToolDigitalOut(0, open_on)
                self.rtde_io.setToolDigitalOut(2, close_on)
            except Exception as e:
                print(f"[Festo] tool DO write failed: {e}")
        
        if open_on:
            self.state = "open"
            self._position = 0
        else:
            self.state = "closed"
            self._position = 255
        print(f"[Festo] state -> {self.state}")

    def grip(self):
        """Close the gripper"""
        self._set_outputs(False, True)

    def release(self):
        """Open the gripper"""
        self._set_outputs(True, False)
    
    def get_current_position(self):
        """Return current gripper position (0=open, 255=closed)"""
        return self._position

class Data(): 
    def __init__(self): 
        self.ROBOT_HOST = "192.168.201.101" 
        self.state_shape = 7 # Adjust state shape here
        self.action_shape = 7 # Adjust action state here

        # Params for D405 hand camera
        self.width = 640
        self.height = 480
        self.fps = 30

        # Reset position (home position for data collection)
        # TCP Pose: [x, y, z, rx, ry, rz]
        self.reset_tcp_pose = [0.2506, -0.2463, 0.3242, 1.1388, -2.9149, -0.0240]
        # Joint Angles: [j1, j2, j3, j4, j5, j6] in radians
        self.reset_joint_pose = [1.9795, -1.4438, 1.3327, -1.4459, -1.5835, 12.2385]

        # start robot and camera
        self.realsense = self.initialize_hand_camera()
        self.rtde_r, self.rtde_c, self.gripper = self.initialize_robot()

        # Params for point cloud processing
        self.handbbox = [-300, -300, -300, 300, 300, 100]

        # Initialize data arrays
        self.clear_buffers()

    def clear_buffers(self):
        """Clear all data buffers"""
        global color_hand_array, depth_hand_array, point_cloud_hand_array
        global joint_state_array, joint_action_array, eef_state_array, eef_action_array, eef_force_array
        
        color_hand_array = []
        depth_hand_array = []
        point_cloud_hand_array = []
        joint_state_array = []
        joint_action_array = []
        eef_state_array = []
        eef_action_array = []
        eef_force_array = []

    def initialize_hand_camera(self):
        """Initialize RealSense D405 hand camera"""
        realsense = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        realsense.start(config)
        
        # Warm up camera
        print("Warming up camera...")
        for _ in range(30):
            realsense.wait_for_frames()
        print("Camera ready!")
        
        return realsense

    def initialize_robot(self): 
        """Initialize robot and Festo gripper"""
        rtde_r = RTDEReceiveInterface(self.ROBOT_HOST)
        rtde_c = RTDEControlInterface(self.ROBOT_HOST)
        rtde_io = RTDEIO(self.ROBOT_HOST)
        print("Initializing Festo gripper...")
        gripper = FestoGripper(rtde_io)
        return rtde_r, rtde_c, gripper
    
    def move_to_reset_position(self):
        """Move robot to reset position using joint control"""
        print("\nMoving to reset position...")
        print(f"Target TCP: {self.reset_tcp_pose}")
        print(f"Target Joints: {self.reset_joint_pose}")
        
        # Move to joint position (safer than TCP for reset)
        velocity = 0.5  # rad/s
        acceleration = 0.5  # rad/s^2
        
        self.rtde_c.moveJ(self.reset_joint_pose, velocity, acceleration)
        
        # Wait for movement to complete
        time.sleep(0.5)
        
        # Verify position
        current_joints = self.rtde_r.getActualQ()
        print(f"Current Joints: {[round(j, 4) for j in current_joints]}")
        print("✓ Reset position reached!\n")

    # Function to get robot joint state
    def get_robot_joint_state(self):
        state = np.array(self.rtde_r.getActualQ())
        action = np.array(self.rtde_r.getTargetQ())
        gripper_state = np.array([self.gripper.get_current_position()]) 
        state = np.concatenate((state, gripper_state))
        action = np.concatenate((action, gripper_state))
        return state, action

    # Function to get robot eef state
    def get_robot_eef_state(self):
        state = np.array(self.rtde_r.getActualTCPPose())
        action = np.array(self.rtde_r.getTargetTCPPose())
        gripper_state = np.array([self.gripper.get_current_position()]) 
        state = np.concatenate((state, gripper_state))
        action = np.concatenate((action, gripper_state))
        return state, action

    # Function to get end-effector force
    def get_robot_eef_force(self):
        force = np.array(self.rtde_r.getActualTCPForce())
        return force

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

    # Function to get visual observations (RGB-D & PC) 
    def get_visual_obs(self):
        """Get visual observations from D405 hand camera"""
        capture_hand = self.realsense.wait_for_frames()

        if capture_hand is not None and capture_hand.get_color_frame() is not None:
            # Capture color_hand image
            color_hand_frame = capture_hand.get_color_frame()
            color_hand_image = np.asanyarray(color_hand_frame.get_data())
            color_hand_image = cv2.cvtColor(color_hand_image, cv2.COLOR_BGR2RGB)

            # Capture depth_hand image
            depth_hand_frame = capture_hand.get_depth_frame()
            depth_hand_image = np.asanyarray(depth_hand_frame.get_data())

            # Capture point cloud from RealSense
            point_cloud_hand = self.get_point_cloud_realsense(capture_hand)

            return color_hand_image, depth_hand_image, point_cloud_hand
        else:
            raise ValueError("RealSense capture is invalid.")

    # Function to handle key presses
    def on_press(self, key):
        global recording, current_time, current_episode_folder

        try:
            if key.char == 'r':
                if not recording:
                    print("\n" + "="*50)
                    print("AUTO RESET - Moving to home position...")
                    print("="*50)
                    self.move_to_reset_position()
                else:
                    print("Cannot reset while recording. Press 's' to stop recording first.")
            
            elif key.char == 'c':
                if not recording:
                    # Clear buffers before starting new episode
                    self.clear_buffers()
                    
                    current_time = time.strftime("%Y%m%d%H%M%S")
                    current_episode_folder = os.path.join(data_folder, current_time)
                    os.makedirs(current_episode_folder, exist_ok=True)

                    recording = True
                    print(f"\n{'='*50}")
                    print(f"Started recording episode {current_time}")
                    print(f"{'='*50}\n")

            elif key.char == 's':
                if recording:
                    print(f"\nStopping recording episode {current_time}...")
                    self.save_data(current_episode_folder)
                    
                    recording = False
                    print(f"Episode {current_time} saved successfully!")
                    print(f"Total frames: {len(color_hand_array)}")
                    print(f"\nReady for next episode. Press 'c' to start recording.\n")
                    time.sleep(0.5)

            elif key.char == 'q':
                if recording:
                    print("\nStopping current recording before quitting...")
                    self.save_data(current_episode_folder)
                    recording = False
                print("Quitting session...")
                return False  # Stop listener

            elif key.char == 'd':
                if not recording:
                    confirmation = input(f"Are you sure you want to delete episode {current_time}? (y/n): ")
                    if 'y' in confirmation.lower() and current_episode_folder:
                        if os.path.exists(current_episode_folder):
                            shutil.rmtree(current_episode_folder)
                            print(f"Deleted episode {current_time}")
                        else:
                            print("Episode folder not found.")
                    else:
                        print("Deletion canceled.")
                else:
                    print("Cannot delete while recording. Press 's' to stop recording first.")

        except AttributeError:
            pass

    # Synchronization mechanism
    def synchronized_capture(self, frequency=5):
        global recording

        interval = 1.0 / frequency

        while True:
            start_time = time.time()

            if recording:
                try:
                    # Get visual observations
                    color_hand_image, depth_hand_image, point_cloud_hand = self.get_visual_obs()

                    # Get robot state
                    joint_state, joint_action = self.get_robot_joint_state()
                    eef_state, eef_action = self.get_robot_eef_state()
                    eef_force = self.get_robot_eef_force()

                    if joint_state is not None and eef_state is not None:
                        # Save data to buffers
                        self.synchronize_data(color_hand_image, depth_hand_image, point_cloud_hand,
                                            joint_state, joint_action, eef_state, eef_action, eef_force)
                        print(f"Frame {len(color_hand_array)} recorded", end='\r')

                except Exception as e:
                    print(f"\nError during capture: {e}")

            elapsed_time = time.time() - start_time
            time_to_sleep = interval - elapsed_time

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def synchronize_data(self, color_hand_image, depth_hand_image, point_cloud_hand,
                        joint_state, joint_action, eef_state, eef_action, eef_force):
        """Add data to buffers"""
        # visual observations
        color_hand_array.append(color_hand_image)
        depth_hand_array.append(depth_hand_image)
        point_cloud_hand_array.append(point_cloud_hand)
            
        # states & actions
        joint_state_array.append(joint_state)
        joint_action_array.append(joint_action)
        eef_state_array.append(eef_state)
        eef_action_array.append(eef_action)
        eef_force_array.append(eef_force)

    def save_data(self, episode_folder):
        """Save all buffered data to disk"""
        # Define file paths
        color_hand_file = os.path.join(episode_folder, "rgb_hand.npy")
        depth_hand_file = os.path.join(episode_folder, "depth_hand.npy")
        point_cloud_hand_file = os.path.join(episode_folder, "point_cloud_hand.npy")

        joint_state_file = os.path.join(episode_folder, "joint_state.npy")
        joint_action_file = os.path.join(episode_folder, "joint_action.npy")
        eef_state_file = os.path.join(episode_folder, "eef_state.npy")
        eef_action_file = os.path.join(episode_folder, "eef_action.npy")
        eef_force_file = os.path.join(episode_folder, "eef_force.npy")

        # List -> array
        color_hand_image = np.array(color_hand_array)
        depth_hand_image = np.array(depth_hand_array)
        point_cloud_hand = np.array(point_cloud_hand_array)

        joint_state = np.array(joint_state_array)
        joint_action = np.array(joint_action_array)
        eef_state = np.array(eef_state_array)
        eef_action = np.array(eef_action_array)
        eef_force = np.array(eef_force_array)

        # Save data
        np.save(color_hand_file, color_hand_image)
        np.save(depth_hand_file, depth_hand_image)
        np.save(point_cloud_hand_file, point_cloud_hand)

        np.save(joint_state_file, joint_state)
        np.save(joint_action_file, joint_action)
        np.save(eef_state_file, eef_state)
        np.save(eef_action_file, eef_action)
        np.save(eef_force_file, eef_force)

        # Save metadata
        metadata = {
            "instruction": instruction,
            "num_frames": len(color_hand_array),
            "timestamp": current_time,
            "camera": "D405_hand",
            "frequency": 5
        }
        with open(os.path.join(episode_folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"\nSaved {len(color_hand_array)} frames to {episode_folder}")


def main():
    # Global variables for controlling the recording state
    global data_folder, recording, instruction, current_episode_folder, current_time
    
    # Set your data folder here
    data_folder = os.path.expanduser("~/pine_data/recordings")    
    # Ensure data folder exists
    os.makedirs(data_folder, exist_ok=True)

    recording = False
    instruction = input("Enter the task instruction: ")
    current_episode_folder = None
    current_time = None

    # Initialize data collection system
    data = Data()

    print("\n" + "="*50)
    print("DATA RECORDING SYSTEM - SINGLE CAMERA MODE")
    print("="*50)
    print("\nControls:")
    print("  'r' - Auto reset to home position")
    print("  'c' - Start recording new episode")
    print("  's' - Stop recording and save")
    print("  'd' - Delete most recent episode")
    print("  'q' - Quit")
    print("\n" + "="*50)
    print(f"\nReset Position:")
    print(f"  TCP: {data.reset_tcp_pose}")
    print(f"  Joints: {data.reset_joint_pose}")
    print("\nReady! Press 'r' to reset, then 'c' to start recording.\n")

    # Start the synchronized capture in a separate thread
    capture_thread = threading.Thread(target=data.synchronized_capture, daemon=True)
    capture_thread.start()

    # Start listening for keyboard inputs
    listener = keyboard.Listener(on_press=data.on_press)
    listener.start()
    listener.join()

    print("\nStopped data recording system.")


if __name__ == '__main__':
    main()
