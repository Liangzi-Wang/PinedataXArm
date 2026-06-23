import time
import json
import numpy as np
import pyrealsense2 as rs
import os
import threading
from pynput import keyboard 
import shutil
import cv2
import h5py

## Script to collect camera data only (no robot control)
## Use this with spacemouse teleoperation script running separately

## Press 'c' to start recording new episode 
## Press 's' to stop recording and save current episode
## Press 'd' to delete most recent episode (only when not recording)
## Press 'q' to quit

class CameraRecorder(): 
    def __init__(self): 
        # Params for D405 hand camera
        self.width = 640
        self.height = 480
        self.fps = 15

        # start camera
        self.realsense = self.initialize_hand_camera()

        # Initialize data arrays
        self.clear_buffers()

    def clear_buffers(self):
        """Clear all data buffers"""
        global color_hand_array, depth_hand_array, timestamp_array
        
        color_hand_array = []
        depth_hand_array = []
        timestamp_array = []

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

    # Function to get visual observations (RGB-D only) 
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

            return color_hand_image, depth_hand_image
        else:
            raise ValueError("RealSense capture is invalid.")

    # Function to handle key presses
    def on_press(self, key):
        global recording, current_time, current_episode_folder

        try:
            if key.char == 'c':
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
    def synchronized_capture(self, frequency=15):
        """Capture camera data at specified frequency"""
        global recording

        interval = 1.0 / frequency

        while True:
            start_time = time.time()

            if recording:
                try:
                    # Get visual observations
                    color_hand_image, depth_hand_image = self.get_visual_obs()
                    timestamp = time.time()

                    # Save data to buffers
                    self.synchronize_data(color_hand_image, depth_hand_image, timestamp)
                    print(f"Frame {len(color_hand_array)} recorded at {timestamp:.3f}", end='\r')

                except Exception as e:
                    print(f"\nError during capture: {e}")

            elapsed_time = time.time() - start_time
            time_to_sleep = interval - elapsed_time

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def synchronize_data(self, color_hand_image, depth_hand_image, timestamp):
        """Add data to buffers"""
        # visual observations
        color_hand_array.append(color_hand_image)
        depth_hand_array.append(depth_hand_image)
        timestamp_array.append(timestamp)

    def save_data(self, episode_folder):
        """Save all buffered data to disk in HDF5 format"""
        # Define file path
        h5_file = os.path.join(episode_folder, "camera_data.h5")

        # List -> array
        color_hand_image = np.array(color_hand_array)
        depth_hand_image = np.array(depth_hand_array)
        timestamps = np.array(timestamp_array)

        # Save data to HDF5 with compression
        print(f"\nSaving data to HDF5 format...")
        with h5py.File(h5_file, 'w') as f:
            # Create datasets with compression
            f.create_dataset('rgb_hand', data=color_hand_image, compression='gzip', compression_opts=4)
            f.create_dataset('depth_hand', data=depth_hand_image, compression='gzip', compression_opts=4)
            f.create_dataset('timestamps', data=timestamps, compression='gzip', compression_opts=4)
            
            # Save metadata as attributes
            f.attrs['instruction'] = instruction
            f.attrs['num_frames'] = len(color_hand_array)
            f.attrs['timestamp'] = current_time
            f.attrs['camera'] = 'D405_hand'
            f.attrs['frequency'] = 15
            f.attrs['start_time'] = float(timestamps[0]) if len(timestamps) > 0 else 0.0
            f.attrs['end_time'] = float(timestamps[-1]) if len(timestamps) > 0 else 0.0
            f.attrs['duration'] = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 0 else 0.0

        print(f"Saved {len(color_hand_array)} frames to {h5_file}")


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

    # Initialize camera recording system
    recorder = CameraRecorder()

    print("\n" + "="*50)
    print("CAMERA RECORDING SYSTEM (No Robot Control)")
    print("="*50)
    print("\nControls:")
    print("  'c' - Start recording new episode")
    print("  's' - Stop recording and save")
    print("  'd' - Delete most recent episode")
    print("  'q' - Quit")
    print("\n" + "="*50)
    print("\nNOTE: Use spacemouse teleoperation script separately")
    print("      to control the robot during recording.")
    print("\nReady! Press 'c' to start recording.\n")

    # Start the synchronized capture in a separate thread
    capture_thread = threading.Thread(target=recorder.synchronized_capture, daemon=True)
    capture_thread.start()

    # Start listening for keyboard inputs
    listener = keyboard.Listener(on_press=recorder.on_press)
    listener.start()
    listener.join()

    print("\nStopped camera recording system.")


if __name__ == '__main__':
    main()
