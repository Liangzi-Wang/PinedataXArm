import time
import json
import numpy as np
import pyrealsense2 as rs
import os
import threading
import robotiq_gripper
from pynput import keyboard 
from rtde_receive import RTDEReceiveInterface 
import shutil
import torch
import pytorch3d.ops as torch3d_ops
import pyk4a
from pyk4a import Config, PyK4A
import sys
import open3d as o3d
import cv2
from PIL import Image

## Script to collect demostrations of task for robot 

## Press 'c' to start recording new episode 
## Press 's' to stop recording 
## Press 'd' to delete most recent episode (only when not recording)
## Press 'q' to quit

class Data(): 
    def __init__(self): 
        self.ROBOT_HOST = "192.168.20.25" 
        self.num_points = 1024 # Adjust number of points to downsample to here
        self.state_shape = 7 # Adjust state shape here
        self.action_shape = 7 # Adjust action state here

        # Params for hand camera
        self.width = 640
        self.height = 480
        self.fps = 15

        # start robot and camera
        self.k4a = self.initialize_camera()
        self.realsense = self.initialize_hand_camera()
        self.rtde_r, self.gripper = self.initialize_robot()

        # Params for point cloud processing
        # self.bbox = [-500, -400, -600, 1000, 200, 1500]
        self.bbox = [-500, -400, -600, 1000, 200, 1000]
        self.plane = [0.00, 0.42, 0.91, -538.08]
        self.distance_threshold = 10
        self.handbbox = [-300, -300, -300, 300, 300, 100]

        # global data variables
        global color_array, depth_array, point_cloud_array, color_hand_array, depth_hand_array, point_cloud_hand_array, point_concatenate_array
        global joint_state_array, joint_action_array, eef_state_array, eef_action_array, eef_force_array, current_array
        color_array, depth_array, point_cloud_array, color_hand_array, depth_hand_array, point_cloud_hand_array, point_concatenate_array = [], [], [], [], [], [], []
        joint_state_array, joint_action_array, eef_state_array, eef_action_array, eef_force_array, current_array = [], [], [], [], [], []

    def initialize_camera(self): 
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                # camera_fps=pyk4a.FPS.FPS_30,
                camera_fps=pyk4a.FPS.FPS_15,
                depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                synchronized_images_only= True,
            )
        )
        k4a.start()
        # Set white balance
        k4a.whitebalance = 4500
        assert k4a.whitebalance == 4500
        return k4a

    def initialize_hand_camera(self):
        realsense = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        realsense.start(config)
        return realsense

    def initialize_robot(self): 
        rtde_r = RTDEReceiveInterface(self.ROBOT_HOST)
        print("Creating gripper...")
        gripper = robotiq_gripper.RobotiqGripper()
        print("Connecting to gripper...")
        gripper.connect(self.ROBOT_HOST, 63352)
        print("Activating gripper...")
        gripper.activate()
        return rtde_r, gripper

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
        action = np.array(self.rtde_r. getTargetTCPPose())
        gripper_state = np.array([self.gripper.get_current_position()]) 
        state = np.concatenate((state, gripper_state))
        action = np.concatenate((action, gripper_state))
        return state, action

    # Function to get end-effector force
    def get_robot_eef_force(self):
        force = np.array(self.rtde_r.getActualTCPForce())
        return force

    # Function to get current
    def get_robot_current(self):
        current = np.array(self.rtde_r.getActualCurrent())
        return current

    #-----------------------------------------------------------------------------------------------------------#
    def downsample_with_fps(self, points):
        # fast point cloud sampling using torch3d
        points = torch.from_numpy(points).unsqueeze(0).cuda()
        self.num_points = torch.tensor([self.num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=self.num_points)
        points = points.squeeze(0).cpu().numpy()
        points = points[sampled_indices.squeeze(0).cpu().numpy()]
        return points
    
    def distance_from_plane(self, points):
        # calculate distance of each point from the plane 
        a, b, c, d = self.plane[0], self.plane[1], self.plane[2], self.plane[3]
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        return distances
    
    # Function to get point cloud
    def get_point_cloud(self, capture):
        if capture is not None and capture.color is not None and capture.depth is not None:
            points = capture.depth_point_cloud.reshape((-1, 3))
            colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) / 255.0 

            # Define bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
            min_bound = np.array(self.bbox[:3])
            max_bound = np.array(self.bbox[3:])  

            # Filter points within bbox
            indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
            points = points[indices]
            colors = colors[indices]
            points = np.concatenate((points, colors), axis=1)
            # print("Final point cloud shapemohadwn:", points.shape)

            # Filter points based on the distance threshold
            distances = self.distance_from_plane(points)
            points = points[distances > self.distance_threshold]

            # downsample
            points = self.downsample_with_fps(points)
            return points
        else:
            raise ValueError("Kinect capture option is None.")
    #-----------------------------------------------------------------------------------------------------------#

    def get_point_cloud_realsense(self, capture):
        """
        Extract a point cloud from a RealSense frameset.
        The output format is (N, 6): first three columns are XYZ coordinates, last three columns are RGB colors.
        This function supports:
            - Bounding box filtering
            - Distance threshold filtering
            - Distance-to-plane filtering
            - FPS-based downsampling
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

        # 
        valid = (points[:, 2] > 720) & (points[:, 2] < 1300)
        points = points[valid]
        colors = colors[valid]

        # # Define bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
        # min_bound = np.array(self.handbbox[:3])
        # max_bound = np.array(self.handbbox[3:])  

        # # Filter points within bbox
        # indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        # points = points[indices]
        # colors = colors[indices]
        points = np.concatenate((points, colors), axis=1)
        # print("Final point cloud shape:", points.shape)

        # # Filter points based on the distance threshold
        # distances = self.distance_from_plane(points)
        # points = points[distances > self.distance_threshold]

        # downsample
        points = self.downsample_with_fps(points)
        # print("Final point cloud shape2:", points.shape)
        return points

    #-----------------------------------------------------------------------------------------------------------#
    # Function to get visual observations (RGB-D & PC) 
    def get_visual_obs(self):
        capture = self.k4a.get_capture()
        capture_hand = self.realsense.wait_for_frames()
        print("capturehand",capture_hand)

        # if capture is not None and capture.color is not None and capture.depth is not None and capture.depth_point_cloud is not None:
        if capture is not None and capture.color is not None and capture.depth is not None and capture.depth_point_cloud is not None and capture_hand is not None and capture_hand.get_color_frame() is not None:
            # Capture color image
            color_image = capture.color[:, :, :3]
            color_image = color_image[:, 280:1000, :3]
            color_image = cv2.resize(color_image, (256, 256))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Capture color_hand image
            color_hand_frame = capture_hand.get_color_frame()
            color_hand_image = np.asarray(color_hand_frame.get_data())
            color_hand_image = color_hand_image[:, 80:560, :3]
            color_hand_image = cv2.resize(color_hand_image, (256, 256))
            color_hand_image = cv2.cvtColor(color_hand_image, cv2.COLOR_BGR2RGB)

            # ----------------------------------------------------------------- #
            # # Visual debug
            # color_display = Image.fromarray(color_image)
            # color_display.show()
            # ----------------------------------------------------------------- #

            # Capture depth image
            depth_image = capture.depth

            # Capture depth_hand image
            depth_hand_frame = capture_hand.get_depth_frame()
            depth_hand_image = np.asarray(depth_hand_frame.get_data())
            # depth_hand_image = depth_hand_image[:, 80:560]
            # depth_hand_image = cv2.resize(depth_hand_image, (256, 256))

            # ----------------------------------------------------------------- #
            # # Visual debug
            # depth_masked = np.where(depth_image == 0, np.nan, depth_image)

            # # normalize to 0~255
            # min_valid = np.nanmin(depth_masked)
            # max_valid = np.nanmax(depth_masked)
            # depth_norm = ((depth_masked - min_valid) / (max_valid - min_valid)) * 255
            # depth_norm = np.nan_to_num(depth_norm, nan=0).astype(np.uint8)

            # depth_display = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            # cv2.imshow("Depth Image", depth_display)
            # cv2.waitKey(1000)
            # ----------------------------------------------------------------- #

            # Capture point cloud
            point_cloud = self.get_point_cloud(capture)

            # Capture point cloud from RealSense
            point_cloud_hand = self.get_point_cloud_realsense(capture_hand)
            # print(type(point_cloud_hand))
            # print(point_cloud_hand.dtype)
            # print(point_cloud_hand.shape)

            point_concatenate = np.concatenate((point_cloud, point_cloud_hand), axis=0)
            point_concatenate = self.downsample_with_fps(point_concatenate)

            # ----------------------------------------------------------------- #
            # # Visual debug
            # pc_o3d = o3d.geometry.PointCloud()
            # pc_o3d.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            # pc_o3d.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
            # o3d.visualization.draw_geometries([pc_o3d],
            #                                    zoom=0.8,
            #                                    front=[-0.4999, -0.1659, -0.8499],
            #                                    lookat=[2.1813, 2.0619, 2.0999],
            #                                    up=[0.1204, -0.9852, 0.1215])
            # ----------------------------------------------------------------- #
            # Visual debug_hand
            # pc_o3d = o3d.geometry.PointCloud()
            # pc_o3d.points = o3d.utility.Vector3dVector(point_concatenate[:, :3])
            # pc_o3d.colors = o3d.utility.Vector3dVector(point_concatenate[:, 3:])
            # o3d.visualization.draw_geometries(
            #     [pc_o3d],
            #     zoom=0.5,
            #     front=[0, 0, -1],   
            #     lookat=[0, 0, 0],   
            #     up=[0, -1, 0]       
            # )
            # exit()

            # ----------------------------------------------------------------- #

            return point_cloud, color_image, depth_image, color_hand_image, depth_hand_image, point_cloud_hand, point_concatenate
        else:
            raise ValueError("Kinect capture option is None.")


    # Function to handle key presses
    def on_press(self, key):
        global recording, current_time

        try:
            if key.char == 'c':
                if not recording:
                    current_time = time.strftime("%Y%m%d%H%M%S")
                    recording = True # start synchronization
                    print(f"Started recording episode ...")

            elif key.char == 's':
                if recording:
                    recording = False
                    episode_folder = os.path.join(data_folder, current_time)
                    print(f"Stopped recording episode ...")

                    time.sleep(1)
                    self.save_data(episode_folder)
                    print("Data saved")

                    time.sleep(1)
                    self.save_instruction(episode_folder)
                    print('Language instructions saved')

            elif key.char == 'q':
                if recording:
                    recording = False
                print("Quitting session...")
                return False  # Stop listener

            elif key.char == 'd':
                if not recording:
                    confirmation = input("Are you sure you want to delete the most recent episode? (y/n):")
                    if 'y' in confirmation:
                        # Delete the episode folder
                        episode_folder = os.path.join(data_folder, current_time)
                        if os.path.exists(episode_folder):
                            shutil.rmtree(episode_folder)
                        print(f"Deleted episode {episode_folder[-14:]}")
                    else:
                        print("Deletion canceled.")
                    

        except AttributeError:
            pass


    # Synchronization mechanism
    def synchronized_capture(self, frequency=5):
        global recording

        interval = 1.0 / frequency

        while True:
            start_time = time.time()

            if recording:
                # Get visual observations
                point_cloud, color_image, depth_image, color_hand_image, depth_hand_image,point_cloud_hand, point_concatenate = self.get_visual_obs()

                # Get robot state
                joint_state, joint_action = self.get_robot_joint_state()
                eef_state, eef_action = self.get_robot_eef_state()
                eef_force = self.get_robot_eef_force()
                current = self.get_robot_current()

                if joint_state is not None and eef_state is not None:
                    # Create episode folder if it doesn't exist
                    episode_folder = os.path.join(data_folder, current_time)
                    if not os.path.exists(episode_folder):
                        os.makedirs(episode_folder)

                    # Save data
                    self.synchronize_data(point_cloud, color_image, depth_image, color_hand_image, depth_hand_image, point_cloud_hand, point_concatenate, \
                                          joint_state, joint_action, eef_state, eef_action, eef_force, current)
                    print("loading")

            elapsed_time = time.time() - start_time
            time_to_sleep = interval - elapsed_time

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def synchronize_data(self, point_cloud, color_image, depth_image, color_hand_image, depth_hand_image,point_cloud_hand, point_concatenate, joint_state, joint_action, eef_state, eef_action, eef_force, current):
        # visual observations
        point_cloud_array.append(point_cloud)
        color_array.append(color_image)
        depth_array.append(depth_image)
        color_hand_array.append(color_hand_image)
        depth_hand_array.append(depth_hand_image)
        point_cloud_hand_array.append(point_cloud_hand)
        point_concatenate_array.append(point_concatenate)
            
        # states & actions
        joint_state_array.append(joint_state)
        joint_action_array.append(joint_action)
        eef_state_array.append(eef_state)
        eef_action_array.append(eef_action)
        eef_force_array.append(eef_force)
        current_array.append(current)

    def save_data(self, episode_folder):
        # Define file paths
        color_file = os.path.join(episode_folder, "rgb.npy")
        depth_file = os.path.join(episode_folder, "depth.npy")
        point_cloud_file = os.path.join(episode_folder, "point_cloud.npy")
        point_cloud_hand_file = os.path.join(episode_folder, "point_cloud_hand.npy")
        point_concatenate_file = os.path.join(episode_folder, "point_concatenate.npy")

        color_hand_file = os.path.join(episode_folder, "rgb_hand.npy")
        depth_hand_file = os.path.join(episode_folder, "depth_hand.npy")

        joint_state_file = os.path.join(episode_folder, "joint_state.npy")
        joint_action_file = os.path.join(episode_folder, "joint_action.npy")
        eef_state_file = os.path.join(episode_folder, "eef_state.npy")
        eef_action_file = os.path.join(episode_folder, "eef_action.npy")
        eef_force_file = os.path.join(episode_folder, "eef_force.npy")
        current_file = os.path.join(episode_folder, "current.npy")

        # List -> array
        point_cloud = np.array(point_cloud_array)
        color_image = np.array(color_array)
        depth_image = np.array(depth_array)
        color_hand_image = np.array(color_hand_array)
        depth_hand_image = np.array(depth_hand_array)
        point_cloud_hand = np.array(point_cloud_hand_array)
        point_concatenate = np.array(point_concatenate_array)

        joint_state = np.array(joint_state_array)
        joint_action = np.array(joint_action_array)
        eef_state = np.array(eef_state_array)
        eef_action = np.array(eef_action_array)
        eef_force = np.array(eef_force_array)
        current = np.array(current_array)

        # Save data into npy format
        np.save(point_cloud_file, point_cloud)
        np.save(point_cloud_hand_file, point_cloud_hand)
        np.save(point_concatenate_file, point_concatenate)
        np.save(color_file, color_image)
        np.save(depth_file, depth_image)
        np.save(color_hand_file, color_hand_image)
        np.save(depth_hand_file, depth_hand_image)

        np.save(joint_state_file, joint_state)
        np.save(joint_action_file, joint_action)
        np.save(eef_state_file, eef_state)
        np.save(eef_action_file, eef_action)
        np.save(eef_force_file, eef_force)
        np.save(current_file, current)

        # Save language instruction into json format
        instruction_json = {"instruction": instruction}
        with open(os.path.join(episode_folder, "instruction.json"), "w") as f:
            json.dump(instruction_json, f, indent=4)

        point_cloud_array.clear()
        point_cloud_hand_array.clear()
        point_concatenate_array.clear()       
        color_array.clear()             
        depth_array.clear()            
        color_hand_array.clear()       
        depth_hand_array.clear()        
        joint_state_array.clear()       
        joint_action_array.clear()      
        eef_state_array.clear()         
        eef_action_array.clear()        
        eef_force_array.clear()
        current_array.clear()      

def main():
    # Global variables for controlling the recording state
    global data_folder, recording, instruction
    data_folder = "/media/mainuser/Extreme Pro/press_button_data"
    # data_folder = "/media/mainuser/Vision 1/UR5_Policy/data"
    # data_folder = "/media/mainuser/a6300fe1-151f-4e9e-8790-c4826f4ee765/data_recording/data_weighing_scale_YuanYao/different_target_object_orientation"

    # data_folder = "/media/mainuser/a6300fe1-151f-4e9e-8790-c4826f4ee765/data_recording/plugnplay_YinYu/6pin_usb_force_fixed_plug_push"
    # data_folder = "/media/mainuser/a6300fe1-151f-4e9e-8790-c4826f4ee765/data_recording/package2package"
    # data_folder = "/media/mainuser/a6300fe1-151f-4e9e-8790-c4826f4ee765/data_recording/cook"
    # data_folder = "/media/mainuser/MohanSSD/hector/data_rec/assemble"
    # data_folder = "/home/mainuser/UR5_Policy/data_recording/test"
    # data_folder = "/home/mainuser/UR5_Policy/data_recording/SUTD"

    recording = False
    instruction = input("Type the instruction of this recording capture: ")

    data = Data()

    print("Start Recording, Press C to start recording episode")

    # Start the synchronized capture in a separate thread
    capture_thread = threading.Thread(target=data.synchronized_capture)
    capture_thread.start()

    # Start listening for keyboard inputs
    listener = keyboard.Listener(on_press=data.on_press)
    listener.start()
    listener.join()

    print("Stopped synchronized capture.")


if __name__ == '__main__':
    main()
