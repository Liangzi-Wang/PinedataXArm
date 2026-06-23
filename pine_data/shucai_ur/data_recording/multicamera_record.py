import time
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
import argparse

## Script to collect demostrations of task for robot 

## Press 'c' to start recording new episode 
## Press 's' to stop recording 
## Press 'd' to delete most recent episode 
## Press 'q' to quit 
## When running script, can set argument to not save certain data

class Data():
    def __init__(self, save_RGB_rs, save_depth_rs, save_RGB_kinect, save_depth_kinect,save_force): 
        ## Adjust relevant parameters here
        self.ROBOT_HOST = "192.168.20.25" 
        self.num_points = 1024 # Adjust number of points to downsample to here
        self.state_shape = 7 # Adjust state shape here
        self.action_shape = 7 # Adjust action state here
        self.record_force = True # Set whether or not to record force data
        self.distance_threshold = 10 # Adjust distance threshold for plane segementation 
        self.frequency = 15 # Adjust recording frequency
        self.img_shape = 84 # Adjust image downsample

        #start robot and camera to receive data
        self.rtde_r, self.gripper = self.start_robot()
        self.k4a, self.pipeline = self.start_camera()

        self.save_RGB_rs = save_RGB_rs
        self.save_depth_rs = save_depth_rs
        self.save_RGB_kinect = save_RGB_kinect
        self.save_depth_kinect = save_depth_kinect
        self.save_force = save_force
       
    def start_camera(self): 
        
        ## Start Azure Kinect 
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_1080P,
                camera_fps=pyk4a.FPS.FPS_30,
                depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                synchronized_images_only= False,
            )
        )
        k4a.start()
        # Set white balance
        k4a.whitebalance = 4500
        assert k4a.whitebalance == 4500

        # Start realsense camera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start streaming
        pipeline.start(config)
      
        return k4a, pipeline
    
    def start_robot(self): 
        rtde_r = RTDEReceiveInterface(self.ROBOT_HOST)
        print("Creating gripper...")
        gripper = robotiq_gripper.RobotiqGripper()
        print("Connecting to gripper...")
        gripper.connect(self.ROBOT_HOST, 63352)
        print("Activating gripper...")
        gripper.activate()
        
        return rtde_r, gripper

    # Function to get robot state
    def get_robot_state(self):
        state = np.array(self.rtde_r.getActualQ())
        action = np.array(self.rtde_r.getTargetQ())
        gripper_state = np.array([self.gripper.get_current_position()]) 
        state = np.concatenate((state,gripper_state))
        action = np.concatenate((action,gripper_state))
        if self.record_force: 
            force = np.array(self.rtde_r.getActualTCPForce())
            return state, action, force

        return state, action

    def get_visual_obs_azure_kinect(self):
      ## Get point cloud from Azure Kinect camera 
      # Wait for a coherent pair of frames: depth and color
      capture = self.k4a.get_capture()
      if capture is not None and capture.depth is not None and capture.color is not None:
        points = capture.depth_point_cloud.reshape((-1, 3))
        #colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) / 255.0 

        # Define bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
        bbox = [-500,-500,-600,1000,250,1200]
        min_bound = np.array(bbox[:3])
        max_bound = np.array(bbox[3:])  

        #crop point clouds
        indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        points = points[indices]
        #colors = colors[indices]
        
        color_image = capture.color[:, :, :3]  # Drop the alpha channel
        depth_image = capture.depth

        # Change depth image to uint16
        depth_image = depth_image*1000
        depth_image = np.uint16(depth_image)

        return points, color_image, depth_image
      
    def get_visual_obs_realsense(self):
      # Get RGB image from realsense camera 
      frames = self.pipeline.wait_for_frames()
      frames.keep()                             ## not sure why this is required, but this prevents it from shutting down after receiving 16 frames
      color_frame = frames.get_color_frame()
      depth_frame = frames.get_depth_frame()

      # Convert color frame to a numpy array
      color_image = np.asanyarray(color_frame.get_data())
      depth_image = np.asanyarray(depth_frame.get_data())
      
      # Change depth image to uint16
      depth_image = depth_image*1000
      depth_image = np.uint16(depth_image)

      #color_image = color_image[...,::-1]
      return color_image, depth_image
  
    def downsample_with_fps(self, points: np.ndarray):
        # fast point cloud sampling using torch3d
        points = torch.from_numpy(points).unsqueeze(0).cuda()
        self.num_points = torch.tensor([self.num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=self.num_points)
        points = points.squeeze(0).cpu().numpy()
        points = points[sampled_indices.squeeze(0).cpu().numpy()]
        return points
    
    def distance_from_plane(self, points):
        #define the plane equation (determined from plane segementation algorithm)
        a = 0.10
        b = 0.63 
        c = 0.77
        d = -619.13 
        #calculate distance of each point from the plane 
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        return distances

    # Function to handle key presses
    def on_press(self, key):
        global recording, current_episode

        try:
            if key.char == 'c':
                if not recording:
                    current_episode += 1
                    global robot_state_array, action_array, point_cloud_array, rgb_array_kinect,depth_array_kinect, rgb_array_rs, depth_array_rs, force_array
                    robot_state_array, action_array, point_cloud_array, rgb_array_kinect,depth_array_kinect, rgb_array_rs, depth_array_rs, force_array = [], [], [], [], [], [] ,[], []
                    recording = True
                    print(f"Started recording episode {current_episode}...")

            elif key.char == 's':
                if recording:
                    recording = False
                    episode_folder = os.path.join(data_folder, f"episode_{current_episode}")
                    print(f"Stopped recording episode {current_episode}")

                    #reload and reshape data
                    time.sleep(1) 
                    self.reshape_data(episode_folder)
                    print("Data saved")

            elif key.char == 'q':
                if recording:
                    recording = False
                print("Quitting session...")
                return False  # Stop listener
            
            elif key.char == 'd':
                if not recording:
                    if current_episode > 0:
                        confirmation = input("Are you sure you want to delete the most recent episode? (y/n): ")
                        if confirmation.lower() == 'y':
                            # Delete the episode folder
                            episode_folder = os.path.join(data_folder, f"episode_{current_episode}")
                            if os.path.exists(episode_folder):
                                shutil.rmtree(episode_folder)
                            print(f"Deleted episode {current_episode}")
                            current_episode -= 1
                        else:
                            print("Deletion canceled.")
                    else:
                        print("No episodes to delete.")


        except AttributeError:
            pass


    # Synchronization mechanism
    def synchronized_capture(self):
        global recording, current_episode

        interval = 1.0 / self.frequency
       
        while True:
            start_time = time.time()
            color_image_rs, depth_image_rs = self.get_visual_obs_realsense()
            # RGB_image = cv2.resize(RGB_image,(self.img_shape,self.img_shape)) # reshape RGB image for saving only
            point_cloud, color_image_kinect, depth_image_kinect  = self.get_visual_obs_azure_kinect()

            cv2.imshow("Realsense", color_image_rs)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            cv2.imshow("Azure Kinect", color_image_kinect)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if recording: 
                robot_state,action,force = self.get_robot_state()
                # Apply plane segmentation to remove table 
                distances = self.distance_from_plane(point_cloud)
                point_cloud = point_cloud[distances > self.distance_threshold]
                
                # Downsample with FPS
                point_cloud = self.downsample_with_fps(np.array(point_cloud))

                if robot_state is not None:
                    # Create episode folder if it doesn't exist
                    episode_folder = os.path.join(data_folder, f"episode_{current_episode}")
                    if not os.path.exists(episode_folder):
                        os.makedirs(episode_folder)

                    # Save data
                    self.save_data(episode_folder, 
                                   robot_state,
                                   action,
                                   point_cloud,
                                   color_image_kinect if self.save_RGB_kinect else None, 
                                   depth_image_kinect if self.save_depth_kinect else None,
                                   color_image_rs if self.save_RGB_rs else None, 
                                   depth_image_rs if self.save_depth_rs else None,
                                   force if self.save_force else None)
                    print("loading")

            elapsed_time = time.time() - start_time
            time_to_sleep = interval - elapsed_time

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def save_data(self, episode_folder, robot_state, action, point_cloud, color_image_kinect, depth_image_kinect, color_image_rs, depth_image_rs,force):
        # Check if the episode folder exists, create it if not
        if not os.path.exists(episode_folder):
            os.makedirs(episode_folder)

        robot_state_array.append(robot_state)
        action_array.append(action)
        point_cloud_array.append(point_cloud)
        
        
        if self.save_RGB_kinect: 
            rgb_array_kinect.append(color_image_kinect)
        
        if self.save_depth_kinect: 
            depth_array_kinect.append(depth_image_kinect)
        
        if self.save_RGB_rs:
            rgb_array_rs.append(color_image_rs)
        
        if self.save_depth_rs: 
            depth_array_rs.append(depth_image_rs)
        
        if self.save_force: 
            force_array.append(force)

    def reshape_data(self, episode_folder):
        #Define file paths 
        robot_state_file = os.path.join(episode_folder, "robot_state.npy")
        action_file = os.path.join(episode_folder, "action.npy")
        point_cloud_file = os.path.join(episode_folder, "point_cloud.npy")

        robot_state = np.array(robot_state_array)
        action = np.array(action_array)
        point_cloud = np.array(point_cloud_array)
    
        #Save all data into npy format
        np.save(point_cloud_file, point_cloud)
        np.save(robot_state_file, robot_state)
        np.save(action_file, action)

        if self.save_RGB_kinect:
            color_image_kinect_file = os.path.join(episode_folder,"color_image_kinect.npy")
            color_image_kinect = np.array(rgb_array_kinect)
            np.save(color_image_kinect_file,color_image_kinect)

        if self.save_depth_kinect: 
            depth_image_kinect_file = os.path.join(episode_folder,"depth_image_kinect.npy")
            depth_image_kinect = np.array(depth_array_kinect)
            np.save(depth_image_kinect_file,depth_image_kinect)
        
        if self.save_RGB_rs:
            color_image_rs_file = os.path.join(episode_folder,"color_image_rs.npy")
            color_image_rs = np.array(rgb_array_rs)
            np.save(color_image_rs_file,color_image_rs)
        
        if self.save_depth_rs:
            depth_image_rs_file = os.path.join(episode_folder,"depth_image_rs.npy")
            depth_image_rs = np.array(depth_array_rs)
            np.save(depth_image_rs_file,depth_image_rs)
        
        if self.save_force: 
            force_file = os.path.join(episode_folder,"force.npy")
            force = np.array(force_array)
            np.save(force_file,force)


def main():

    global current_episode, data_folder, recording
    # Global variables for controlling the recording state
    data_folder = "data"
    recording = False
    current_episode = 0

    def str2bool(v):
        if v.lower() in ('true', '1'):
            return True
        elif v.lower() in ('false', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Data Collection Script")
    parser.add_argument('--save-RGB-rs', type=str2bool, default=True, help='Set to False to not receive RGB image from Realsense Camera')
    parser.add_argument('--save-depth-rs', type=str2bool, default=True, help='Set to False to not receive depth image from Realsense Camera')
    parser.add_argument('--save-RGB-kinect', type=str2bool, default=True, help='Set to False to not receive RGB image from Azure Kinect Camera')
    parser.add_argument('--save-depth-kinect', type=str2bool, default=True, help='Set to False to not receive depth image from Azure Kinect Camera')
    parser.add_argument('--save-force', type=str2bool, default=True, help='Set to False to not receive force data from robot')


    args = parser.parse_args()

    data = Data(save_RGB_rs = args.save_RGB_rs,
                save_depth_rs = args.save_depth_rs,
                save_RGB_kinect = args.save_RGB_kinect,
                save_depth_kinect = args.save_depth_kinect,
                save_force = args.save_force)
    

    # Check for the latest episode and continue from it, ensure data folder exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    episodes = [int(folder.split('_')[-1]) for folder in os.listdir(data_folder) if folder.startswith('episode')]
    if episodes:
        current_episode = max(episodes)
        print(f"Resuming from episode {current_episode}")
    else:
        current_episode = 0
        print("No previous episodes found. Starting from episode 1.")

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
