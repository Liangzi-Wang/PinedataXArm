## This folder contains the scripts for recording demostrations

1. record_pc.py - records point clouds, robot_state and action
2. record_all_data.py - records point clouds, RGB, Depth, robot_state and action
3. convert_to_zarr.py - converts npy format from the above two scripts to zarr
4. robotiq_gripper.py - script provided by ur_rtde to control/receive gripper information
5. multicamera_record.py - record data with Azure Kinect (Point Cloud) and Realsense (RGB), robot_state, force and action 
6. tools - scripts to check data 

## There are two point cloud segmentations done for recording scripts 
1. Cropping by bounding volume, alter bbox value if necessary
2. Plane segementation using RANSAC - to filter out the table 
 - In the tools folder, there is a plane_segmentation.py provided to view the point clouds after segementation, there is also a portion of code commented out which can be used to determine the equation of the plane 
 - Edit the coefficients in the method distance_from_plane() and the distance_threshold if necessary
 - Refer to https://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html for more details

## Downsampling 
As mentioned in the DP3 paper, downsampling is done via FPS, edit self.num_points if necessary. For most task 1024 seems to be the sweet spot. 

To set up azure kinect on a new ubuntu 22.04 workstation, refer to this link: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1790
