import numpy as np
import os
import open3d as o3d
import cv2

# Function to read data from the episode folder
def read_data(episode_folder):
    robot_state = np.load(os.path.join(episode_folder, "robot_state.npy"))
    action = np.load(os.path.join(episode_folder, "action.npy"))
    point_cloud = np.load(os.path.join(episode_folder, "point_cloud.npy"))
    return robot_state, action, point_cloud

def read_rgbd(episode_folder):
    color_image = np.load(os.path.join(episode_folder,"rgb.npy"))
    # depth_image = np.load(os.path.join(episode_folder,"depth.npy"))

    return color_image

# Function to save points as a PCD file without RGB data
def save_as_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(filename, pcd)

# Example usage:
episode_folder = '/home/mainuser/dp3_ws/dp3_setup_scripts/data_recording_(azure kinect)/data/episode_90'  # Path to the episode folder
robot_state, action, point_cloud = read_data(episode_folder)


# Print the shapes of the loaded arrays
print(f"Point Cloud Shape: {point_cloud.shape}")
print(f"Action Shape: {action.shape}")
print(f"Robot State Shape: {robot_state.shape}")

#read force 
# force = np.load(os.path.join(episode_folder,'force.npy'))
# print(f"Force shape: {force.shape}")

# Extract the first point cloud from the loaded data
points = point_cloud[0]
print(f"First Point Cloud Shape: {points.shape}")

# Save the first point cloud as a PCD file
save_as_pcd(points, "processed_point_cloud.pcd")


# For RGBD visualisation
color_image = read_rgbd(episode_folder)
print(f"RGB Shape: {color_image.shape}")
# print(f"Depth Shape: {depth_image.shape}")

interval = 5
for image in color_image:
    cv2.imshow("image", image)
    if cv2.waitKey(interval) & 0xFF == ord('q'):
        break
