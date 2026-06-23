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
    depth_image = np.load(os.path.join(episode_folder,"depth.npy"))

    return color_image, depth_image


# Function to save points as a PCD file without RGB data
def save_as_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(filename, pcd)

# Example usage:
episode_folder = '/home/mainuser/UR5_Policy/data_recording/data_cube_plate/20250613141250'  # Path to the episode folder
robot_state, action, point_cloud = read_data(episode_folder)


# Print the shapes of the loaded arrays
print(f"Point Cloud Shape: {point_cloud.shape}")
print(f"Action Shape: {action.shape}")
print(f"Robot State Shape: {robot_state.shape}")

# Extract the first point cloud from the loaded data
points = point_cloud[123]
print(f"First Point Cloud Shape: {points.shape}")

# Save the first point cloud as a PCD file
save_as_pcd(points, "processed_point_cloud.pcd")


# # For RGBD visualisation
# color_image, depth_image = read_rgbd(episode_folder)
# print(f"RGB Shape: {color_image.shape}")
# print(f"Depth Shape: {depth_image.shape}")

# # Show RGB image 
# cv2.imshow("image", color_image[90])
# # Wait for a key press and close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()

