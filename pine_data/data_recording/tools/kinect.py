import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops

def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_2160P,
            camera_fps=pyk4a.FPS.FPS_30,
            depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # Set white balance
    k4a.whitebalance = 4500    
    assert k4a.whitebalance == 4500

    # Initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    capture = k4a.get_capture()
    if capture is not None and capture.depth is not None and capture.color is not None:
        points = capture.depth_point_cloud.reshape((-1, 3))
        colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) / 255.0 

        # Define bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
        bbox = [-500,-500,-600,1000,250,1200]
        min_bound = np.array(bbox[:3])
        max_bound = np.array(bbox[3:])  
        #crop point clouds
        indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        points = points[indices]
        colors = colors[indices]

        #points= downsample_with_fps(points)
        print(points.shape)
        print(colors.shape)
        # Save the point cloud data as a PCD file
        save_point_cloud_as_pcd("point_cloud.pcd", points, colors)


def save_point_cloud_as_pcd(filename, points, colors):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pc)

def downsample_with_fps(points: np.ndarray, num_points: int = 4096):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points

if __name__ == "__main__":
    main()
