import zarr
import visualizer

zarr_file = zarr.open('/home/mainuser/robot_arm/testing_code/hardware/azure_kinect/zarr_data', mode='r')

print(zarr_file.tree())

#actions = zarr_file['action']

#print(actions.info)

#actions = actions[:]

point_cloud = zarr_file['data']['point_cloud']
point_cloud = point_cloud[:]
print(point_cloud[0].shape)
 
action = zarr_file['data']['action']
action = action[:]

ep_end = zarr_file['meta']['episode_ends']
ep_end = ep_end[:]
print(ep_end)
