import os 
import re
import zarr 
import numpy as np

## Script to convert data collected in npy to zarr for model training with batch processing

# data_path = '/media/mainuser/MohanSSD/hector/data_rec/plug'  # path of collected demonstrations
# save_path = '/media/mainuser/MohanSSD/hector/data_rec/plug.zarr'  # path to save zarr data
data_path = '/media/mainuser/a6300fe1-151f-4e9e-8790-c4826f4ee765/data_recording/data_weighing_scale_YuanYao/remove_vegetable_after_off'  # path of collected demonstrations
save_path = '/media/mainuser/1A64-8D68/remove_vegetable_after_off.zarr'  # path to save zarr data

# Open or create the Zarr root group
zarr_root = zarr.open_group(save_path, mode='a')

# Check for 'data' group
if 'data' not in zarr_root:
    zarr_data = zarr_root.create_group('data')
    print("Created 'data' group")
else:
    zarr_data = zarr_root['data']
    print("'data' group already exists")

# Check for 'meta' group
if 'meta' not in zarr_root:
    zarr_meta = zarr_root.create_group('meta')
    print("Created 'meta' group")
else:
    zarr_meta = zarr_root['meta']
    print("'meta' group already exists")

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

# Regular expression to match and extract episode number
episode_pattern = re.compile(r'episode_(\d+)')

# List all items in the main folder
all_items = os.listdir(data_path)

# Initialize arrays to store batches' data
rgb_image_array_batch = []
hand_image_array_batch = []
point_cloud_array_batch = []
state_array_batch = []
action_array_batch = []
depth_array_batch = []  # New: Depth array batch
eef_force_array_batch = []  # New: Force array batch
episode_end_batch = []

# Batch size
batch_size = 1000

# Find and sort episode folders
episode_folders = sorted(
    [item for item in all_items if os.path.isdir(os.path.join(data_path, item)) and episode_pattern.match(item)],
    key=lambda x: int(episode_pattern.match(x).group(1))
)

# Gather data from every episode in batches
for episode_folder in episode_folders:
    episode_path = os.path.join(data_path, episode_folder)
    print(f"Opening folder: {episode_path}")

    # Load data
    rgb_image = np.load(os.path.join(episode_path, "rgb.npy"))
    hand_image = np.load(os.path.join(episode_path, "rgb_hand.npy"))
    point_cloud = np.load(os.path.join(episode_path, "point_concatenate.npy"))
    # point_cloud = np.load(os.path.join(episode_path, "point_cloud.npy"))
    robot_state = np.load(os.path.join(episode_path, "joint_state.npy"))
    action = np.load(os.path.join(episode_path, "joint_action.npy"))
    depth_image = np.load(os.path.join(episode_path, "depth_hand.npy"))  # New: Load depth
    eef_force = np.load(os.path.join(episode_path, "eef_force.npy"))  # New: Load force

    # Process each batch within the episode
    for i in range(0, len(rgb_image), batch_size):
        # Get the batch
        batch_rgb_image = rgb_image[i:i + batch_size]
        batch_hand_image = hand_image[i:i + batch_size]
        batch_point_cloud = point_cloud[i:i + batch_size]
        batch_state = robot_state[i:i + batch_size]
        batch_action = action[i:i + batch_size]
        batch_depth_image = depth_image[i:i + batch_size]  # New: Process depth in batch
        batch_force = eef_force[i:i + batch_size]  # New: Process force in batch

        # Append batch data to the batch arrays
        rgb_image_array_batch.append(batch_rgb_image)
        hand_image_array_batch.append(batch_hand_image)
        point_cloud_array_batch.append(batch_point_cloud)
        state_array_batch.append(batch_state)
        action_array_batch.append(batch_action)
        depth_array_batch.append(batch_depth_image)  # New: Append depth batch
        eef_force_array_batch.append(batch_force)  # New: Append force batch

        # Track the episode end index
        if episode_end_batch:
            episode_end_batch.append(episode_end_batch[-1] + batch_rgb_image.shape[0])
        else:
            episode_end_batch.append(batch_rgb_image.shape[0])

    print(f"Processed episode: {episode_folder}")

# Stop early if no batches were collected (e.g. empty folder or regex mismatch)
if not rgb_image_array_batch:
    raise RuntimeError(
        f"No episodes were processed under {data_path}. "
        "Ensure the path is correct and contains episode_* folders with data."
    )

for i, arr in enumerate(rgb_image_array_batch):
    print(f"Shape of array {i}: {arr.shape}")

# Convert lists to numpy arrays after processing all episodes
rgb_image_array_batch = np.concatenate(rgb_image_array_batch, axis=0)
hand_image_array_batch = np.concatenate(hand_image_array_batch, axis=0)
point_cloud_array_batch = np.concatenate(point_cloud_array_batch, axis=0)
state_array_batch = np.concatenate(state_array_batch, axis=0)
action_array_batch = np.concatenate(action_array_batch, axis=0)
depth_array_batch = np.concatenate(depth_array_batch, axis=0)  # New: Concatenate depth array batch
eef_force_array_batch = np.concatenate(eef_force_array_batch, axis=0)  # New: Concatenate force array batch
episode_end_batch = np.array(episode_end_batch, dtype=object)

# Define chunk sizes
rgb_image_chunk_size = (batch_size, rgb_image_array_batch.shape[1], rgb_image_array_batch.shape[2], rgb_image_array_batch.shape[3])
hand_image_chunk_size = (batch_size, hand_image_array_batch.shape[1], hand_image_array_batch.shape[2], hand_image_array_batch.shape[3])
point_cloud_chunk_size = (batch_size, point_cloud_array_batch.shape[1], point_cloud_array_batch.shape[2])
action_chunk_size = (batch_size, action_array_batch.shape[1])
state_chunk_size = (batch_size, state_array_batch.shape[1])
depth_chunk_size = (batch_size, depth_array_batch.shape[1], depth_array_batch.shape[2])  # New: Depth chunk size
eef_force_chunk_size = (batch_size, eef_force_array_batch.shape[1])  # New: Force chunk size

# Create zarr datasets
zarr_data.create_dataset('img', data=rgb_image_array_batch, chunks=rgb_image_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('rgb_hand', data=hand_image_array_batch, chunks=hand_image_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('point_cloud', data=point_cloud_array_batch, chunks=point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_array_batch, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_array_batch, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('depth', data=depth_array_batch, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)  # New: Save depth
zarr_data.create_dataset('eef_force', data=eef_force_array_batch, chunks=eef_force_chunk_size, dtype='float32', overwrite=True, compressor=compressor)  # New: Save force

# Save episode end indices
zarr_meta.create_dataset('episode_ends', data=episode_end_batch, chunks=(batch_size,), dtype='int64', overwrite=True, compressor=compressor)

print("Data conversion to zarr completed successfully.")
