import os
import shutil

# Define the source and target directories
source_dir = "/home/mainuser/UR5_Policy/current_data"  # Replace with your source directory
target_dir = "/home/mainuser/UR5_Policy/rlds_data_builder/UR5_dataset/data/train"  # Replace with your target directory

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Iterate through each episode folder in the source directory
for episode_folder in os.listdir(source_dir):
    episode_folder_path = os.path.join(source_dir, episode_folder)
    
    # Check if it is a directory and matches the 'episode_x' naming pattern
    if os.path.isdir(episode_folder_path) and episode_folder.startswith("episode_"):
        # Iterate through each .npy file in the current episode folder
        for file_name in os.listdir(episode_folder_path):
            if file_name.startswith("episode_"):
                # Construct the source file path
                source_file_path = os.path.join(episode_folder_path, file_name)
                
                # Construct the new file name (e.g., episode_001.npy)
                episode_number = episode_folder.split("_")[1]
                new_file_name = f"episode_{int(episode_number)}.npy"
                
                # Construct the target file path
                target_file_path = os.path.join(target_dir, new_file_name)
                
                # Copy the .npy file to the target directory
                shutil.copy(source_file_path, target_file_path)
                print(f"Copied: {source_file_path} -> {target_file_path}")