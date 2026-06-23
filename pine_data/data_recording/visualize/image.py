import matplotlib.pyplot as plt
import numpy as np
import os

# Set your folder path
folder_path = "/media/mainuser/MohanSSD/data/assemble/20251028161306/rgb_hand.npy"

# Get first .npy file
# npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
# first_npy_path = os.path.join(folder_path, folder_path[0])

# Load numpy array and display
image = np.load(folder_path)
image = image[0]
plt.imshow(image)
# plt.title(folder_path[0])
plt.savefig('/home/mainuser/UR5_Policy/data_recording/visualize/image_hand.png')

plt.imshow(image)
