import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
state = np.load('/home/mainuser/UR5_Policy/data_recording/data/20250515150138/eef_state.npy')  # shape: (N, H, W, 3) expected
action = np.load('/home/mainuser/UR5_Policy/data_recording/data/20250515150138/eef_action.npy')

print("state:", state)
print("action:", action)

# Loop over each image
# for idx, img in enumerate(rgb_array):
#     plt.imshow(img.astype(np.uint8))  # cast if necessary
#     plt.title(f"Image {idx}")
#     plt.axis('off')
#     plt.show()
