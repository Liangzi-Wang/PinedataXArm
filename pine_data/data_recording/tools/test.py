import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
data = np.load('/home/mainuser/UR5_Policy/data_recording/data/20250717155352/joint_state.npy')

first_image = data[0]
print(first_image)
