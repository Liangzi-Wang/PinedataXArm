import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# Set your folder path
folder_path = "/media/mainuser/a6300fe1-151f-4e9e-8790-c4826f4ee765/data_recording/data_weighing_scale_YuanYao/different_target_object_orientation/20251121123043/rgb.npy"
# folder_path = "/media/mainuser/a6300fe1-151f-4e9e-8790-c4826f4ee765/data_recording/data_weighing_scale_YuanYao/different_target_object_orientation/20251121101214/rgb_hand.npy"


# Load numpy array and display
image = np.load(folder_path)

for frame in image:
    # Ensure the frame is in uint8 (values between 0 and 255)
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow('Frame', frame)
    # Wait for 10 ms for key press
    key = cv2.waitKey(int(100/15))
    
    # If 'q' is pressed, exit the loop
    if key == ord('q'):
        break

# Close the display window
cv2.destroyAllWindows()