# source /home/mainuser/UR5_Policy/data_record_env/bin/activate
# python /home/mainuser/UR5_Policy/spacemouse_teleoperation/3DConnexion_UR5_Teleop_Gripper_YinYu.py
# python /home/mainuser/UR5_Policy/data_recording/record_data.py

# k4aviewer
# realsense-viewer


from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface 
from rtde_io import RTDEIOInterface as RTDEIO
import robotiq_gripper
from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from threading import Thread, Event
from collections import defaultdict
import numpy as np
import time

#Instructions: Insert the plug into the fixed left socket and then pull it out.
#Instructions: Insert the plug into the fixed left socket and then pull it out with flexible plug location.
#Instructions: Insert the plug into the fixed left socket and then pull it out with flexible socket location.

#Instructions: Insert the 6-pin usb into the usb box.

# The only difference from 3DConnexion_UR5_Teleop.py is the ability to control the gripper position.

## ROBOT_HOST -> The robot's IP Address
## SCALE_FACTOR -> To increase/decrease the robot velocity 
## The acceleration in rtde_c.speedL() -> If there is any latency in robot movement (increasing acceleration = increasing deceleration)

class Spacemouse(Thread):
    def __init__(self, max_value=500, deadzone=(0,0,0,0,0,0), dtype=np.float32):
        """
        Continuously listen to 3D connection space naviagtor events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
        
        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        super().__init__()
        self.stop_event = Event()
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.motion_event = SpnavMotionEvent([0,0,0], [0,0,0], 0)
        self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

    def get_motion_state(self): #this method gets the movement of the mouse 
        me = self.motion_event
        state = np.array(me.translation + me.rotation, 
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
    
    def get_motion_state_transformed(self): #transforms get_motion_state 
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back
        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]

        # Set values lesser than 0.3 to 0 for better control
        tf_state[np.abs(tf_state) < 0.3] = 0
        tf_state = tf_state * SCALE_FACTOR

        return tf_state

    def is_button_pressed(self, button_id):
        return self.button_state[button_id]

    def stop(self):
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        spnav_open()
        try:
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    self.motion_event = event
                elif isinstance(event, SpnavButtonEvent):
                    self.button_state[event.bnum] = event.press
                else:
                    time.sleep(1/200)
        finally:
            spnav_close()

# Define robot parameters
ROBOT_HOST = "192.168.201.101"  # IP address of the robot controller
SCALE_FACTOR = 0.5 # Scale factor for velocity command

# =========================
# Force Mode Functions
# =========================

def zero_ft_sensor(rtde_c):
    # Zero the force/torque sensor readings to avoid initial deviation
    print("Zeroing FT Sensor...")
    rtde_c.zeroFtSensor()

def enter_force_mode(rtde_c, rtde_r):
    # Enter constant-force control along Z (downward), with Z the only compliant axis.
    print("Entering force mode...")
    task_frame = rtde_r.getActualTCPPose()  # Current TCP pose as reference frame
    zero_ft_sensor(rtde_c)
    selection = [0, 0, 1, 0, 0, 0]          # Only Z compliant
    Fz = 10.0                                # Target downward force (N)
    wrench = [0.0, 0.0, Fz, 0.0, 0.0, 0.0]
    type = 2                                 # 2 = force control mode
    limits = [float('inf'), float('inf'), 0.02,   # Z speed ≤ 2 cm/s (compliant axis)
              float('inf'), float('inf'), float('inf')]
    rtde_c.forceMode(task_frame, selection, wrench, type, limits)

# --- original exit_force_mode (kept for reference) ---
# def exit_force_mode(rtde_c):
#     print("Exiting force mode...")
#     rtde_c.forceModeStop()

def exit_force_mode(rtde_c):
    # Safer exit: stop motion first, short settle, then stop force-mode (avoids thread/state hiccups)
    print("Exiting force mode...")
    try:
        rtde_c.speedStop()
        time.sleep(0.02)
    except Exception as e:
        print("[warn] speedStop during exit:", e)
    try:
        rtde_c.forceModeStop()
    except Exception as e:
        print("[warn] forceModeStop:", e)

def main():
    sm = Spacemouse()
    sm.start()
    # Initialize RTDE interfaces
    rtde_c = RTDEControlInterface(ROBOT_HOST)
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)
    rtde_io = RTDEIO(ROBOT_HOST)
    
    print("Creating gripper...")
    gripper = robotiq_gripper.RobotiqGripper()
    print("Connecting to gripper...")
    gripper.connect(ROBOT_HOST, 63352)
    print("Activating gripper...")
    gripper.activate()
    gripper_position = gripper.get_current_position()
    gripper_max = gripper.get_max_position()
    gripper_min = gripper.get_min_position()

    # Force-mode state
    force_mode_active = False

    # --- New: two-button combo toggle (debounced) ---
    COMBO_WINDOW  = 0.12   # s: edges within this window → combo
    COMBO_COOLDOWN = 0.60  # s: prevent repeated toggles while held
    SINGLE_SUPPRESS = 0.25 # s: briefly suppress single open/close after combo

    prev_b0 = False
    prev_b1 = False
    b0_down_t = 0.0
    b1_down_t = 0.0
    combo_block_until = 0.0
    combo_cooldown_until = 0.0

    try:
        while True:
            if rtde_r.getRobotMode() == 7:
                # Read teleop inputs and robot state
                motion_state = sm.get_motion_state_transformed()
                TCP_pose = rtde_r.getActualTCPPose()
                Force = rtde_r.getActualTCPForce()

                print("Current Force reading: ", np.round(Force, 2))
                print("Cartesian Coordinates: ", np.round(TCP_pose, 2))
                print("Motion state: ", motion_state)

                # --- Original height-gated downward-only slow-down (kept, but commented) ---
                # if TCP_pose[0] > -0.2:
                #     if TCP_pose[2] < 0.26 and motion_state[2] < 0:
                #         motion_state *= 0.025  # slow all 6 DOF only when moving down

                # --- Replacement: linear-only slow-down near plane (all directions), extra-slow for Z-down ---

                # plug in socket
                # if TCP_pose[2] < 0.26:
                #     motion_state[:3] *= 0.25         # slow X/Y/Z linears
                #     if motion_state[2] < 0:
                #         motion_state[2] *= 0.40      # extra damping for downward Z (0.25*0.40 ≈ 0.10)

                # for picking usb plug
                if TCP_pose[0] < -0.20:
                    if TCP_pose[2] < 0.23:
                        motion_state[:3] *= 0.25         # slow X/Y/Z linears
                        if motion_state[2] < 0:
                            motion_state[2] *= 0.40      # extra damping for downward Z (0.25*0.40 ≈ 0.10)

                # plug usb box standing
                if TCP_pose[0] > -0.15:
                    if TCP_pose[2] < 0.38:
                        motion_state[:3] *= 0.25         # slow X/Y/Z linears
                        if motion_state[2] < 0:
                            motion_state[2] *= 0.40      # extra damping for downward Z (0.25*0.40 ≈ 0.10)

                # --- While in force-mode, don't fight Z force: block teleop Z only (rotations untouched) ---
                if force_mode_active:
                    motion_state[2] = 0.0

                # Send velocity command
                rtde_c.speedL(motion_state, acceleration=15, time=0.01)

                # Optional velocity print (filtered)
                actual_velocity = rtde_r.getActualTCPSpeed()
                actual_velocity = [0 if abs(x) < 0.01 else x for x in actual_velocity]
                print("Current velocity vector: ", actual_velocity)

                # --------- Button handling ---------
                now = time.time()
                b0 = sm.is_button_pressed(0)  # open
                b1 = sm.is_button_pressed(1)  # close

                # Rising edges (for combo timing)
                if b0 and not prev_b0:
                    b0_down_t = now
                if b1 and not prev_b1:
                    b1_down_t = now

                # Edge-triggered combo: both down within window and not in cooldown
                combo_edge = False
                if (b0 and b1) and (now >= combo_cooldown_until):
                    if (abs(b0_down_t - b1_down_t) <= COMBO_WINDOW) and ((not prev_b0) or (not prev_b1)):
                        combo_edge = True

                if combo_edge:
                    combo_block_until = now + SINGLE_SUPPRESS
                    combo_cooldown_until = now + COMBO_COOLDOWN
                    # Toggle Force-Mode with safe exit
                    if force_mode_active:
                        exit_force_mode(rtde_c)
                        force_mode_active = False
                        print("[FM] OFF")
                    else:
                        enter_force_mode(rtde_c, rtde_r)
                        force_mode_active = True
                        print("[FM] ON")

                # Single-button gripper actions (suppressed briefly after combo)
                if now >= combo_block_until:
                    if b0 and not b1:
                        gripper_position += 3
                        gripper.move(gripper_position, 155, 255)
                    if b1 and not b0:
                        gripper_position -= 3
                        gripper.move(gripper_position, 155, 255)

                # Update prev button states
                prev_b0, prev_b1 = b0, b1

                # Clamp gripper range
                if gripper_position < gripper_min:
                    gripper_position = gripper_min
                if gripper_position > gripper_max:
                    gripper_position = gripper_max

                print(f"Gripper Position ({gripper_min} to {gripper_max}): {gripper.get_current_position()}")
                if gripper.is_gripping():
                    print("Gripping object")
                else:
                    print("Not gripping object")

                time.sleep(1/100)

            else:
                print("Robot is not ready.")
                time.sleep(1)  # Wait longer if robot is not ready

    except KeyboardInterrupt:
        # Graceful shutdown
        print("Stopping robot")
        try:
            if force_mode_active:
                exit_force_mode(rtde_c)
        except:
            pass
        rtde_c.stopScript()
        sm.stop()

if __name__ == "__main__":
    main()
