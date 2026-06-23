# source /home/mainuser/UR5_Policy/data_record_env/bin/activate
# python /home/mainuser/UR5_Policy/spacemouse_teleoperation/3DConnexion_UR5_Teleop_Gripper_YinYu.py
# python /home/mainuser/UR5_Policy/data_recording/record_data.py

# k4aviewer
# realsense-viewer


from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface 
from rtde_io import RTDEIOInterface as RTDEIO
# import robotiq_gripper  # robotiq gripper disabled
from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from threading import Thread, Event
from collections import defaultdict
from pynput import keyboard
import numpy as np
import time

import h5py
import os
import re
import argparse
from pathlib import Path

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
        print("2")
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
        
        # Freeze yaw and pitch BEFORE transformation
        # Keep only rotation around Z axis in SpaceMouse frame (which becomes X after transform)
        state[3] = 0  # Freeze rotation around SpaceMouse X
        state[5] = 0  # Freeze rotation around SpaceMouse Y
        # state[5] is rotation around SpaceMouse Z - keep it (becomes roll around robot X)
        # 
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

# Festo gripper DO mapping (update to match your wiring)
FESTO_OPEN_DO = 0   # energize this line to open
FESTO_CLOSE_DO = 2  # energize this line to close
# Set to True if your Festo valves are wired to the TOOL connector instead of controller I/Os
FESTO_USE_TOOL_DO = False


# =========================
# Festo Gripper Helper
# =========================

class FestoGripper:
    def __init__(self, rtde_io, open_do=FESTO_OPEN_DO, close_do=FESTO_CLOSE_DO):
        self.rtde_io = rtde_io
        self.open_do = open_do
        self.close_do = close_do
        self.state = None
        print(f"[Festo] connected via RTDEIO (open_do={open_do}, close_do={close_do}, use_tool={FESTO_USE_TOOL_DO})")
        self.grip()
        self.release()  # start open by default

    def _set_outputs(self, open_on: bool, close_on: bool):
        # Simple 2-solenoid drive; adjust to match your manifold logic.
        # Drive controller DOs
        self.rtde_io.setStandardDigitalOut(self.open_do, open_on)
        self.rtde_io.setStandardDigitalOut(self.close_do, close_on)
        # Optionally drive tool DOs (many Festo valves are wired here)
        if FESTO_USE_TOOL_DO:
            try:
                self.rtde_io.setToolDigitalOut(0, open_on)   # tool DO0
                self.rtde_io.setToolDigitalOut(2, close_on)  # tool DO1
            except Exception as e:
                print("[Festo] tool DO write failed:", e)
        self.state = "open" if open_on else "closed"
        print(f"[Festo] state -> {self.state} (open_do={open_on}, close_do={close_on})")

    def grip(self):
        self._set_outputs(False, True)

    def release(self):
        self._set_outputs(True, False)

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
        
        
###
def save_trajectory_to_h5(tcp_poses, joint_poses, forces, timestamps, episode_time, data_folder):
    """Save robot trajectory data to HDF5 file"""
    os.makedirs(data_folder, exist_ok=True)
    h5_file = os.path.join(data_folder, f"trajectory_{episode_time}.h5")
    
    # Convert lists to numpy arrays
    tcp_poses = np.array(tcp_poses)
    joint_poses = np.array(joint_poses)
    forces = np.array(forces)
    timestamps = np.array(timestamps)
    
    print(f"\nSaving trajectory data to HDF5 format...")
    with h5py.File(h5_file, 'w') as f:
        # Create datasets with compression
        f.create_dataset('tcp_pose', data=tcp_poses, compression='gzip', compression_opts=4)
        f.create_dataset('joint_pose', data=joint_poses, compression='gzip', compression_opts=4)
        f.create_dataset('force', data=forces, compression='gzip', compression_opts=4)
        f.create_dataset('timestamps', data=timestamps, compression='gzip', compression_opts=4)
        
        # Save metadata as attributes
        # f.attrs['instruction'] = instruction
        # f.attrs['num_frames'] = len(tcp_poses)
        # f.attrs['timestamp'] = time.strftime("%Y%m%d%H%M%S")
        # f.attrs['robot'] = 'UR5'
        # f.attrs['control_frequency'] = 200  # Hz
        # f.attrs['start_time'] = float(timestamps[0]) if len(timestamps) > 0 else 0.0
        # f.attrs['end_time'] = float(timestamps[-1]) if len(timestamps) > 0 else 0.0
        # f.attrs['duration'] = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 0 else 0.0
    
    print(f"Saved {len(tcp_poses)} frames to {h5_file}")
    
###

def main():
    parser = argparse.ArgumentParser(description="UR5 teleoperation + trajectory recording.")
    parser.add_argument(
        "--root",
        default="./recordings",
        help="Root directory for saved data (default: ./recordings).",
    )
    args = parser.parse_args()

    # Reset position configuration
    RESET_TCP_POSE = [0.2506, -0.2463, 0.3242, 1.1388, -2.9149, -0.0240]
    RESET_JOINT_POSE = [1.9795, -1.4438, 1.3327, -1.4459, -1.5835, 12.2385]
    
    raw_instruction = input("Enter the task instruction: ").strip()
    # Sanitize instruction for safe folder name
    safe_instruction = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_instruction).strip("._-")
    instruction = safe_instruction if safe_instruction else "untitled"
    date_str = time.strftime("%Y%m%d")
    base_data_folder = Path(args.root).expanduser().resolve()
    data_folder = str(base_data_folder / date_str / instruction / "trajs_h5")
    
    sm = Spacemouse()
    sm.start()
    # Initialize RTDE interfaces
    rtde_c = RTDEControlInterface(ROBOT_HOST)
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)
    rtde_io = RTDEIO(ROBOT_HOST)

    # Festo gripper
    festo = FestoGripper(rtde_io)
    print("[Festo] initial state:", festo.state)

    # Force-mode state
    force_mode_active = False
    
    # Reset flag
    reset_requested = False

    # --- Data recording buffers ---
    tcp_pose_array = []
    joint_pose_array = []
    force_array = []
    timestamp_array = []
        
    # Initialize recording state
    recording = False
    current_episode_time = None
    current_episode_folder = None
    
    def on_key_press(key):
        nonlocal reset_requested, recording, current_episode_time, current_episode_folder
        nonlocal tcp_pose_array, joint_pose_array, force_array, timestamp_array
        
        try:
            if key.char == 'r':
                reset_requested = True
                print("\n[RESET] Reset to home position requested...")
                
            elif key.char == 'c':
                if not recording:
                    # Clear buffers and start new episode
                    tcp_pose_array = []
                    joint_pose_array = []
                    force_array = []
                    timestamp_array = []
                    
                    current_episode_time = time.strftime("%Y%m%d%H%M%S")
                    current_episode_folder = data_folder
                    
                    recording = True
                    print(f"\n{'='*60}")
                    print(f"[RECORDING] Started episode {current_episode_time}")
                    print(f"[RECORDING] Saving to: {current_episode_folder}")
                    print(f"{'='*60}\n")
                else:
                    print("[RECORDING] Already recording! Press 's' to stop first.")
                    
            elif key.char == 's':
                if recording:
                    print(f"\n[RECORDING] Stopping episode {current_episode_time}...")
                    
                    # Save trajectory to HDF5
                    save_trajectory_to_h5(tcp_pose_array, joint_pose_array, force_array,
                                         timestamp_array, current_episode_time, data_folder)
                    
                    recording = False
                    print(f"[RECORDING] Episode {current_episode_time} saved!")
                    print(f"[RECORDING] Total frames: {len(tcp_pose_array)}")
                    print(f"[RECORDING] Ready for next episode. Press 'c' to start.\n")
                else:
                    print("[RECORDING] Not recording. Press 'c' to start recording.")
                    
            elif key.char == 'q':
                if recording:
                    print("\n[RECORDING] Stopping current episode before quitting...")
                    save_trajectory_to_h5(tcp_pose_array, joint_pose_array, force_array,
                                         timestamp_array, current_episode_time, data_folder)
                    recording = False
                print("Quitting...")
                return False  # Stop listener
                
        except AttributeError:
            pass
    
    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    # --- Position saving state ---
    position_file = "robot_trajectory.txt"
    frame_count = 0
    
    # Clear/create new file
    with open(position_file, 'w') as f:
        f.write("# Robot Trajectory Recording\n")
        f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Format: Frame, Timestamp, TCP_Pose(6), Joint_Angles(6), Force(6)\n\n")
    
    # --- New: two-button combo toggle (debounced) ---
    COMBO_WINDOW  = 0.20   # s: edges within this window → combo (wider = easier to trigger)
    COMBO_COOLDOWN = 0.35  # s: prevent repeated toggles while held (shorter = more responsive)
    SINGLE_SUPPRESS = 0.08 # s: briefly suppress single spacemouse_teleoperation/3DConnexion_UR5_Teleop_Gripper_pine.pyopen/close after combo (shorter = quicker resume)

    prev_b0 = False
    prev_b1 = False
    b0_down_t = 0.0
    b1_down_t = 0.0
    combo_block_until = 0.0
    combo_cooldown_until = 0.0

    print("\n" + "="*60)
    print("SPACEMOUSE TELEOPERATION WITH CONTINUOUS RECORDING")
    print("="*60)
    print("Controls:")
    print("  - Move SpaceMouse to control robot")
    print("  - SpaceMouse rotation: ROLL only (pitch/yaw frozen)")
    print("  - Button 0: Open gripper")
    print("  - Button 1: Close gripper")
    print("  - Press 'r': Reset to home position")
    print("  - Ctrl+C: Quit and save trajectory")
    print(f"Recording to: {position_file}")
    print(f"Home Position: TCP={RESET_TCP_POSE[:3]}")
    print("="*60 + "\n")
    
    try:
        while True:
            if rtde_r.getRobotMode() == 7:
                # Check for reset request
                if reset_requested:
                    print("\n" + "="*60)
                    print("[RESET] Moving to home position...")
                    print(f"Target TCP: {RESET_TCP_POSE}")
                    print(f"Target Joints: {RESET_JOINT_POSE}")
                    print("="*60)
                    
                    # Stop current motion
                    rtde_c.speedStop()
                    time.sleep(0.1)
                    
                    # Lift TCP vertically by 2cm before reset
                    print("[RESET] Lifting TCP to plug out\n")
                    current_tcp = rtde_r.getActualTCPPose()
                    lift_tcp = current_tcp.copy()
                    lift_tcp[2] += 0.02  # Add 2cm (0.02m) to Z coordinate
                    rtde_c.moveL(lift_tcp, speed=0.1, acceleration=0.3)
                    time.sleep(0.2)  # Wait for lift to complete
                    
                    # Move to reset position using joint control (safer)
                    velocity = 0.9  # rad/s
                    acceleration = 0.9  # rad/s^2
                    rtde_c.moveJ(RESET_JOINT_POSE, velocity, acceleration)
                    
                    # Wait for movement to complete
                    time.sleep(0.5)
                    
                    # Verify position
                    current_tcp = rtde_r.getActualTCPPose()
                    current_joints = rtde_r.getActualQ()
                    print(f"[RESET] Current TCP: {[round(x, 4) for x in current_tcp]}")
                    print(f"[RESET] Current Joints: {[round(j, 4) for j in current_joints]}")
                    print("[RESET] ✓ Reset complete!\n")
                    
                    reset_requested = False
                
                # Read teleop inputs and robot statesource data_record_env/bin/activate 

                motion_state = sm.get_motion_state_transformed()
                TCP_pose = rtde_r.getActualTCPPose()
                joint_pose = rtde_r.getActualQ()
                Force = rtde_r.getActualTCPForce()
                
                # Save current position and force to file (every frame)
                frame_count += 1
                timestamp = time.time()
                with open(position_file, 'a') as f:
                    f.write(f"Frame_{frame_count:06d},{timestamp:.6f},")
                    f.write(f"{','.join(map(str, TCP_pose))},")
                    f.write(f"{','.join(map(str, joint_pose))},")
                    f.write(f"{','.join(map(str, Force))}\n")
                    
                # print(f"Frame {frame_count} | Force: {np.round(Force, 2)} | TCP: {np.round(TCP_pose, 2)}")
                # print("Motion state: ", motion_state)
                
                # Save current data to buffers (line by line recording)
                if recording:
                    tcp_pose_array.append(TCP_pose)
                    joint_pose_array.append(joint_pose)
                    force_array.append(Force)
                    timestamp_array.append(timestamp)
                    # Print recording status
                    print(f"[REC] Frame {len(tcp_pose_array)} | Force: {np.round(Force, 2)} | TCP: {np.round(TCP_pose, 2)}", end='\r')
                else:
                    # Print normal status when not recording
                    print(f"Force: {np.round(Force, 2)} | TCP: {np.round(TCP_pose, 2)}", end='\r')

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
                # if force_mode_active:
                #     motion_state[2] = 0.0

                # Send velocity command
                rtde_c.speedL(motion_state, acceleration=15, time=0.01)

                # Optional velocity print (filtered)
                actual_velocity = rtde_r.getActualTCPSpeed()
                actual_velocity = [0 if abs(x) < 0.01 else x for x in actual_velocity]
                # print("Current velocity vector: ", actual_velocity)

                # --------- Button handling ---------
                now = time.time()
                b0 = sm.is_button_pressed(0)  # open
                b1 = sm.is_button_pressed(1)  # close

                # Rising edges (for combo timing)
                if b0 and not prev_b0:
                    b0_down_t = now
                    print("[BTN] b0 down")
                if b1 and not prev_b1:
                    b1_down_t = now
                    print("[BTN] b1 down")
                if (not b0) and prev_b0:
                    print("[BTN] b0 up")
                if (not b1) and prev_b1:
                    print("[BTN] b1 up")

                # Edge-triggered combo: both down within window and not in cooldown
                combo_edge = False
                if (b0 and b1) and (now >= combo_cooldown_until):
                    if (abs(b0_down_t - b1_down_t) <= COMBO_WINDOW) and ((not prev_b0) or (not prev_b1)):
                        combo_edge = True

                # if combo_edge:
                #     combo_block_until = now + SINGLE_SUPPRESS
                #     combo_cooldown_until = now + COMBO_COOLDOWN
                #     # Toggle Force-Mode with safe exit
                #     if force_mode_active:
                #         exit_force_mode(rtde_c)
                #         force_mode_active = False
                #         print("[FM] OFF")
                #     else:
                #         enter_force_mode(rtde_c, rtde_r)
                #         force_mode_active = True
                #         print("[FM] ON")

                if now >= combo_block_until:
                    if b0 and not b1:
                        print("~~~~~~~~~~~~~~~~~~!!!!!!!!!!!!!!!!!!!11GRIPPER OPEN")
                        festo.release()  # open
                    if b1 and not b0:
                        festo.grip()     # close
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1GRIPPER cLOse")

                # Update prev button states
                prev_b0, prev_b1 = b0, b1

                time.sleep(1/200)

            else:
                print("Robot is not ready.")
                time.sleep(1)  # Wait longer if robot is not ready

    except KeyboardInterrupt:
        # Graceful shutdown
        print("\n\nStopping robot...")
        try:
            if force_mode_active:
                exit_force_mode(rtde_c)
        except:
            pass
        rtde_c.stopScript()
        sm.stop()
        listener.stop()
        
        # Save summary to file
        with open(position_file, 'a') as f:
            f.write(f"\n# Recording ended: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total frames recorded: {frame_count}\n")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total frames recorded: {frame_count}")
        print(f"Trajectory saved to: {position_file}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
