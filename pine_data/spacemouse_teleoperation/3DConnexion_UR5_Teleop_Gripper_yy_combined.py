from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface 
from rtde_io import RTDEIOInterface as RTDEIO
import robotiq_gripper
from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from threading import Thread, Event
from collections import defaultdict
import numpy as np
import time
import argparse

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
ROBOT_HOST = "192.168.20.25"  # IP address of the robot controller
SCALE_FACTOR = 0.5 # Scale factor for velocity command

def zero_ft_sensor(rtde_c):
    # Zero the force/torque sensor readings to avoid initial deviation
    print("Zeroing FT Sensor...")
    rtde_c.zeroFtSensor()

def enter_force_mode(rtde_c, rtde_r):
    print("Entering force mode...")
    task_frame = rtde_r.getActualTCPPose()  # Get current TCP pose for force mode reference

    zero_ft_sensor(rtde_c)
    
    # Only Z is compliant (apply downward force along Z)
    selection = [0, 0, 1, 0, 0, 0]  # Only Z compliant
    
    # Define the force wrench (apply downward force along Z)
    Fz = 15.0  # Downward force in Newtons (adjust as needed)
    wrench = [0.0, 0.0, Fz, 0.0, 0.0, 0.0]  # Apply force along Z

    # Force mode type (2 = constant-force control)
    type = 2

    # Limits for compliant and non-compliant axes
    limits = [float('inf'), float('inf'), 0.02,   # Z speed ≤ 2 cm/s (for the compliant Z axis)
              float('inf'), float('inf'), float('inf')]  # No limits

    # Start force mode with the specified settings
    rtde_c.forceMode(task_frame, selection, wrench, type, limits)


def exit_force_mode(rtde_c):
    print("Exiting force mode...")
    rtde_c.forceModeStop()

def main(task_type):
    sm = Spacemouse()
    sm.start()
    # Initialize RTDEControlInterface
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

    force_mode_active = False  # Ensure this is initialized outside the loop
    last_button_press_time = 0.0

    try:
        while True:
            if rtde_r.getRobotMode() == 7:
                # Read motion state from SpaceMouse
                motion_state = sm.get_motion_state_transformed()

                # Approaching socket, slow down
                TCP_pose = rtde_r.getActualTCPPose()
                force = rtde_r.getActualTCPForce()
                print("Current Force reading: ", np.round(force, 2))
                print("Cartesian Coordinates: ", np.round(TCP_pose, 2))
                print("Motion state: ", motion_state)

                # Task-specific logic based on the task_type argument
                if task_type == "force_mode_task":
                    # Force Mode Task: Enter or Exit force mode based on button presses
                    if sm.is_button_pressed(0) and sm.is_button_pressed(1):  # Both buttons pressed together
                        current_time = time.time()
                        if current_time - last_button_press_time > 1.0:
                            print("Toggling Force Mode...")
                            if force_mode_active:
                                exit_force_mode(rtde_c)
                            else:
                                enter_force_mode(rtde_c, rtde_r)
                            last_button_press_time = current_time

                    # **Slow Down Logic**: Slow down the downward movement when close to the socket
                    if TCP_pose[0] > -0.2:  # Assuming the socket is at X = -0.2
                        if TCP_pose[2] < 0.26 and motion_state[2] < 0:  # Close to socket and moving down
                            motion_state *= 0.025  # Slow down downward movement

                    # Normal movement command when force mode is not active
                    if not force_mode_active:
                        rtde_c.speedL(motion_state, acceleration=15, time=0.01)

                elif task_type == "weighing_scale_task":
                    # Weighing Scale Task: Apply logic for weighing scale (force detection, moving slowly)
                    if TCP_pose[0] > -0.2:
                        if TCP_pose[2] < 0.26 and motion_state[2] < 0:
                            motion_state *= 0.025  # Slow down downward movement when close to table
                        if TCP_pose[2] < 0.23 and force[2] > 50 and motion_state[2] < 0:
                            motion_state[2] = 0  # Stop downward movement when contact is detected

                #send command to robot 
                rtde_c.speedL(motion_state, acceleration = 15, time = 0.01) #adjust the acceleration if required 

                #get TCP velocity of robot
                actual_velocity = rtde_r.getActualTCPSpeed()
                actual_velocity = [0 if abs(x) < 0.01 else x for x in actual_velocity] #filter out extremely small numbers
                print("Current velocity vector: " , actual_velocity)

                #get TCP pose of robot
                #actual_pose = rtde_r.getActualTCPPose()
                #print(actual_pose)

                #get joint pose of robot 
                # joint_pose = rtde_r.getActualQ()
                # print("Current Joint Pose:", joint_pose)

  
                if sm.is_button_pressed(0):
                    gripper_position += 3
                    gripper.move(gripper_position, 155, 255)

                if sm.is_button_pressed(1):
                    gripper_position -= 3
                    gripper.move(gripper_position, 155, 255)

                # Detect button combination for Force Mode toggle
                if sm.is_button_pressed(0) and sm.is_button_pressed(1):  # Both buttons pressed together
                    current_time = time.time()
                    # Long press check (button hold for 1 second)
                    if current_time - last_button_press_time > 1.0:
                        print("Toggling Force Mode...")
                        if force_mode_active:
                            exit_force_mode(rtde_c)
                        else:
                            enter_force_mode(rtde_c, rtde_r)
                        force_mode_active = not force_mode_active
                        last_button_press_time = current_time

                if gripper_position < gripper_min:
                    gripper_position = gripper_min

                if gripper_position > gripper_max:
                    gripper_position = gripper_max

                print(f"Gripper Position ({gripper_min} to {gripper_max}): {gripper.get_current_position()}")
                
                if gripper.is_gripping(): 
                    print("Gripping object")
                
                else: 
                    print("Not gripping object")

                #wait awhile before proceeding 
                time.sleep(1/100)

            else:
                print("Robot is not ready.")
                time.sleep(1)  # Wait longer if robot is not ready

    except KeyboardInterrupt:
        # Handle graceful shutdown here
        print("Stopping robot")
        rtde_c.stopScript()
        sm.stop()


if __name__ == "__main__":
    # Argument parsing to specify the task type
    parser = argparse.ArgumentParser(description="Robot Task Control")
    parser.add_argument("task", choices=["force_mode_task", "weighing_scale_task"], help="Choose the task to perform.")
    args = parser.parse_args()
    
    main(args.task)
