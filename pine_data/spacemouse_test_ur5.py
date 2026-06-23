""" Test the UR5 teleoperation with Festo gripper. """
import time
import numpy as np
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface as RTDEIO
import sys
sys.path.append('/home/pine/pine_data/spacemouse_teleoperation')
from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from collections import defaultdict
from threading import Thread, Event


class Spacemouse(Thread):
    """Minimal SpaceMouse class for testing"""
    def __init__(self, max_value=500, deadzone=0.0, dtype=np.float32):
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

    def get_motion_state_transformed(self):
        state = self.get_motion_state()
        state[3] = 0  # Freeze pitch
        state[5] = 0  # Freeze yaw
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        tf_state[np.abs(tf_state) < 0.3] = 0
        return tf_state

    def get_motion_state(self):
        me = self.motion_event
        state = np.array(me.translation + me.rotation, dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def is_button_pressed(self, button_id):
        return self.button_state[button_id]

    def stop(self):
        self.stop_event.set()
        self.join()

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


def test_spacemouse_only():
    """Test SpaceMouse output only (no robot connection)"""
    print("\n" + "="*60)
    print("SPACEMOUSE TEST - Device Only")
    print("="*60)
    print("This test reads SpaceMouse input without connecting to robot")
    print("Move the SpaceMouse and press buttons to see the output")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    sm = Spacemouse()
    sm.start()
    
    try:
        with np.printoptions(precision=3, suppress=True):
            while True:
                motion = sm.get_motion_state_transformed()
                b0 = sm.is_button_pressed(0)
                b1 = sm.is_button_pressed(1)
                print(f"Motion: {motion} | Buttons: [0]={b0} [1]={b1}")
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        sm.stop()
        print("Test completed!")


def test_robot_connection():
    """Test robot connection and state reading"""
    ROBOT_HOST = "192.168.201.101"
    
    print("\n" + "="*60)
    print("ROBOT CONNECTION TEST")
    print("="*60)
    print(f"Connecting to robot at {ROBOT_HOST}...")
    print("="*60 + "\n")
    
    try:
        rtde_r = RTDEReceiveInterface(ROBOT_HOST)
        rtde_io = RTDEIO(ROBOT_HOST)
        
        print("✓ Connection successful!\n")
        
        for i in range(10):
            tcp_pose = rtde_r.getActualTCPPose()
            joint_pose = rtde_r.getActualQ()
            force = rtde_r.getActualTCPForce()
            robot_mode = rtde_r.getRobotMode()
            
            print(f"Frame {i+1}:")
            print(f"  TCP Pose: {np.round(tcp_pose, 3)}")
            print(f"  Joints: {np.round(joint_pose, 3)}")
            print(f"  Force: {np.round(force, 2)}")
            print(f"  Robot Mode: {robot_mode} {'(RUNNING)' if robot_mode == 7 else '(NOT READY)'}")
            print()
            time.sleep(0.5)
        
        print("✓ Robot connection test completed!")
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")


def test_festo_gripper():
    """Test Festo gripper control"""
    ROBOT_HOST = "192.168.201.101"
    FESTO_OPEN_DO = 0
    FESTO_CLOSE_DO = 2
    
    print("\n" + "="*60)
    print("FESTO GRIPPER TEST")
    print("="*60)
    print(f"Connecting to robot at {ROBOT_HOST}...")
    print("="*60 + "\n")
    
    try:
        rtde_io = RTDEIO(ROBOT_HOST)
        print("✓ Connection successful!\n")
        
        print("Testing gripper open/close cycle...")
        
        # Open
        print("1. Opening gripper...")
        rtde_io.setStandardDigitalOut(FESTO_OPEN_DO, True)
        rtde_io.setStandardDigitalOut(FESTO_CLOSE_DO, False)
        time.sleep(2)
        
        # Close
        print("2. Closing gripper...")
        rtde_io.setStandardDigitalOut(FESTO_OPEN_DO, False)
        rtde_io.setStandardDigitalOut(FESTO_CLOSE_DO, True)
        time.sleep(2)
        
        # Open again
        print("3. Opening gripper...")
        rtde_io.setStandardDigitalOut(FESTO_OPEN_DO, True)
        rtde_io.setStandardDigitalOut(FESTO_CLOSE_DO, False)
        time.sleep(2)
        
        print("\n✓ Gripper test completed!")
        
    except Exception as e:
        print(f"✗ Gripper test failed: {e}")


def test_integrated():
    """Test SpaceMouse + Robot + Gripper together"""
    ROBOT_HOST = "192.168.201.101"
    FESTO_OPEN_DO = 0
    FESTO_CLOSE_DO = 2
    
    print("\n" + "="*60)
    print("INTEGRATED TEST - SpaceMouse + Robot + Gripper")
    print("="*60)
    print("This test reads all inputs without sending motion commands")
    print("Move SpaceMouse and press buttons to test")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        # Initialize
        sm = Spacemouse()
        sm.start()
        rtde_r = RTDEReceiveInterface(ROBOT_HOST)
        rtde_io = RTDEIO(ROBOT_HOST)
        
        print("✓ All systems connected!\n")
        
        prev_b0 = False
        prev_b1 = False
        
        with np.printoptions(precision=3, suppress=True):
            while True:
                # Read SpaceMouse
                motion = sm.get_motion_state_transformed()
                b0 = sm.is_button_pressed(0)
                b1 = sm.is_button_pressed(1)
                
                # Read Robot
                tcp_pose = rtde_r.getActualTCPPose()
                force = rtde_r.getActualTCPForce()
                
                # Button edge detection
                if b0 and not prev_b0:
                    print(">>> Opening gripper")
                    rtde_io.setStandardDigitalOut(FESTO_OPEN_DO, True)
                    rtde_io.setStandardDigitalOut(FESTO_CLOSE_DO, False)
                if b1 and not prev_b1:
                    print(">>> Closing gripper")
                    rtde_io.setStandardDigitalOut(FESTO_OPEN_DO, False)
                    rtde_io.setStandardDigitalOut(FESTO_CLOSE_DO, True)
                
                prev_b0, prev_b1 = b0, b1
                
                # Display
                print(f"Motion: {motion}")
                print(f"TCP: {np.round(tcp_pose, 3)} | Force: {np.round(force, 2)}")
                print(f"Buttons: [0]={b0} [1]={b1}")
                print("-" * 60)
                
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nStopping...")
        sm.stop()
        print("✓ Integrated test completed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")


def main():
    """Interactive test menu"""
    print("\n" + "="*60)
    print("UR5 TELEOPERATION TEST SUITE")
    print("="*60)
    print("\nSelect a test to run:")
    print("  1. SpaceMouse only (no robot connection)")
    print("  2. Robot connection test")
    print("  3. Festo gripper test")
    print("  4. Integrated test (all systems)")
    print("  5. Exit")
    print("="*60)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        test_spacemouse_only()
    elif choice == '2':
        test_robot_connection()
    elif choice == '3':
        test_festo_gripper()
    elif choice == '4':
        test_integrated()
    elif choice == '5':
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")


if __name__ == "__main__":
    main()