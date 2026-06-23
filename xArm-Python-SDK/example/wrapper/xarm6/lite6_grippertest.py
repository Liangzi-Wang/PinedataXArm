import os
import sys
import time
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI


#######################################################
"""
Just for test example
"""
if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
    except:
        ip = input('Please input the xArm ip address:')
        if not ip:
            print('input error, exit')
            sys.exit(1)
########################################################


arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

arm.open_lite6_gripper()
print("gripper opened")
time.sleep(1)

print("tgpio digital1: ", arm.get_tgpio_digital(1))
print("tgpio digital0: ", arm.get_tgpio_digital(0))

print("tgpio output digital1: ", arm.get_tgpio_output_digital(1))
print("tgpio output digital0: ", arm.get_tgpio_output_digital(0))
time.sleep(1)

arm.close_lite6_gripper()
print("gripper closed")
time.sleep(1)

print("tgpio digital1: ", arm.get_tgpio_digital(1))
print("tgpio digital0: ", arm.get_tgpio_digital(0))

print("tgpio output digital1: ", arm.get_tgpio_output_digital(1))
print("tgpio output digital0: ", arm.get_tgpio_output_digital(0))
time.sleep(1)