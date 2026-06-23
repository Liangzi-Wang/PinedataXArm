import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI
from configparser import ConfigParser
parser = ConfigParser()
parser.read('../robot.conf')
try:
    ip = parser.get('xArm', 'ip')
except:
    ip = input('Please input the xArm ip address[192.168.1.194]:')
    if not ip:
        ip = '192.168.1.194'


arm = XArmAPI(ip)
arm.motion_enable(True)
arm.clean_error()
arm.set_mode(0)
arm.set_state(0)
time.sleep(1)

code = arm.set_bio_gripper_enable(True)
print('set_bio_gripper_enable, code={}'.format(code))

code = arm.set_bio_gripper_speed(300)
print('set_bio_gripper_speed, code={}'.format(code))

code = arm.set_bio_gripper_control_mode(mode=1)
print('set_bio_gripper_speed, code={}'.format(code))

code = arm.set_bio_gripper_g2_position(pos=124, speed=850, force=100, wait=True, timeout=3)
print('set_bio_gripper_speed, code={}'.format(code), 'pos 1 reached')

code = arm.set_bio_gripper_g2_position(pos=75, speed=850, force=100, wait=True, timeout=3)
print('set_bio_gripper_speed, code={}'.format(code), 'pos 2 reached')

code = arm.set_bio_gripper_g2_position(pos=150, speed=850, force=100, wait=True, timeout=3)
print('set_bio_gripper_speed, code={}'.format(code), 'pos 3 reached')

arm.robotiq_close()