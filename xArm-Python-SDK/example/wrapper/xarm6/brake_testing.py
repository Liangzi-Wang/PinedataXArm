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
arm.move_gohome(wait=True)

x=375
y =0
z=325
r=180
p=0
yaw=0

arm.set_position(x, y, z, r, p, yaw, wait=True)
print("pos: ", arm.get_position())

time.sleep(1)
arm.motion_enable(False)
time.sleep(1)
arm.motion_enable(True)
print("pos after stop: ", arm.get_position())