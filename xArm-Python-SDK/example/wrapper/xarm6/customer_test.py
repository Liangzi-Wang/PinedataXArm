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

time.sleep(5)

#move to pick the object

arm.set_position(*[250, 0, 800, 180, -90, 0], wait=True)
arm.set_position(*[500, 0, 800, 180, -90, 0], wait=True)
arm.set_position(*[0, 0, 800, 180, -90, 0], wait=True)


#move to place the object
arm.set_position(*[0, 0, 700, 180, -90, 0], wait=True)
arm.set_position(*[-500, 0, 700, 180, -90, 0], wait=True)
arm.set_position(*[-250, 0, 700, 180, -90, 0], wait=True)
