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

arm.set_servo_angle(angle=[0, 15, -20, 0, -35, 0])

i=0
speed = 10
for i in range(17):
    angle1 = [0, 15, -20, 0, -123, 0]
    angle2 = [0, 15, -20, 0, 123, 0]
    j=0
    for j in range(10):
        arm.set_servo_angle(angle=angle1, speed=speed)
        arm.set_servo_angle(angle=angle2, speed=speed)
        j+=1
    speed+=10
    i+=1