import os
import sys
import time
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

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


arm.clean_error()
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)
time.sleep(.5)

xtarget, ytarget, ztarget = 342.6, 3.8, 86.3

# Lower speed and acceleration reduces end-of-motion snap
arm.set_tcp_jerk(30)          # very smooth jerk
arm.set_tcp_maxacc(150)       # low acceleration

arm.set_position(
    x=xtarget, y=ytarget, z=ztarget,
    roll=180, pitch=-90, yaw=0,
    speed=10, mvacc=150, radius=None,
    wait=False
)
print('resetdone')
time.sleep(2)

ztarget+=15
arm.set_position(
    x=xtarget, y=ytarget, z=ztarget,
    roll=180, pitch=-90, yaw=0,
    speed=10, mvacc=150, radius=None,
    wait=False
)
time.sleep(2)
print('Done1')

ztarget-=15
arm.set_position(
    x=xtarget, y=ytarget, z=ztarget,
    roll=180, pitch=-90, yaw=0,
    speed=10, mvacc=150, radius=None,
    wait=False
)
time.sleep(2)
print('Done2')

ztarget+=15
arm.set_position(
    x=xtarget, y=ytarget, z=ztarget,
    roll=180, pitch=-90, yaw=0,
    speed=10, mvacc=150, radius=None,
    wait=False
)
time.sleep(2)
print('Done3')

arm.set_position(
    x=xtarget, y=ytarget, z=ztarget,
    roll=180, pitch=-90, yaw=0,
    speed=10, mvacc=150, radius=None,
    wait=False
)

time.sleep(5)
print('Done4')
arm.disconnect()