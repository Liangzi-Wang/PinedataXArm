#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Move Arc Joint
"""

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

speed = 50

arm.set_teach_sensitivity(value=1)
arm.save_conf()
arm.set_mode(mode=2)
arm.set_mode(mode=2)
arm.set_state(state=0)

print("manual mode on sensitivity 1")

time.sleep(10)
print("arm is about to be reset")
time.sleep(5)
arm.reset(wait=False)
print("arm has been reset")

arm.set_teach_sensitivity(value=5)
arm.save_conf()
print("teach sensitivity 5")
arm.set_mode(mode=2)
arm.set_mode(mode=2)
arm.set_state(state=0)
time.sleep(10)