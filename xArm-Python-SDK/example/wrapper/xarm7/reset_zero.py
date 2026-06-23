import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

arm = XArmAPI('192.168.1.210')
time.sleep(0.5)
if arm.warn_code != 0:
    arm.clean_warn()
if arm.error_code != 0:
    arm.clean_error()

def bytes_to_u16(data):
    """big-endian byte sequence"""
    data_u16 = data[0] << 8 | data[1]
    return data_u16
def u16_to_bytes(data):
    """big-endian byte sequence"""
    bts = bytes([data // 256 % 256])
    bts += bytes([data % 256])
    return bts

print(arm.get_gripper_version())

ret = arm.core.gripper_modbus_r16s(0x0105, 1)
# print(ret)
print("gripper io ctrl mode:%d"%bytes_to_u16(ret[5:7]))

ret=arm.core.gripper_modbus_w16s(0x1105,u16_to_bytes(1),1)
print(ret)


ret = arm.core.gripper_modbus_r16s(0x0105, 1)
# print(ret)
print("gripper io ctrl mode:%d"%bytes_to_u16(ret[5:7]))

ret=arm.core.gripper_modbus_w16s(0x1817,u16_to_bytes(1),1)
print(ret)