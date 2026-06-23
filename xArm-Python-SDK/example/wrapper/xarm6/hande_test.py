#!/usr/bin/env python3
"""
Robotiq Hand-E integration for UFactory xArm via RS485 Modbus RTU.

Importable module:
    from hande_test import hande_activate, hande_position, close_hande, open_hande

Slave ID : 0x09  (Robotiq default)
Register : 0x03E8 (ACTION REQUEST, 3 registers)
"""

import time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crc16(data: list) -> int:
    """CRC-16/IBM (Modbus) — returns 16-bit CRC."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc & 0xFFFF


def _send(arm, payload: list):
    """Append Modbus CRC and transmit over RS485."""
    crc = _crc16(payload)
    arm.set_rs485_data(
        datas=payload + [crc & 0xFF, (crc >> 8) & 0xFF],
        protocol='modbus_rtu'
    )


# ---------------------------------------------------------------------------
# Hand-E status
# ---------------------------------------------------------------------------

def hande_is_activated(arm) -> bool:
    """Return True when the Hand-E reports gSTA=3 (activation complete)."""
    status = arm.set_rs485_data(
        datas=[0x09, 0x04, 0x07, 0xD0, 0x00, 0x01, 0x30, 0x0F],
        protocol='modbus_rtu'
    )
    print('activation status:', status)
    if status and isinstance(status[1], list) and len(status[1]) >= 5:
        gSTA = (status[1][3] >> 4) & 0x03
        return gSTA == 3
    return False


def hande_get_status(arm) -> dict:
    """Read 3 status registers; return dict with gOBJ, gSTA, gPO, gCU."""
    resp = arm.set_rs485_data(
        datas=[0x09, 0x04, 0x07, 0xD0, 0x00, 0x03, 0xB1, 0xCE],
        protocol='modbus_rtu'
    )
    if resp and isinstance(resp[1], list) and len(resp[1]) >= 9:
        data = resp[1]
        return {
            'gOBJ': (data[3] >> 6) & 0x03,
            'gSTA': (data[3] >> 4) & 0x03,
            'gPO':   data[7],
            'gCU':   data[8],
        }
    return {}


def hande_is_motion_complete(arm) -> bool:
    """Return True when fingers have stopped (gOBJ != 0)."""
    s = hande_get_status(arm)
    print('hande status:', s)
    return s.get('gOBJ', 0) != 0


# ---------------------------------------------------------------------------
# Hand-E commands
# ---------------------------------------------------------------------------

def hande_activate(arm):
    """Reset then activate the Hand-E; block until activation is confirmed."""
    arm.set_rs485_timeout(1000)
    arm.set_rs485_baudrate(baud=115200, target='robot')

    _send(arm, [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    time.sleep(0.5)

    _send(arm, [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00])

    print('Waiting for Hand-E activation...')
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if hande_is_activated(arm):
            print('Hand-E activated.')
            return
        time.sleep(0.2)
    print('Hand-E activation timed out — continuing anyway')


def hande_position(arm, position: int, speed: int = 255, force: int = 150):
    """
    Send a position command to the Hand-E.

    position : 0 = fully open, 255 = fully closed
    speed    : 0–255
    force    : 0–255
    """
    position = max(0, min(255, position))
    speed    = max(0, min(255, speed))
    force    = max(0, min(255, force))
    _send(arm, [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                0x09, 0x00, 0x00, position, speed, force])


def close_hande(arm, speed: int = 100, force: int = 100, wait: bool = True):
    """Close the Hand-E. speed/force: 0–100 (mapped to 0–255)."""
    hande_position(arm, 255, round(speed * 2.55), round(force * 2.55))
    if wait:
        while not hande_is_motion_complete(arm):
            time.sleep(0.1)
        print('Close complete.')


def open_hande(arm, speed: int = 100, force: int = 100, wait: bool = True):
    """Open the Hand-E. speed/force: 0–100 (mapped to 0–255)."""
    hande_position(arm, 0, round(speed * 2.55), round(force * 2.55))
    if wait:
        while not hande_is_motion_complete(arm):
            time.sleep(0.1)
        print('Open complete.')


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    import sys

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
        except Exception:
            ip = input('Please input the xArm ip address: ')
            if not ip:
                print('input error, exit')
                sys.exit(1)

    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    hande_activate(arm)
    close_hande(arm, speed=10, force=10)
    time.sleep(2)
    open_hande(arm, speed=50, force=25)
    close_hande(arm, speed=10, force=10)
