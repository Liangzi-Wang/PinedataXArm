"""Reproduce set_ft_sensor_mode(1) returning 10 during admittance setup."""

import argparse
import time

from xarm import XArmAPI

LINEAR_STIFFNESS = (150, 150, 150)
ANGULAR_STIFFNESS = (100, 100, 100)
MASS = (0.5, 0.5, 0.5)
MOMENT_OF_INERTIA = (0.005, 0.005, 0.005)
LINEAR_THRESHOLD = (10, 10, 10)
ANGULAR_THRESHOLD = (2, 2, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.1.208")
    ip = parser.parse_args().ip

    robot = XArmAPI(
        ip, is_radian=True, enable_report=True, report_type="rich", max_callback_thread_count=-1
    )
    robot.connect()
    if not robot.connected:
        raise ConnectionError(f"Failed to connect to robot at {ip}")

    try:
        robot.set_state(0)
        robot.motion_enable(enable=True)
        robot.set_gripper_mode(0)
        robot.set_gripper_enable(True)
        robot.set_tcp_maxacc(100.0)
        robot.set_joint_maxacc(4 * 3.141592653589793)
        time.sleep(1)
        robot.clean_warn()
        robot.clean_error()
        robot.set_mode(1)
        robot.set_state(state=0)
        time.sleep(0.5)

        try:
            print(f"set_ft_sensor_enable(1) -> {robot.set_ft_sensor_enable(1)}")

            c_axis = [int(bool(s)) for s in (*LINEAR_STIFFNESS, *ANGULAR_STIFFNESS)]
            code = robot.set_ft_sensor_admittance_parameters(
                coord=0,
                c_axis=c_axis,
                M=[*MASS, *MOMENT_OF_INERTIA],
                K=[*LINEAR_STIFFNESS, *ANGULAR_STIFFNESS],
                B=[0.0] * 6,
                params_limit=False,
            )
            print(f"set_ft_sensor_admittance_parameters(...) -> {code}")

            code = robot.set_ft_admittance_ctrl_threshold([*LINEAR_THRESHOLD, *ANGULAR_THRESHOLD])
            print(f"set_ft_admittance_ctrl_threshold(...) -> {code}")

            print(f"  mode={robot.mode} state={robot.state} err={robot.error_code}")
            print(f"set_ft_sensor_mode(1) -> {robot.set_ft_sensor_mode(1)}")
        finally:
            time.sleep(0.5)
            robot.clean_error()
            time.sleep(0.5)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
