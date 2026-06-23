#!/usr/bin/env python3
"""Reset UR3e TCP to a predefined pose via ur-rtde."""

import argparse

from scipy.spatial.transform import Rotation as R


# [x, y, z, qw, qx, qy, qz]
RESET_POSE_QUAT_WXYZ = (
    0.2506,
    -0.2463,
    0.3242,
    0.0060212706,
    0.3638795806,
    -0.9313949679,
    -0.0076686951,
)


def _load_rtde_interfaces():
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
        return RTDEControlInterface, RTDEReceiveInterface
    except ImportError:
        from ur_rtde import rtde_control as _rtde_control
        from ur_rtde import rtde_receive as _rtde_receive

        return _rtde_control.RTDEControlInterface, _rtde_receive.RTDEReceiveInterface


def main() -> None:
    parser = argparse.ArgumentParser(description="Move UR3e to reset TCP pose.")
    parser.add_argument("--host", default="192.168.201.101", help="UR robot IP")
    parser.add_argument("--robot_port", type=int, default=30004, help="RTDE port")
    parser.add_argument("--speed", type=float, default=0.25, help="moveL speed")
    parser.add_argument("--accel", type=float, default=0.5, help="moveL acceleration")
    parser.add_argument("--async_move", action="store_true", help="Use async moveL")
    args = parser.parse_args()

    RTDEControlInterface, RTDEReceiveInterface = _load_rtde_interfaces()
    rtde_c = RTDEControlInterface(args.host, args.robot_port)
    rtde_r = RTDEReceiveInterface(args.host, args.robot_port)

    x, y, z, qw, qx, qy, qz = RESET_POSE_QUAT_WXYZ
    rpy = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)
    rotvec = R.from_euler("xyz", rpy, degrees=False).as_rotvec()
    target_tcp = [float(x), float(y), float(z), float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]

    print("Current TCP:", rtde_r.getActualTCPPose())
    print("Target TCP:", target_tcp)
    rtde_c.moveL(target_tcp, args.speed, args.accel, args.async_move)
    print("Done. Current TCP:", rtde_r.getActualTCPPose())

    if hasattr(rtde_c, "stopScript"):
        rtde_c.stopScript()
    if hasattr(rtde_c, "disconnect"):
        rtde_c.disconnect()
    if hasattr(rtde_r, "disconnect"):
        rtde_r.disconnect()


if __name__ == "__main__":
    main()
