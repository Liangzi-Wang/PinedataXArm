#!/usr/bin/env python3
"""Quick D405 + D455 health check by capturing N frames from each camera."""

import argparse
import sys
import time

import numpy as np
import pyrealsense2 as rs


# In our setup D405 may appear as 0B5B (common) or 0B07.
D405_PIDS = {"0B5B", "0B07"}
D455_PIDS = {"0B5C"}


def list_realsense_devices():
    devices = []
    for dev in rs.context().query_devices():
        try:
            devices.append(
                {
                    "name": dev.get_info(rs.camera_info.name),
                    "serial": dev.get_info(rs.camera_info.serial_number),
                    "pid": dev.get_info(rs.camera_info.product_id),
                }
            )
        except Exception:
            continue
    return devices


def find_serial_by_pid(pid_set, exclude_serial=None):
    for dev in list_realsense_devices():
        if exclude_serial is not None and dev["serial"] == exclude_serial:
            continue
        if dev["pid"] in pid_set:
            return dev["serial"]
    return None


def start_pipeline(serial, width, height, fps):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipe.start(cfg)
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Test D405 and D455 by capturing 10 frames.")
    parser.add_argument("--frames", type=int, default=10, help="Frames to capture from each camera.")
    parser.add_argument("--width", type=int, default=640, help="Stream width.")
    parser.add_argument("--height", type=int, default=480, help="Stream height.")
    parser.add_argument("--fps", type=int, default=15, help="Stream FPS.")
    args = parser.parse_args()

    devices = list_realsense_devices()
    if not devices:
        print("No RealSense devices detected.")
        return 2

    print("Detected RealSense devices:")
    for d in devices:
        print(f"  - {d['name']} (pid={d['pid']}, serial={d['serial']})")

    d405_serial = find_serial_by_pid(D405_PIDS)
    d455_serial = find_serial_by_pid(D455_PIDS, exclude_serial=d405_serial)

    if d405_serial is None:
        print("D405 not found (expected pid in {'0B5B','0B07'}).")
        return 2
    if d455_serial is None:
        print("D455 not found (expected pid in {'0B5C'}).")
        return 2

    print(f"\nUsing D405 serial: {d405_serial}")
    print(f"Using D455 serial: {d455_serial}")

    hand_pipe = None
    ext_pipe = None
    try:
        hand_pipe = start_pipeline(d405_serial, args.width, args.height, args.fps)
        ext_pipe = start_pipeline(d455_serial, args.width, args.height, args.fps)
    except RuntimeError as exc:
        print(f"\nFailed to start camera pipeline: {exc}")
        print("Hint: close realsense-viewer/other scripts that may hold the device.")
        return 1

    hand_ok = 0
    ext_ok = 0

    try:
        # Small warmup.
        for _ in range(5):
            hand_pipe.wait_for_frames()
            ext_pipe.wait_for_frames()

        print(f"\nCapturing {args.frames} frames per camera...")
        t0 = time.time()
        for i in range(args.frames):
            h_frames = hand_pipe.wait_for_frames()
            e_frames = ext_pipe.wait_for_frames()

            h_color = h_frames.get_color_frame()
            h_depth = h_frames.get_depth_frame()
            e_color = e_frames.get_color_frame()
            e_depth = e_frames.get_depth_frame()

            if h_color and h_depth:
                hc = np.asanyarray(h_color.get_data())
                hd = np.asanyarray(h_depth.get_data())
                if hc.size > 0 and hd.size > 0:
                    hand_ok += 1

            if e_color and e_depth:
                ec = np.asanyarray(e_color.get_data())
                ed = np.asanyarray(e_depth.get_data())
                if ec.size > 0 and ed.size > 0:
                    ext_ok += 1

            print(
                f"Frame {i + 1:02d}/{args.frames}: "
                f"D405 ok={hand_ok} | D455 ok={ext_ok}",
                end="\r",
                flush=True,
            )

        dt = time.time() - t0
        print("\n")
        print("Test summary:")
        print(f"  D405 valid frames: {hand_ok}/{args.frames}")
        print(f"  D455 valid frames: {ext_ok}/{args.frames}")
        print(f"  Elapsed: {dt:.2f}s")

        if hand_ok == args.frames and ext_ok == args.frames:
            print("PASS: Both cameras are working.")
            return 0

        print("FAIL: One or both cameras missed frames.")
        return 1

    finally:
        if hand_pipe is not None:
            hand_pipe.stop()
        if ext_pipe is not None:
            ext_pipe.stop()


if __name__ == "__main__":
    sys.exit(main())
