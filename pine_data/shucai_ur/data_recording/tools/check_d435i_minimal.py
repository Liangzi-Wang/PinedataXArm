#!/usr/bin/env python3
"""Minimal pyrealsense2 check script for Intel RealSense D435i."""

import argparse
import sys
import time

import numpy as np
import pyrealsense2 as rs


DEFAULT_D435I_PIDS = {"0B3A"}


def list_devices():
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        try:
            devices.append(
                {
                    "name": dev.get_info(rs.camera_info.name),
                    "serial": dev.get_info(rs.camera_info.serial_number),
                    "pid": dev.get_info(rs.camera_info.product_id).upper(),
                }
            )
        except Exception:
            continue
    return devices


def pick_d435i_serial(devices, serial_override=None):
    if serial_override:
        for d in devices:
            if d["serial"] == serial_override:
                return serial_override
        return None

    for d in devices:
        if d["pid"] in DEFAULT_D435I_PIDS:
            return d["serial"]
    return None


def main():
    parser = argparse.ArgumentParser(description="Minimal D435i pyrealsense2 check")
    parser.add_argument("--serial", default=None, help="Camera serial to open")
    parser.add_argument("--fps", type=int, default=15, help="Capture fps")
    parser.add_argument("--frames", type=int, default=30, help="Frames to capture")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    args = parser.parse_args()

    devices = list_devices()
    if not devices:
        print("No RealSense devices found.")
        return 1

    print("Detected RealSense devices:")
    for d in devices:
        print(f"  - {d['name']} | serial={d['serial']} | pid={d['pid']}")

    serial = pick_d435i_serial(devices, serial_override=args.serial)
    if serial is None:
        if args.serial:
            print(f"Requested serial not found: {args.serial}")
        else:
            print("No D435i found automatically. Use --serial to force one.")
        return 2

    print(f"Using camera serial: {serial}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    try:
        pipeline.start(config)
        print("Pipeline started. Warming up...")
        for _ in range(10):
            pipeline.wait_for_frames()

        ok = 0
        start = time.time()
        for i in range(args.frames):
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                print(f"Frame {i + 1}: invalid")
                continue

            color_np = np.asanyarray(color.get_data())
            depth_np = np.asanyarray(depth.get_data())
            ok += 1
            print(
                f"Frame {i + 1}/{args.frames}: "
                f"color={color_np.shape}, depth={depth_np.shape}, "
                f"depth_mean={float(depth_np.mean()):.2f}"
            )

        elapsed = max(1e-6, time.time() - start)
        print(f"Done. valid_frames={ok}/{args.frames}, observed_fps={ok / elapsed:.2f}")
        return 0 if ok > 0 else 3

    except Exception as exc:
        print(f"Capture failed: {exc}")
        return 4
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
