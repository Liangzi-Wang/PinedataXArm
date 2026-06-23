#!/usr/bin/env python3
"""Lossless episode vision compressor/decompressor.

Compress:
  - rgb_*.npy    -> .mkv via FFV1 (lossless)
  - depth_*.npy  -> raw .mkv via FFV1 + gray16le (lossless)

Decompress:
  - restore .npy files from the encoded files in the same episode folder

This utility stores reconstruction metadata in metadata.json under:
  - vision_storage
  - vision_files
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


RGB_CODEC = "ffv1"
DEPTH_CODEC = "ffv1"
WEBAPP_RGB_CHANNEL_ORDER = "rgb"

RGB_JOBS = (
    ("rgb_hand", "rgb_hand.npy", "rgb_hand.mkv"),
    ("rgb_external", "rgb_external.npy", "rgb_external.mkv"),
)
DEPTH_JOBS = (
    ("depth_hand", "depth_hand.npy", "depth_hand_raw.mkv"),
    ("depth_external", "depth_external.npy", "depth_external_raw.mkv"),
)


def require_ffmpeg(binary_name: str) -> str:
    binary_path = shutil.which(binary_name)
    if binary_path:
        return binary_path
    raise RuntimeError(
        f"Required executable '{binary_name}' was not found in PATH. "
        "Install ffmpeg or pass --ffmpeg-bin with the correct path."
    )


def load_metadata(episode_dir: Path) -> Dict:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_metadata(episode_dir: Path, metadata: Dict) -> Path:
    metadata_path = episode_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def infer_fps(episode_dir: Path, metadata: Dict) -> int:
    frequency = metadata.get("frequency")
    if isinstance(frequency, int) and frequency > 0:
        return frequency
    if isinstance(frequency, float) and frequency > 0:
        return max(1, int(round(frequency)))

    for ts_name in ("timestamps_hand.npy", "timestamps_external.npy"):
        ts_path = episode_dir / ts_name
        if not ts_path.exists():
            continue
        timestamps = np.load(ts_path)
        if timestamps.size < 2:
            continue
        diffs = np.diff(timestamps.astype(np.float64))
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            continue
        return max(1, int(round(1.0 / float(np.median(diffs)))))

    return 15


def vision_entry(
    *,
    path: str,
    storage: str,
    codec: str | None,
    shape: Sequence[int],
    dtype: str,
    fps: int,
    verified_lossless: bool,
    channel_order: str | None = None,
) -> Dict:
    entry = {
        "path": path,
        "storage": storage,
        "codec": codec,
        "shape": [int(dim) for dim in shape],
        "dtype": dtype,
        "fps": int(fps),
        "verified_lossless": bool(verified_lossless),
    }
    if channel_order is not None:
        entry["channel_order"] = channel_order.upper()
        entry["channel_convention"] = "webapp_rgb_no_swap"
    return entry


def normalize_depth_frame(frame: np.ndarray) -> np.ndarray:
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return np.zeros(frame.shape, dtype=np.uint8)

    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.uint8)

    scaled = np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def run_ffmpeg_encode(
    *,
    ffmpeg_bin: str,
    frames: np.ndarray,
    output_path: Path,
    fps: int,
    input_pix_fmt: str,
    codec: str,
    extra_args: Sequence[str],
) -> None:
    frame_array = np.ascontiguousarray(frames)
    if frame_array.ndim < 3:
        raise ValueError(f"Expected frame array with at least 3 dimensions, got {frame_array.shape}")

    height = int(frame_array.shape[1])
    width = int(frame_array.shape[2])
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        input_pix_fmt,
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
        *extra_args,
        str(output_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        for frame in frame_array:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
        return_code = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise

    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed while writing {output_path.name}: {stderr_text.strip() or 'unknown ffmpeg error'}"
        )


def read_exact_stdout(stream, expected_bytes: int) -> bytes:
    chunks = []
    total = 0
    while total < expected_bytes:
        chunk = stream.read(min(1024 * 1024, expected_bytes - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total != expected_bytes:
        raise RuntimeError(f"Expected {expected_bytes} bytes from ffmpeg, received {total} bytes.")
    return b"".join(chunks)


def decode_video_to_array(
    *,
    ffmpeg_bin: str,
    input_path: Path,
    shape: Sequence[int],
    output_pix_fmt: str,
    dtype: np.dtype,
) -> np.ndarray:
    frame_count = int(shape[0])
    height = int(shape[1])
    width = int(shape[2])
    channels = int(shape[3]) if len(shape) == 4 else 1
    bytes_per_value = np.dtype(dtype).itemsize
    expected_bytes = frame_count * height * width * channels * bytes_per_value

    cmd = [
        ffmpeg_bin,
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-frames:v",
        str(frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        output_pix_fmt,
        "-vsync",
        "0",
        "-",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdout is not None
        raw_bytes = read_exact_stdout(proc.stdout, expected_bytes)
        stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
        return_code = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise

    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed while reading {input_path.name}: {stderr_text.strip() or 'unknown ffmpeg error'}"
        )

    array = np.frombuffer(raw_bytes, dtype=dtype)
    return array.reshape(tuple(int(dim) for dim in shape)).copy()


def assert_lossless_roundtrip(original: np.ndarray, decoded: np.ndarray, label: str) -> None:
    if original.shape != decoded.shape:
        raise RuntimeError(f"{label}: shape mismatch {original.shape} vs {decoded.shape}")
    if original.dtype != decoded.dtype:
        raise RuntimeError(f"{label}: dtype mismatch {original.dtype} vs {decoded.dtype}")
    if not np.array_equal(original, decoded):
        diff = original.astype(np.int64) - decoded.astype(np.int64)
        max_abs_diff = int(np.max(np.abs(diff)))
        raise RuntimeError(f"{label}: decoded data differs from source (max abs diff={max_abs_diff})")


def encode_rgb(
    frames: np.ndarray,
    output_path: Path,
    fps: int,
    ffmpeg_bin: str,
    verify: bool,
    channel_order: str,
) -> Dict:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"RGB frames must be uint8 with shape (N,H,W,3), got {frames.dtype} {frames.shape}")
    input_pix_fmt = "rgb24" if channel_order.lower() == "rgb" else "bgr24"

    run_ffmpeg_encode(
        ffmpeg_bin=ffmpeg_bin,
        frames=frames,
        output_path=output_path,
        fps=fps,
        input_pix_fmt=input_pix_fmt,
        codec=RGB_CODEC,
        extra_args=["-pix_fmt", input_pix_fmt],
    )

    if verify:
        decoded = decode_video_to_array(
            ffmpeg_bin=ffmpeg_bin,
            input_path=output_path,
            shape=frames.shape,
            output_pix_fmt=input_pix_fmt,
            dtype=np.uint8,
        )
        assert_lossless_roundtrip(frames, decoded, output_path.name)

    return vision_entry(
        path=output_path.name,
        storage="mkv",
        codec=RGB_CODEC,
        shape=frames.shape,
        dtype=str(frames.dtype),
        fps=fps,
        verified_lossless=verify,
        channel_order=channel_order,
    )


def encode_depth(frames: np.ndarray, output_path: Path, fps: int, ffmpeg_bin: str, verify: bool) -> Dict:
    if frames.dtype != np.uint16 or frames.ndim != 3:
        raise ValueError(f"Depth frames must be uint16 with shape (N,H,W), got {frames.dtype} {frames.shape}")

    run_ffmpeg_encode(
        ffmpeg_bin=ffmpeg_bin,
        frames=frames,
        output_path=output_path,
        fps=fps,
        input_pix_fmt="gray16le",
        codec=DEPTH_CODEC,
        extra_args=["-pix_fmt", "gray16le"],
    )

    if verify:
        decoded = decode_video_to_array(
            ffmpeg_bin=ffmpeg_bin,
            input_path=output_path,
            shape=frames.shape,
            output_pix_fmt="gray16le",
            dtype=np.uint16,
        )
        assert_lossless_roundtrip(frames, decoded, output_path.name)

    return vision_entry(
        path=output_path.name,
        storage="mkv",
        codec=DEPTH_CODEC,
        shape=frames.shape,
        dtype=str(frames.dtype),
        fps=fps,
        verified_lossless=verify,
    )


def compress_episode(
    episode_dir: Path,
    *,
    ffmpeg_bin: str,
    delete_raw: bool,
    verify: bool,
    rgb_channel_order: str,
) -> None:
    ffmpeg_path = require_ffmpeg(ffmpeg_bin)
    metadata = load_metadata(episode_dir)
    fps = infer_fps(episode_dir, metadata)
    vision_files = dict(metadata.get("vision_files") or {})
    wrote_any = False

    for key, npy_name, video_name in RGB_JOBS:
        npy_path = episode_dir / npy_name
        if not npy_path.exists():
            continue
        frames = np.load(npy_path)
        output_path = episode_dir / video_name
        print(f"Compressing {npy_name} -> {video_name}")
        vision_files[key] = encode_rgb(
            frames,
            output_path,
            fps,
            ffmpeg_path,
            verify,
            rgb_channel_order,
        )
        wrote_any = True
        if delete_raw:
            npy_path.unlink()

    for key, npy_name, raw_video_name in DEPTH_JOBS:
        npy_path = episode_dir / npy_name
        if not npy_path.exists():
            continue
        frames = np.load(npy_path)
        raw_output_path = episode_dir / raw_video_name
        print(f"Compressing {npy_name} -> {raw_video_name}")
        entry = encode_depth(frames, raw_output_path, fps, ffmpeg_path, verify)
        vision_files[key] = entry
        wrote_any = True
        if delete_raw:
            npy_path.unlink()

    if not wrote_any:
        raise RuntimeError(f"No rgb/depth npy files found under {episode_dir}")

    metadata["vision_storage"] = "lossless-video"
    metadata["vision_files"] = vision_files
    metadata["vision_codec_tool"] = "webapp/episode_codec.py"
    save_metadata(episode_dir, metadata)
    print(f"Updated metadata: {episode_dir / 'metadata.json'}")


def decode_entry(episode_dir: Path, entry: Dict, output_path: Path, ffmpeg_bin: str) -> None:
    shape = entry.get("shape")
    dtype_name = entry.get("dtype")
    if not shape or not dtype_name:
        raise RuntimeError(f"Cannot decode {output_path.name}: missing shape/dtype metadata.")

    input_path = episode_dir / str(entry.get("path", ""))
    if not input_path.exists():
        raise RuntimeError(f"Cannot decode {output_path.name}: source file missing: {input_path}")

    if entry.get("storage") == "mp4":
        channel_order = str(entry.get("channel_order", "RGB")).lower()
        output_pix_fmt = "rgb24" if channel_order == "rgb" else "bgr24"
        array = decode_video_to_array(
            ffmpeg_bin=ffmpeg_bin,
            input_path=input_path,
            shape=shape,
            output_pix_fmt=output_pix_fmt,
            dtype=np.dtype(dtype_name),
        )
    elif entry.get("storage") == "mkv":
        if len(shape) == 4 and int(shape[3]) in {3, 4}:
            channel_order = str(entry.get("channel_order", "RGB")).lower()
            output_pix_fmt = "rgb24" if channel_order == "rgb" else "bgr24"
            array = decode_video_to_array(
                ffmpeg_bin=ffmpeg_bin,
                input_path=input_path,
                shape=shape,
                output_pix_fmt=output_pix_fmt,
                dtype=np.dtype(dtype_name),
            )
        else:
            array = decode_video_to_array(
                ffmpeg_bin=ffmpeg_bin,
                input_path=input_path,
                shape=shape,
                output_pix_fmt="gray16le",
                dtype=np.dtype(dtype_name),
            )
    else:
        raise RuntimeError(f"Unsupported storage '{entry.get('storage')}' for {output_path.name}")

    np.save(output_path, array)


def decompress_episode(episode_dir: Path, *, ffmpeg_bin: str) -> None:
    ffmpeg_path = require_ffmpeg(ffmpeg_bin)
    metadata = load_metadata(episode_dir)
    vision_files = metadata.get("vision_files") or {}
    if not vision_files:
        raise RuntimeError(
            f"No vision_files metadata found in {episode_dir / 'metadata.json'}. "
            "This episode does not look like a compressed episode."
        )

    decode_jobs = (
        ("rgb_hand", "rgb_hand.npy"),
        ("rgb_external", "rgb_external.npy"),
        ("depth_hand", "depth_hand.npy"),
        ("depth_external", "depth_external.npy"),
    )
    decoded_any = False
    for key, npy_name in decode_jobs:
        entry = vision_files.get(key)
        if not entry:
            continue
        print(f"Decompressing {entry.get('path', key)} -> {npy_name}")
        decode_entry(episode_dir, entry, episode_dir / npy_name, ffmpeg_path)
        decoded_any = True

    if not decoded_any:
        raise RuntimeError(f"No encoded vision files found to decode under {episode_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress/decompress one episode folder losslessly in place."
    )
    parser.add_argument(
        "mode",
        choices=("compress", "decompress"),
        help="compress raw npy vision files, or decompress encoded files back to npy",
    )
    parser.add_argument(
        "episode_folder",
        help="Episode folder containing metadata.json and rgb/depth files.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable name or absolute path.",
    )
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete raw rgb/depth npy after verified compression.",
    )
    parser.add_argument(
        "--rgb-channel-order",
        choices=("rgb", "bgr"),
        default=WEBAPP_RGB_CHANNEL_ORDER,
        help=(
            "Channel order of rgb_*.npy arrays before compression. "
            "Webapp replay uses Image.fromarray(frame) with no channel swap, "
            "so episodes from record_multi_camera_npy.py / the webapp use 'rgb'."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip decode-back verification after compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episode_dir = Path(args.episode_folder).expanduser().resolve()
    if not episode_dir.is_dir():
        raise RuntimeError(f"Episode folder does not exist: {episode_dir}")

    if args.mode == "compress":
        compress_episode(
            episode_dir,
            ffmpeg_bin=args.ffmpeg_bin,
            delete_raw=args.delete_raw,
            verify=not args.no_verify,
            rgb_channel_order=args.rgb_channel_order,
        )
    else:
        decompress_episode(
            episode_dir,
            ffmpeg_bin=args.ffmpeg_bin,
        )


if __name__ == "__main__":
    main()
