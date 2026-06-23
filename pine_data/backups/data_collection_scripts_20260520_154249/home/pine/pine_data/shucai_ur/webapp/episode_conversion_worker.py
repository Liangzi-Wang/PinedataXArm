#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from episode_codec import compress_episode


CONVERSION_MARKER = ".conversion_in_progress"
CONVERSION_ERROR_FILE = ".conversion_error.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run episode conversion in a detached background process."
    )
    parser.add_argument(
        "mode",
        choices=("compress",),
        help="Conversion mode. Only 'compress' is supported.",
    )
    parser.add_argument(
        "episode_folder",
        help="Episode folder containing metadata.json and rgb/depth npy files.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable name or absolute path.",
    )
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete raw rgb/depth npy files after verified compression.",
    )
    parser.add_argument(
        "--rgb-channel-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="Channel order of rgb_*.npy arrays before compression.",
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
    marker_path = episode_dir / CONVERSION_MARKER
    error_path = episode_dir / CONVERSION_ERROR_FILE

    if not episode_dir.is_dir():
        raise RuntimeError(f"Episode folder does not exist: {episode_dir}")

    print(f"[conversion-worker] start: {episode_dir}")
    try:
        if error_path.exists():
            error_path.unlink()
        compress_episode(
            episode_dir,
            ffmpeg_bin=args.ffmpeg_bin,
            delete_raw=args.delete_raw,
            verify=not args.no_verify,
            rgb_channel_order=args.rgb_channel_order,
        )
        print(f"[conversion-worker] success: {episode_dir}")
    except Exception as exc:
        error_text = "".join(traceback.format_exception(exc))
        try:
            error_path.write_text(error_text, encoding="utf-8")
        except OSError:
            pass
        print(f"[conversion-worker] failed: {episode_dir}")
        print(error_text)
    finally:
        try:
            marker_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
