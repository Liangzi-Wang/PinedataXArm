#!/usr/bin/env python3
"""
检查当前 recordings 管线下各 episode 内部的时间戳流是否对齐。

当前脚本检查：
- timestamps_hand.npy
- timestamps_external.npy
- timestamps_robot.npy

目录约定：
recordings/YYYYMMDD/<instruction>/camera_npy/YYYYMMDDHHMMSS/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


STREAM_FILES = {
    "hand": "timestamps_hand.npy",
    "external": "timestamps_external.npy",
    "robot": "timestamps_robot.npy",
}


@dataclass(frozen=True)
class Bounds:
    name: str
    path: Path
    count: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _is_timestamp_dir(name: str) -> bool:
    return len(name) == 14 and name.isdigit()


def _iter_episode_dirs(root: Path) -> list[Path]:
    episodes: list[Path] = []
    for camera_dir in root.rglob("camera_npy"):
        if not camera_dir.is_dir():
            continue
        for child in sorted(camera_dir.iterdir()):
            if child.is_dir() and _is_timestamp_dir(child.name):
                episodes.append(child)
    return sorted(set(episodes))


def _load_bounds(path: Path, name: str) -> Bounds | None:
    if not path.is_file():
        return None
    try:
        timestamps = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
    except Exception as exc:
        print(f"读取失败: {path} ({exc})")
        return None
    if timestamps.ndim != 1 or timestamps.size == 0:
        return None
    return Bounds(
        name=name,
        path=path,
        count=int(timestamps.size),
        start=float(timestamps[0]),
        end=float(timestamps[-1]),
    )


def _load_episode_streams(episode_dir: Path) -> dict[str, Bounds]:
    streams: dict[str, Bounds] = {}
    for name, filename in STREAM_FILES.items():
        bounds = _load_bounds(episode_dir / filename, name)
        if bounds is not None:
            streams[name] = bounds
    return streams


def _pairwise(items: Iterable[Bounds]) -> Iterable[tuple[Bounds, Bounds]]:
    ordered = list(items)
    for idx, left in enumerate(ordered):
        for right in ordered[idx + 1 :]:
            yield left, right


def _compare_streams(left: Bounds, right: Bounds, threshold: float) -> dict[str, float | bool | str]:
    start_diff = abs(left.start - right.start)
    end_diff = abs(left.end - right.end)
    overlap_s = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    aligned = overlap_s > 0.0 and start_diff <= threshold and end_diff <= threshold
    return {
        "pair": f"{left.name}/{right.name}",
        "aligned": aligned,
        "start_diff": start_diff,
        "end_diff": end_diff,
        "overlap_s": overlap_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="检查当前 recordings 管线下各 episode 内部时间戳是否对齐")
    parser.add_argument("--root", default="./recordings", help="recordings 根目录，默认: ./recordings")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="判定开始/结束时间差是否可接受的阈值（秒），默认 1.0",
    )
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="默认仅显示异常 episode；加上此参数可显示全部 episode",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return 1

    episodes = _iter_episode_dirs(root)
    if not episodes:
        print(f"未找到 episode 目录: {root}")
        return 1

    ok_count = 0
    warn_count = 0
    robot_count = 0
    single_camera_count = 0

    print(f"root: {root}")
    print(f"episodes: {len(episodes)}")
    print(f"threshold: {args.threshold:.3f}s")

    for episode_dir in episodes:
        streams = _load_episode_streams(episode_dir)
        if not streams:
            warn_count += 1
            print(f"\n[WARN] {episode_dir}: 没有找到任何时间戳文件")
            continue

        has_robot = "robot" in streams
        has_hand = "hand" in streams
        has_external = "external" in streams
        if has_robot:
            robot_count += 1
        if int(has_hand) + int(has_external) <= 1:
            single_camera_count += 1

        comparisons = [_compare_streams(left, right, args.threshold) for left, right in _pairwise(streams.values())]
        aligned = all(item["aligned"] for item in comparisons) if comparisons else True

        if aligned:
            ok_count += 1
        else:
            warn_count += 1

        if aligned and not args.show_ok:
            continue

        rel_episode = episode_dir.relative_to(root)
        status = "OK" if aligned else "WARN"
        print(f"\n[{status}] {rel_episode}")
        for name in ("hand", "external", "robot"):
            bounds = streams.get(name)
            if bounds is None:
                continue
            print(
                f"  - {name:<8} count={bounds.count:<6} "
                f"start={bounds.start:.6f} end={bounds.end:.6f} duration={bounds.duration:.3f}s"
            )

        if not comparisons:
            print("  - 仅存在一个时间戳流，无法进行对齐比较。")
            continue

        for item in comparisons:
            print(
                f"  - {item['pair']:<13} aligned={str(item['aligned']):<5} "
                f"Δstart={item['start_diff']:.3f}s Δend={item['end_diff']:.3f}s overlap={item['overlap_s']:.3f}s"
            )

    print("\nSummary")
    print(f"  - OK episodes: {ok_count}")
    print(f"  - WARN episodes: {warn_count}")
    print(f"  - Episodes with robot stream: {robot_count}")
    print(f"  - Episodes with <=1 camera stream: {single_camera_count}")
    return 0 if warn_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
