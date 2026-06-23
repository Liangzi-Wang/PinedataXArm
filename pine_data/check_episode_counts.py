#!/usr/bin/env python3
"""
统计当前 recordings 管线下各任务的 episode 数量。

当前统计基于：
recordings/YYYYMMDD/<instruction>/camera_npy/YYYYMMDDHHMMSS/

输出：
- 每个 instruction 的总 episode 数
- 其中包含机器人状态的 episode 数
- 今日新增的 episode 数
- 今日新增且包含机器人状态的 episode 数
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def _is_date_dir(name: str) -> bool:
    return len(name) == 8 and name.isdigit()


def _is_episode_dir(path: Path) -> bool:
    return path.is_dir() and len(path.name) == 14 and path.name.isdigit()


@dataclass
class Stat:
    episodes: int = 0
    robot_episodes: int = 0


def _episode_dirs(camera_root: Path) -> list[Path]:
    if not camera_root.is_dir():
        return []
    return [path for path in sorted(camera_root.iterdir()) if _is_episode_dir(path)]


def _has_robot_stream(episode_dir: Path) -> bool:
    return (episode_dir / "timestamps_robot.npy").is_file()


def _format_row(cols: list[str], widths: list[int]) -> str:
    return " | ".join(value.ljust(width) for value, width in zip(cols, widths))


def main() -> int:
    parser = argparse.ArgumentParser(description="统计当前 recordings 管线下各 instruction 的 episode 数量")
    parser.add_argument("--root", default="./recordings", help="recordings 根目录，默认: ./recordings")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"路径不存在: {root}")
        return 1

    today = datetime.now().strftime("%Y%m%d")
    total_stats: dict[str, Stat] = defaultdict(Stat)
    today_stats: dict[str, Stat] = defaultdict(Stat)

    date_dirs = [path for path in sorted(root.iterdir()) if path.is_dir() and _is_date_dir(path.name)]
    if not date_dirs:
        print(f"未找到日期目录（YYYYMMDD）: {root}")
        return 1

    for date_dir in date_dirs:
        for instruction_dir in sorted(path for path in date_dir.iterdir() if path.is_dir()):
            instruction = instruction_dir.name
            episodes = _episode_dirs(instruction_dir / "camera_npy")
            if not episodes:
                continue

            total_stats[instruction].episodes += len(episodes)
            total_stats[instruction].robot_episodes += sum(1 for episode in episodes if _has_robot_stream(episode))

            if date_dir.name == today:
                today_stats[instruction].episodes += len(episodes)
                today_stats[instruction].robot_episodes += sum(1 for episode in episodes if _has_robot_stream(episode))

    if not total_stats:
        print(f"未发现有效 episode: {root}")
        return 1

    headers = [
        "instruction",
        "total_episode",
        "total_robot_episode",
        "today_episode",
        "today_robot_episode",
    ]

    rows: list[list[str]] = []
    for instruction, stat in sorted(
        total_stats.items(),
        key=lambda item: (item[1].episodes, item[1].robot_episodes, item[0]),
        reverse=True,
    ):
        today_stat = today_stats.get(instruction, Stat())
        rows.append([
            instruction,
            str(stat.episodes),
            str(stat.robot_episodes),
            str(today_stat.episodes),
            str(today_stat.robot_episodes),
        ])

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    print(f"root: {root}")
    print(f"today: {today}")
    print(_format_row(headers, widths))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(_format_row(row, widths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
