#!/usr/bin/env python3
"""List subfolders under a given path and print a summary count."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def list_subfolders(root: Path) -> list[Path]:
    """Return all subfolders under root (recursive), excluding root itself."""
    return sorted(path for path in root.rglob("*") if path.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show all subfolders under a given path and print folder count."
    )
    parser.add_argument("path", help="Path to search for subfolders")
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f"Error: path does not exist: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        return 1

    folders = list_subfolders(root)

    for folder in folders:
        print(folder)

    print(f"\nTotal folders: {len(folders)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
