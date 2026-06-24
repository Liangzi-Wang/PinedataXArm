from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_pinedata_spacemouse_queue_publisher", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SpaceMouse publisher: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_status(status_file: Path, status: dict[str, Any], message: str) -> None:
    payload = {
        "initialized": bool(status.get("connected")),
        "message": message,
        "last_error": str(status.get("last_error") or ""),
        "spacemouse": status,
    }
    status_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_file.with_suffix(status_file.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, status_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the queue publisher and mirror its status for the Pine UI.")
    parser.add_argument("--publisher-script", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--host", default=os.getenv("SPACEMOUSE_QUEUE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SPACEMOUSE_QUEUE_PORT", "8765")))
    parser.add_argument("--authkey", default=os.getenv("SPACEMOUSE_QUEUE_AUTHKEY", "spacemouse"))
    parser.add_argument("--publish-hz", type=float, default=float(os.getenv("SPACEMOUSE_QUEUE_PUBLISH_HZ", "100")))
    parser.add_argument(
        "--status-hz",
        type=float,
        default=float(os.getenv("SPACEMOUSE_STATUS_FILE_HZ", "20")),
    )
    args = parser.parse_args()

    publisher_path = Path(args.publisher_script).expanduser().resolve()
    status_path = Path(args.status_file).expanduser().resolve()
    if not publisher_path.is_file():
        raise RuntimeError(f"SpaceMouse publisher script not found: {publisher_path}")

    module = _load_module(publisher_path)
    original_replace = module.replace_latest_status
    status_period = 1.0 / max(1.0, args.status_hz)
    last_status_write = 0.0
    latest_status: dict[str, Any] = {}

    def replace_and_publish(status: dict[str, Any]) -> None:
        nonlocal last_status_write, latest_status
        latest_status = dict(status)
        original_replace(status)
        now = time.monotonic()
        if now - last_status_write >= status_period:
            _write_status(status_path, latest_status, "Queue SpaceMouse publisher running.")
            last_status_write = now

    def handle_shutdown(_signum, _frame) -> None:
        raise KeyboardInterrupt

    module.replace_latest_status = replace_and_publish
    signal.signal(signal.SIGTERM, handle_shutdown)
    sys.argv = [
        str(publisher_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--authkey",
        args.authkey,
        "--publish-hz",
        str(args.publish_hz),
    ]

    try:
        module.main()
    finally:
        if latest_status:
            stopped_status = dict(latest_status)
            stopped_status.update({
                "connected": False,
                "available": False,
                "active": False,
                "status": "disconnected",
                "motion_state": [0.0] * 6,
            })
        else:
            stopped_status = {
                "requested": True,
                "enabled": True,
                "connected": False,
                "available": False,
                "active": False,
                "stale": True,
                "status": "disconnected",
                "latest_timestamp": time.time(),
                "motion_state": [0.0] * 6,
                "buttons": {"left": False, "right": False},
                "last_error": "",
            }
        _write_status(status_path, stopped_status, "Queue SpaceMouse publisher stopped.")


if __name__ == "__main__":
    main()
