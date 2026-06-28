from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def _select_queue_port(host: str, requested_port: int) -> int:
    if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        return requested_port
    bind_host = "127.0.0.1" if host == "localhost" else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((bind_host, requested_port))
            return requested_port
        except OSError:
            probe.bind((bind_host, 0))
            selected_port = int(probe.getsockname()[1])
    print(
        f"[QueueTeleop] queue port {requested_port} is busy; using {selected_port} instead.",
        flush=True,
    )
    return selected_port


def _write_failure_status(status_file: Path, message: str) -> None:
    payload = {
        "initialized": False,
        "message": f"Queue teleop failed: {message}",
        "last_error": message,
        "spacemouse": {
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
            "last_error": message,
        },
    }
    status_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_file.with_suffix(status_file.suffix + f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, status_file)


def _wait_for_port(host: str, port: int, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SpaceMouse publisher exited with code {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for SpaceMouse queue at {host}:{port}")


def _stop_process(process: subprocess.Popen | None, timeout_s: float = 2.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise the SpaceMouse queue publisher and xArm consumer.")
    parser.add_argument("--publisher-script", required=True)
    parser.add_argument("--teleop-script", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--sdk-path", required=True)
    parser.add_argument("--host", default=os.getenv("SPACEMOUSE_QUEUE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SPACEMOUSE_QUEUE_PORT", "8765")))
    parser.add_argument("--authkey", default=os.getenv("SPACEMOUSE_QUEUE_AUTHKEY", "spacemouse"))
    parser.add_argument("--publish-hz", type=float, default=float(os.getenv("SPACEMOUSE_QUEUE_PUBLISH_HZ", "200")))
    parser.add_argument("--poll-hz", type=float, default=float(os.getenv("XARM_QUEUE_POLL_HZ", "250")))
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    args = parser.parse_args()
    args.port = _select_queue_port(args.host, args.port)

    base_dir = Path(__file__).resolve().parent
    publisher_path = Path(args.publisher_script).expanduser().resolve()
    teleop_path = Path(args.teleop_script).expanduser().resolve()
    sdk_path = Path(args.sdk_path).expanduser().resolve()
    status_path = Path(args.status_file).expanduser().resolve()
    for path, label in (
        (publisher_path, "SpaceMouse publisher"),
        (teleop_path, "xArm queue teleop"),
        (sdk_path, "xArm SDK"),
    ):
        if not path.exists():
            raise RuntimeError(f"{label} not found: {path}")

    env = os.environ.copy()
    python_path = [str(sdk_path)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env["PYTHONUNBUFFERED"] = "1"
    env["XARM_IP"] = env.get("XARM_IP", env.get("ROBOT_IP", "192.168.1.206"))
    runtime_home = Path(
        env.get("XARM_SDK_HOME", str(base_dir / ".runtime" / "xarm_home"))
    ).expanduser().resolve()
    runtime_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(runtime_home)

    publisher: subprocess.Popen | None = None
    teleop: subprocess.Popen | None = None
    stop_requested = threading.Event()

    def request_stop(_signum=None, _frame=None) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def read_commands() -> None:
        for raw_line in sys.stdin:
            command = raw_line.strip()
            if command.lower() == "q":
                stop_requested.set()
                return
            if teleop is not None and teleop.poll() is None and teleop.stdin is not None:
                try:
                    teleop.stdin.write(raw_line)
                    teleop.stdin.flush()
                except BrokenPipeError:
                    pass

    threading.Thread(target=read_commands, daemon=True, name="queue-teleop-stdin").start()

    try:
        publisher = subprocess.Popen(
            [
                sys.executable,
                str(publisher_path),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--authkey",
                args.authkey,
                "--publish-hz",
                str(args.publish_hz),
                "--status-file",
                str(status_path),
            ],
            env=env,
        )
        _wait_for_port(args.host, args.port, publisher, args.startup_timeout)

        teleop = subprocess.Popen(
            [
                sys.executable,
                str(teleop_path),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--authkey",
                args.authkey,
                "--poll-hz",
                str(args.poll_hz),
                "--status-file",
                str(status_path),
            ],
            env=env,
            stdin=subprocess.PIPE,
            text=True,
        )
        print(
            f"[QueueTeleop] publisher={publisher_path.name}, controller={teleop_path.name}, "
            f"queue={args.host}:{args.port}, robot={env['XARM_IP']}"
        )

        while not stop_requested.is_set():
            publisher_code = publisher.poll()
            teleop_code = teleop.poll()
            if publisher_code is not None:
                raise RuntimeError(f"SpaceMouse publisher exited with code {publisher_code}")
            if teleop_code is not None:
                if teleop_code != 0:
                    raise RuntimeError(f"xArm queue teleop exited with code {teleop_code}")
                return 0
            time.sleep(0.1)
        return 0
    except Exception as exc:
        _stop_process(teleop)
        _stop_process(publisher)
        _write_failure_status(status_path, str(exc))
        raise
    finally:
        _stop_process(teleop)
        _stop_process(publisher)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[QueueTeleop] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
