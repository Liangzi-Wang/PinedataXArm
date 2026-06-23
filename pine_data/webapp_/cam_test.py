from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import pyrealsense2 as rs
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


DEFAULT_HAND_SERIAL = os.getenv("HAND_SERIAL", "218622270687").strip()
DEFAULT_EXTERNAL_SERIAL = os.getenv("EXTERNAL_SERIAL", "409122274280").strip()
HAND_PIDS = {"0B5B", "0B07"}
EXTERNAL_PIDS = {"0B5B"}

RGB_CANDIDATES = [
    {"width": 1280, "height": 720, "fps": 15},
    {"width": 1280, "height": 720, "fps": 30},
    {"width": 640, "height": 480, "fps": 15},
    {"width": 640, "height": 480, "fps": 30},
    {"width": 480, "height": 270, "fps": 15},
    {"width": 480, "height": 270, "fps": 5},
    {"width": 640, "height": 480, "fps": 5},
    {"width": 424, "height": 240, "fps": 15},
    {"width": 424, "height": 240, "fps": 5},
]


def _optional_positive(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def describe_device(device: rs.device) -> str:
    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    pid = device.get_info(rs.camera_info.product_id)
    usb = device.get_info(rs.camera_info.usb_type_descriptor) if device.supports(rs.camera_info.usb_type_descriptor) else "unknown"
    return f"{name} serial={serial} pid={pid} usb={usb}"


def render_status_frame(title: str, message: str, width: int = 960, height: int = 540) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (24, 29, 35)
    cv2.putText(image, title, (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (226, 232, 240), 2, cv2.LINE_AA)
    y = 140
    for line in message.splitlines():
        cv2.putText(image, line[:90], (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (248, 180, 95), 2, cv2.LINE_AA)
        y += 38
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode placeholder frame.")
    return encoded.tobytes()


def supported_rgb_modes(device: rs.device) -> set[tuple[int, int, int]]:
    modes: set[tuple[int, int, int]] = set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if str(profile.stream_type()) != "stream.color":
                continue
            try:
                video = profile.as_video_stream_profile()
            except Exception:
                continue
            modes.add((int(video.width()), int(video.height()), int(profile.fps())))
    return modes


def candidate_profiles(
    device: rs.device,
    preferred_width: int | None = None,
    preferred_height: int | None = None,
    preferred_fps: int | None = None,
) -> list[dict[str, int]]:
    supported = supported_rgb_modes(device)
    result: list[dict[str, int]] = []
    seen: set[tuple[int, int, int]] = set()
    candidates: list[dict[str, int]] = []

    width = _optional_positive(preferred_width)
    height = _optional_positive(preferred_height)
    fps = _optional_positive(preferred_fps)
    if width is not None and height is not None and fps is not None:
        candidates.append({"width": width, "height": height, "fps": fps})

    candidates.extend(RGB_CANDIDATES)
    for candidate in candidates:
        key = (candidate["width"], candidate["height"], candidate["fps"])
        if key in seen:
            continue
        if key in supported:
            depth_width, depth_height = preferred_depth_resolution(
                device,
                candidate["width"],
                candidate["height"],
            )
            result.append(
                {
                    "width": candidate["width"],
                    "height": candidate["height"],
                    "fps": candidate["fps"],
                    "depth_width": depth_width,
                    "depth_height": depth_height,
                }
            )
            seen.add(key)
    return result


def find_device(serial: str | None, product_ids: set[str], exclude_serials: Iterable[str] = ()) -> rs.device:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise RuntimeError("No RealSense devices detected.")

    excluded = {item for item in exclude_serials if item}
    if serial:
        for device in devices:
            if device.get_info(rs.camera_info.serial_number) == serial:
                return device
        known = ", ".join(describe_device(device) for device in devices)
        raise RuntimeError(f"Requested camera serial '{serial}' not found. Devices: {known}")

    for device in devices:
        device_serial = device.get_info(rs.camera_info.serial_number)
        if device_serial in excluded:
            continue
        pid = device.get_info(rs.camera_info.product_id).upper()
        if pid in product_ids:
            return device

    known = ", ".join(describe_device(device) for device in devices)
    raise RuntimeError(f"No matching camera found for product IDs {sorted(product_ids)}. Devices: {known}")


def start_rgb_pipeline(serial: str, candidate: dict[str, int]) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color,
        candidate["width"],
        candidate["height"],
        rs.format.bgr8,
        candidate["fps"],
    )
    config.enable_stream(
        rs.stream.depth,
        candidate.get("depth_width", candidate["width"]),
        candidate.get("depth_height", candidate["height"]),
        rs.format.z16,
        candidate["fps"],
    )
    pipeline.start(config)
    return pipeline


def preferred_depth_resolution(
    device: rs.device,
    color_width: int,
    color_height: int,
) -> tuple[int, int]:
    pid = device.get_info(rs.camera_info.product_id).upper()
    if pid == "0B07" and (color_width > 848 or color_height > 480):
        return 848, 480
    return color_width, color_height


def choose_pipeline(
    serial: str,
    device: rs.device,
    timeout_ms: int,
    warmup_attempts: int,
    preferred_width: int | None = None,
    preferred_height: int | None = None,
    preferred_fps: int | None = None,
) -> tuple[rs.pipeline, dict[str, int]]:
    candidates = candidate_profiles(
        device,
        preferred_width=preferred_width,
        preferred_height=preferred_height,
        preferred_fps=preferred_fps,
    )
    if not candidates:
        raise RuntimeError("No supported RGB profiles found for this camera.")

    failures: list[str] = []
    for candidate in candidates:
        pipeline = None
        profile_name = f"{candidate['width']}x{candidate['height']}@{candidate['fps']}"
        try:
            pipeline = start_rgb_pipeline(serial, candidate)
            for _ in range(max(1, warmup_attempts)):
                frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
                color_frame = frames.get_color_frame()
                if color_frame:
                    selected_pipeline = pipeline
                    pipeline = None
                    return selected_pipeline, candidate
            failures.append(f"{profile_name} -> no RGB frames received")
        except Exception as exc:
            failures.append(f"{profile_name} -> {exc}")
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass

    raise RuntimeError("No working RGB profile found.\n  - " + "\n  - ".join(failures))


@dataclass
class CameraFeed:
    name: str
    serial_override: str
    product_ids: set[str]
    exclude_serials: tuple[str, ...]
    timeout_ms: int
    warmup_attempts: int
    latest_jpeg: bytes
    latest_depth_jpeg: bytes
    preferred_width: int | None = None
    preferred_height: int | None = None
    preferred_fps: int | None = None
    status: str = "starting"
    detail: str = "Feed is starting."
    serial: str = ""
    device_name: str = ""
    profile_name: str = ""
    last_frame_ts: float = 0.0
    last_depth_frame_ts: float = 0.0

    def __post_init__(self) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"{self.name}_feed")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3.0)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            age = time.time() - self.last_frame_ts if self.last_frame_ts else None
            depth_age = time.time() - self.last_depth_frame_ts if self.last_depth_frame_ts else None
            requested_profile = ""
            req_w = _optional_positive(self.preferred_width)
            req_h = _optional_positive(self.preferred_height)
            req_fps = _optional_positive(self.preferred_fps)
            if req_w is not None and req_h is not None and req_fps is not None:
                requested_profile = f"{req_w}x{req_h}@{req_fps}"
            return {
                "name": self.name,
                "status": self.status,
                "detail": self.detail,
                "serial": self.serial,
                "device_name": self.device_name,
                "profile": self.profile_name,
                "requested_profile": requested_profile,
                "seconds_since_frame": age,
                "seconds_since_depth_frame": depth_age,
            }

    def get_latest_jpeg(self) -> bytes:
        with self.lock:
            return self.latest_jpeg

    def get_latest_depth_jpeg(self) -> bytes:
        with self.lock:
            return self.latest_depth_jpeg

    def _set_state(
        self,
        *,
        status: str,
        detail: str,
        jpeg: bytes | None = None,
        depth_jpeg: bytes | None = None,
    ) -> None:
        with self.lock:
            self.status = status
            self.detail = detail
            if jpeg is not None:
                self.latest_jpeg = jpeg
            if depth_jpeg is not None:
                self.latest_depth_jpeg = depth_jpeg

    def _run(self) -> None:
        pipeline = None
        try:
            self._set_state(
                status="connecting",
                detail="Detecting camera...",
                jpeg=render_status_frame(self.name.title(), "Detecting camera..."),
                depth_jpeg=render_status_frame(f"{self.name.title()} Depth", "Detecting camera..."),
            )
            device = find_device(self.serial_override or None, self.product_ids, exclude_serials=self.exclude_serials)
            serial = device.get_info(rs.camera_info.serial_number)
            self.serial = serial
            self.device_name = device.get_info(rs.camera_info.name)
            self._set_state(
                status="connecting",
                detail=f"Found {self.device_name} ({serial}). Choosing RGB profile...",
                jpeg=render_status_frame(self.name.title(), f"Found {self.device_name}\nSerial: {serial}\nChoosing RGB profile..."),
                depth_jpeg=render_status_frame(f"{self.name.title()} Depth", f"Found {self.device_name}\nSerial: {serial}\nChoosing RGB+Depth profile..."),
            )
            pipeline, candidate = choose_pipeline(
                serial,
                device,
                self.timeout_ms,
                self.warmup_attempts,
                preferred_width=self.preferred_width,
                preferred_height=self.preferred_height,
                preferred_fps=self.preferred_fps,
            )
            self.profile_name = f"{candidate['width']}x{candidate['height']}@{candidate['fps']}"
            requested_text = ""
            req_w = _optional_positive(self.preferred_width)
            req_h = _optional_positive(self.preferred_height)
            req_fps = _optional_positive(self.preferred_fps)
            if req_w is not None and req_h is not None and req_fps is not None:
                requested = f"{req_w}x{req_h}@{req_fps}"
                if requested != self.profile_name:
                    requested_text = f" (requested {requested}, fallback applied)"
            self._set_state(
                status="streaming",
                detail=f"Streaming with {self.profile_name}{requested_text}",
            )

            while not self.stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        self._set_state(
                            status="connected",
                            detail="Connected, but no RGB frame arrived.",
                            jpeg=render_status_frame(self.name.title(), "Connected\nNo RGB frame arrived."),
                        )
                        continue

                    color = np.asanyarray(color_frame.get_data())
                    ok, encoded = cv2.imencode(".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if not ok:
                        raise RuntimeError("Failed to encode RGB frame.")

                    depth_encoded_bytes = None
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth = np.asanyarray(depth_frame.get_data())
                        depth_colored = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth, alpha=0.03),
                            cv2.COLORMAP_JET,
                        )
                        ok_depth, depth_encoded = cv2.imencode(
                            ".jpg",
                            depth_colored,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                        )
                        if ok_depth:
                            depth_encoded_bytes = depth_encoded.tobytes()

                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()
                        if depth_encoded_bytes is not None:
                            self.latest_depth_jpeg = depth_encoded_bytes
                            self.last_depth_frame_ts = time.time()
                        self.status = "streaming"
                        self.detail = f"Streaming with {self.profile_name}{requested_text}"
                        self.last_frame_ts = time.time()
                except Exception as exc:
                    self._set_state(
                        status="connected",
                        detail=f"Connected, frame read failed: {exc}",
                        jpeg=render_status_frame(self.name.title(), f"Connected\nFrame read failed:\n{exc}"),
                        depth_jpeg=render_status_frame(f"{self.name.title()} Depth", f"Connected\nDepth read failed:\n{exc}"),
                    )
                    time.sleep(0.2)
        except Exception as exc:
            self._set_state(
                status="error",
                detail=str(exc),
                jpeg=render_status_frame(self.name.title(), f"Error\n{exc}"),
                depth_jpeg=render_status_frame(f"{self.name.title()} Depth", f"Error\n{exc}"),
            )
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass


class CameraTestApp:
    def __init__(
        self,
        hand_serial: str,
        external_serial: str,
        timeout_ms: int,
        warmup_attempts: int,
        external_width: int | None,
        external_height: int | None,
        external_fps: int | None,
    ) -> None:
        placeholder = render_status_frame("Camera", "Starting RGB preview...")
        depth_placeholder = render_status_frame("Camera Depth", "Starting depth preview...")
        self.hand = CameraFeed(
            name="hand camera",
            serial_override=hand_serial.strip(),
            product_ids=HAND_PIDS,
            exclude_serials=(),
            timeout_ms=timeout_ms,
            warmup_attempts=warmup_attempts,
            latest_jpeg=placeholder,
            latest_depth_jpeg=depth_placeholder,
        )
        self.external = CameraFeed(
            name="external camera",
            serial_override=external_serial.strip(),
            product_ids=EXTERNAL_PIDS,
            exclude_serials=(hand_serial.strip(),),
            timeout_ms=timeout_ms,
            warmup_attempts=warmup_attempts,
            latest_jpeg=placeholder,
            latest_depth_jpeg=depth_placeholder,
            preferred_width=external_width,
            preferred_height=external_height,
            preferred_fps=external_fps,
        )
        self.app = FastAPI(title="Camera Test")
        self._install_routes()

    def _install_routes(self) -> None:
        @self.app.on_event("startup")
        def _startup() -> None:
            self.hand.start()
            self.external.start()

        @self.app.on_event("shutdown")
        def _shutdown() -> None:
            self.hand.stop()
            self.external.stop()

        @self.app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Test</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1720;
      --panel: #17212b;
      --line: #2a3642;
      --text: #e5eef7;
      --muted: #98a8b8;
      --accent: #58c4dc;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      background: linear-gradient(180deg, #101820, #0c1218);
      color: var(--text);
    }
    .page {
      width: min(96vw, 1800px);
      margin: 0 auto;
      padding: 20px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 1.9rem;
    }
    p {
      margin: 0 0 20px;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 18px 36px rgba(0, 0, 0, 0.25);
    }
    .head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .title {
      font-size: 1.05rem;
      font-weight: 700;
      text-transform: capitalize;
    }
    .badge {
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 700;
      background: rgba(88, 196, 220, 0.14);
      color: var(--accent);
      white-space: nowrap;
    }
    img {
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #05080c;
    }
    .meta {
      padding: 12px 16px 16px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.55;
      white-space: pre-wrap;
      user-select: text;
    }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="page">
    <h1>Camera Test</h1>
    <p>Standalone RGB + depth preview for the hand and external RealSense cameras.</p>
    <section class="grid">
      <article class="panel">
        <div class="head">
          <div class="title">Hand Camera</div>
          <div class="badge" id="handBadge">starting</div>
        </div>
                <img src="/stream/hand" alt="Hand camera RGB preview">
                <img src="/stream/hand/depth" alt="Hand camera depth preview">
        <div class="meta" id="handMeta">Loading...</div>
      </article>
      <article class="panel">
        <div class="head">
          <div class="title">External Camera</div>
          <div class="badge" id="externalBadge">starting</div>
        </div>
                <img src="/stream/external" alt="External camera RGB preview">
                <img src="/stream/external/depth" alt="External camera depth preview">
        <div class="meta" id="externalMeta">Loading...</div>
      </article>
    </section>
  </main>
  <script>
    function hasActiveTextSelection() {
      const selection = window.getSelection ? window.getSelection() : null;
      if (!selection || selection.isCollapsed) {
        return false;
      }
      return Boolean(selection.toString().trim());
    }

    function setTextWhenSafe(element, text) {
      if (!element) {
        return;
      }
      if (hasActiveTextSelection()) {
        element.dataset.pendingText = text;
        return;
      }
      if (element.textContent !== text) {
        element.textContent = text;
      }
      delete element.dataset.pendingText;
    }

    function flushDeferredText() {
      if (hasActiveTextSelection()) {
        return;
      }
      for (const key of ["hand", "external"]) {
        const element = document.getElementById(`${key}Meta`);
        if (!element) {
          continue;
        }
        const pending = element.dataset.pendingText;
        if (typeof pending === "string") {
          element.textContent = pending;
          delete element.dataset.pendingText;
        }
      }
    }

    async function refreshStatus() {
      if (hasActiveTextSelection()) {
        return;
      }
      try {
        const response = await fetch("/status");
        const payload = await response.json();
        for (const key of ["hand", "external"]) {
          const item = payload[key];
          document.getElementById(`${key}Badge`).textContent = item.status;
          setTextWhenSafe(document.getElementById(`${key}Meta`),
            `Device: ${item.device_name || "-"}\\n` +
            `Serial: ${item.serial || "-"}\\n` +
            `Requested: ${item.requested_profile || "-"}\\n` +
            `Profile: ${item.profile || "-"}\\n` +
            `Detail: ${item.detail || "-"}\\n` +
                        `Seconds since RGB frame: ${item.seconds_since_frame == null ? "-" : item.seconds_since_frame.toFixed(1)}\n` +
                        `Seconds since depth frame: ${item.seconds_since_depth_frame == null ? "-" : item.seconds_since_depth_frame.toFixed(1)}`);
        }
      } catch (_error) {
      }
    }
    refreshStatus();
    setInterval(() => {
      flushDeferredText();
      refreshStatus();
    }, 1000);
  </script>
</body>
</html>
"""

        @self.app.get("/status", response_class=JSONResponse)
        def status() -> dict[str, object]:
            return {
                "hand": self.hand.snapshot(),
                "external": self.external.snapshot(),
            }

        @self.app.get("/stream/{camera_name}")
        def stream(camera_name: str) -> StreamingResponse:
            feed = self._feed_for_name(camera_name)

            def generator() -> Iterable[bytes]:
                while True:
                    frame = feed.get_latest_jpeg()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                    time.sleep(0.08)

            return StreamingResponse(
                generator(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.get("/stream/{camera_name}/depth")
        def stream_depth(camera_name: str) -> StreamingResponse:
            feed = self._feed_for_name(camera_name)

            def generator() -> Iterable[bytes]:
                while True:
                    frame = feed.get_latest_depth_jpeg()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                    time.sleep(0.08)

            return StreamingResponse(
                generator(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

    def _feed_for_name(self, camera_name: str) -> CameraFeed:
        name = camera_name.strip().lower()
        if name == "hand":
            return self.hand
        if name == "external":
            return self.external
        raise HTTPException(status_code=404, detail=f"Unknown camera '{camera_name}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone web page for hand/external RGB camera previews.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8011, help="Port to bind.")
    parser.add_argument("--hand-serial", default=DEFAULT_HAND_SERIAL, help="Hand camera serial.")
    parser.add_argument("--external-serial", default=DEFAULT_EXTERNAL_SERIAL, help="External camera serial.")
    parser.add_argument("--external-width", type=int, default=1280, help="Requested external camera width.")
    parser.add_argument("--external-height", type=int, default=720, help="Requested external camera height.")
    parser.add_argument("--external-fps", type=int, default=15, help="Requested external camera FPS.")
    parser.add_argument("--timeout-ms", type=int, default=1500, help="Frame wait timeout in milliseconds.")
    parser.add_argument("--warmup-attempts", type=int, default=8, help="Warmup attempts per RGB profile.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camera_test = CameraTestApp(
        hand_serial=args.hand_serial,
        external_serial=args.external_serial,
        timeout_ms=args.timeout_ms,
        warmup_attempts=args.warmup_attempts,
        external_width=args.external_width,
        external_height=args.external_height,
        external_fps=args.external_fps,
    )
    uvicorn.run(camera_test.app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
