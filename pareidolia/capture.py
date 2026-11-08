"""Frame sources: a webcam read on its own thread, a video file, or a still
image with a slow pan (for testing without a camera)."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .params import Config

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class Source:
    realtime = True

    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class Camera(Source):
    """Reads frames continuously on a background thread; read() returns the newest."""

    def __init__(self, cfg: Config, device: int | None = None):
        c = cfg.camera
        self.size = (int(c["width"]), int(c["height"]))
        self.mirror = bool(c["mirror"])
        dev = int(c["device"]) if device is None else device
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(dev)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {dev}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.cap.set(cv2.CAP_PROP_FPS, int(c["fps"]))
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._count = 0
        self._last_read = -1
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        if c["lock_exposure"]:
            threading.Thread(target=self._lock_exposure, daemon=True).start()

    def _lock_exposure(self) -> None:
        time.sleep(2.0)  # let auto exposure settle first
        try:
            exp = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1 = manual
            if exp > 0:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except cv2.error:
            pass

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._count += 1

    def read(self) -> np.ndarray | None:
        """Newest frame; waits briefly for a new one so frames are not repeated.
        The first frame can take over a second while the camera starts."""
        with self._lock:
            started = self._frame is not None
        deadline = time.monotonic() + (0.1 if started else 5.0)
        while True:
            with self._lock:
                if self._frame is not None and self._count != self._last_read:
                    self._last_read = self._count
                    return self._frame
            if time.monotonic() > deadline:
                with self._lock:
                    return self._frame
            time.sleep(0.001)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class VideoFile(Source):
    """Loops a video file, paced to its frame rate unless realtime=False."""

    def __init__(self, path: str | Path, size: tuple[int, int], realtime: bool = True):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.size = size
        self.realtime = realtime
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.period = 1.0 / fps if fps and fps > 1 else 1 / 30
        self.fps = 1.0 / self.period
        self._next = time.monotonic()

    def read(self) -> np.ndarray | None:
        if self.realtime:
            wait = self._next - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next = max(self._next + self.period, time.monotonic() - self.period)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return frame

    def close(self) -> None:
        self.cap.release()


class StillImage(Source):
    """A slow pan and zoom over a still image, so the tracker sees motion."""

    def __init__(self, path: str | Path, size: tuple[int, int], realtime: bool = True, fps: float = 30):
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"cannot read image {path}")
        W, H = size
        scale = max(W / img.shape[1], H / img.shape[0]) * 1.25
        self.img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)
        self.size = size
        self.realtime = realtime
        self.period = 1.0 / fps
        self.t = 0.0
        self._next = time.monotonic()

    def read(self) -> np.ndarray | None:
        if self.realtime:
            wait = self._next - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next = max(self._next + self.period, time.monotonic() - self.period)
        self.t += self.period
        W, H = self.size
        ih, iw = self.img.shape[:2]
        zoom = 1.0 + 0.08 * math.sin(self.t * 0.21)
        cw, ch = int(W * zoom * 1.1), int(H * zoom * 1.1)
        cw, ch = min(cw, iw), min(ch, ih)
        x = int((iw - cw) * (0.5 + 0.5 * math.sin(self.t * 0.13)))
        y = int((ih - ch) * (0.5 + 0.5 * math.sin(self.t * 0.17 + 1.0)))
        return cv2.resize(self.img[y:y + ch, x:x + cw], self.size, interpolation=cv2.INTER_AREA)


def open_source(spec: str | None, cfg: Config, realtime: bool = True) -> Source:
    """None or a number -> camera; an image path -> StillImage; else a video file."""
    size = (int(cfg.camera["width"]), int(cfg.camera["height"]))
    if spec is None:
        return Camera(cfg)
    if spec.isdigit():
        return Camera(cfg, int(spec))
    p = Path(spec)
    if p.suffix.lower() in IMAGE_EXT:
        return StillImage(p, size, realtime)
    return VideoFile(p, size, realtime)
