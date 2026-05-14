import glob
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    device: str | int | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: int = cv2.CAP_V4L2
    warmup_frames: int = 5


class CameraSource:
    def __init__(self, config: CameraConfig):
        self.config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._attempted_devices: list[str | int] = []

    def open(self) -> None:
        self.close()
        self._attempted_devices = []
        for device in self._candidate_devices():
            self._attempted_devices.append(device)
            capture = cv2.VideoCapture(device, self.config.backend)
            if not capture.isOpened():
                capture.release()
                continue
            self._configure_capture(capture)
            self._capture = capture
            self._warmup()
            return

        attempted = ", ".join(repr(device) for device in self._attempted_devices) or "none"
        raise RuntimeError(f"Camera open failed, attempted devices: {attempted}")

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._capture is None or not self._capture.isOpened():
            return False, None
        ok, frame = self._capture.read()
        if not ok:
            return False, None
        return True, frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def _candidate_devices(self) -> list[str | int]:
        if self.config.device is not None:
            return [self.config.device]

        devices: list[str | int] = []
        devices.extend(sorted(glob.glob("/dev/v4l/by-id/*")))
        devices.extend(sorted(glob.glob("/dev/video*")))
        devices.extend(range(4))
        return devices

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)

    def _warmup(self) -> None:
        if self._capture is None:
            return
        for _ in range(max(0, self.config.warmup_frames)):
            self._capture.read()


__all__ = ["CameraConfig", "CameraSource"]
