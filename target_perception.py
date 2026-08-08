"""Frame-only purple target worker used by the unified camera pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from purple_target import find_purple_target_offset


@dataclass(frozen=True)
class PurpleTargetObservation:
    found: bool
    offset_x_px: int | None
    offset_y_px: int | None
    capture_time_s: float
    processed_time_s: float
    error: str | None = None

    def age_s(self, now_s: float | None = None) -> float:
        now_s = time.monotonic() if now_s is None else now_s
        if self.capture_time_s <= 0.0:
            return float("inf")
        return max(0.0, now_s - self.capture_time_s)


class PurpleTargetInferenceThread:
    """Consume the camera's latest-frame subscription without opening a device."""

    def __init__(
        self,
        camera_thread,
        *,
        max_dimension: int = 256,
        min_area_ratio: float = 0.005,
        poll_interval_s: float = 0.005,
        stale_timeout_s: float = 1.0,
    ):
        if int(max_dimension) < 32:
            raise ValueError("max_dimension must be at least 32")
        if not 0.0 < float(min_area_ratio) <= 1.0:
            raise ValueError("min_area_ratio must be within (0, 1]")
        self._frames = (
            camera_thread.subscribe_frames()
            if hasattr(camera_thread, "subscribe_frames")
            else camera_thread.frame_buffer
        )
        self._max_dimension = int(max_dimension)
        self._min_area_ratio = float(min_area_ratio)
        self._poll_interval_s = float(poll_interval_s)
        self._stale_timeout_s = float(stale_timeout_s)
        self.target_buffer = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.detection_count = 0
        self.error_count = 0

    def start(self) -> None:
        if self._running:
            return
        from perception_pipeline import SharedLatest
        self.target_buffer = SharedLatest()
        self._running = True
        self._thread = threading.Thread(
            target=self._task,
            name="purple-target",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def latest_result(self, max_age_s: float | None = None):
        max_age_s = self._stale_timeout_s if max_age_s is None else float(max_age_s)
        result, timestamp = (
            self.target_buffer.latest()
            if self.target_buffer is not None
            else (None, 0.0)
        )
        age = (
            max(0.0, time.monotonic() - timestamp)
            if timestamp > 0
            else float("inf")
        )
        return result, age, age > max_age_s

    def _task(self) -> None:
        last_frame_id = id(None)
        while self._running:
            frame, frame_ts = self._frames.latest()
            if frame is None or id(frame) == last_frame_id:
                time.sleep(self._poll_interval_s)
                continue
            last_frame_id = id(frame)
            error = None
            try:
                offset = find_purple_target_offset(
                    frame,
                    color_order="bgr",
                    max_dimension=self._max_dimension,
                    min_area_ratio=self._min_area_ratio,
                )
            except Exception as exc:  # detector failures must not stop road following
                self.error_count += 1
                offset = None
                error = f"{type(exc).__name__}: {exc}"
            result = PurpleTargetObservation(
                found=offset is not None,
                offset_x_px=offset[0] if offset is not None else None,
                offset_y_px=offset[1] if offset is not None else None,
                capture_time_s=float(frame_ts),
                processed_time_s=time.monotonic(),
                error=error,
            )
            self.target_buffer.publish(result, frame_ts)
            self.detection_count += 1


__all__ = ["PurpleTargetInferenceThread", "PurpleTargetObservation"]
