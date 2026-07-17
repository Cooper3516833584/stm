"""Road perception pipeline with camera capture and segmentation inference threads.

Replaces the single-threaded blocking ``cap.read() → segmentation inference``
sequence with separate parallel threads, so the main control loop only
reads the latest results without blocking.

Architecture::

    CameraThread ──frame──→ Segmentation thread ──result──→ Control Loop
       (独立)                  (独立)                        (主线程, 10Hz)

    SharedLatest buffers decouple producers from consumers with a
    lock-protected "latest is greatest" policy — the control loop
    always gets the most recent result without waiting for either
    camera or YOLO to finish.

Usage::

    pipeline = PerceptionPipeline(
        camera_index=7,
        model_path="FlightController/Solutions/model/road_yolo11n_seg_128.onnx",
        npu_model_path="FlightController/Solutions/model/new_road_seg_v3_final_fp32.nb",
        inference_backend="npu",
        flight_height_m=1.0,
    )
    pipeline.start()
    try:
        while True:
            perception, age_s = pipeline.latest_perception()
            # perception is a RoadPerceptionResult or None (lost / not ready)
            desired = follower.update(perception, now_s=time.monotonic())
            pipeline.log_summary()
            time.sleep(0.1)  # 10 Hz
    finally:
        pipeline.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np


def _open_camera_capture(cv2, index: int, width: int, height: int, fps: int):
    """Open the road camera through V4L2 with an explicit MJPG profile."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    buffer_size_property = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
    if buffer_size_property is not None:
        cap.set(buffer_size_property, 1)
    return cap


def _capture_profile(cap, cv2) -> str:
    """Describe the profile accepted by the camera driver for diagnostics."""
    try:
        backend = cap.getBackendName()
    except Exception:
        backend = "unknown"
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    return f"backend={backend} profile={width}x{height}@{fps:.1f} fourcc={fourcc!r}"


# ── SharedLatest ────────────────────────────────────────────────────────

class SharedLatest:
    """Thread-safe single-writer / multi-reader ``latest is greatest`` buffer.

    - ``publish(value, timestamp)``:  writer calls, atomic replace.
    - ``latest() → (value, timestamp)``:  reader calls, returns snapshot.
    - ``age_s(now_s) → float``:  seconds since last publish.

    The lock is held only for pointer assignment — readers and writers
    never block on each other for more than a few microseconds.
    """

    __slots__ = ("_value", "_timestamp", "_lock")

    def __init__(self) -> None:
        self._value: Any = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()

    # ── writer API ──────────────────────────────────────────────────

    def publish(self, value: Any, timestamp: float) -> None:
        """Atomically replace the stored value."""
        with self._lock:
            self._value = value
            self._timestamp = timestamp

    # ── reader API ──────────────────────────────────────────────────

    def latest(self) -> tuple[Any, float]:
        """Return ``(value, timestamp)`` snapshot."""
        with self._lock:
            return self._value, self._timestamp

    def age_s(self, now_s: float) -> float:
        """Seconds since the last ``publish()``."""
        with self._lock:
            ts = self._timestamp
        return max(0.0, now_s - ts) if ts > 0.0 else float("inf")

    @property
    def has_value(self) -> bool:
        """True if ``publish()`` has been called at least once."""
        with self._lock:
            return self._value is not None


# ── CameraThread ────────────────────────────────────────────────────────

class CameraThread:
    """Independent camera capture thread via OpenCV V4L2.

    Publishes raw BGR frames into ``self.frame_buffer`` at the camera's
    native frame rate (typically 30 fps).  No preprocessing — that is
    the responsibility of the YOLO thread or the control loop.
    """

    def __init__(
        self,
        camera_index: int = 7,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._index = camera_index
        self._width = width
        self._height = height
        self._fps = fps

        self.frame_buffer = SharedLatest()
        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._ok: bool = False
        self.capture_count: int = 0
        self.capture_error_count: int = 0
        self._last_log_s: float = 0.0
        self._last_log_count: int = 0
        self._interval_read_ms_total: float = 0.0
        self._interval_read_ms_max: float = 0.0

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        """Open the camera device and start the capture thread."""
        import cv2

        if self._running:
            return

        cap = _open_camera_capture(
            cv2,
            self._index,
            self._width,
            self._height,
            self._fps,
        )
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open road camera index {self._index} through V4L2")

        # Warm-up: read and discard a few frames
        for _ in range(5):
            cap.read()
        accepted_profile = _capture_profile(cap, cv2)

        self._cap = cap
        self._ok = True
        self._running = True
        self._last_log_s = time.monotonic()
        self._last_log_count = self.capture_count
        self._thread = threading.Thread(
            target=self._capture_task, name="camera", daemon=True
        )
        self._thread.start()
        self._log(
            f"started requested={self._width}x{self._height}@{self._fps} "
            f"{accepted_profile}"
        )

    def stop(self) -> None:
        """Signal the thread to stop, join it, and release the camera."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._ok = False
        self._log("stopped")

    # ── reader API (called from other threads) ───────────────────────

    def latest_frame(self, max_age_s: float = 0.5) -> tuple[Any, float]:
        """Return ``(frame, timestamp)`` or ``(None, 0)`` if stale."""
        frame, ts = self.frame_buffer.latest()
        if frame is None:
            return None, 0.0
        if self.frame_buffer.age_s(time.monotonic()) > max_age_s:
            return None, 0.0
        return frame, ts

    @property
    def ok(self) -> bool:
        return self._ok

    # ── internal ────────────────────────────────────────────────────

    def _capture_task(self) -> None:
        """Loop: read frame → publish to buffer."""
        import cv2

        consecutive_failures = 0
        while self._running:
            read_started_s = time.monotonic()
            ok, frame = self._cap.read()
            now = time.monotonic()
            read_ms = max(0.0, now - read_started_s) * 1000.0

            if ok and frame is not None:
                consecutive_failures = 0
                self.frame_buffer.publish(frame, now)
                self.capture_count += 1
                self._interval_read_ms_total += read_ms
                self._interval_read_ms_max = max(self._interval_read_ms_max, read_ms)
                if not self._ok:
                    self._ok = True
            else:
                consecutive_failures += 1
                self.capture_error_count += 1
                if consecutive_failures >= 30:
                    self._ok = False
                if consecutive_failures == 1:
                    self._log(f"camera read failed (consecutive={consecutive_failures})")
                time.sleep(0.033)  # ~30 fps retry

            if now - self._last_log_s >= 5.0:
                elapsed_s = max(1e-6, now - self._last_log_s)
                completed = self.capture_count - self._last_log_count
                mean_read_ms = self._interval_read_ms_total / max(1, completed)
                self._log(
                    f"fps~{completed / elapsed_s:.1f} read_ms~{mean_read_ms:.1f} "
                    f"read_max_ms={self._interval_read_ms_max:.1f} "
                    f"errors={self.capture_error_count}"
                )
                self._last_log_s = now
                self._last_log_count = self.capture_count
                self._interval_read_ms_total = 0.0
                self._interval_read_ms_max = 0.0

    def _log(self, msg: str) -> None:
        try:
            from loguru import logger
            logger.info(f"[camera] {msg}")
        except ImportError:
            pass


# ── YOLOInferenceThread ─────────────────────────────────────────────────

class YOLOInferenceThread:
    """Independent road segmentation inference thread.

    Polls ``CameraThread.frame_buffer`` for the latest frame, runs
    ``road_perception.get_road_perception()``, and publishes the result
    to ``self.road_buffer``.  If the camera produces frames faster than
    YOLO can process them, intermediate frames are silently dropped.

    On the STM32MP257 with the .nb 128 model, each inference takes
    ~59 ms, yielding an effective rate of ~17 fps.
    """

    def __init__(
        self,
        camera_thread: CameraThread,
        model_path: str,
        npu_model_path: str = "FlightController/Solutions/model/new_road_seg_v3_final_fp32.nb",
        inference_backend: str = "npu",
        postprocess_mode: str = "fast-main",
        flight_height_m: float = 1.0,
        wb_enable: bool = False,
        wb_r: float = 2.78,
        wb_g: float = 1.00,
        wb_b: float = 1.26,
        offset_comp_config=None,
        poll_interval_s: float = 0.005,
        stale_timeout_s: float = 1.0,
    ) -> None:
        self._camera = camera_thread
        self._model_path = model_path
        self._npu_model_path = npu_model_path
        self._inference_backend = inference_backend
        self._postprocess_mode = postprocess_mode
        self._flight_height_m = flight_height_m
        self._wb_enable = wb_enable
        self._wb_r = wb_r
        self._wb_g = wb_g
        self._wb_b = wb_b
        self._offset_comp_config = offset_comp_config
        self._poll_interval_s = poll_interval_s
        self._stale_timeout_s = stale_timeout_s

        self.road_buffer = SharedLatest()
        self._thread: threading.Thread | None = None
        self._running = False

        # Statistics
        self.inference_count: int = 0
        self.skip_count: int = 0
        self.error_count: int = 0
        self._last_log_s: float = 0.0
        self._last_log_count: int = 0
        self._interval_inference_ms_total: float = 0.0
        self._interval_inference_ms_max: float = 0.0
        self._interval_input_age_ms_max: float = 0.0

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        """Load the model and start the inference thread."""
        if self._running:
            return

        # Configure road_perception's module-level session before the worker
        # starts.  NPU is the production default; CPU retains the legacy
        # lightweight YOLO implementation.
        import road_perception
        road_perception.configure_model(
            backend=self._inference_backend,
            cpu_model_path=self._model_path,
            npu_model_path=self._npu_model_path,
            postprocess_mode=self._postprocess_mode,
        )
        io_info = road_perception.get_model_io_info()

        self._running = True
        self._last_log_s = time.monotonic()
        self._last_log_count = self.inference_count
        self._thread = threading.Thread(
            target=self._inference_task, name="yolo", daemon=True
        )
        self._thread.start()
        selected_model = (
            self._npu_model_path
            if self._inference_backend == "npu"
            else self._model_path
        )
        self._log(
            f"started backend={self._inference_backend} model={selected_model} "
            f"provider={io_info['provider']} kind={io_info['model_kind']} "
            f"postprocess={io_info['postprocess_mode']}"
        )

    def stop(self) -> None:
        """Signal the thread to stop and join it."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._log(
            f"stopped — inferences={self.inference_count} "
            f"skipped={self.skip_count} errors={self.error_count}"
        )

    # ── reader API (called from control thread) ──────────────────────

    def latest_result(
        self, max_age_s: float | None = None
    ) -> tuple[Any, float, bool]:
        """Return ``(perception_result, age_s, is_stale)``.

        ``perception_result`` is a ``RoadPerceptionResult`` or ``None``
        (before the first inference completes, or on error).
        """
        if max_age_s is None:
            max_age_s = self._stale_timeout_s
        result, ts = self.road_buffer.latest()
        now = time.monotonic()
        age = max(0.0, now - ts) if ts > 0.0 else float("inf")
        is_stale = age > max_age_s
        return result, age, is_stale

    # ── internal ────────────────────────────────────────────────────

    def _inference_task(self) -> None:
        """Loop: wait for new frame → infer → publish."""
        import road_perception

        # Configure white balance if enabled
        wb_config = None
        if self._wb_enable:
            from road_perception import CameraWhiteBalanceConfig
            wb_config = CameraWhiteBalanceConfig(
                enabled=True,
                r_gain=self._wb_r,
                g_gain=self._wb_g,
                b_gain=self._wb_b,
            )

        # The global session is loaded on first call inside road_perception
        last_frame_id = id(None)  # sentinel — any real frame will differ

        while self._running:
            frame, frame_ts = self._camera.frame_buffer.latest()
            if frame is None:
                time.sleep(self._poll_interval_s)
                continue

            # Skip if same frame we already processed
            current_id = id(frame)
            if current_id == last_frame_id:
                time.sleep(self._poll_interval_s)
                self.skip_count += 1
                continue

            last_frame_id = current_id

            try:
                inference_started_s = time.monotonic()
                result = road_perception.get_road_perception(
                    frame,
                    flight_height_m=self._flight_height_m,
                    debug_save_path=None,
                    offset_comp_config=self._offset_comp_config,
                    wb_config=wb_config,
                )
                inference_finished_s = time.monotonic()
            except Exception as exc:
                self.error_count += 1
                import road_perception as rp
                result = rp._lost_result(f"YOLO thread: {type(exc).__name__}: {exc}")
                self._log(f"inference error: {exc}")
                time.sleep(self._poll_interval_s)
                continue

            self.road_buffer.publish(result, frame_ts)
            self.inference_count += 1
            inference_ms = max(0.0, inference_finished_s - inference_started_s) * 1000.0
            input_age_ms = max(0.0, inference_finished_s - frame_ts) * 1000.0
            self._interval_inference_ms_total += inference_ms
            self._interval_inference_ms_max = max(
                self._interval_inference_ms_max,
                inference_ms,
            )
            self._interval_input_age_ms_max = max(
                self._interval_input_age_ms_max,
                input_age_ms,
            )

            # Periodic log
            now = time.monotonic()
            if now - self._last_log_s >= 5.0:
                elapsed_s = max(1e-6, now - self._last_log_s)
                completed = self.inference_count - self._last_log_count
                fps = completed / elapsed_s
                self._last_log_s = now
                self._last_log_count = self.inference_count
                state = result.road_state if result else "?"
                mean_inference_ms = (
                    self._interval_inference_ms_total / max(1, completed)
                )
                self._log(
                    f"fps~{fps:.1f} "
                    f"infer_ms~{mean_inference_ms:.1f} "
                    f"infer_max_ms={self._interval_inference_ms_max:.1f} "
                    f"input_age_max_ms={self._interval_input_age_ms_max:.1f} "
                    f"state={state}"
                )
                self._interval_inference_ms_total = 0.0
                self._interval_inference_ms_max = 0.0
                self._interval_input_age_ms_max = 0.0

    def _log(self, msg: str) -> None:
        try:
            from loguru import logger
            logger.info(f"[yolo] {msg}")
        except ImportError:
            pass


# ── PerceptionPipeline (convenience) ─────────────────────────────────────

class PerceptionPipeline:
    """Thin wrapper that owns a CameraThread + YOLOInferenceThread.

    One-shot ``start()`` / ``stop()`` and a single ``latest_perception()``
    read point for the control loop.
    """

    def __init__(
        self,
        camera_index: int = 7,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        model_path: str = "FlightController/Solutions/model/road_yolo11n_seg_128.onnx",
        npu_model_path: str = "FlightController/Solutions/model/new_road_seg_v3_final_fp32.nb",
        inference_backend: str = "npu",
        postprocess_mode: str = "fast-main",
        flight_height_m: float = 1.0,
        wb_enable: bool = False,
        wb_r: float = 1.00,
        wb_g: float = 1.00,
        wb_b: float = 1.00,
        offset_comp_config=None,
    ) -> None:
        self.camera = CameraThread(
            camera_index=camera_index,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )
        self.yolo = YOLOInferenceThread(
            camera_thread=self.camera,
            model_path=model_path,
            npu_model_path=npu_model_path,
            inference_backend=inference_backend,
            postprocess_mode=postprocess_mode,
            flight_height_m=flight_height_m,
            wb_enable=wb_enable,
            wb_r=wb_r,
            wb_g=wb_g,
            wb_b=wb_b,
            offset_comp_config=offset_comp_config,
        )

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        self.camera.start()
        self.yolo.start()

    def stop(self) -> None:
        # Stop YOLO first (so it doesn't try to read frames while
        # camera is shutting down)
        self.yolo.stop()
        self.camera.stop()

    # ── control-loop API ────────────────────────────────────────────

    def latest_perception(self) -> tuple[Any, float, bool]:
        """Return ``(perception, age_s, is_stale)`` for the control loop."""
        return self.yolo.latest_result()

    def latest_frame(self) -> tuple[Any, float]:
        """Return ``(frame, timestamp)`` — for recording / debug."""
        return self.camera.latest_frame()

    @property
    def camera_ok(self) -> bool:
        return self.camera.ok
