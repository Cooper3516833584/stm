"""Spawn-based process runtime for camera/NPU and dual-radar acquisition.

The control process only consumes latest-value snapshots. Large camera frames
and radar point arrays live in shared-memory rings; queues carry small metadata
messages and are always drained to the newest item.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
import queue
import statistics
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameRef:
    slot: int
    generation: int
    sequence: int
    capture_time_s: float
    shape: tuple[int, int, int]
    dtype: str = "uint8"


@dataclass(frozen=True)
class VisionSnapshot:
    sequence: int
    capture_time_s: float
    completed_time_s: float
    camera_ok: bool
    perception: Any | None
    frame_ref: FrameRef | None
    inference_ms: float = 0.0
    normalize_ms: float = 0.0
    preprocess_ms: float = 0.0
    npu_ms: float = 0.0
    postprocess_ms: float = 0.0
    error_count: int = 0
    publish_drops: int = 0

    def age_s(self, now_s: float | None = None) -> float:
        now_s = time.perf_counter() if now_s is None else now_s
        return max(0.0, now_s - self.capture_time_s)


@dataclass(frozen=True)
class RadarSnapshot:
    sequence: int
    published_time_s: float
    last_frame_time_s: float
    points_slot: int
    points_generation: int
    point_count: int
    connected: bool
    fresh: bool
    radar_health: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    crc_errors: int = 0
    parse_buffer_bytes: int = 0
    publish_drops: int = 0

    def age_s(self, now_s: float | None = None) -> float:
        now_s = time.perf_counter() if now_s is None else now_s
        if self.last_frame_time_s <= 0.0:
            return float("inf")
        return max(0.0, now_s - self.last_frame_time_s)


@dataclass(frozen=True)
class RuntimeMetrics:
    target_hz: float
    achieved_hz: float
    work_p50_ms: float
    work_p95_ms: float
    work_p99_ms: float
    work_max_ms: float
    jitter_p99_ms: float
    deadline_misses: int
    samples: int


@dataclass(frozen=True)
class RuntimeHealth:
    vision_alive: bool
    radar_alive: bool
    vision_ready: bool
    radar_ready: bool
    vision_age_s: float
    radar_age_s: float
    vision_publish_drops: int
    radar_publish_drops: int
    vision_queue_depth: int | None
    radar_queue_depth: int | None


@dataclass(frozen=True)
class ProcessRuntimeConfig:
    camera_index: int = 7
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    model_path: str = "FlightController/Solutions/model/road_yolo11n_seg_128.onnx"
    npu_model_path: str = "FlightController/Solutions/model/new_road_seg_v5_final_fp32.nb"
    inference_backend: str = "npu"
    postprocess_mode: str = "fast-main"
    instance_selection: str = "highest-confidence"
    flight_height_m: float = 1.0
    offset_comp_config: Any | None = None
    wb_enable: bool = False
    wb_r: float = 1.0
    wb_g: float = 1.0
    wb_b: float = 1.0
    upper_port: str = "/dev/ttySTM4"
    lower_port: str = "/dev/ttySTM9"
    radar_timeout_s: float = 0.5
    radar_publish_hz: float = 50.0
    frame_slots: int = 8
    radar_slots: int = 2
    radar_max_points: int = 2160
    enable_vision: bool = True
    enable_radar: bool = True
    apply_cpu_affinity: bool = True


class LoopRateMonitor:
    """Low-overhead rolling control-loop timing statistics."""

    def __init__(self, target_hz: float, capacity: int = 4096):
        self.target_hz = max(0.1, float(target_hz))
        self.period_s = 1.0 / self.target_hz
        self._work_ms: deque[float] = deque(maxlen=max(32, capacity))
        self._period_ms: deque[float] = deque(maxlen=max(32, capacity))
        self._last_start_s: float | None = None
        self._deadline_misses = 0

    def record(self, loop_start_s: float, loop_end_s: float) -> None:
        work_s = max(0.0, loop_end_s - loop_start_s)
        self._work_ms.append(work_s * 1000.0)
        if work_s > self.period_s:
            self._deadline_misses += 1
        if self._last_start_s is not None:
            self._period_ms.append(max(0.0, loop_start_s - self._last_start_s) * 1000.0)
        self._last_start_s = loop_start_s

    def snapshot(self) -> RuntimeMetrics:
        work = list(self._work_ms)
        periods = list(self._period_ms)
        target_ms = self.period_s * 1000.0
        jitter = [abs(value - target_ms) for value in periods]
        achieved = 1000.0 / statistics.fmean(periods) if periods else 0.0
        return RuntimeMetrics(
            target_hz=self.target_hz,
            achieved_hz=achieved,
            work_p50_ms=_percentile(work, 50.0),
            work_p95_ms=_percentile(work, 95.0),
            work_p99_ms=_percentile(work, 99.0),
            work_max_ms=max(work, default=0.0),
            jitter_p99_ms=_percentile(jitter, 99.0),
            deadline_misses=self._deadline_misses,
            samples=len(work),
        )


class _SharedArrayRing:
    """Single-writer ring with generation validation for overwrite detection."""

    def __init__(self, *, slots: int, shape: tuple[int, ...], dtype: np.dtype, create: bool = True,
                 data_name: str | None = None, generation_name: str | None = None):
        self.slots = int(slots)
        self.shape = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)
        data_bytes = self.slots * int(np.prod(self.shape)) * self.dtype.itemsize
        generation_bytes = self.slots * np.dtype(np.uint64).itemsize
        self.data_shm = SharedMemory(create=create, size=data_bytes if create else 0, name=data_name)
        self.generation_shm = SharedMemory(
            create=create,
            size=generation_bytes if create else 0,
            name=generation_name,
        )
        self.data = np.ndarray((self.slots, *self.shape), dtype=self.dtype, buffer=self.data_shm.buf)
        self.generations = np.ndarray((self.slots,), dtype=np.uint64, buffer=self.generation_shm.buf)
        if create:
            self.generations.fill(0)

    def descriptor(self) -> dict[str, Any]:
        return {
            "slots": self.slots,
            "shape": self.shape,
            "dtype": self.dtype.str,
            "data_name": self.data_shm.name,
            "generation_name": self.generation_shm.name,
        }

    @classmethod
    def attach(cls, descriptor: dict[str, Any]) -> "_SharedArrayRing":
        return cls(
            slots=descriptor["slots"],
            shape=tuple(descriptor["shape"]),
            dtype=np.dtype(descriptor["dtype"]),
            create=False,
            data_name=descriptor["data_name"],
            generation_name=descriptor["generation_name"],
        )

    def write(self, sequence: int, value: np.ndarray) -> tuple[int, int]:
        slot = int(sequence % self.slots)
        generation = int(sequence + 1)
        self.generations[slot] = np.uint64(generation * 2 - 1)
        self.data[slot][...] = value
        self.generations[slot] = np.uint64(generation * 2)
        return slot, generation * 2

    def read(self, slot: int, generation: int) -> np.ndarray | None:
        before = int(self.generations[slot])
        if before != int(generation) or before & 1:
            return None
        value = self.data[slot].copy()
        after = int(self.generations[slot])
        if after != before:
            return None
        return value

    def read_prefix(self, slot: int, generation: int, count: int) -> np.ndarray | None:
        before = int(self.generations[slot])
        if before != int(generation) or before & 1:
            return None
        value = self.data[slot][: max(0, int(count))].copy()
        after = int(self.generations[slot])
        if after != before:
            return None
        return value

    def close(self) -> None:
        self.data_shm.close()
        self.generation_shm.close()

    def unlink(self) -> None:
        for shm in (self.data_shm, self.generation_shm):
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


class ProcessRuntime:
    """Parent-side owner of isolated vision and radar workers."""

    def __init__(self, config: ProcessRuntimeConfig | None = None):
        self.config = config or ProcessRuntimeConfig()
        self._ctx = mp.get_context("spawn")
        self._vision_stop_event = self._ctx.Event()
        self._radar_stop_event = self._ctx.Event()
        self._vision_ready = self._ctx.Event()
        self._radar_ready = self._ctx.Event()
        # One-slot queues make publication genuinely latest-only. A pipe can
        # eventually fill and stall the sensor worker when the control process
        # is busy, which is precisely the coupling this runtime must avoid.
        self._vision_queue = self._ctx.Queue(maxsize=1)
        self._radar_queue = self._ctx.Queue(maxsize=1)
        self._frame_ring = _SharedArrayRing(
            slots=self.config.frame_slots,
            shape=(self.config.camera_height, self.config.camera_width, 3),
            dtype=np.uint8,
        )
        self._radar_ring = _SharedArrayRing(
            slots=self.config.radar_slots,
            shape=(self.config.radar_max_points, 2),
            dtype=np.float32,
        )
        self._vision_process: mp.Process | None = None
        self._radar_process: mp.Process | None = None
        self._latest_vision: VisionSnapshot | None = None
        self._latest_radar: RadarSnapshot | None = None
        self._started = False
        self._closed = False

    def start(self, timeout_s: float = 15.0) -> None:
        if self._closed:
            raise RuntimeError("process runtime cannot be restarted after stop()")
        if self._started:
            return
        if self.config.enable_vision:
            from .VisionProcess import vision_worker_main
            self._vision_process = self._ctx.Process(
                target=vision_worker_main,
                args=(self.config, self._frame_ring.descriptor(), self._vision_queue,
                      self._vision_ready, self._vision_stop_event),
                name="vision-worker",
                daemon=True,
            )
            self._vision_process.start()
        if self.config.enable_radar:
            from FlightController.Components.RadarProcess import radar_worker_main
            self._radar_process = self._ctx.Process(
                target=radar_worker_main,
                args=(self.config, self._radar_ring.descriptor(), self._radar_queue,
                      self._radar_ready, self._radar_stop_event),
                name="radar-worker",
                daemon=True,
            )
            self._radar_process.start()
        # Pin the parent only after spawning. Linux children inherit affinity;
        # pinning earlier would prevent the vision worker from selecting CPU1.
        _set_affinity(0, bool(self.config.apply_cpu_affinity))
        self._started = True
        deadline = time.perf_counter() + max(0.0, timeout_s)
        for enabled, ready, label in (
            (self.config.enable_vision, self._vision_ready, "vision"),
            (self.config.enable_radar, self._radar_ready, "radar"),
        ):
            while enabled and not ready.is_set():
                if time.perf_counter() >= deadline:
                    self.stop()
                    raise TimeoutError(f"{label} worker did not become ready")
                time.sleep(0.02)

    @property
    def frame_ring_descriptor(self) -> dict[str, Any]:
        return self._frame_ring.descriptor()

    def latest_vision(self) -> VisionSnapshot | None:
        self._latest_vision = _drain_latest(self._vision_queue, self._latest_vision)
        return self._latest_vision

    def latest_radar(self) -> tuple[RadarSnapshot | None, np.ndarray]:
        self._latest_radar = _drain_latest(self._radar_queue, self._latest_radar)
        snapshot = self._latest_radar
        if snapshot is None:
            return None, np.empty((0, 2), dtype=np.float32)
        points = self._radar_ring.read_prefix(
            snapshot.points_slot,
            snapshot.points_generation,
            snapshot.point_count,
        )
        if points is None:
            return snapshot, np.empty((0, 2), dtype=np.float32)
        return snapshot, points

    def read_frame(self, frame_ref: FrameRef | None) -> np.ndarray | None:
        if frame_ref is None:
            return None
        frame = self._frame_ring.read(frame_ref.slot, frame_ref.generation)
        if frame is None:
            return None
        return frame.reshape(frame_ref.shape)

    def health(self, now_s: float | None = None) -> RuntimeHealth:
        now_s = time.perf_counter() if now_s is None else now_s
        vision = self.latest_vision()
        radar, _ = self.latest_radar()
        return RuntimeHealth(
            vision_alive=bool(self._vision_process is None or self._vision_process.is_alive()),
            radar_alive=bool(self._radar_process is None or self._radar_process.is_alive()),
            vision_ready=bool(not self.config.enable_vision or self._vision_ready.is_set()),
            radar_ready=bool(not self.config.enable_radar or self._radar_ready.is_set()),
            vision_age_s=vision.age_s(now_s) if vision is not None else float("inf"),
            radar_age_s=radar.age_s(now_s) if radar is not None else float("inf"),
            vision_publish_drops=vision.publish_drops if vision is not None else 0,
            radar_publish_drops=radar.publish_drops if radar is not None else 0,
            vision_queue_depth=_queue_depth(self._vision_queue),
            radar_queue_depth=_queue_depth(self._radar_queue),
        )

    def stop(self) -> None:
        if self._closed:
            return
        self.stop_workers()
        for output_queue in (self._vision_queue, self._radar_queue):
            try:
                output_queue.close()
                output_queue.join_thread()
            except (OSError, ValueError):
                pass
        self._frame_ring.close()
        self._radar_ring.close()
        self._frame_ring.unlink()
        self._radar_ring.unlink()
        self._started = False
        self._closed = True

    def stop_workers(self) -> None:
        """Stop sensor producers while keeping shared rings readable.

        Recorder shutdown calls this first, drains its queued FrameRef jobs,
        and only then calls :meth:`stop` to unlink the shared memory.
        """
        self.stop_vision_worker()
        self.stop_radar_worker()

    def stop_vision_worker(self) -> None:
        self._vision_stop_event.set()
        _join_process(self._vision_process)

    def stop_radar_worker(self) -> None:
        self._radar_stop_event.set()
        _join_process(self._radar_process)

    def __enter__(self) -> "ProcessRuntime":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class ProcessVisionPipeline:
    """Compatibility view matching :class:`perception_pipeline.PerceptionPipeline`."""

    def __init__(self, runtime: ProcessRuntime, stale_timeout_s: float = 1.0):
        self.runtime = runtime
        self.stale_timeout_s = float(stale_timeout_s)
        self._last: VisionSnapshot | None = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.runtime.stop_vision_worker()

    def latest_perception(self):
        snapshot = self.runtime.latest_vision()
        if snapshot is not None:
            self._last = snapshot
        if self._last is None:
            return None, float("inf"), True
        age = self._last.age_s()
        return self._last.perception, age, age > self.stale_timeout_s

    def latest_frame(self):
        snapshot = self.runtime.latest_vision()
        if snapshot is not None:
            self._last = snapshot
        if self._last is None or self._last.frame_ref is None:
            return None, 0.0
        return self._last.frame_ref, self._last.frame_ref.capture_time_s

    @property
    def camera_ok(self) -> bool:
        snapshot = self.runtime.latest_vision()
        if snapshot is not None:
            self._last = snapshot
        return bool(self._last is not None and self._last.camera_ok)


class ProcessRadarClient:
    """Compatibility view matching the MultiRadar APIs used by control code."""

    def __init__(self, runtime: ProcessRuntime, max_age_s: float = 0.5):
        self.runtime = runtime
        self.max_age_s = float(max_age_s)
        self._last_snapshot: RadarSnapshot | None = None
        self._last_points = np.empty((0, 2), dtype=np.float32)
        self._points_valid = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.runtime.stop_radar_worker()

    def _refresh(self) -> None:
        snapshot, points = self.runtime.latest_radar()
        if snapshot is not None:
            self._last_snapshot = snapshot
            if len(points) or snapshot.point_count == 0:
                self._last_points = points
                self._points_valid = True
            else:
                self._points_valid = False

    @property
    def running(self) -> bool:
        return bool(self.runtime._radar_process is not None and self.runtime._radar_process.is_alive())

    @property
    def connected(self) -> bool:
        self._refresh()
        return bool(
            self.running
            and self._last_snapshot is not None
            and self._last_snapshot.connected
        )

    def is_fresh(self, max_age_s: float = 0.5, now_s: float | None = None) -> bool:
        self._refresh()
        return bool(
            self.running
            and self._last_snapshot is not None
            and self._last_snapshot.connected
            and self._points_valid
            and self._last_snapshot.crc_errors == 0
            and self._last_snapshot.age_s(now_s) <= float(max_age_s)
        )

    def get_obstacle_points_body_cm(self, max_distance_cm: float | None = None) -> np.ndarray:
        self._refresh()
        points = self._last_points.copy()
        if max_distance_cm is not None and len(points):
            squared = np.einsum("ij,ij->i", points, points)
            points = points[squared <= float(max_distance_cm) ** 2]
        return points

    def get_health_snapshot(self, now_s: float | None = None, max_age_s: float = 0.5) -> dict[str, Any]:
        self._refresh()
        if self._last_snapshot is None:
            return {"connected": False, "fresh": False, "max_age_s": max_age_s, "radars": []}
        return {
            "connected": bool(self.running and self._last_snapshot.connected),
            "fresh": bool(
                self.running
                and self._last_snapshot.connected
                and self._points_valid
                and self._last_snapshot.crc_errors == 0
                and self._last_snapshot.age_s(now_s) <= max_age_s
            ),
            "max_age_s": max_age_s,
            "radars": list(self._last_snapshot.radar_health),
            "sequence": self._last_snapshot.sequence,
            "crc_errors": self._last_snapshot.crc_errors,
            "parse_buffer_bytes": self._last_snapshot.parse_buffer_bytes,
        }


def _drain_latest(output_queue, current):
    while True:
        try:
            current = output_queue.get_nowait()
        except queue.Empty:
            break
        except (EOFError, OSError, ValueError):
            break
    return current


def _send_latest(output_queue, value: Any) -> int:
    """Best-effort latest-only publish without blocking a sensor worker."""
    try:
        output_queue.put_nowait(value)
        return 0
    except queue.Full:
        pass
    except (BrokenPipeError, EOFError, OSError, ValueError):
        return 1

    # Discard the stale queued snapshot and try exactly once. The feeder
    # thread may still report Full momentarily; dropping the new snapshot is
    # preferable to ever blocking camera/NPU or radar acquisition.
    try:
        output_queue.get_nowait()
    except (queue.Empty, EOFError, OSError, ValueError):
        return 1
    try:
        output_queue.put_nowait(value)
    except (queue.Full, BrokenPipeError, EOFError, OSError, ValueError):
        return 1
    return 1


def _set_affinity(core: int, enabled: bool) -> None:
    if not enabled or os.name != "posix" or not hasattr(os, "sched_setaffinity"):
        return
    try:
        available = sorted(os.sched_getaffinity(0))
        if len(available) > core:
            os.sched_setaffinity(0, {available[core]})
    except OSError:
        pass


def _join_process(process) -> None:
    if process is None:
        return
    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)


def _queue_depth(output_queue) -> int | None:
    try:
        return int(output_queue.qsize())
    except (AttributeError, NotImplementedError, OSError, ValueError):
        return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


__all__ = [
    "FrameRef",
    "LoopRateMonitor",
    "ProcessRuntime",
    "ProcessRuntimeConfig",
    "ProcessRadarClient",
    "ProcessVisionPipeline",
    "RadarSnapshot",
    "RuntimeHealth",
    "RuntimeMetrics",
    "VisionSnapshot",
]
