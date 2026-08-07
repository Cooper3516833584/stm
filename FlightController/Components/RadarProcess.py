"""Batch-oriented dual-radar worker and NumPy-native polar map kernel."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import time
from typing import Any

import numpy as np

from FlightController.Components.Utils import calculate_crc8
from FlightController.Runtime.ProcessRuntime import (
    RadarSnapshot,
    _SharedArrayRing,
    _send_latest,
    _set_affinity,
)


FRAME_HEADER = b"\x54\x2c"
FRAME_LENGTH = 47
PAYLOAD_STRUCT = struct.Struct("<HH" + "HB" * 12 + "HH")


@dataclass(frozen=True)
class ParsedRadarBatch:
    rotation_speed_deg_s: np.ndarray
    start_degree: np.ndarray
    stop_degree: np.ndarray
    timestamps_ms: np.ndarray
    distances_mm: np.ndarray
    confidences: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.distances_mm.shape[0])


class RadarFrameParser:
    """Incremental D500 parser accepting arbitrary split/concatenated chunks."""

    def __init__(self):
        self.buffer = bytearray()
        self.frames_ok = 0
        self.crc_errors = 0
        self.discarded_bytes = 0

    def feed(self, chunk: bytes | bytearray | memoryview) -> ParsedRadarBatch | None:
        self.buffer.extend(chunk)
        rows: list[tuple[int, ...]] = []
        while len(self.buffer) >= FRAME_LENGTH:
            index = self.buffer.find(FRAME_HEADER)
            if index < 0:
                keep = len(FRAME_HEADER) - 1
                discarded = max(0, len(self.buffer) - keep)
                self.discarded_bytes += discarded
                if discarded:
                    del self.buffer[:discarded]
                break
            if index:
                self.discarded_bytes += index
                del self.buffer[:index]
            if len(self.buffer) < FRAME_LENGTH:
                break
            frame = bytes(self.buffer[:FRAME_LENGTH])
            if calculate_crc8(frame[:-1]) != frame[-1]:
                self.crc_errors += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            try:
                rows.append(PAYLOAD_STRUCT.unpack(frame[2:-1]))
            except struct.error:
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            del self.buffer[:FRAME_LENGTH]
            self.frames_ok += 1

        if not rows:
            return None
        count = len(rows)
        speed = np.empty(count, dtype=np.float64)
        starts = np.empty(count, dtype=np.float64)
        stops = np.empty(count, dtype=np.float64)
        timestamps = np.empty(count, dtype=np.uint16)
        distances = np.empty((count, 12), dtype=np.int32)
        confidences = np.empty((count, 12), dtype=np.uint8)
        for row_index, row in enumerate(rows):
            speed[row_index] = row[0]
            starts[row_index] = row[1] * 0.01
            stops[row_index] = row[26] * 0.01
            timestamps[row_index] = row[27]
            distances[row_index] = [row[2 + point * 2] for point in range(12)]
            confidences[row_index] = [row[3 + point * 2] for point in range(12)]
        return ParsedRadarBatch(speed, starts, stops, timestamps, distances, confidences)


class RadarTimestampTracker:
    """Extend the radar's 30-second device timestamp across wraparound."""

    def __init__(self, wrap_ms: int = 30000):
        self.wrap_ms = int(wrap_ms)
        self.last_raw_ms: int | None = None
        self.wrap_offset_ms = 0
        self.wrap_count = 0

    def update_many(self, raw_timestamps: np.ndarray) -> np.ndarray:
        extended = np.empty(len(raw_timestamps), dtype=np.int64)
        half_wrap = self.wrap_ms / 2.0
        for index, raw_value in enumerate(raw_timestamps):
            raw_ms = int(raw_value)
            if self.last_raw_ms is not None:
                delta = raw_ms - self.last_raw_ms
                if delta < -half_wrap:
                    self.wrap_offset_ms += self.wrap_ms
                    self.wrap_count += 1
                elif delta > half_wrap:
                    # Device restart or an out-of-order stream: match the
                    # legacy latency tracker by resetting the unwrap origin.
                    self.wrap_offset_ms = 0
            self.last_raw_ms = raw_ms
            extended[index] = self.wrap_offset_ms + raw_ms
        return extended


class BatchPolarMap:
    """Production MODE_MIN map whose hot update path executes in NumPy C."""

    ACC = 3
    REMAP = 2
    TOTAL_BINS = 360 * ACC

    def __init__(
        self,
        *,
        confidence_threshold: int = 0,
        distance_threshold_mm: int = 10,
        timeout_s: float = 0.15,
        cleanup_period_s: float = 0.04,
    ):
        self.confidence_threshold = int(confidence_threshold)
        self.distance_threshold_mm = int(distance_threshold_mm)
        self.timeout_s = float(timeout_s)
        self.cleanup_period_s = min(0.04, max(0.001, float(cleanup_period_s)))
        self.data = np.full(self.TOTAL_BINS, -1, dtype=np.int32)
        self.time_stamp = np.zeros(self.TOTAL_BINS, dtype=np.float64)
        radians = np.deg2rad(np.arange(self.TOTAL_BINS, dtype=np.float32) / self.ACC)
        self.cos_arr = np.cos(radians).astype(np.float32)
        self.sin_arr = np.sin(radians).astype(np.float32)
        self.rotation_rpm = 0.0
        self.update_count = 0
        self._last_cleanup_s = 0.0
        self._offsets = np.arange(-self.REMAP, self.REMAP + 1, dtype=np.int32)
        self._point_numbers = np.arange(12, dtype=np.float64)
        self._sentinel = np.iinfo(np.int32).max
        self._scratch = np.full(self.TOTAL_BINS, self._sentinel, dtype=np.int32)

    def update_batch(self, batch: ParsedRadarBatch, now_s: float | None = None) -> None:
        if batch.frame_count == 0:
            return
        now_s = time.perf_counter() if now_s is None else float(now_s)
        steps = ((batch.stop_degree - batch.start_degree) % 360.0) / 11.0
        degrees = (
            batch.start_degree[:, None]
            + steps[:, None] * self._point_numbers[None, :]
        ) % 360.0
        valid = (
            (batch.distances_mm >= self.distance_threshold_mm)
            & (batch.confidences >= self.confidence_threshold)
        )
        if np.any(valid):
            bases_all = np.rint(degrees * self.ACC).astype(np.int32)
            # Preserve sequential-frame overwrite semantics without allocating
            # and scanning a frame_count x 1080 matrix for every serial read.
            # The only Python loop is per complete 47-byte frame; point/bin
            # expansion and MODE_MIN reduction remain in NumPy C.
            for frame_index in range(batch.frame_count):
                frame_valid = valid[frame_index]
                if not np.any(frame_valid):
                    continue
                bases = bases_all[frame_index, frame_valid]
                values = batch.distances_mm[frame_index, frame_valid]
                indices = (
                    (bases[:, None] + self._offsets[None, :]) % self.TOTAL_BINS
                ).reshape(-1)
                mapped_values = np.repeat(values, len(self._offsets)).astype(
                    np.int32, copy=False
                )
                touched = np.unique(indices)
                np.minimum.at(self._scratch, indices, mapped_values)
                self.data[touched] = self._scratch[touched]
                self.time_stamp[touched] = now_s
                self._scratch[touched] = self._sentinel
        self.rotation_rpm = float(batch.rotation_speed_deg_s[-1] / 360.0 * 60.0)
        self.update_count += batch.frame_count
        self.expire(now_s)

    def expire(self, now_s: float | None = None) -> None:
        """Clear stale bins at most once per 40 ms, even if input stops."""
        now_s = time.perf_counter() if now_s is None else float(now_s)
        if now_s - self._last_cleanup_s >= self.cleanup_period_s:
            self.data[self.time_stamp < now_s - self.timeout_s] = -1
            self._last_cleanup_s = now_s

    def points_xy_cm(self, max_distance_cm: float | None = None) -> np.ndarray:
        selected = self.data != -1
        if not np.any(selected):
            return np.empty((0, 2), dtype=np.float32)
        distances_cm = self.data[selected].astype(np.float32) * 0.1
        points = np.column_stack(
            (distances_cm * self.cos_arr[selected], -distances_cm * self.sin_arr[selected])
        ).astype(np.float32, copy=False)
        if max_distance_cm is not None:
            squared = np.einsum("ij,ij->i", points, points)
            points = points[squared <= float(max_distance_cm) ** 2]
        return points


def _body_points(
    radar_map: BatchPolarMap,
    *,
    mount_xy_cm: tuple[float, float],
    mount_yaw_deg: float,
    mount_mirror_y: bool,
    max_distance_cm: float | None,
) -> np.ndarray:
    points = radar_map.points_xy_cm(max_distance_cm=max_distance_cm)
    if not len(points):
        return points
    if mount_mirror_y:
        points[:, 1] *= -1.0
    radians = np.deg2rad(float(mount_yaw_deg))
    rotation = np.asarray(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]],
        dtype=np.float32,
    )
    return points @ rotation.T + np.asarray(mount_xy_cm, dtype=np.float32)


def radar_worker_main(config, radar_ring_descriptor, output, ready_event, stop_event) -> None:
    """Own both serial devices and publish one latest merged point cloud."""
    _set_affinity(0, bool(config.apply_cpu_affinity))
    ring = _SharedArrayRing.attach(radar_ring_descriptor)
    serials = []
    states: list[dict[str, Any]] = []
    mounts = [
        ((0.0, 0.0), 0.0, False),
        ((0.96, 0.15), 0.0, True),
    ]
    try:
        import serial

        for name, port in (("upper", config.upper_port), ("lower", config.lower_port)):
            device = serial.Serial(port, baudrate=230400, timeout=0)
            serials.append(device)
            states.append(
                {
                    "name": name,
                    "parser": RadarFrameParser(),
                    "map": BatchPolarMap(),
                    "last_frame_time_s": 0.0,
                    "bytes_read": 0,
                    "read_batches": 0,
                    "in_waiting_peak": 0,
                    "last_device_timestamp_ms": None,
                    "last_device_timestamp_extended_ms": None,
                    "timestamp_tracker": RadarTimestampTracker(),
                }
            )
        ready_event.set()
        sequence = 0
        publish_drops = 0
        publish_period_s = 1.0 / max(1.0, float(config.radar_publish_hz))
        next_publish_s = time.perf_counter()
        while not stop_event.is_set():
            did_work = False
            for device, state in zip(serials, states):
                waiting = int(device.in_waiting)
                state["in_waiting_peak"] = max(state["in_waiting_peak"], waiting)
                if waiting <= 0:
                    continue
                chunk = device.read(waiting)
                read_time_s = time.perf_counter()
                state["bytes_read"] += len(chunk)
                state["read_batches"] += 1
                batch = state["parser"].feed(chunk)
                if batch is not None:
                    state["map"].update_batch(batch, read_time_s)
                    state["last_frame_time_s"] = read_time_s
                    state["last_device_timestamp_ms"] = int(batch.timestamps_ms[-1])
                    extended = state["timestamp_tracker"].update_many(batch.timestamps_ms)
                    state["last_device_timestamp_extended_ms"] = int(extended[-1])
                did_work = True

            now_s = time.perf_counter()
            if now_s >= next_publish_s:
                point_sets = []
                for state, (mount_xy, mount_yaw, mirror) in zip(states, mounts):
                    state["map"].expire(now_s)
                    point_sets.append(
                        _body_points(
                            state["map"],
                            mount_xy_cm=mount_xy,
                            mount_yaw_deg=mount_yaw,
                            mount_mirror_y=mirror,
                            max_distance_cm=300.0,
                        )
                    )
                merged = np.vstack([points for points in point_sets if len(points)]).astype(
                    np.float32, copy=False
                ) if any(len(points) for points in point_sets) else np.empty((0, 2), dtype=np.float32)
                max_points = int(radar_ring_descriptor["shape"][0])
                if len(merged) > max_points:
                    merged = merged[:max_points]
                sequence += 1
                slot, generation = ring.write_prefix(sequence, merged)
                latest_frame_s = min((float(state["last_frame_time_s"]) for state in states), default=0.0)
                health = tuple(
                    {
                        "name": state["name"],
                        "connected": state["last_frame_time_s"] > 0.0,
                        "last_frame_age_s": (
                            max(0.0, now_s - state["last_frame_time_s"])
                            if state["last_frame_time_s"] > 0.0 else None
                        ),
                        "frames_ok_total": state["parser"].frames_ok,
                        "crc_errors": state["parser"].crc_errors,
                        "parse_buffer_bytes": len(state["parser"].buffer),
                        "in_waiting_peak": state["in_waiting_peak"],
                        "serial_bytes_read": state["bytes_read"],
                        "serial_read_batches": state["read_batches"],
                        "last_device_timestamp_ms": state["last_device_timestamp_ms"],
                        "last_device_timestamp_extended_ms": (
                            state["last_device_timestamp_extended_ms"]
                        ),
                        "timestamp_wrap_count": state["timestamp_tracker"].wrap_count,
                    }
                    for state in states
                )
                connected = all(item["connected"] for item in health)
                fresh = bool(
                    connected
                    and all(
                        item["last_frame_age_s"] is not None
                        and item["last_frame_age_s"] <= config.radar_timeout_s
                        for item in health
                    )
                )
                publish_drops += _send_latest(
                    output,
                    RadarSnapshot(
                        sequence=sequence,
                        published_time_s=now_s,
                        last_frame_time_s=latest_frame_s,
                        points_slot=slot,
                        points_generation=generation,
                        point_count=len(merged),
                        connected=connected,
                        fresh=fresh,
                        radar_health=health,
                        crc_errors=sum(item["crc_errors"] for item in health),
                        parse_buffer_bytes=sum(item["parse_buffer_bytes"] for item in health),
                        publish_drops=publish_drops,
                    ),
                )
                while next_publish_s <= now_s:
                    next_publish_s += publish_period_s
            if not did_work:
                time.sleep(0.001)
    finally:
        ready_event.clear()
        for device in serials:
            try:
                device.close()
            except Exception:
                pass
        ring.close()
        try:
            output.close()
        except (OSError, ValueError):
            pass


__all__ = [
    "BatchPolarMap",
    "ParsedRadarBatch",
    "RadarFrameParser",
    "RadarTimestampTracker",
    "radar_worker_main",
]
