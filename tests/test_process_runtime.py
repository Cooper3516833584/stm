import struct
import queue

import numpy as np

from FlightController.Components.RadarProcess import (
    BatchPolarMap,
    RadarFrameParser,
    RadarTimestampTracker,
    _body_points,
)
from FlightController.Components.Utils import calculate_crc8
from FlightController.Runtime.ProcessRuntime import (
    LoopRateMonitor,
    ProcessRuntime,
    ProcessRuntimeConfig,
    _SharedArrayRing,
    _drain_latest,
    _send_latest,
)


def _frame(start_centideg: int, distances, timestamp: int = 100) -> bytes:
    values = [3600, start_centideg]
    for distance in distances:
        values.extend((int(distance), 200))
    values.extend(((start_centideg + 1100) % 36000, timestamp))
    payload = struct.pack("<HH" + "HB" * 12 + "HH", *values)
    without_crc = b"\x54\x2c" + payload
    return without_crc + bytes([calculate_crc8(without_crc)])


def test_radar_parser_handles_split_concat_garbage_and_crc_error():
    parser = RadarFrameParser()
    first = _frame(1000, range(1000, 1012), 10)
    second = _frame(2000, range(1100, 1112), 20)
    broken = bytearray(_frame(3000, range(1200, 1212), 30))
    broken[-1] ^= 0xFF

    assert parser.feed(b"garbage" + first[:13]) is None
    batch = parser.feed(first[13:] + bytes(broken) + second)

    assert batch is not None
    assert batch.frame_count == 2
    assert batch.timestamps_ms.tolist() == [10, 20]
    assert batch.distances_mm[0].tolist() == list(range(1000, 1012))
    assert parser.frames_ok == 2
    assert parser.crc_errors == 1
    assert len(parser.buffer) < 47


def test_radar_timestamp_tracker_extends_30_second_wrap():
    tracker = RadarTimestampTracker()
    extended = tracker.update_many(np.array([29980, 29995, 7, 20], dtype=np.uint16))
    assert extended.tolist() == [29980, 29995, 30007, 30020]
    assert tracker.wrap_count == 1


def test_batch_map_matches_legacy_sequential_last_frame_semantics():
    parser = RadarFrameParser()
    # Both frames touch the same bins. The second/farther frame must overwrite
    # the first rather than taking the minimum across the whole read batch.
    first = _frame(0, [1000] * 12, 10)
    second = _frame(0, [2000] * 12, 20)
    batch = parser.feed(first + second)
    assert batch is not None

    radar_map = BatchPolarMap(timeout_s=1.0)
    radar_map.update_batch(batch, now_s=5.0)
    assert radar_map.data[0] == 2000
    assert radar_map.update_count == 2


def test_batch_map_expires_without_receiving_another_frame():
    parser = RadarFrameParser()
    batch = parser.feed(_frame(0, [1000] * 12, 10))
    assert batch is not None
    radar_map = BatchPolarMap(timeout_s=0.15, cleanup_period_s=0.04)
    radar_map.update_batch(batch, now_s=5.0)
    assert np.any(radar_map.data != -1)
    radar_map.expire(now_s=5.20)
    assert not np.any(radar_map.data != -1)


def test_batch_map_bins_match_sequential_python_reference():
    frames = [
        _frame(1234, range(700, 712), 10),
        _frame(35800, range(900, 912), 20),
        _frame(1300, range(1100, 1112), 30),
    ]
    parser = RadarFrameParser()
    batch = parser.feed(b"".join(frames))
    assert batch is not None
    radar_map = BatchPolarMap(timeout_s=1.0)
    radar_map.update_batch(batch, now_s=5.0)

    reference = np.full(1080, -1, dtype=np.int32)
    for frame_index in range(batch.frame_count):
        frame_values = {}
        step = (batch.stop_degree[frame_index] - batch.start_degree[frame_index]) % 360 / 11
        for point_index in range(12):
            degree = (batch.start_degree[frame_index] + point_index * step) % 360
            base = round(degree * 3)
            distance = int(batch.distances_mm[frame_index, point_index])
            for offset in range(-2, 3):
                index = (base + offset) % 1080
                frame_values.setdefault(index, []).append(distance)
        for index, values in frame_values.items():
            reference[index] = min(values)
    np.testing.assert_array_equal(radar_map.data, reference)


def test_body_transform_matches_lower_radar_mirror_and_offset():
    radar_map = BatchPolarMap(timeout_s=1.0)
    radar_map.data[270] = 1000  # 90 degrees: radar-frame (0, -100) cm
    transformed = _body_points(
        radar_map,
        mount_xy_cm=(0.96, 0.15),
        mount_yaw_deg=0.0,
        mount_mirror_y=True,
        max_distance_cm=300.0,
    )
    assert transformed.shape == (1, 2)
    np.testing.assert_allclose(transformed[0], [0.96, 100.15], atol=1e-4)


def test_shared_ring_rejects_overwritten_generation():
    ring = _SharedArrayRing(slots=2, shape=(4,), dtype=np.int32)
    try:
        slot, generation = ring.write(0, np.asarray([1, 2, 3, 4], dtype=np.int32))
        assert ring.read(slot, generation).tolist() == [1, 2, 3, 4]
        assert ring.read_prefix(slot, generation, 2).tolist() == [1, 2]
        ring.write(2, np.asarray([5, 6, 7, 8], dtype=np.int32))
        assert ring.read(slot, generation) is None
    finally:
        ring.close()
        ring.unlink()


def test_latest_only_channel_discards_stale_item_without_blocking():
    output = queue.Queue(maxsize=1)
    assert _send_latest(output, "old") == 0
    assert _send_latest(output, "new") == 1
    assert _drain_latest(output, None) == "new"


def test_loop_rate_monitor_reports_work_and_deadlines():
    monitor = LoopRateMonitor(20.0)
    for index in range(10):
        started = index * 0.05
        monitor.record(started, started + (0.01 if index < 9 else 0.06))
    result = monitor.snapshot()
    assert result.achieved_hz == 20.0
    assert result.samples == 10
    assert result.deadline_misses == 1
    assert result.work_max_ms == 60.0


def test_runtime_without_workers_starts_and_stops_idempotently():
    runtime = ProcessRuntime(
        ProcessRuntimeConfig(enable_vision=False, enable_radar=False)
    )
    runtime.start(timeout_s=0.1)
    health = runtime.health()
    assert health.vision_ready
    assert health.radar_ready
    runtime.stop()
    runtime.stop()
    try:
        runtime.start()
    except RuntimeError:
        pass
    else:
        raise AssertionError("stopped process runtime must not restart")
