"""
Measure raw D500 radar serial latency with the radar packet timestamp.

This tool bypasses Map_Circle and the avoidance planner. It reads D500 frames
directly from the UART, unwraps the radar's 30s packet timestamp, and reports
how far each frame is behind the best observed host/radar clock offset.
"""

from __future__ import annotations

import argparse
import statistics
import struct
import sys
import time
from collections import deque
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (root, root.parent):
        value = str(p)
        if value not in sys.path:
            sys.path.insert(0, value)


class RadarTimestampLatency:
    WRAP_MS = 30000.0

    def __init__(self) -> None:
        self.last_raw_ms: int | None = None
        self.wrap_offset_ms = 0.0
        self.min_delta_ms: float | None = None
        self.first_host_ms: float | None = None
        self.first_device_ms: float | None = None
        self.last_host_ms = 0.0
        self.last_device_ms = 0.0
        self.latest_ms = 0.0
        self.max_ms = 0.0
        self.samples = 0

    def update(self, raw_timestamp_ms: int, host_time_s: float) -> float:
        raw_timestamp_ms = int(raw_timestamp_ms)
        host_ms = host_time_s * 1000.0
        if self.last_raw_ms is not None:
            delta_raw = raw_timestamp_ms - self.last_raw_ms
            if delta_raw < -self.WRAP_MS / 2:
                self.wrap_offset_ms += self.WRAP_MS
            elif delta_raw > self.WRAP_MS / 2:
                self.wrap_offset_ms = 0.0
                self.min_delta_ms = None
                self.first_host_ms = None
                self.first_device_ms = None

        device_ms = self.wrap_offset_ms + raw_timestamp_ms
        delta_ms = host_ms - device_ms
        if self.min_delta_ms is None or delta_ms < self.min_delta_ms:
            self.min_delta_ms = delta_ms
        latency_ms = max(0.0, delta_ms - self.min_delta_ms)

        if self.first_host_ms is None:
            self.first_host_ms = host_ms
            self.first_device_ms = device_ms

        self.last_raw_ms = raw_timestamp_ms
        self.last_host_ms = host_ms
        self.last_device_ms = device_ms
        self.latest_ms = latency_ms
        self.max_ms = max(self.max_ms, latency_ms)
        self.samples += 1
        return latency_ms

    def device_rate_pct(self) -> float:
        if self.first_host_ms is None or self.first_device_ms is None:
            return 0.0
        host_elapsed = self.last_host_ms - self.first_host_ms
        device_elapsed = self.last_device_ms - self.first_device_ms
        return device_elapsed / host_elapsed * 100.0 if host_elapsed > 0 else 0.0

    def clock_drift_ms(self) -> float:
        if self.first_host_ms is None or self.first_device_ms is None:
            return 0.0
        host_elapsed = self.last_host_ms - self.first_host_ms
        device_elapsed = self.last_device_ms - self.first_device_ms
        return host_elapsed - device_elapsed


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * pct))
    return values[idx]


def _frame_timestamp_ms(frame: bytes) -> int:
    return struct.unpack_from("<H", frame, 44)[0]


def _frame_rpm(frame: bytes) -> float:
    rotation_spd = struct.unpack_from("<H", frame, 2)[0]
    return rotation_spd / 360.0 * 60.0


def main() -> None:
    _setup_path()

    import serial
    from FlightController.Components.Utils import calculate_crc8

    parser = argparse.ArgumentParser(description="Diagnose raw D500 radar UART latency.")
    parser.add_argument("--port", default="/dev/ttySTM4", help="Radar serial device path.")
    parser.add_argument("--baudrate", type=int, default=230400, help="Radar UART baudrate.")
    parser.add_argument("--duration", type=float, default=180.0, help="Run duration in seconds; 0 means forever.")
    parser.add_argument("--report-interval", type=float, default=1.0, help="Report interval in seconds.")
    parser.add_argument(
        "--idle-sleep-ms",
        type=float,
        default=0.0,
        help="Sleep when no bytes are waiting. Use 1.0 to mimic the current driver on the board.",
    )
    args = parser.parse_args()

    start_bit = b"\x54\x2C"
    frame_length = 47
    buf = b""
    model = RadarTimestampLatency()
    window: deque[float] = deque(maxlen=5000)

    valid_frames = 0
    crc_errors = 0
    dropped_bytes = 0
    read_batches = 0
    bytes_read = 0
    in_waiting_peak = 0
    last_valid = 0
    last_crc = 0
    last_bytes = 0
    last_report = time.perf_counter()
    start = last_report
    latest_rpm = 0.0

    print(
        f"[RAW_LATENCY] start port={args.port} baudrate={args.baudrate} "
        f"idle_sleep_ms={args.idle_sleep_ms}",
        flush=True,
    )

    with serial.Serial(args.port, baudrate=args.baudrate, timeout=0) as ser:
        try:
            while args.duration <= 0 or time.perf_counter() - start < args.duration:
                waiting = ser.in_waiting
                if waiting > 0:
                    in_waiting_peak = max(in_waiting_peak, waiting)
                    chunk = ser.read(waiting)
                    host_read_time = time.perf_counter()
                    read_batches += 1
                    bytes_read += len(chunk)
                    buf += chunk

                    while len(buf) >= frame_length:
                        idx = buf.find(start_bit)
                        if idx == -1:
                            keep = len(start_bit) - 1
                            dropped_bytes += max(0, len(buf) - keep)
                            buf = buf[-keep:] if len(buf) >= keep else buf
                            break
                        if idx > 0:
                            dropped_bytes += idx
                            buf = buf[idx:]
                        frame = buf[:frame_length]
                        buf = buf[frame_length:]
                        if calculate_crc8(frame[:-1]) != frame[-1]:
                            crc_errors += 1
                            continue
                        raw_ts = _frame_timestamp_ms(frame)
                        latest_rpm = _frame_rpm(frame)
                        latency_ms = model.update(raw_ts, host_read_time)
                        window.append(latency_ms)
                        valid_frames += 1
                elif args.idle_sleep_ms > 0:
                    time.sleep(args.idle_sleep_ms / 1000.0)
                else:
                    time.sleep(0)

                now = time.perf_counter()
                if now - last_report >= args.report_interval:
                    elapsed = now - last_report
                    values = list(window)
                    valid_rate = (valid_frames - last_valid) / elapsed
                    crc_rate = (crc_errors - last_crc) / elapsed
                    byte_rate = (bytes_read - last_bytes) / elapsed
                    last_valid = valid_frames
                    last_crc = crc_errors
                    last_bytes = bytes_read
                    last_report = now

                    if values:
                        p50 = statistics.median(values)
                        p95 = _percentile(values, 0.95)
                        interval_max = max(values)
                        window.clear()
                    else:
                        p50 = p95 = interval_max = 0.0

                    print(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [RAW_LATENCY] "
                        f"run={now-start:.0f}s valid={valid_rate:.0f}/s crc={crc_rate:.1f}/s "
                        f"bytes={byte_rate:.0f}/s rpm={latest_rpm:.1f} | "
                        f"age_ms latest={model.latest_ms:.1f} p50={p50:.1f} "
                        f"p95={p95:.1f} max={interval_max:.1f} global_max={model.max_ms:.1f} | "
                        f"dev_rate={model.device_rate_pct():.2f}% drift={model.clock_drift_ms():.1f}ms | "
                        f"in_waiting_peak={in_waiting_peak}B parse_buf={len(buf)}B "
                        f"dropped={dropped_bytes}B total_valid={valid_frames} total_crc={crc_errors}",
                        flush=True,
                    )
                    in_waiting_peak = 0
        except KeyboardInterrupt:
            pass

    print(
        f"[RAW_LATENCY] done run={time.perf_counter()-start:.1f}s "
        f"valid={valid_frames} crc={crc_errors} max_age_ms={model.max_ms:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
