#!/usr/bin/env python3
"""Poll system resources on STM32MP257 and log to a CSV file.

Usage (board-side)::

    # Log every 1 second, until Ctrl+C
    python FlightController/tools/monitor_resources.py --interval 1 monitor.csv

    # Run in background and kill after the main test finishes
    python FlightController/tools/monitor_resources.py -i 2 monitor.csv &
    MONITOR_PID=$!
    python road_follow_main.py --no-fc --dry-run --loop-hz 5.0
    kill $MONITOR_PID

Columns written to the CSV::

    timestamp, cpu_total_pct, cpu0_pct, cpu1_pct, mem_used_mb, mem_total_mb,
    mem_pct, npu_usage, gpu_usage, load_1m
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────

def _cpu_samples() -> list[list[int]]:
    """Return list of per-CPU time fields from /proc/stat."""
    with open("/proc/stat", "r") as f:
        cpus: list[list[int]] = []
        for line in f:
            if line.startswith("cpu"):
                parts = line.strip().split()
                cpus.append([int(x) for x in parts[1:]])
        return cpus


def _cpu_diff(prev: list[list[int]], curr: list[list[int]]) -> list[float]:
    """Return per-CPU utilisation percentages between two samples."""
    result: list[float] = []
    for i in range(min(len(prev), len(curr))):
        prev_idle = prev[i][3] + (prev[i][4] if len(prev[i]) > 4 else 0)
        curr_idle = curr[i][3] + (curr[i][4] if len(curr[i]) > 4 else 0)
        prev_total = sum(prev[i])
        curr_total = sum(curr[i])
        delta_idle = max(0, curr_idle - prev_idle)
        delta_total = max(1, curr_total - prev_total)
        result.append(round(100.0 * (1.0 - delta_idle / delta_total), 1))
    return result


def _mem_info() -> dict[str, int]:
    """Read /proc/meminfo, return selected fields in kB."""
    fields = {"MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached"}
    result: dict[str, int] = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            name = line.split(":")[0]
            if name in fields:
                val = int(line.split(":")[1].strip().split()[0])
                result[name] = val
    return result


def _npu_util() -> float | None:
    """Try to read NPU utilisation percentage.

    On STM32MP257 the VIP9000 driver exposes statistics through debugfs.
    """
    candidates = [
        "/sys/kernel/debug/gc/idle",  # galcore idle status
        "/sys/devices/platform/npu/utilization",  # some BSPs
        "/sys/devices/platform/vsi_npu/utilization",
    ]
    for path in candidates:
        try:
            with open(path, "r") as f:
                val = f.read().strip()
            return float(val)
        except (OSError, ValueError):
            continue

    # Fallback: check if /dev/galcore was opened (heuristic)
    try:
        if os.path.exists("/dev/galcore"):
            return 0.0  # driver loaded but no direct util metric
    except OSError:
        pass
    return None


def _gpu_util() -> float | None:
    """Try to read GPU (GC8000 / GC7000) utilisation if exposed."""
    candidates = [
        "/sys/class/drm/card0/device/gpu_busy_percent",
        "/sys/kernel/debug/dri/0/gpu_busy",
    ]
    for path in candidates:
        try:
            with open(path, "r") as f:
                return float(f.read().strip())
        except (OSError, ValueError):
            continue
    return None


def _load_1m() -> float:
    """Return 1-minute load average."""
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return 0.0


# ── main ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output CSV path")
    parser.add_argument("-i", "--interval", type=float, default=1.0,
                        help="Poll interval in seconds (default: 1.0)")
    parser.add_argument("--no-header", action="store_true",
                        help="Omit CSV header row")
    parser.add_argument("--once", action="store_true",
                        help="Run one sample and exit (useful for spot checks)")
    args = parser.parse_args()

    fieldnames = [
        "timestamp", "cpu_total_pct", "cpu0_pct", "cpu1_pct",
        "mem_used_mb", "mem_total_mb", "mem_pct",
        "npu_usage", "gpu_usage", "load_1m",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.output, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if not args.no_header:
        writer.writeheader()

    prev_cpu = _cpu_samples()
    cycle = 0

    try:
        while True:
            cycle += 1
            tstamp = time.strftime("%H:%M:%S")

            # CPU
            curr_cpu = _cpu_samples()
            cpu_pcts = _cpu_diff(prev_cpu, curr_cpu)
            prev_cpu = curr_cpu

            cpu_total = cpu_pcts[0] if cpu_pcts else 0.0
            cpu0 = cpu_pcts[1] if len(cpu_pcts) > 1 else 0.0
            cpu1 = cpu_pcts[2] if len(cpu_pcts) > 2 else 0.0

            # Memory
            mem = _mem_info()
            total_kb = mem.get("MemTotal", 0)
            avail_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
            used_kb = total_kb - avail_kb
            used_mb = round(used_kb / 1024.0, 1)
            total_mb = round(total_kb / 1024.0, 1)
            mem_pct = round(100.0 * used_kb / max(1, total_kb), 1)

            row = {
                "timestamp": tstamp,
                "cpu_total_pct": cpu_total,
                "cpu0_pct": cpu0,
                "cpu1_pct": cpu1,
                "mem_used_mb": used_mb,
                "mem_total_mb": total_mb,
                "mem_pct": mem_pct,
                "npu_usage": _npu_util() or "",
                "gpu_usage": _gpu_util() or "",
                "load_1m": round(_load_1m(), 2),
            }
            writer.writerow(row)
            fh.flush()

            print(
                f"[{tstamp}] CPU: {cpu_total:5.1f}% | "
                f"MEM: {used_mb:5.0f}/{total_mb:.0f} MB ({mem_pct:.0f}%) | "
                f"load: {row['load_1m']:.2f}  ",
                end="\r",
            )

            if args.once:
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        fh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
