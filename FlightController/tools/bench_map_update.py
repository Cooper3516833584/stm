"""
Micro-benchmark: measure per-step cost of Map_Circle.update() on the real board.

Usage:
    PYTHONPATH=. python -u FlightController/tools/bench_map_update.py
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (root, root.parent):
        value = str(p)
        if value not in sys.path:
            sys.path.insert(0, value)


def main() -> None:
    _setup_path()

    import numpy as np
    from FlightController.Components.LDRadar_Resolver import Map_Circle, Radar_Package, Point_2D

    N_FRAMES = 300
    WARMUP = 30

    map_ = Map_Circle()
    map_.timeout_clear = True
    map_.timeout_time = 0.15

    # -------- build a realistic 12-point Radar_Package --------
    pkg = Radar_Package()
    pkg.rotation_spd = 35800  # centi-degree/s  → 358 deg/s → ~10rps
    pkg.time_stamp = 1000     # ms, D500 raw timestamp
    for i in range(12):
        deg = (i * 30.0 + 10.0) % 360.0
        pkg.points.append(Point_2D(deg, 1500.0, 200))  # 1.5m, conf=200

    # ---------- warmup ----------
    for _ in range(WARMUP):
        map_.update(pkg)

    # ---------- step-by-step timing ----------
    # We instrument a copy of update() logic to measure each phase.
    data = pkg  # alias
    timings: dict[str, list[float]] = {
        "dict_build": [],
        "dict_apply": [],
        "timeout_clear": [],
        "avail_points": [],
        "total": [],
    }

    for _ in range(N_FRAMES):
        t_total = time.perf_counter()

        # --- Phase 1: build deg_values_dict (pure Python dict) ---
        t1 = time.perf_counter()
        deg_values_dict = {}
        for point in data.points:
            if point.distance < map_.distance_threshold or point.confidence < map_.confidence_threshold:
                continue
            base = round(point.degree * map_.ACC)
            if map_.REMAP == 0:
                base %= 360 * map_.ACC
                try:
                    deg_values_dict[base].append(point.distance)
                except KeyError:
                    deg_values_dict[base] = [point.distance]
            else:
                for offset in range(-map_.REMAP, map_.REMAP + 1):
                    deg = (base + offset) % (360 * map_.ACC)
                    try:
                        deg_values_dict[deg].append(point.distance)
                    except KeyError:
                        deg_values_dict[deg] = [point.distance]
        t2 = time.perf_counter()

        # --- Phase 2: apply to data[] grid ---
        for deg, values in deg_values_dict.items():
            if map_.update_mode == map_.MODE_MIN:
                map_.data[deg] = np.min(values)
            elif map_.update_mode == map_.MODE_MAX:
                map_.data[deg] = np.max(values)
            elif map_.update_mode == map_.MODE_AVG:
                map_.data[deg] = np.round(np.mean(values))
            if map_.timeout_clear:
                map_.time_stamp[deg] = time.perf_counter()
        t3 = time.perf_counter()

        # --- Phase 3: timeout_clear (full array scan) ---
        if map_.timeout_clear:
            map_.data[map_.time_stamp < time.perf_counter() - map_.timeout_time] = -1
        t4 = time.perf_counter()

        # --- Phase 4: avail_points (full array scan) ---
        map_.avail_points = np.count_nonzero(map_.data != -1)
        t5 = time.perf_counter()

        map_.rotation_spd = data.rotation_spd / 360 * 60
        map_.update_count += 1

        timings["dict_build"].append((t2 - t1) * 1000)
        timings["dict_apply"].append((t3 - t2) * 1000)
        timings["timeout_clear"].append((t4 - t3) * 1000)
        timings["avail_points"].append((t5 - t4) * 1000)
        timings["total"].append((t5 - t_total) * 1000)

    # ---------- report ----------
    def _pct(values, p):
        s = sorted(values)
        return s[int(round((len(s) - 1) * p))]

    print()
    print(f"{'Step':<22} {'p50(ms)':>8} {'p95(ms)':>8} {'max(ms)':>8} {'sum/s':>10}")
    print("-" * 56)
    total_per_sec = 0.0
    for name in ("dict_build", "dict_apply", "timeout_clear", "avail_points", "total"):
        vals = timings[name]
        p50 = _pct(vals, 0.5)
        p95 = _pct(vals, 0.95)
        mx = max(vals)
        per_sec = sum(vals) / max(N_FRAMES, 1) * 300  # extrapolate to 300fps
        print(f"{name:<22} {p50:8.3f} {p95:8.3f} {mx:8.3f} {per_sec:8.0f}ms")
        total_per_sec += per_sec if name != "total" else 0
    print("-" * 56)
    print(f"{'extrapolated @300fps':<22} {'':>8} {'':>8} {'':>8} {total_per_sec:8.0f}ms")
    print(f"{'CPU load (single core)':<22} {'':>8} {'':>8} {'':>8} {total_per_sec/10:7.1f}%")
    print()

    # ---------- optimized mode projection ----------
    print("--- projected with timeout_clear+avail reduced to every 12th frame ---")
    base_per_frame = (_pct(timings["dict_build"], 0.5)
                      + _pct(timings["dict_apply"], 0.5)
                      + _pct(timings["timeout_clear"], 0.5) / 12
                      + _pct(timings["avail_points"], 0.5) / 12)
    per_sec_opt = base_per_frame * 300
    print(f"Projected per-frame: {base_per_frame:.3f}ms  →  @300fps: {per_sec_opt:.0f}ms/s  "
          f"CPU: {per_sec_opt/10:.1f}%")
    print()


if __name__ == "__main__":
    main()
