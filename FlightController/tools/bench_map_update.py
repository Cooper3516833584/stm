"""
Incremental micro-benchmark: measure the impact of each dict_apply / dict_build
optimization across 6 variants.

Usage:
    PYTHONPATH=. python -u FlightController/tools/bench_map_update.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (root, root.parent):
        value = str(p)
        if value not in sys.path:
            sys.path.insert(0, value)


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------
N_FRAMES = 300
WARMUP = 30
ACC = 3          # 360 * 3 = 1080 bins
REMAP = 2        # ±2° mapping
TOTAL_BINS = 360 * ACC  # 1080


def _build_input_frames(n: int) -> list[list[tuple[int, int]]]:
    """Return `n` frames, each a list of (degree_x10, distance_mm).  Degrees
    rotate 1.0° per frame to exercise fresh dict keys every frame."""
    frames = []
    for f in range(n):
        pts = []
        base_deg = (f * 10) % 3600  # degrees × 10
        for i in range(12):
            d = base_deg + i * 300  # 30° spacing
            dist = 1500 + (i * 23) % 500
            pts.append((d % 3600, dist))
        frames.append(pts)
    return frames


# ---------------------------------------------------------------------------
# Variant implementations — each returns (dict_build_ms, dict_apply_ms)
# ---------------------------------------------------------------------------

def _v_baseline(frames: list[list[tuple[int, int]]]) -> tuple[list[float], list[float]]:
    import numpy as np
    data = np.ones(TOTAL_BINS, dtype=np.int64) * -1
    ts_arr = np.zeros(TOTAL_BINS, dtype=np.float64)

    build_times, apply_times = [], []
    for pts in frames:
        # dict_build
        t0 = time.perf_counter()
        deg_values_dict: dict[int, list[int]] = {}
        for deg_x10, dist in pts:
            base = round(deg_x10 / 10 * ACC)
            if REMAP == 0:
                base %= TOTAL_BINS
                try:
                    deg_values_dict[base].append(dist)
                except KeyError:
                    deg_values_dict[base] = [dist]
            else:
                for offset in range(-REMAP, REMAP + 1):
                    deg = (base + offset) % TOTAL_BINS
                    try:
                        deg_values_dict[deg].append(dist)
                    except KeyError:
                        deg_values_dict[deg] = [dist]
        t1 = time.perf_counter()

        # dict_apply
        for deg, values in deg_values_dict.items():
            data[deg] = np.min(values)
            ts_arr[deg] = time.perf_counter()
        t2 = time.perf_counter()

        build_times.append((t1 - t0) * 1000)
        apply_times.append((t2 - t1) * 1000)

    return build_times, apply_times


def _v_apply_builtins(frames: list[list[tuple[int, int]]]) -> tuple[list[float], list[float]]:
    import numpy as np
    data = np.ones(TOTAL_BINS, dtype=np.int64) * -1
    ts_arr = np.zeros(TOTAL_BINS, dtype=np.float64)

    build_times, apply_times = [], []
    for pts in frames:
        # dict_build (same as baseline)
        t0 = time.perf_counter()
        deg_values_dict: dict[int, list[int]] = {}
        for deg_x10, dist in pts:
            base = round(deg_x10 / 10 * ACC)
            if REMAP == 0:
                base %= TOTAL_BINS
                try:
                    deg_values_dict[base].append(dist)
                except KeyError:
                    deg_values_dict[base] = [dist]
            else:
                for offset in range(-REMAP, REMAP + 1):
                    deg = (base + offset) % TOTAL_BINS
                    try:
                        deg_values_dict[deg].append(dist)
                    except KeyError:
                        deg_values_dict[deg] = [dist]
        t1 = time.perf_counter()

        # dict_apply: Python builtins
        for deg, values in deg_values_dict.items():
            data[deg] = min(values)
            ts_arr[deg] = time.perf_counter()
        t2 = time.perf_counter()

        build_times.append((t1 - t0) * 1000)
        apply_times.append((t2 - t1) * 1000)

    return build_times, apply_times


def _v_apply_one_ts(frames: list[list[tuple[int, int]]]) -> tuple[list[float], list[float]]:
    import numpy as np
    data = np.ones(TOTAL_BINS, dtype=np.int64) * -1
    ts_arr = np.zeros(TOTAL_BINS, dtype=np.float64)

    build_times, apply_times = [], []
    for pts in frames:
        # dict_build (same as baseline)
        t0 = time.perf_counter()
        deg_values_dict: dict[int, list[int]] = {}
        for deg_x10, dist in pts:
            base = round(deg_x10 / 10 * ACC)
            if REMAP == 0:
                base %= TOTAL_BINS
                try:
                    deg_values_dict[base].append(dist)
                except KeyError:
                    deg_values_dict[base] = [dist]
            else:
                for offset in range(-REMAP, REMAP + 1):
                    deg = (base + offset) % TOTAL_BINS
                    try:
                        deg_values_dict[deg].append(dist)
                    except KeyError:
                        deg_values_dict[deg] = [dist]
        t1 = time.perf_counter()

        # dict_apply: single time.perf_counter()
        _now = time.perf_counter()
        for deg, values in deg_values_dict.items():
            data[deg] = np.min(values)
            ts_arr[deg] = _now
        t2 = time.perf_counter()

        build_times.append((t1 - t0) * 1000)
        apply_times.append((t2 - t1) * 1000)

    return build_times, apply_times


def _v_apply_both(frames: list[list[tuple[int, int]]]) -> tuple[list[float], list[float]]:
    import numpy as np
    data = np.ones(TOTAL_BINS, dtype=np.int64) * -1
    ts_arr = np.zeros(TOTAL_BINS, dtype=np.float64)

    build_times, apply_times = [], []
    for pts in frames:
        # dict_build (same as baseline)
        t0 = time.perf_counter()
        deg_values_dict: dict[int, list[int]] = {}
        for deg_x10, dist in pts:
            base = round(deg_x10 / 10 * ACC)
            if REMAP == 0:
                base %= TOTAL_BINS
                try:
                    deg_values_dict[base].append(dist)
                except KeyError:
                    deg_values_dict[base] = [dist]
            else:
                for offset in range(-REMAP, REMAP + 1):
                    deg = (base + offset) % TOTAL_BINS
                    try:
                        deg_values_dict[deg].append(dist)
                    except KeyError:
                        deg_values_dict[deg] = [dist]
        t1 = time.perf_counter()

        # dict_apply: Python builtins + single timestamp
        _now = time.perf_counter()
        for deg, values in deg_values_dict.items():
            data[deg] = min(values)
            ts_arr[deg] = _now
        t2 = time.perf_counter()

        build_times.append((t1 - t0) * 1000)
        apply_times.append((t2 - t1) * 1000)

    return build_times, apply_times


def _v_build_setdefault(frames: list[list[tuple[int, int]]]) -> tuple[list[float], list[float]]:
    import numpy as np
    data = np.ones(TOTAL_BINS, dtype=np.int64) * -1
    ts_arr = np.zeros(TOTAL_BINS, dtype=np.float64)

    build_times, apply_times = [], []
    for pts in frames:
        # dict_build: setdefault
        t0 = time.perf_counter()
        deg_values_dict: dict[int, list[int]] = {}
        for deg_x10, dist in pts:
            base = round(deg_x10 / 10 * ACC)
            if REMAP == 0:
                base %= TOTAL_BINS
                deg_values_dict.setdefault(base, []).append(dist)
            else:
                for offset in range(-REMAP, REMAP + 1):
                    deg = (base + offset) % TOTAL_BINS
                    deg_values_dict.setdefault(deg, []).append(dist)
        t1 = time.perf_counter()

        # dict_apply (same as baseline)
        for deg, values in deg_values_dict.items():
            data[deg] = np.min(values)
            ts_arr[deg] = time.perf_counter()
        t2 = time.perf_counter()

        build_times.append((t1 - t0) * 1000)
        apply_times.append((t2 - t1) * 1000)

    return build_times, apply_times


def _v_full(frames: list[list[tuple[int, int]]]) -> tuple[list[float], list[float]]:
    import numpy as np
    data = np.ones(TOTAL_BINS, dtype=np.int64) * -1
    ts_arr = np.zeros(TOTAL_BINS, dtype=np.float64)

    build_times, apply_times = [], []
    for pts in frames:
        # dict_build: setdefault
        t0 = time.perf_counter()
        deg_values_dict: dict[int, list[int]] = {}
        for deg_x10, dist in pts:
            base = round(deg_x10 / 10 * ACC)
            if REMAP == 0:
                base %= TOTAL_BINS
                deg_values_dict.setdefault(base, []).append(dist)
            else:
                for offset in range(-REMAP, REMAP + 1):
                    deg = (base + offset) % TOTAL_BINS
                    deg_values_dict.setdefault(deg, []).append(dist)
        t1 = time.perf_counter()

        # dict_apply: Python builtins + single timestamp
        _now = time.perf_counter()
        for deg, values in deg_values_dict.items():
            data[deg] = min(values)
            ts_arr[deg] = _now
        t2 = time.perf_counter()

        build_times.append((t1 - t0) * 1000)
        apply_times.append((t2 - t1) * 1000)

    return build_times, apply_times


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[int(round((len(s) - 1) * p))]


def _run_variant(name: str, fn, frames, warmup_idx: int) -> list[dict]:
    """Run variant, return per-step rows for reporting."""
    # warmup
    fn(frames[:warmup_idx])
    # measure
    build, apply = fn(frames[warmup_idx:])
    total = [b + a for b, a in zip(build, apply)]
    n = len(build)

    def _row(step: str, vals: list[float]) -> dict:
        p50 = _pct(vals, 0.5)
        p95 = _pct(vals, 0.95)
        mx = max(vals)
        per_sec = sum(vals) / n * 300  # extrapolate to 300fps
        return {
            "variant": name,
            "step": step,
            "p50": p50,
            "p95": p95,
            "max": mx,
            "sum_s": per_sec,
        }

    return [
        _row("dict_build", build),
        _row("dict_apply", apply),
        _row("total", total),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANTS = [
    ("BASELINE", _v_baseline),
    ("APPLY_BUILTINS", _v_apply_builtins),
    ("APPLY_ONE_TS", _v_apply_one_ts),
    ("APPLY_BOTH", _v_apply_both),
    ("BUILD_SETDEFAULT", _v_build_setdefault),
    ("FULL_OPT", _v_full),
]


def main() -> None:
    _setup_path()

    print(f"\n{'='*80}")
    print(f"  map.update() incremental benchmark  ({N_FRAMES} frames + {WARMUP} warmup)")
    print(f"{'='*80}\n")

    frames_all = _build_input_frames(N_FRAMES + WARMUP)

    all_rows = []
    for name, fn in VARIANTS:
        rows = _run_variant(name, fn, frames_all, WARMUP)
        all_rows.extend(rows)
        p50_total = rows[-1]["p50"]
        cpu = rows[-1]["sum_s"] / 10
        print(f"  {name:<18}  total p50={p50_total:7.3f}ms  @300fps={rows[-1]['sum_s']:5.0f}ms/s  CPU={cpu:5.1f}%")
    print()

    # Detailed table
    header = f"  {'Variant':<18} {'Step':<12} {'p50(ms)':>8} {'p95(ms)':>8} {'max(ms)':>8} {'sum/s':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    baseline_total = None
    for row in all_rows:
        line = f"  {row['variant']:<18} {row['step']:<12} {row['p50']:8.3f} {row['p95']:8.3f} {row['max']:8.3f} {row['sum_s']:8.0f}"
        print(line)
        if row["variant"] == "BASELINE" and row["step"] == "total":
            baseline_total = row
    print()

    # Savings summary
    if baseline_total:
        full_row = [r for r in all_rows if r["variant"] == "FULL_OPT" and r["step"] == "total"][0]
        save_ms = baseline_total["p50"] - full_row["p50"]
        save_pct = save_ms / baseline_total["p50"] * 100 if baseline_total["p50"] > 0 else 0
        save_cpu = (baseline_total["sum_s"] - full_row["sum_s"]) / 10
        print(f"  FULL_OPT vs BASELINE:")
        print(f"    p50:  {baseline_total['p50']:.3f}ms → {full_row['p50']:.3f}ms  ({save_ms:.3f}ms, {save_pct:.0f}%)")
        print(f"    CPU:  {baseline_total['sum_s']/10:.1f}% → {full_row['sum_s']/10:.1f}%  (saved {save_cpu:.1f}%)")
        print(f"    @300fps: {full_row['sum_s']:.0f}ms/s  (headroom: {100-full_row['sum_s']/10:.0f}%)")
    print()


if __name__ == "__main__":
    main()
