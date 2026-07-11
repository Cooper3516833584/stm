#!/usr/bin/env python3
"""Board-side benchmark: compare ONNX FP32 vs .nb float16 on STM32MP257 CPU.

Run this on the STM32MP257 board (not PC) to measure real inference performance.

This is Step 0 of the YOLO11n-seg 128×128 lightweighting plan.  Before shrinking
the model, we answer: *on the CPU without NPU acceleration, which backend is faster —
onnxruntime FP32 or stai_mpu float16?*

Usage (on STM32MP257 board)::

    # Method A — use x-linux-ai-benchmark (ST's standard tool) if available:
    mkdir -p /tmp/model_bench
    cp FlightController/Solutions/model/road_yolo11n_seg.onnx /tmp/model_bench/
    cp FlightController/Solutions/model/road_yolo11n_seg_1.nb   /tmp/model_bench/
    x-linux-ai-benchmark -d /tmp/model_bench --cpu_cores 2 --export_results

    # Method B — fall back to the project's own tools:
    python FlightController/tools/bench_onnx_vs_nb.py

Output
------
The script prints a side-by-side comparison table: model size, inference latency,
preprocessing overhead, and total end-to-end time.  It also writes a JSON report
to ``FlightController/Solutions/model/bench_onnx_vs_nb.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths (board-side)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "FlightController" / "Solutions" / "model"
# Old 416 models
ONNX_PATH_416 = MODEL_DIR / "road_yolo11n_seg.onnx"
NB_PATH_416 = MODEL_DIR / "road_yolo11n_seg_1.nb"
# New 128 lightweight models
ONNX_PATH_128 = MODEL_DIR / "road_yolo11n_seg_128.onnx"
NB_PATH_128 = MODEL_DIR / "road_yolo11n_seg_128_1.nb"
# Default test targets — ordered by expected speed
DEFAULT_MODELS: list[tuple[str, Path, int]] = [
    ("ONNX 128", ONNX_PATH_128, 128),
    (".nb  128", NB_PATH_128, 128),
    ("ONNX 416", ONNX_PATH_416, 416),
    (".nb  416", NB_PATH_416, 416),
]
TEST_IMAGE_DIR = REPO_ROOT / "tests" / "roads"
REPORT_PATH = MODEL_DIR / "bench_onnx_vs_nb.json"

WARMUP_RUNS = 3
BENCH_RUNS = 15


def _find_test_image() -> Path | None:
    """Return the first available test road image, or None."""
    if not TEST_IMAGE_DIR.is_dir():
        return None
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidates = sorted(TEST_IMAGE_DIR.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    return None


def _preprocess(frame: np.ndarray, input_size: int) -> np.ndarray:
    """Mirror of road_perception._preprocess for standalone benchmark."""
    import cv2

    h, w = frame.shape[:2]
    scale = min(input_size / float(w), input_size / float(h))
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    img = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - resized_w) / 2.0
    pad_y = (input_size - resized_h) / 2.0
    img[
        int(round(pad_y)) : int(round(pad_y)) + resized_h,
        int(round(pad_x)) : int(round(pad_x)) + resized_w,
    ] = resized
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


def _bench_onnx(
    model_path: Path,
    input_size: int,
    blob: np.ndarray,
    warmup_runs: int = WARMUP_RUNS,
    bench_runs: int = BENCH_RUNS,
) -> dict[str, Any]:
    """Benchmark onnxruntime inference."""
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if "XnnpackExecutionProvider" in ort.get_available_providers():
        providers = ["XnnpackExecutionProvider", "CPUExecutionProvider"]

    t0 = time.monotonic()
    session = ort.InferenceSession(str(model_path), providers=providers)
    load_ms = (time.monotonic() - t0) * 1000.0

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    # Warmup
    for _ in range(warmup_runs):
        session.run(output_names, {input_name: blob})

    # Timed runs
    latencies: list[float] = []
    for _ in range(bench_runs):
        start = time.monotonic()
        session.run(output_names, {input_name: blob})
        latencies.append((time.monotonic() - start) * 1000.0)

    return {
        "backend": "onnxruntime",
        "provider": providers[0],
        "load_ms": round(load_ms, 1),
        "input_size": input_size,
        "latency_ms_mean": round(float(np.mean(latencies)), 1),
        "latency_ms_std": round(float(np.std(latencies)), 1),
        "latency_ms_min": round(float(np.min(latencies)), 1),
        "latency_ms_max": round(float(np.max(latencies)), 1),
    }


def _bench_nb(
    model_path: Path,
    input_size: int,
    blob: np.ndarray,
    warmup_runs: int = WARMUP_RUNS,
    bench_runs: int = BENCH_RUNS,
) -> dict[str, Any]:
    """Benchmark .nb model via nb_graph.NBGraphSession (stai_mpu fallback)."""
    # Add repo root so we can import nb_graph
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from nb_graph import NBGraphSession

    t0 = time.monotonic()
    session = NBGraphSession(str(model_path))
    load_ms = (time.monotonic() - t0) * 1000.0

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    # Warmup
    for _ in range(warmup_runs):
        session.run(output_names, {input_name: blob})

    # Timed runs
    latencies: list[float] = []
    for _ in range(bench_runs):
        start = time.monotonic()
        session.run(output_names, {input_name: blob})
        latencies.append((time.monotonic() - start) * 1000.0)

    return {
        "backend": "stai_mpu (NBGraphSession)",
        "load_ms": round(load_ms, 1),
        "input_size": input_size,
        "latency_ms_mean": round(float(np.mean(latencies)), 1),
        "latency_ms_std": round(float(np.std(latencies)), 1),
        "latency_ms_min": round(float(np.min(latencies)), 1),
        "latency_ms_max": round(float(np.max(latencies)), 1),
    }


def _print_table(onnx_result: dict, nb_result: dict | None, nb_error: str | None) -> None:
    """Print a side-by-side comparison."""
    print()
    print("=" * 78)
    print("  ONNX (FP32/CPU)  vs  .nb (float16/CPU)  —  STM32MP257 Benchmark")
    print("=" * 78)

    rows = [
        ("Backend", onnx_result.get("backend", "?"), (nb_result or {}).get("backend", nb_error or "N/A")),
        ("Provider", onnx_result.get("provider", "-"), "-"),
        ("Load time", f"{onnx_result.get('load_ms', '?')} ms", f"{(nb_result or {}).get('load_ms', nb_error or '?')} ms"),
        ("Input size", str(onnx_result.get('input_size', '?')), str((nb_result or {}).get('input_size', 'N/A'))),
        ("", "", ""),
        ("Mean latency", f"{onnx_result.get('latency_ms_mean', '?')} ms", f"{(nb_result or {}).get('latency_ms_mean', nb_error or '?')} ms"),
        ("Std latency", f"{onnx_result.get('latency_ms_std', '?')} ms", f"{(nb_result or {}).get('latency_ms_std', nb_error or '?')} ms"),
        ("Min latency", f"{onnx_result.get('latency_ms_min', '?')} ms", f"{(nb_result or {}).get('latency_ms_min', nb_error or '?')} ms"),
        ("Max latency", f"{onnx_result.get('latency_ms_max', '?')} ms", f"{(nb_result or {}).get('latency_ms_max', nb_error or '?')} ms"),
    ]

    for label, onnx_val, nb_val in rows:
        print(f"  {label:<18} {onnx_val:<24} {nb_val:<24}")

    if nb_result and onnx_result:
        speedup = onnx_result["latency_ms_mean"] / max(0.001, nb_result["latency_ms_mean"])
        faster = ".nb" if speedup > 1.01 else "ONNX"
        ratio = max(speedup, 1.0 / max(0.001, speedup))
        print()
        print(f"  → {faster} is {ratio:.1f}× faster on CPU")

    print("=" * 78)


def _print_results_table(results: list[dict]) -> None:
    """Print a multi-model comparison table."""
    print()
    print("=" * 78)
    print("  YOLO11n-seg Model Benchmarks — STM32MP257 CPU")
    print("=" * 78)
    header = f"  {'Model':<14} {'Input':<8} {'Backend':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}"
    print(header)
    print("  " + "-" * 76)

    best_latency = min(
        (r["latency_ms_mean"] for r in results if "latency_ms_mean" in r),
        default=1.0,
    )

    for r in results:
        label = r.get("label", "?")
        inp = str(r.get("input_size", "?"))
        backend = r.get("backend", "?")
        if "error" in r:
            print(f"  {label:<14} {inp:<8} {backend:<22} {'ERROR':>8}  ({r['error'][:40]})")
            continue
        mean = f"{r['latency_ms_mean']:.0f} ms"
        std = f"{r['latency_ms_std']:.0f} ms"
        lo = f"{r['latency_ms_min']:.0f} ms"
        hi = f"{r['latency_ms_max']:.0f} ms"
        ratio = r["latency_ms_mean"] / max(0.001, best_latency)
        tag = " ★ best" if ratio < 1.02 else f" ×{ratio:.1f}"
        print(f"  {label:<14} {inp:<8} {backend:<22} {mean:>8} {std:>8} {lo:>8} {hi:>8}{tag}")

    print("=" * 78)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="YOLO11n-seg multi-model CPU benchmark")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Explicit (label, path, input_size) triples.  Default: ONNX 128, ONNX 416, .nb 416.",
    )
    parser.add_argument("--image", default=None, help="Test image path")
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=BENCH_RUNS)
    parser.add_argument(
        "--no-nb",
        action="store_true",
        help="Skip .nb models (e.g. on PC without stai_mpu).",
    )
    parser.add_argument(
        "--no-416",
        action="store_true",
        help="Skip 416 models to save time.",
    )
    args = parser.parse_args()

    warmup = args.warmup
    bench_runs = args.runs

    # Build list of models to test
    targets: list[tuple[str, Path, int]] = []
    for label, path, inp_size in DEFAULT_MODELS:
        if args.no_416 and "416" in label:
            continue
        if args.no_nb and ".nb" in label:
            continue
        if path.is_file():
            targets.append((label, path, inp_size))
        else:
            print(f"SKIP {label}: file not found ({path})")

    if not targets:
        print("ERROR: no models to benchmark.")
        return 1

    print(f"Models to benchmark: {len(targets)}")
    for label, path, inp_size in targets:
        print(f"  {label}: {path}  ({inp_size}×{inp_size})")
    print(f"  Runs: {warmup} warmup + {bench_runs} timed")

    # Find test image
    test_img = Path(args.image) if args.image else _find_test_image()
    if test_img is None:
        env_img = os.environ.get("TEST_IMAGE")
        test_img = Path(env_img) if env_img else None
    if test_img is None or not test_img.is_file():
        print("ERROR: no test image.  Use --image or place a JPEG in tests/roads/")
        return 1
    print(f"  Test image: {test_img}")

    import cv2
    frame = cv2.imread(str(test_img))
    if frame is None:
        print(f"ERROR: cannot read: {test_img}")
        return 1

    results: list[dict] = []

    for label, model_path, inp_size in targets:
        print(f"\n[{len(results)+1}/{len(targets)}] Benchmarking: {label} ...")

        try:
            blob = _preprocess(frame, inp_size)

            if ".nb" in label or model_path.suffix == ".nb":
                # Lazy import nb_graph
                if str(REPO_ROOT) not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT))
                result = _bench_nb(model_path, inp_size, blob, warmup, bench_runs)
                result["backend"] = "stai_mpu"
            else:
                result = _bench_onnx(model_path, inp_size, blob, warmup, bench_runs)

            result["label"] = label
            result["input_size"] = inp_size
            result["load_ms"] = result.get("load_ms", 0)
            results.append(result)
            print(f"  Mean latency: {result['latency_ms_mean']:.0f} ms")

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {err_msg}")
            results.append({"label": label, "input_size": inp_size, "error": err_msg})

    # Print table and save report
    _print_results_table(results)

    report = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_image": str(test_img),
        "warmup_runs": warmup,
        "bench_runs": bench_runs,
        "results": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nReport saved to: {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
