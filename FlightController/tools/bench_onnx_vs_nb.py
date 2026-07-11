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
ONNX_PATH = MODEL_DIR / "road_yolo11n_seg.onnx"
NB_PATH = MODEL_DIR / "road_yolo11n_seg_1.nb"
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


def _bench_onnx(model_path: Path, input_size: int, blob: np.ndarray) -> dict[str, Any]:
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
    for _ in range(WARMUP_RUNS):
        session.run(output_names, {input_name: blob})

    # Timed runs
    latencies: list[float] = []
    for _ in range(BENCH_RUNS):
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


def _bench_nb(model_path: Path, input_size: int, blob: np.ndarray) -> dict[str, Any]:
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
    for _ in range(WARMUP_RUNS):
        session.run(output_names, {input_name: blob})

    # Timed runs
    latencies: list[float] = []
    for _ in range(BENCH_RUNS):
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


def main() -> int:
    print("YOLO11n-seg ONNX vs .nb CPU benchmark")
    print(f"  ONNX model : {ONNX_PATH}  ({'exists' if ONNX_PATH.is_file() else 'MISSING'})")
    print(f"  .nb model  : {NB_PATH}  ({'exists' if NB_PATH.is_file() else 'MISSING'})")
    print(f"  Runs       : {WARMUP_RUNS} warmup + {BENCH_RUNS} timed")

    # Find a test image
    test_img = _find_test_image()
    if test_img is None:
        print("\nERROR: no test image found.  Place a road JPEG in tests/roads/")
        print("  or export TEST_IMAGE=/path/to/road.jpg")
        env_img = os.environ.get("TEST_IMAGE")
        if env_img:
            test_img = Path(env_img)
        else:
            return 1
    print(f"  Test image : {test_img}")

    import cv2
    frame = cv2.imread(str(test_img))
    if frame is None:
        print(f"ERROR: cannot read test image: {test_img}")
        return 1

    # ------------------------------------------------------------------
    # ONNX benchmark
    # ------------------------------------------------------------------
    if not ONNX_PATH.is_file():
        print(f"\nERROR: ONNX model not found: {ONNX_PATH}")
        return 1

    print("\n[1/2] Benchmarking ONNX (onnxruntime CPU) ...")
    try:
        # Detect input size from existing ONNX model
        import onnxruntime as ort
        temp = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        onnx_input_shape = temp.get_inputs()[0].shape
        onnx_input_size = onnx_input_shape[2] if len(onnx_input_shape) >= 4 else 320
    except Exception:
        onnx_input_size = 320

    blob = _preprocess(frame, onnx_input_size)
    onnx_result = _bench_onnx(ONNX_PATH, onnx_input_size, blob)
    print(f"  ONNX mean latency: {onnx_result['latency_ms_mean']:.1f} ms")

    # ------------------------------------------------------------------
    # .nb benchmark
    # ------------------------------------------------------------------
    nb_result: dict | None = None
    nb_error: str | None = None

    if not NB_PATH.is_file():
        nb_error = "model file missing"
        print(f"\n[2/2] Skipping .nb benchmark — {nb_error}")
    else:
        print("\n[2/2] Benchmarking .nb (stai_mpu CPU fallback) ...")
        try:
            # .nb model has its own expected input size; use 416 (matching calibration
            # manifest) or detect from the model.
            nb_input_size = 416  # per calibration_manifest.json
            blob_nb = _preprocess(frame, nb_input_size)
            nb_result = _bench_nb(NB_PATH, nb_input_size, blob_nb)
            print(f"  .nb mean latency: {nb_result['latency_ms_mean']:.1f} ms")
        except Exception as exc:
            nb_error = f"{type(exc).__name__}: {exc}"
            print(f"  .nb benchmark failed: {nb_error}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    _print_table(onnx_result, nb_result, nb_error)

    report = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "onnx_model": str(ONNX_PATH),
        "nb_model": str(NB_PATH),
        "test_image": str(test_img),
        "warmup_runs": WARMUP_RUNS,
        "bench_runs": BENCH_RUNS,
        "onnx": onnx_result,
        "nb": nb_result,
        "nb_error": nb_error,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
