"""Smoke-test a .nb NPU model on the development board.

Usage:

    PYTHONPATH=. python3 FlightController/tools/test_nb_model.py
    PYTHONPATH=. python3 FlightController/tools/test_nb_model.py --model FlightController/Solutions/model/road_yolo11n_seg_1.nb
    PYTHONPATH=. python3 FlightController/tools/test_nb_model.py --model ... --image adjustment/roads/road_0000.jpg

Stages:
  1. stai_mpu import check
  2. Load .nb model → print input/output metadata
  3. Single inference with zero / random / real image
  4. Report timing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_DEFAULT_NB = "FlightController/Solutions/model/road_yolo11n_seg_1.nb"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test a .nb NPU model.")
    p.add_argument("--model", default=_DEFAULT_NB,
                   help=f"Path to .nb model (default: {_DEFAULT_NB})")
    p.add_argument("--image", default=None,
                   help="Optional BGR image path for realistic input")
    p.add_argument("--runs", type=int, default=5,
                   help="Number of timed inference iterations (default: 5)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warmup iterations (default: 1)")
    return p.parse_args()


# ── stage 1: import check ────────────────────────────────────────

def check_stai_mpu() -> bool:
    print("=" * 60)
    print("  Stage 1 — stai_mpu import")
    print("=" * 60)
    try:
        from stai_mpu import stai_mpu_network  # noqa: F401
        print("  [OK]  stai_mpu.stai_mpu_network imported")
        return True
    except ImportError as e:
        print(f"  [FAIL]  {e}")
        print("  Fix:  apt install python3-libstai-mpu")
        return False


# ── stage 2: load model ──────────────────────────────────────────

def load_and_inspect(model_path: str):
    print()
    print("=" * 60)
    print("  Stage 2 — Load .nb model")
    print("=" * 60)
    if not os.path.isfile(model_path):
        print(f"  [FAIL]  File not found: {model_path}")
        return None

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Model : {model_path}  ({size_mb:.2f} MiB)")

    from nb_graph import NBGraphSession

    t0 = time.perf_counter()
    try:
        sess = NBGraphSession(model_path)
    except Exception as e:
        print(f"  [FAIL]  {e}")
        return None
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Load time : {load_ms:.1f} ms")
    print(f"  Backend   : {sess.backend}")
    print()

    # Inputs
    inputs = sess.get_inputs()
    print(f"  INPUTS  ({len(inputs)}):")
    for i, inp in enumerate(inputs):
        shape_str = " x ".join(str(d) for d in inp.shape)
        print(f"    [{i}]  name={inp.name}  shape=[{shape_str}]  type={inp.type}")

    # Outputs
    outputs = sess.get_outputs()
    print(f"  OUTPUTS ({len(outputs)}):")
    for i, out in enumerate(outputs):
        shape_str = " x ".join(str(d) for d in out.shape)
        print(f"    [{i}]  name={out.name}  shape=[{shape_str}]  type={out.type}")

    # Quick sanity checks
    inp0 = inputs[0]
    if len(inp0.shape) != 4 or inp0.shape[0] != 1:
        print(f"  [WARN]  Unexpected input shape: {inp0.shape}")
    if len(outputs) < 2:
        print(f"  [WARN]  Expected 2 outputs for YOLO-seg, got {len(outputs)}")

    print("  [OK]  Model loaded")
    return sess


# ── stage 3: inference ───────────────────────────────────────────

def run_inference(sess, image_path: str | None, runs: int, warmup: int):
    print()
    print("=" * 60)
    print("  Stage 3 — Inference")
    print("=" * 60)

    inputs = sess.get_inputs()
    inp0 = inputs[0]
    input_name = inp0.name
    input_shape = inp0.shape

    if len(input_shape) != 4:
        print(f"  [FAIL]  Expected 4D input, got shape {input_shape}")
        return False

    _, c, h, w = input_shape
    print(f"  Input  : {input_name}  [1, {c}, {h}, {w}]")

    # Build input blob
    import cv2
    if image_path is not None and os.path.isfile(image_path):
        print(f"  Source : {image_path}")
        frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  [WARN]  Could not read {image_path}, using random")
            blob = np.random.rand(1, c, h, w).astype(np.float32)
        else:
            # Use the same preprocessing as road_perception.py
            blob = _preprocess_like_road_perception(frame, int(h))
    else:
        print("  Source : random float32 [0, 1]")
        blob = np.random.rand(1, c, h, w).astype(np.float32)

    # Warmup
    for _ in range(max(0, warmup)):
        sess.run(None, {input_name: blob})

    # Timed runs
    times_ms: list[float] = []
    for i in range(max(1, runs)):
        t0 = time.perf_counter()
        outputs = sess.run(None, {input_name: blob})
        dt = (time.perf_counter() - t0) * 1000
        times_ms.append(dt)

    print()
    for j, out in enumerate(outputs):
        arr = np.asarray(out)
        print(f"  Output[{j}]:  shape={arr.shape}  dtype={arr.dtype}  "
              f"all_finite={bool(np.isfinite(arr).all())}  "
              f"min={float(arr.min()):.4f}  max={float(arr.max()):.4f}")

    times_arr = np.array(times_ms)
    print()
    print(f"  Timing ({runs} runs):")
    print(f"    min  = {float(times_arr.min()):.2f} ms")
    print(f"    mean = {float(times_arr.mean()):.2f} ms")
    print(f"    max  = {float(times_arr.max()):.2f} ms")
    print(f"    FPS  = {1000.0 / float(times_arr.mean()):.1f}")
    print("  [OK]  Inference completed")
    return True


def _preprocess_like_road_perception(frame: np.ndarray, input_size: int) -> np.ndarray:
    """Mimic road_perception._preprocess() but for arbitrary input_size."""
    import cv2
    h_img, w_img = frame.shape[:2]
    scale = min(input_size / float(w_img), input_size / float(h_img))
    rw = max(1, int(round(w_img * scale)))
    rh = max(1, int(round(h_img * scale)))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)

    img = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    left = int(round((input_size - rw) / 2.0))
    top = int(round((input_size - rh) / 2.0))
    img[top:top + rh, left:left + rw] = resized

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0).astype(np.float32)


# ── main ─────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if not check_stai_mpu():
        return 1

    sess = load_and_inspect(args.model)
    if sess is None:
        return 2

    ok = run_inference(sess, args.image, args.runs, args.warmup)
    if not ok:
        return 3

    print()
    print("=" * 60)
    print("  ALL STAGES PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
