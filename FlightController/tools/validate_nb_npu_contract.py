"""Validate that a compiled .nb model looks like a real STM32MP2 NPU path.

Run this on the OpenSTLinux board.  This script checks the model contract
that Python can observe directly: tensor metadata and inference latency.
Use the documented strace wrapper around this command to prove /dev/galcore
ioctl activity.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "FlightController/Solutions/model/road_yolo11n_seg_vsinpu_fp32_opt.nb"
QUANTIZED_TYPES = ("tensor(int8)", "tensor(uint8)")
FLOAT_TYPES = ("tensor(float)", "tensor(float16)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate .nb tensor contract and latency on STM32MP2."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=".nb model path.")
    parser.add_argument("--image", default=None, help="Optional BGR image path.")
    parser.add_argument("--runs", type=int, default=10, help="Timed inference runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs.")
    parser.add_argument(
        "--max-mean-ms",
        type=float,
        default=80.0,
        help="Fail if mean latency is above this threshold.",
    )
    parser.add_argument(
        "--allow-float-io",
        action="store_true",
        help="Do not fail on float input/output metadata.",
    )
    parser.add_argument(
        "--profile-raw-stai",
        action="store_true",
        help="Also time raw stai_mpu set_input/run/get_output without NBGraphSession conversions.",
    )
    return parser.parse_args()


def _format_meta(items) -> list[str]:
    return [
        f"{item.name} shape={list(item.shape)} type={item.type}"
        for item in items
    ]


def _has_float_tensor(items) -> bool:
    return any(str(item.type).lower() in FLOAT_TYPES for item in items)


def _has_quantized_tensor(items) -> bool:
    return any(str(item.type).lower() in QUANTIZED_TYPES for item in items)


def _make_input(session, image_path: str | None) -> tuple[str, np.ndarray]:
    inp = session.get_inputs()[0]
    input_name = inp.name
    shape = list(inp.shape)
    if len(shape) != 4:
        raise ValueError(f"Expected 4D NCHW input, got {shape}")
    _, channels, height, width = shape

    if image_path is None:
        blob = np.random.rand(1, channels, height, width).astype(np.float32)
        return input_name, blob

    import cv2

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    src_h, src_w = frame.shape[:2]
    scale = min(width / float(src_w), height / float(src_h))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    img = np.full((height, width, 3), 114, dtype=np.uint8)
    left = int(round((width - resized_w) / 2.0))
    top = int(round((height - resized_h) / 2.0))
    img[top : top + resized_h, left : left + resized_w] = resized
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return input_name, np.expand_dims(img, axis=0).astype(np.float32)


def _dtype_from_meta_type(type_name: str) -> np.dtype:
    lower = str(type_name).lower()
    if "uint8" in lower:
        return np.dtype(np.uint8)
    if "int8" in lower:
        return np.dtype(np.int8)
    if "float16" in lower:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


def _profile_raw_stai(model_path: str, runs: int, warmup: int) -> None:
    """Print raw STAI timings without Python-side quant/dequant wrappers."""
    from stai_mpu import stai_mpu_network  # type: ignore[import-untyped]

    print()
    print("=" * 72)
    print("  Raw stai_mpu profile")
    print("=" * 72)

    network = stai_mpu_network(model_path=model_path, use_hw_acceleration=True)
    input_infos = network.get_input_infos()
    output_infos = network.get_output_infos()

    raw_inputs: list[np.ndarray] = []
    for index, info in enumerate(input_infos):
        shape = list(info.get_shape())
        dtype = _dtype_from_meta_type(info.get_dtype())
        if dtype == np.int8:
            arr = np.random.randint(-128, 127, size=shape, dtype=np.int8)
        elif dtype == np.uint8:
            arr = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        else:
            arr = np.random.rand(*shape).astype(dtype)
        raw_inputs.append(np.ascontiguousarray(arr))
        print(f"raw_input[{index}]: shape={shape} dtype={raw_inputs[-1].dtype}")

    for index, info in enumerate(output_infos):
        print(
            f"raw_output[{index}]: shape={list(info.get_shape())} "
            f"dtype={_dtype_from_meta_type(info.get_dtype())}"
        )

    for index, arr in enumerate(raw_inputs):
        network.set_input(index, arr)

    for _ in range(max(0, warmup)):
        network.run()
        for index in range(len(output_infos)):
            _ = network.get_output(index)

    run_times: list[float] = []
    get_times: list[float] = []
    set_times: list[float] = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        for index, arr in enumerate(raw_inputs):
            network.set_input(index, arr)
        set_times.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        network.run()
        run_times.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        for index in range(len(output_infos)):
            _ = network.get_output(index)
        get_times.append((time.perf_counter() - start) * 1000.0)

    print(
        "raw_set_input_ms: "
        f"mean={float(np.mean(set_times)):.2f} "
        f"min={float(np.min(set_times)):.2f} "
        f"max={float(np.max(set_times)):.2f}"
    )
    print(
        "raw_run_ms: "
        f"mean={float(np.mean(run_times)):.2f} "
        f"min={float(np.min(run_times)):.2f} "
        f"max={float(np.max(run_times)):.2f}"
    )
    print(
        "raw_get_output_ms: "
        f"mean={float(np.mean(get_times)):.2f} "
        f"min={float(np.min(get_times)):.2f} "
        f"max={float(np.max(get_times)):.2f}"
    )


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.model):
        print(f"[FAIL] model not found: {args.model}")
        return 2

    from nb_graph import NBGraphSession

    print("=" * 72)
    print("  .nb NPU contract validation")
    print("=" * 72)
    print(f"model: {args.model}")
    print(f"size : {Path(args.model).stat().st_size / 1024 / 1024:.2f} MiB")

    start = time.perf_counter()
    session = NBGraphSession(args.model)
    load_ms = (time.perf_counter() - start) * 1000.0

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"load_ms: {load_ms:.2f}")
    print("inputs:")
    for line in _format_meta(inputs):
        print(f"  - {line}")
    print("outputs:")
    for line in _format_meta(outputs):
        print(f"  - {line}")

    input_name, blob = _make_input(session, args.image)
    for _ in range(max(0, args.warmup)):
        session.run(None, {input_name: blob})

    timings: list[float] = []
    last_outputs = None
    for _ in range(max(1, args.runs)):
        start = time.perf_counter()
        last_outputs = session.run(None, {input_name: blob})
        timings.append((time.perf_counter() - start) * 1000.0)

    assert last_outputs is not None
    finite = [bool(np.isfinite(np.asarray(item)).all()) for item in last_outputs]
    times = np.asarray(timings, dtype=np.float64)
    mean_ms = float(times.mean())
    print("output_arrays:")
    for index, item in enumerate(last_outputs):
        arr = np.asarray(item)
        print(f"  - [{index}] shape={list(arr.shape)} dtype={arr.dtype}")
    print(f"finite_outputs: {finite}")
    print(
        "latency_ms: "
        f"min={float(times.min()):.2f} "
        f"mean={mean_ms:.2f} "
        f"max={float(times.max()):.2f}"
    )

    failures: list[str] = []
    if not args.allow_float_io:
        if _has_float_tensor(inputs) or _has_float_tensor(outputs):
            failures.append("model exposes float/float16 I/O; expected quantized .nb path")
        if not (_has_quantized_tensor(inputs) or _has_quantized_tensor(outputs)):
            failures.append("no int8/uint8 input/output metadata observed")
    if mean_ms > args.max_mean_ms:
        failures.append(f"mean latency {mean_ms:.2f} ms exceeds {args.max_mean_ms:.2f} ms")
    if not all(finite):
        failures.append("non-finite output detected")

    if failures:
        print("[FAIL]")
        for failure in failures:
            print(f"  - {failure}")
        if args.profile_raw_stai:
            _profile_raw_stai(args.model, args.runs, args.warmup)
        return 1

    print("[PASS] tensor metadata and latency look NPU-compatible")
    print("Next proof: run this command under strace and grep for /dev/galcore ioctl.")
    if args.profile_raw_stai:
        _profile_raw_stai(args.model, args.runs, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
