"""Validate an ONNX model with the STM32MP257 VSINPU execution provider.

Run this on the OpenSTLinux board:

    PYTHONPATH=. python3 FlightController/tools/validate_vsinpu_model.py \
      --model FlightController/Solutions/model/npu_quantization/road_yolo11n_seg_vsinpu_int8_qdq.onnx \
      --image adjustment/roads/IPC_2026-06-14.10.32.58.1790.jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ONNX model on VSINPU.")
    parser.add_argument("--model", required=True, type=Path, help="ONNX model path.")
    parser.add_argument("--image", type=Path, default=None, help="Optional BGR image.")
    parser.add_argument(
        "--provider",
        default="VSINPUExecutionProvider",
        help="Execution provider to request.",
    )
    parser.add_argument(
        "--allow-cpu-provider",
        action="store_true",
        help="Also register CPUExecutionProvider after the requested provider.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Timed inference runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup inference runs.")
    return parser.parse_args()


def preprocess_image(image_path: Path, input_size: int) -> np.ndarray:
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = frame.shape[:2]
    scale = min(input_size / float(width), input_size / float(height))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    image = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    left = int(round((input_size - resized_w) / 2.0))
    top = int(round((input_size - resized_h) / 2.0))
    image[top : top + resized_h, left : left + resized_w] = resized

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0).astype(np.float32)


def make_input(image_path: Path | None, input_size: int) -> np.ndarray:
    if image_path is not None:
        return preprocess_image(image_path, input_size)
    return np.zeros((1, 3, input_size, input_size), dtype=np.float32)


def main() -> int:
    args = parse_args()
    providers = [args.provider]
    if args.allow_cpu_provider:
        providers.append("CPUExecutionProvider")

    print("Available providers:", ort.get_available_providers())
    print("Requested providers:", providers)
    session = ort.InferenceSession(str(args.model), providers=providers)
    print("Session providers:", session.get_providers())

    model_input = session.get_inputs()[0]
    input_shape = model_input.shape
    input_size = int(input_shape[2])
    print("Input:", model_input.name, input_shape, model_input.type)
    print("Outputs:", [(item.name, item.shape, item.type) for item in session.get_outputs()])

    blob = make_input(args.image, input_size)
    for _ in range(max(0, args.warmup)):
        session.run(None, {model_input.name: blob})

    timings_ms: list[float] = []
    outputs = None
    for _ in range(max(1, args.runs)):
        start = time.perf_counter()
        outputs = session.run(None, {model_input.name: blob})
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    assert outputs is not None
    print("Output arrays:", [(tuple(item.shape), str(item.dtype)) for item in outputs])
    print("Finite outputs:", [bool(np.isfinite(item).all()) for item in outputs])
    print(
        "Latency ms: "
        f"min={min(timings_ms):.2f} "
        f"mean={float(np.mean(timings_ms)):.2f} "
        f"max={max(timings_ms):.2f}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
