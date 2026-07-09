"""Render masks from a DeepLab-style .nb segmentation model.

This is an offline/board-side diagnostic tool. It does not connect to the
flight controller, radar, or camera devices. It loads one compiled `.nb` model,
runs it on still images, and writes class masks plus an overlay image.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _setup_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)
    return root


ROOT = _setup_path()

from nb_graph import NBGraphSession  # noqa: E402


DEFAULT_MODEL = (
    ROOT
    / "FlightController/Solutions/model/st_deeplabv3_mnv2_a050_s16_asppv2_256_qdq_int8_1.nb"
)


def parse_args() -> argparse.Namespace:
    default_output = (
        Path("/media/sdcard/npu_debug/deeplab_overlay")
        if Path("/media/sdcard").is_dir()
        else ROOT / "debug/deeplab_overlay"
    )
    parser = argparse.ArgumentParser(
        description="Render DeepLab .nb class masks and overlay images."
    )
    parser.add_argument("images", nargs="+", type=Path, help="Input BGR image files.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=".nb model path.")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--road-class",
        type=int,
        default=1,
        help="Class index to use for the primary overlay. Try 0 or 1.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Overlay alpha for the selected class mask.",
    )
    return parser.parse_args()


def _letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    height, width = frame.shape[:2]
    scale = min(size / float(width), size / float(height))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    image = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_w) / 2.0
    pad_y = (size - resized_h) / 2.0
    left = int(round(pad_x))
    top = int(round(pad_y))
    image[top : top + resized_h, left : left + resized_w] = resized
    return image, scale, pad_x, pad_y


def _preprocess(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    image, scale, pad_x, pad_y = _letterbox(frame, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0).astype(np.float32), scale, pad_x, pad_y


def _crop_and_resize_mask(
    mask_square: np.ndarray,
    *,
    orig_w: int,
    orig_h: int,
    input_size: int,
    pad_x: float,
    pad_y: float,
) -> np.ndarray:
    crop_x1 = max(0, min(input_size, int(round(pad_x))))
    crop_y1 = max(0, min(input_size, int(round(pad_y))))
    crop_x2 = max(crop_x1, min(input_size, int(round(input_size - pad_x))))
    crop_y2 = max(crop_y1, min(input_size, int(round(input_size - pad_y))))
    cropped = mask_square[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped.size == 0:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def _decode_class_map(output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(output)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected output [1,C,H,W] or [C,H,W], got {output.shape}")
    class_map = np.argmax(arr, axis=0).astype(np.uint8)
    return class_map, arr.astype(np.float32, copy=False)


def _write_outputs(
    frame: np.ndarray,
    class_map: np.ndarray,
    logits: np.ndarray,
    *,
    output_dir: Path,
    stem: str,
    road_class: int,
    alpha: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = int(logits.shape[0])
    hist = np.bincount(class_map.reshape(-1), minlength=classes)
    total = max(1, int(class_map.size))
    print(
        "  class_pixels:",
        " ".join(f"class{idx}={int(count)}({count / total:.3f})" for idx, count in enumerate(hist)),
    )
    for idx in range(classes):
        mask = (class_map == idx).astype(np.uint8) * 255
        cv2.imwrite(str(output_dir / f"{stem}_class{idx}_mask.png"), mask)

    selected = (class_map == road_class).astype(np.uint8) * 255
    color = np.zeros_like(frame)
    color[selected > 0] = (0, 180, 255)
    overlay = cv2.addWeighted(frame, 1.0, color, float(alpha), 0.0)
    cv2.imwrite(str(output_dir / f"{stem}_class{road_class}_overlay.jpg"), overlay)


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    session = NBGraphSession(str(args.model))
    input_meta = session.get_inputs()[0]
    shape = list(input_meta.shape)
    if len(shape) != 4:
        raise ValueError(f"Expected NCHW model input, got {shape}")
    input_name = input_meta.name
    input_size = int(shape[2])
    print(f"model: {args.model}")
    print(f"input: {input_name} shape={shape} type={input_meta.type}")
    print("outputs:", [(item.name, list(item.shape), item.type) for item in session.get_outputs()])
    print(f"output_dir: {args.output_dir}")

    for image_path in args.images:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] could not read image: {image_path}")
            continue
        orig_h, orig_w = frame.shape[:2]
        blob, _scale, pad_x, pad_y = _preprocess(frame, input_size)
        start = time.perf_counter()
        outputs = session.run(None, {input_name: blob})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        class_map_square, logits = _decode_class_map(outputs[0])
        class_map = _crop_and_resize_mask(
            class_map_square,
            orig_w=orig_w,
            orig_h=orig_h,
            input_size=input_size,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        print(f"[OK] {image_path.name}: inference={elapsed_ms:.2f} ms")
        _write_outputs(
            frame,
            class_map,
            logits,
            output_dir=args.output_dir,
            stem=image_path.stem,
            road_class=args.road_class,
            alpha=args.alpha,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
