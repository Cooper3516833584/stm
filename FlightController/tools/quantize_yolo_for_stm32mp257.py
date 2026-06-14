"""Build calibration packs and INT8 QDQ ONNX models for STM32MP257 NPU tests.

The calibration preprocessing mirrors the current YOLO runtime path:
OpenCV BGR frame -> letterbox with 114 padding -> RGB -> float32 / 255 -> NCHW.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "FlightController" / "Solutions" / "model"
ADJUSTMENT_DIR = ROOT / "adjustment"
OUTPUT_DIR = MODEL_DIR / "npu_quantization"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class QuantJob:
    name: str
    model_path: Path
    image_dir: Path


JOBS = (
    QuantJob(
        name="road_yolo11n_seg",
        model_path=MODEL_DIR / "road_yolo11n_seg.onnx",
        image_dir=ADJUSTMENT_DIR / "roads",
    ),
    QuantJob(
        name="tree_furniture",
        model_path=MODEL_DIR / "tree_furniture.onnx",
        image_dir=ADJUSTMENT_DIR / "trees",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create calibration NPZ files and INT8 QDQ ONNX models."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for calibration packs, quantized models, and manifest.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Use at most this many evenly sampled calibration images per model.",
    )
    parser.add_argument(
        "--road-model",
        type=Path,
        default=MODEL_DIR / "road_yolo11n_seg.onnx",
        help="Road YOLO ONNX model to quantize.",
    )
    parser.add_argument(
        "--tree-model",
        type=Path,
        default=MODEL_DIR / "tree_furniture.onnx",
        help="Tree/furniture YOLO ONNX model to quantize.",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Suffix inserted before output file roles, for example '_vsinpu'.",
    )
    parser.add_argument(
        "--skip-npz",
        action="store_true",
        help="Skip writing compressed ST Cloud calibration .npz files.",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Only create calibration packs and manifest.",
    )
    parser.add_argument(
        "--keep-preprocessed",
        action="store_true",
        help="Keep ONNX shape-inference intermediates next to output models.",
    )
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> tuple[QuantJob, ...]:
    return (
        QuantJob(
            name=f"road_yolo11n_seg{args.name_suffix}",
            model_path=args.road_model,
            image_dir=ADJUSTMENT_DIR / "roads",
        ),
        QuantJob(
            name=f"tree_furniture{args.name_suffix}",
            model_path=args.tree_model,
            image_dir=ADJUSTMENT_DIR / "trees",
        ),
    )


def manifest_path_for(output_dir: Path, name_suffix: str) -> Path:
    if name_suffix:
        return output_dir / f"calibration_manifest{name_suffix}.json"
    return output_dir / "calibration_manifest.json"


def workspace_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def get_model_io(model_path: Path) -> tuple[str, int, list[dict[str, object]], list[dict[str, object]]]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = [
        {"name": item.name, "shape": list(item.shape), "type": item.type}
        for item in session.get_inputs()
    ]
    outputs = [
        {"name": item.name, "shape": list(item.shape), "type": item.type}
        for item in session.get_outputs()
    ]
    if len(inputs) != 1:
        raise ValueError(f"{model_path} must have exactly one input, got {len(inputs)}")

    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    if len(shape) < 4 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise ValueError(f"{model_path} has unsupported input shape: {shape}")
    if shape[2] != shape[3]:
        raise ValueError(f"{model_path} input must be square NCHW, got: {shape}")
    return input_meta.name, int(shape[2]), inputs, outputs


def list_images(image_dir: Path) -> list[Path]:
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No calibration images found in {image_dir}")
    return images


def evenly_sample(paths: list[Path], max_images: int) -> list[Path]:
    if max_images <= 0 or len(paths) <= max_images:
        return paths
    indices = np.linspace(0, len(paths) - 1, max_images)
    return [paths[int(round(index))] for index in indices]


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


def write_npz(
    output_path: Path,
    input_name: str,
    image_paths: list[Path],
    input_size: int,
) -> None:
    batches = []
    for index, path in enumerate(image_paths, start=1):
        batches.append(preprocess_image(path, input_size)[0])
        if index % 25 == 0 or index == len(image_paths):
            print(f"  packed {index}/{len(image_paths)} images for {output_path.name}", flush=True)
    stacked = np.stack(batches, axis=0).astype(np.float32)
    np.savez_compressed(output_path, **{input_name: stacked})


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str, image_paths: Iterable[Path], input_size: int):
        self.input_name = input_name
        self.image_paths = iter(image_paths)
        self.input_size = input_size

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            image_path = next(self.image_paths)
        except StopIteration:
            return None
        return {self.input_name: preprocess_image(image_path, self.input_size)}


def quantize_model(
    model_path: Path,
    output_path: Path,
    input_name: str,
    image_paths: list[Path],
    input_size: int,
    keep_preprocessed: bool,
) -> None:
    preprocessed_path = output_path.with_suffix(".preprocessed.onnx")
    print(f"  running ONNX shape inference: {preprocessed_path.name}", flush=True)
    try:
        quant_pre_process(
            input_model=model_path,
            output_model_path=preprocessed_path,
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
        )
        quant_input = preprocessed_path
    except Exception as exc:
        print(f"  shape inference skipped after error: {exc}", flush=True)
        quant_input = model_path

    reader = ImageCalibrationReader(input_name, image_paths, input_size)
    print(f"  quantizing {model_path.name} -> {output_path.name}", flush=True)
    quantize_static(
        model_input=quant_input,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CPUExecutionProvider"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    if preprocessed_path.exists() and not keep_preprocessed:
        preprocessed_path.unlink()


def verify_model(model_path: Path, input_name: str, input_size: int) -> list[dict[str, object]]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    sample = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    outputs = session.run(None, {input_name: sample})
    return [
        {
            "index": index,
            "shape": list(output.shape),
            "dtype": str(output.dtype),
        }
        for index, output in enumerate(outputs)
    ]


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "created_at_epoch": int(time.time()),
        "preprocess": {
            "layout": "NCHW",
            "letterbox_pad": 114,
            "color": "BGR input converted to RGB",
            "normalization": "float32 / 255.0",
            "quant_format": "QDQ",
            "activation_type": "QInt8 symmetric",
            "weight_type": "QInt8 symmetric",
            "per_channel": False,
        },
        "jobs": [],
    }

    for job in build_jobs(args):
        print(f"[{job.name}] reading model metadata", flush=True)
        input_name, input_size, inputs, outputs = get_model_io(job.model_path)
        all_images = list_images(job.image_dir)
        selected_images = evenly_sample(all_images, args.max_images)
        print(
            f"[{job.name}] input={input_name} size={input_size} "
            f"images={len(selected_images)}/{len(all_images)}",
            flush=True,
        )

        npz_path = output_dir / f"{job.name}_calibration.npz"
        quantized_path = output_dir / f"{job.name}_int8_qdq.onnx"

        if not args.skip_npz:
            print(f"[{job.name}] writing calibration npz", flush=True)
            write_npz(npz_path, input_name, selected_images, input_size)

        verification = None
        if not args.skip_quantize:
            print(f"[{job.name}] running static quantization", flush=True)
            quantize_model(
                job.model_path,
                quantized_path,
                input_name,
                selected_images,
                input_size,
                args.keep_preprocessed,
            )
            print(f"[{job.name}] verifying quantized model", flush=True)
            verification = verify_model(quantized_path, input_name, input_size)

        manifest["jobs"].append(
            {
                "name": job.name,
                "model": workspace_path(job.model_path),
                "image_dir": workspace_path(job.image_dir),
                "source_image_count": len(all_images),
                "selected_image_count": len(selected_images),
                "max_images": args.max_images,
                "input_name": input_name,
                "input_size": input_size,
                "inputs": inputs,
                "outputs": outputs,
                "calibration_npz": workspace_path(npz_path) if not args.skip_npz else None,
                "calibration_npz_key": input_name if not args.skip_npz else None,
                "quantized_model": workspace_path(quantized_path)
                if not args.skip_quantize
                else None,
                "quantized_cpu_verification": verification,
                "selected_images": [
                    workspace_path(path) for path in selected_images
                ],
            }
        )

    manifest_path = manifest_path_for(output_dir, args.name_suffix)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
