"""Quantize YOLO ONNX models to INT8 QDQ using PC GPU (DirectML) acceleration.

Reads preprocessed calibration data from .npz files and produces per-tensor
QInt8 QDQ ONNX models compatible with ST Edge AI Cloud Optimize → .nb pipeline.

Usage::

    PYTHONPATH=. python FlightController/tools/quantize_yolo_gpu.py

    PYTHONPATH=. python FlightController/tools/quantize_yolo_gpu.py \\
        --road-model FlightController/Solutions/model/road_yolo11n_seg.onnx \\
        --road-npz D:/drone2/road_yolo11n_seg_calibration.npz \\
        --tree-model FlightController/Solutions/model/tree_furniture.onnx \\
        --tree-npz D:/drone2/tree_furniture_calibration.npz \\
        --gpu-provider DmlExecutionProvider
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

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


# ── defaults ─────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "FlightController" / "Solutions" / "model"
OUTPUT_DIR = MODEL_DIR / "npu_quantization"

ROAD_MODEL_DEFAULT = MODEL_DIR / "road_yolo11n_seg.onnx"
TREE_MODEL_DEFAULT = MODEL_DIR / "tree_furniture.onnx"

ROAD_NPZ_DEFAULT = Path("D:/drone2/road_yolo11n_seg_calibration.npz")
TREE_NPZ_DEFAULT = Path("D:/drone2/tree_furniture_calibration.npz")

GPU_PROVIDER_DEFAULT = "DmlExecutionProvider"


# ── CLI ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantize YOLO ONNX models → INT8 QDQ with GPU acceleration."
    )
    p.add_argument("--road-model", type=Path, default=ROAD_MODEL_DEFAULT,
                   help=f"Road FP32 ONNX model (default: {ROAD_MODEL_DEFAULT})")
    p.add_argument("--road-npz", type=Path, default=ROAD_NPZ_DEFAULT,
                   help=f"Road calibration .npz (default: {ROAD_NPZ_DEFAULT})")
    p.add_argument("--tree-model", type=Path, default=TREE_MODEL_DEFAULT,
                   help=f"Tree FP32 ONNX model (default: {TREE_MODEL_DEFAULT})")
    p.add_argument("--tree-npz", type=Path, default=TREE_NPZ_DEFAULT,
                   help=f"Tree calibration .npz (default: {TREE_NPZ_DEFAULT})")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Output directory (default: {OUTPUT_DIR})")
    p.add_argument("--gpu-provider", default=GPU_PROVIDER_DEFAULT,
                   choices=["DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider"],
                   help="GPU provider for calibration (default: DmlExecutionProvider)")
    p.add_argument("--max-calibration-samples", type=int, default=200,
                   help="Max calibration samples per model (default: 200)")
    p.add_argument("--skip-road", action="store_true", help="Skip road model")
    p.add_argument("--skip-tree", action="store_true", help="Skip tree model")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU calibration")
    p.add_argument("--keep-preprocessed", action="store_true",
                   help="Keep preprocessed ONNX intermediates")
    return p.parse_args()


# ── npz calibration reader ───────────────────────────────────────

class NpzCalibrationReader(CalibrationDataReader):
    """Yield preprocessed images from a .npz file for ORT calibration."""

    def __init__(self, input_name: str, images: np.ndarray):
        self._input_name = input_name
        self._images = images          # (N, C, H, W) float32
        self._pos = 0
        self._n = len(images)

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._pos >= self._n:
            return None
        blob = self._images[self._pos : self._pos + 1]  # keep batch dim [1,C,H,W]
        self._pos += 1
        return {self._input_name: blob.astype(np.float32)}


# ── helpers ──────────────────────────────────────────────────────

def _available_gpu_providers() -> list[str]:
    available = ort.get_available_providers()
    return [p for p in available if p != "CPUExecutionProvider"
            and any(kw in p for kw in ("Dml", "CUDA", "ROCm", "TensorRT"))]


def _best_gpu_provider(requested: str) -> str | None:
    available = set(ort.get_available_providers())
    candidates = [requested] + ["DmlExecutionProvider", "CUDAExecutionProvider",
                                  "TensorrtExecutionProvider", "ROCMExecutionProvider"]
    for c in candidates:
        if c in available:
            return c
    return None


def _validate_model(model_path: Path) -> tuple[str, int, list[dict], list[dict]]:
    """Return (input_name, input_size, inputs_meta, outputs_meta)."""
    import onnx
    m = onnx.load(str(model_path))
    inp0 = m.graph.input[0]
    shape = [d.dim_value for d in inp0.type.tensor_type.shape.dim]
    name = inp0.name
    if len(shape) != 4 or shape[0] != 1 or shape[2] != shape[3]:
        raise ValueError(f"{model_path.name}: bad input shape {shape}, need [1,3,S,S]")
    input_size = int(shape[2])

    out_meta = []
    for out in m.graph.output:
        oshape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        out_meta.append({"name": out.name, "shape": oshape, "type": "tensor(float)"})

    in_meta = [{"name": name, "shape": shape, "type": "tensor(float)"}]
    return name, input_size, in_meta, out_meta


def _resize_calibration(images: np.ndarray, target_size: int) -> np.ndarray:
    """Resize (N,C,H,W) calibration array from its current size to target_size.

    Uses bilinear interpolation via OpenCV on each channel of each image.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4D calibration array, got shape {images.shape}")

    _, _, h, w = images.shape
    if h == target_size and w == target_size:
        return images

    import cv2
    out = np.empty((len(images), images.shape[1], target_size, target_size),
                   dtype=np.float32)
    for i in range(len(images)):
        for c in range(images.shape[1]):
            out[i, c] = cv2.resize(
                images[i, c], (target_size, target_size),
                interpolation=cv2.INTER_LINEAR,
            )
    return out


# ── quantization pipeline ────────────────────────────────────────

def quantize_one(
    model_path: Path,
    npz_path: Path,
    output_name: str,
    output_dir: Path,
    gpu_provider: str | None,
    max_samples: int,
    keep_preprocessed: bool,
) -> dict:
    """Run the full pipeline for one model. Returns a result dict."""
    result: dict = {"model": str(model_path), "output": "", "status": "FAILED"}
    t_total_start = time.perf_counter()

    # ── validate ──
    if not model_path.is_file():
        result["error"] = f"model not found: {model_path}"
        return result
    if not npz_path.is_file():
        result["error"] = f"calibration npz not found: {npz_path}"
        return result

    input_name, model_size, inputs_meta, outputs_meta = _validate_model(model_path)
    print(f"\n{'='*60}")
    print(f"  {output_name}")
    print(f"{'='*60}")
    print(f"  Model      : {model_path.name}  ({os.path.getsize(model_path)/(1024*1024):.1f} MiB)")
    print(f"  Calibration: {npz_path.name}  ({os.path.getsize(npz_path)/(1024*1024):.0f} MiB)")
    print(f"  Input      : {input_name}  [1,3,{model_size},{model_size}]  float32")
    print(f"  Outputs    : {len(outputs_meta)}")
    print(f"  GPU        : {gpu_provider or 'CPU only'}")

    # ── load calibration ──
    print(f"\n  [1/4] Loading calibration data...", flush=True)
    t0 = time.perf_counter()
    data = np.load(str(npz_path))
    npz_key = data.files[0]
    calib_images = data[npz_key][:max_samples].astype(np.float32)  # (N,C,H,W)
    if calib_images.ndim == 3:
        # (W, H, C) → batch
        calib_images = calib_images[np.newaxis]
    if calib_images.ndim != 4:
        result["error"] = f"Unexpected npz shape: {calib_images.shape}, need 4D"
        return result
    if calib_images.shape[1] != 3:
        # (N, H, W, C) → (N, C, H, W)
        calib_images = np.transpose(calib_images, (0, 3, 1, 2))
    print(f"  Loaded {calib_images.shape[0]} samples  shape={calib_images.shape}  dtype={calib_images.dtype}", flush=True)
    print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

    # ── resize if needed ──
    calib_h, calib_w = calib_images.shape[2], calib_images.shape[3]
    if calib_h != model_size or calib_w != model_size:
        print(f"\n  Resizing calibration {calib_h}x{calib_w} → {model_size}x{model_size}...", flush=True)
        t0 = time.perf_counter()
        calib_images = _resize_calibration(calib_images, model_size)
        print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

    # ── step 1: ONNX shape inference ──
    preprocessed_path = output_dir / f"{output_name}.preprocessed.onnx"
    print(f"\n  [2/4] ONNX shape inference...", flush=True)
    t0 = time.perf_counter()
    try:
        quant_pre_process(
            input_model=str(model_path),
            output_model_path=str(preprocessed_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
        )
        quant_input = preprocessed_path
        print(f"  Done in {time.perf_counter()-t0:.1f}s  →  {preprocessed_path.name}", flush=True)
    except Exception as exc:
        print(f"  Shape inference skipped: {exc}", flush=True)
        quant_input = model_path

    # ── step 2: calibration providers ──
    calib_providers = [gpu_provider, "CPUExecutionProvider"] if gpu_provider else ["CPUExecutionProvider"]

    # ── step 3: quantize ──
    quantized_path = output_dir / f"{output_name}_int8_qdq.onnx"
    print(f"\n  [3/4] Static quantization → {quantized_path.name}", flush=True)
    t0 = time.perf_counter()
    reader = NpzCalibrationReader(input_name, calib_images)
    quantize_static(
        model_input=str(quant_input),
        model_output=str(quantized_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=calib_providers,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )
    quant_ms = (time.perf_counter() - t0) * 1000
    size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"  Done in {quant_ms/1000:.1f}s  →  {quantized_path.name}  ({size_mb:.1f} MiB)", flush=True)

    # ── step 4: verify ──
    print(f"\n  [4/4] Verifying quantized model...", flush=True)
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    sample = np.zeros((1, 3, model_size, model_size), dtype=np.float32)
    outputs = session.run(None, {input_name: sample})
    ok = all(np.isfinite(o).all() for o in outputs)
    for i, o in enumerate(outputs):
        print(f"    output[{i}]: shape={o.shape}  dtype={o.dtype}  "
              f"finite={bool(np.isfinite(o).all())}  "
              f"range=[{float(o.min()):.4f}, {float(o.max()):.4f}]", flush=True)

    verify_t = time.perf_counter() - t0
    total_t = time.perf_counter() - t_total_start

    if ok:
        print(f"\n  PASSED  ({total_t:.0f}s total)", flush=True)
        result["status"] = "OK"
    else:
        print(f"\n  WARNING: non-finite outputs detected", flush=True)
        result["status"] = "NON_FINITE"

    result["output"] = str(quantized_path)
    result["input_name"] = input_name
    result["input_size"] = model_size
    result["inputs"] = inputs_meta
    result["outputs"] = outputs_meta
    result["calibration_samples"] = len(calib_images)
    result["quantize_time_s"] = quant_ms / 1000
    result["verify_time_s"] = verify_t

    # cleanup
    if preprocessed_path.exists() and not keep_preprocessed:
        preprocessed_path.unlink()

    return result


# ── main ─────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpu = None if args.no_gpu else _best_gpu_provider(args.gpu_provider)

    print("=" * 60)
    print("  YOLO INT8 Quantizer (GPU accelerated)")
    print("=" * 60)
    print(f"  GPU provider : {gpu or 'CPU (disabled)'}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Per-channel  : False")
    print(f"  Quant format : QDQ  |  QInt8 symmetric  |  per-tensor")

    results: list[dict] = []

    if not args.skip_road:
        results.append(quantize_one(
            model_path=args.road_model,
            npz_path=args.road_npz,
            output_name="road_yolo11n_seg",
            output_dir=args.output_dir,
            gpu_provider=gpu,
            max_samples=args.max_calibration_samples,
            keep_preprocessed=args.keep_preprocessed,
        ))

    if not args.skip_tree:
        results.append(quantize_one(
            model_path=args.tree_model,
            npz_path=args.tree_npz,
            output_name="tree_furniture",
            output_dir=args.output_dir,
            gpu_provider=gpu,
            max_samples=args.max_calibration_samples,
            keep_preprocessed=args.keep_preprocessed,
        ))

    # ── summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status_icon = "[OK]" if r["status"] == "OK" else "[FAIL]"
        out_name = Path(r.get("output", "?")).name if r.get("output") else "?"
        print(f"  {status_icon}  {r.get('model','?')}  →  {out_name}")
        if "error" in r:
            print(f"       error: {r['error']}")

    manifest = {
        "created_at_epoch": int(time.time()),
        "gpu_provider": gpu,
        "quant_config": {
            "format": "QDQ",
            "per_channel": False,
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "symmetric": True,
            "reduce_range": False,
            "method": "MinMax",
        },
        "results": results,
    }
    manifest_path = args.output_dir / "gpu_quantize_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"\n  Manifest: {manifest_path}")

    all_ok = all(r["status"] == "OK" for r in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
