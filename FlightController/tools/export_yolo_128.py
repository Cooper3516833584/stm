"""Export YOLO11n-seg at 128x128 input resolution for CPU inference.

This script re-exports the Ultralytics YOLO11n-seg model at a reduced
input resolution (128x128) to produce a lighter-weight ONNX model suitable
for CPU-only inference on the STM32MP257 (Cortex-A35 dual-core).

Usage::

    conda activate yolo11
    pip install ultralytics onnxsim  # install if not already present
    python FlightController/tools/export_yolo_128.py

Output::

    FlightController/Solutions/model/road_yolo11n_seg_128.onnx

The exported model has:
  - input:  [1, 3, 128, 128] float32 (NCHW, values [0, 1])
  - output0: [1, 37, N] detection head (N ~336 candidates at 128px)
  - output1: [1, 32, 32, 32] prototype masks

Compared to the original 320x320 model (~11.5 MB, ~1800ms on CPU),
the 128x128 model is ~2.8 MB with an estimated 200-400ms inference time
(FLOPs reduced by ~6.25x).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "FlightController" / "Solutions" / "model"
OUTPUT_NAME = "road_yolo11n_seg_128"


def _check_ultralytics() -> None:
    """Ensure ultralytics is installed; print a helpful message if not."""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print(
            "ultralytics is not installed. Install it with:\n"
            "    pip install ultralytics\n"
            "Then re-run this script."
        )
        sys.exit(1)


def export_yolo_128(
    output_dir: Path | None = None,
    imgsz: int = 128,
    half: bool = False,
    weights: str | None = None,
) -> Path:
    """Export YOLO11n-seg to ONNX and return the output path.

    Parameters
    ----------
    output_dir:
        Directory for the exported ONNX file.  Defaults to ``MODEL_DIR``.
    imgsz:
        Input resolution (square).  128 by default; use 160 for a
        quality-vs-speed trade-off.
    half:
        If True, export FP16 weights (smaller file, faster on some ARM CPUs
        with fp16 SIMD support).  Defaults to False (FP32).

    Returns
    -------
    Path
        Absolute path to the exported ONNX file.
    """
    from ultralytics import YOLO

    if output_dir is None:
        output_dir = MODEL_DIR
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / OUTPUT_NAME

    weights_path = weights or "yolo11n-seg.pt"
    print(f"Loading YOLO11n-seg weights: {weights_path}")
    model = YOLO(weights_path)

    print(f"Exporting ONNX @ imgsz={imgsz}, half={half} → {output_path}.onnx")
    exported = model.export(
        format="onnx",
        opset=14,
        simplify=True,
        dynamic=False,
        imgsz=imgsz,
        batch=1,
        half=half,
        nms=False,
    )

    # Ultralytics .export() returns the path as a string.
    exported_path = Path(str(exported))
    if exported_path != output_path.with_suffix(".onnx"):
        import shutil

        shutil.move(str(exported_path), str(output_path.with_suffix(".onnx")))
        exported_path = output_path.with_suffix(".onnx")

    print(f"Exported: {exported_path}  ({exported_path.stat().st_size / 1e6:.1f} MB)")
    return exported_path


def optimize_onnx(model_path: Path) -> Path:
    """Run onnxsim constant-folding and graph simplification.

    Returns the (possibly replaced) path to the optimized model.
    """
    try:
        from onnxsim import simplify
    except ImportError:
        print("onnxsim not installed — skipping graph simplification.")
        print("  Install with: pip install onnxsim")
        return model_path

    import onnx

    print(f"Simplifying with onnxsim: {model_path.name} ...")
    original = onnx.load(str(model_path))
    simplified, check_ok = simplify(original)

    if not check_ok:
        print(
            "  WARNING: onnxsim consistency check failed — "
            "the simplified model may produce different outputs."
        )

    optimized_path = model_path.with_name(model_path.stem + "_opt.onnx")
    onnx.save(simplified, str(optimized_path))

    size_before = model_path.stat().st_size
    size_after = optimized_path.stat().st_size
    reduction = (1.0 - size_after / max(1, size_before)) * 100.0
    print(
        f"  Optimized: {optimized_path.name}  "
        f"({size_before / 1e6:.1f} → {size_after / 1e6:.1f} MB, "
        f"-{reduction:.1f}%)"
    )

    return optimized_path


def validate_model(model_path: Path, imgsz: int) -> bool:
    """Run a quick sanity-check inference with onnxruntime CPU."""
    import numpy as np
    import onnxruntime as ort

    print(f"Validating with onnxruntime CPU: {model_path.name} ...")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    print(f"  Input : name={input_name!r}  shape={input_shape}  type={input_meta.type}")

    for i, out in enumerate(session.get_outputs()):
        print(f"  Output {i}: name={out.name!r}  shape={out.shape}  type={out.type}")

    # Create a dummy input and run inference.
    dummy = np.zeros(input_shape, dtype=np.float32)
    outputs = session.run(None, {input_name: dummy})

    for i, arr in enumerate(outputs):
        print(f"  Output {i} actual shape: {arr.shape}  dtype: {arr.dtype}")

    # Basic shape checks.
    out0 = outputs[0]
    out1 = outputs[1]
    ok = True

    if out0.ndim != 3 or out0.shape[0] != 1:
        print(f"  FAIL: output0 should be [1, C, N], got {out0.shape}")
        ok = False
    if out1.ndim != 4 or out1.shape[1] != 32:
        print(f"  FAIL: output1 should be [1, 32, H, W], got {out1.shape}")
        ok = False
    if input_shape[2] != imgsz or input_shape[3] != imgsz:
        print(f"  FAIL: input spatial != {imgsz}, got {input_shape[2]}x{input_shape[3]}")
        ok = False

    if ok:
        print("  PASS: model structure looks valid.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export YOLO11n-seg at reduced resolution for CPU inference",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=128,
        help="Square input resolution (default: 128).  Use 160 for quality trade-off.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 weights (smaller file, may be faster on ARM with fp16 SIMD).",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip onnxsim graph simplification.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: FlightController/Solutions/model/).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Path to YOLO11n-seg .pt weights file. "
            "If not provided, downloads the COCO-pretrained 'yolo11n-seg.pt' "
            "from Ultralytics.  To export a road-fine-tuned model, pass the "
            "path to your custom .pt file."
        ),
    )
    args = parser.parse_args()

    _check_ultralytics()

    exported = export_yolo_128(
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        half=args.half,
        weights=args.weights,
    )

    final = exported
    if not args.no_optimize:
        final = optimize_onnx(exported)

    ok = validate_model(final, args.imgsz)
    if not ok:
        print("\nValidation FAILED — inspect the exported model before deploying.")
        return 1

    print(f"\nDone. Model ready: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
