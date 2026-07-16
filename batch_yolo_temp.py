"""Batch YOLO inference on temp/ images using road_yolo11n_seg.onnx.

Usage:  python batch_yolo_temp.py
"""

import glob
import os
import sys
import time

# Ensure the project root is on sys.path for road_perception imports.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2

import road_perception

# ── Force ONNX runtime (CPU) — skip NPU .nb model on desktop ─────
road_perception._AUTO_USE_NPU = False

TEMP_DIR = os.path.join(_PROJECT_ROOT, "temp")
PATTERN = os.path.join(TEMP_DIR, "*.jpg")


def main() -> int:
    image_paths = sorted(
        p for p in glob.glob(PATTERN)
        if "_yoloed" not in os.path.basename(p)
    )
    if not image_paths:
        print(f"[SKIP] No .jpg images found in {TEMP_DIR} (excluding *_yoloed)")
        return 0

    print(f"Found {len(image_paths)} image(s) to process.\n")

    for idx, img_path in enumerate(image_paths, start=1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(TEMP_DIR, f"{base}_yoloed.jpg")

        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[{idx:02d}/{len(image_paths):02d}] SKIP  cannot read: {img_path}")
            continue

        t0 = time.perf_counter()
        result = road_perception.get_road_perception(frame, debug_save_path=out_path)
        elapsed = time.perf_counter() - t0

        print(
            f"[{idx:02d}/{len(image_paths):02d}] {base}  "
            f"state={result.road_state}  conf={result.confidence:.3f}  "
            f"({elapsed:.2f}s)"
        )

    print(f"\nDone. Output files saved to {TEMP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
