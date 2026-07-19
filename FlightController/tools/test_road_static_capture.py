"""Capture one road-camera image and render its single-road geometry.

This is a vision-only board-side test.  It never imports or connects to the
flight controller, radar, or navigation controller, so it cannot arm motors
or send a control frame.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _setup_path() -> None:
    root = str(REPOSITORY_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision-only static road capture: save one frame and its centerline overlay."
    )
    parser.add_argument("--camera-index", type=int, default=7)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument(
        "--road-model-backend",
        choices=["npu", "cpu"],
        default="npu",
    )
    parser.add_argument(
        "--road-postprocess-mode",
        choices=["fast-main", "full"],
        default="fast-main",
    )
    parser.add_argument(
        "--model",
        default="FlightController/Solutions/model/road_yolo11n_seg_128.onnx",
        help="Legacy CPU ONNX model path.",
    )
    parser.add_argument(
        "--model-npu",
        default="FlightController/Solutions/model/new_road_seg_v5_final_fp32.nb",
        help="NPU semantic segmentation model path.",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/sdcard/road_static_test",
        help="Board directory for the captured JPEG and annotated JPEG.",
    )
    return parser.parse_args()


def _capture_one_frame(args: argparse.Namespace):
    """Open the same OpenCV camera path as the live perception pipeline."""
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open road camera index {args.camera_index}")

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        for _ in range(max(0, args.warmup_frames)):
            cap.read()

        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Road camera did not return a frame after warm-up")
        return frame
    finally:
        cap.release()


def main() -> int:
    args = parse_args()
    _setup_path()

    import road_perception

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"{timestamp}_road_raw.jpg"
    overlay_path = output_dir / f"{timestamp}_road_centerline.jpg"

    print("Vision-only test: no FC, radar, arming, takeoff, or control commands are used.")
    print(f"Capturing one frame from camera {args.camera_index}...")
    frame = _capture_one_frame(args)
    if not cv2.imwrite(str(raw_path), frame):
        raise RuntimeError(f"Failed to save captured frame: {raw_path}")

    road_perception.configure_model(
        backend=args.road_model_backend,
        cpu_model_path=args.model,
        npu_model_path=args.model_npu,
        postprocess_mode=args.road_postprocess_mode,
    )
    result = road_perception.get_road_perception(
        frame,
        debug_save_path=str(overlay_path),
    )

    print(f"Raw image: {raw_path}")
    print(f"Centerline overlay: {overlay_path}")
    print(
        "result: found={} state={} confidence={:.3f} err={:.1f}px corr={:.1f}px "
        "angle={:.1f}deg points={}".format(
            result.is_road_found,
            result.road_state,
            result.confidence,
            result.pixel_error,
            result.corrected_pixel_error,
            result.centerline_angle,
            len(result.centerline_points),
        )
    )
    if result.debug_msg:
        print(f"detail: {result.debug_msg}")
    return 0 if result.is_road_found else 2


if __name__ == "__main__":
    raise SystemExit(main())
