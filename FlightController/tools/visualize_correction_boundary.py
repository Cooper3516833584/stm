"""Visualize max_correction_px boundary on a camera frame.

Captures one frame from the road-following camera, draws red lines at
±max_correction_px from center, and saves the result to the SD card.

Usage:
    PYTHONPATH=. python FlightController/tools/visualize_correction_boundary.py
    PYTHONPATH=. python FlightController/tools/visualize_correction_boundary.py --index 7 --max-px 120
"""

import argparse
import os
import sys

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize max_correction_px on camera frame"
    )
    parser.add_argument("--index", type=int, default=7,
                        help="cv2 camera index (default: 7, road-following cam)")
    parser.add_argument("--max-px", type=float, default=120.0,
                        help="max_correction_px value (default: 120)")
    parser.add_argument("--out", default="/media/sdcard/max_correction_px.jpg",
                        help="output path (default: /media/sdcard/max_correction_px.jpg)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {args.index}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Discard warmup frames
    for _ in range(5):
        cap.read()

    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("ERROR: failed to read frame", file=sys.stderr)
        return 2

    h, w = frame.shape[:2]
    cx = w // 2
    left_boundary = int(cx - args.max_px)
    right_boundary = int(cx + args.max_px)

    # Draw center line (green)
    cv2.line(frame, (cx, 0), (cx, h - 1), (0, 255, 0), 2)

    # Draw ±max_correction_px lines (red)
    cv2.line(frame, (left_boundary, 0), (left_boundary, h - 1), (0, 0, 255), 2)
    cv2.line(frame, (right_boundary, 0), (right_boundary, h - 1), (0, 0, 255), 2)

    # Annotate
    cv2.putText(frame, f"center", (cx + 4, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"-{args.max_px:.0f}px", (left_boundary - 60, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, f"+{args.max_px:.0f}px", (right_boundary + 4, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, f"max_correction = {args.max_px:.0f}px  ({args.max_px / w * 100:.1f}% of {w}px)",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.out, frame)
    print(f"Saved: {args.out}  ({w}x{h})")
    print(f"  Green = center ({cx}px)")
    print(f"  Red   = +/-{args.max_px:.0f}px boundary  ({left_boundary}px .. {right_boundary}px)")
    print(f"  Range = {args.max_px / w * 100:.1f}% of image width")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
