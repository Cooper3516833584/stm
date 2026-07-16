"""Measure actual road perception frame rate.

Captures frames from the road-following camera and runs the full perception
pipeline (preprocess → ONNX inference → mask decode → centerline extraction →
offset compensation) to report the realistic processing speed.

Usage:
    PYTHONPATH=. python FlightController/tools/bench_vision_fps.py
    PYTHONPATH=. python FlightController/tools/bench_vision_fps.py --frames 100 --index 7
"""

import argparse
import math
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark road perception pipeline FPS"
    )
    parser.add_argument("--frames", type=int, default=50,
                        help="number of frames to process (default: 50)")
    parser.add_argument("--index", type=int, default=7,
                        help="cv2 camera index (default: 7)")
    parser.add_argument("--model", default=None,
                        help="ONNX model path (default: auto)")
    parser.add_argument("--model-npu", default=None,
                        help=".nb NPU compiled model path")
    parser.add_argument("--flight-height-m", type=float, default=1.0,
                        help="flight height for m/px calc (default: 1.0)")
    parser.add_argument("--no-offset-comp", action="store_true",
                        help="disable offset compensation")
    args = parser.parse_args()

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(args.index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {args.index}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(5):
        cap.read()

    # --- import road perception (triggers model session load) ---
    import road_perception
    if args.model:
        road_perception.MODEL_PATH = args.model
        road_perception._AUTO_USE_NPU = False
    if args.model_npu:
        road_perception.MODEL_PATH_NPU = args.model_npu
        road_perception._AUTO_USE_NPU = True

    t0 = time.perf_counter()
    from road_perception import (
        CameraOffsetCompensationConfig,
        get_road_perception,
    )
    t_import = time.perf_counter() - t0
    print(f"import + ONNX session load: {t_import:.2f}s")

    offset_cfg: CameraOffsetCompensationConfig | None = None
    if not args.no_offset_comp:
        offset_cfg = CameraOffsetCompensationConfig(
            enabled=True,
            cam_forward_offset_m=-0.0787,
        )

    # --- warmup (3 frames, exclude from stats) ---
    for _ in range(3):
        ok, frame = cap.read()
        if not ok:
            print("ERROR: warmup frame read failed", file=sys.stderr)
            cap.release()
            return 2
        get_road_perception(frame, flight_height_m=args.flight_height_m,
                            offset_comp_config=offset_cfg)

    # --- benchmark ---
    times: list[float] = []
    valid_frames = 0
    total_frames = 0
    stage_stats: dict[str, list[float]] = {
        "capture": [], "perception": [], "total": [],
    }

    while total_frames < args.frames:
        t_cap_start = time.perf_counter()
        ok, frame = cap.read()
        t_cap_end = time.perf_counter()

        if not ok:
            print(f"WARN: frame {total_frames} read failed, skipping", file=sys.stderr)
            continue

        total_frames += 1
        t_perc_start = time.perf_counter()
        result = get_road_perception(
            frame,
            flight_height_m=args.flight_height_m,
            offset_comp_config=offset_cfg,
        )
        t_perc_end = time.perf_counter()

        capture_ms = (t_cap_end - t_cap_start) * 1000
        perception_ms = (t_perc_end - t_perc_start) * 1000
        total_ms = capture_ms + perception_ms

        stage_stats["capture"].append(capture_ms)
        stage_stats["perception"].append(perception_ms)
        stage_stats["total"].append(total_ms)
        times.append(total_ms)

        if getattr(result, "is_road_found", False):
            valid_frames += 1

        elapsed = time.perf_counter() - t0 - t_import
        if total_frames % 10 == 0:
            recent_avg = sum(times[-10:]) / len(times[-10:])
            print(f"  frame {total_frames:4d}/{args.frames}  "
                  f"total={total_ms:6.1f}ms  avg10={recent_avg:6.1f}ms  "
                  f"fps10={1000/recent_avg:5.1f}  "
                  f"road_found={valid_frames}/{total_frames}")

    cap.release()

    # --- report ---
    arr = np.array(times)
    cap_arr = np.array(stage_stats["capture"])
    perc_arr = np.array(stage_stats["perception"])

    print()
    print("=" * 62)
    print("  Road Perception FPS Benchmark Results")
    print("=" * 62)
    print(f"  frames processed : {total_frames}")
    print(f"  road found       : {valid_frames} ({valid_frames/max(total_frames,1)*100:.0f}%)")
    print(f"  offset comp      : {'enabled' if offset_cfg and offset_cfg.enabled else 'disabled'}")
    print(f"  flight height    : {args.flight_height_m}m")
    print()
    print(f"  {'stage':<16} {'p50':>7} {'p95':>7} {'p99':>7} {'mean':>7} {'max':>7}")
    print(f"  {'─'*16} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for label, a in [("capture", cap_arr), ("perception", perc_arr), ("total", arr)]:
        print(f"  {label:<16} "
              f"{np.percentile(a, 50):6.1f}ms "
              f"{np.percentile(a, 95):6.1f}ms "
              f"{np.percentile(a, 99):6.1f}ms "
              f"{np.mean(a):6.1f}ms "
              f"{np.max(a):6.1f}ms")
    print()
    fps_mean = 1000.0 / max(np.mean(arr), 0.001)
    fps_p50 = 1000.0 / max(np.percentile(arr, 50), 0.001)
    print(f"  mean FPS         : {fps_mean:5.1f}  (loop_hz upper bound)")
    print(f"  p50  FPS         : {fps_p50:5.1f}  (typical achievable)")
    print()
    print(f"  Suggested loop_hz: {max(1, math.floor(fps_mean * 0.7))}  (70% of mean, safety margin)")
    print()
    print("  ── Interpretation ──")
    if fps_mean >= 30:
        print("  Vision pipeline can sustain 30 Hz → align with radar loop_hz.")
    elif fps_mean >= 15:
        print("  Vision pipeline can sustain ~15-20 Hz → 10 Hz is too conservative.")
    elif fps_mean >= 8:
        print("  Vision pipeline ~10 Hz → current default is appropriate.")
    else:
        print("  Vision pipeline < 8 Hz → consider model optimization or smaller input size.")
    print("=" * 62)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
