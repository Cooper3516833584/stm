"""Profile per-frame road-perception stages on the development board.

This diagnostic opens only the camera and road model.  It does not import the
flight-controller or radar paths and cannot send motion commands.

Example::

    PYTHONPATH=. python -u FlightController/tools/profile_vision_stages.py \
        --frames 20 --index 7
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import io
import pstats
import sys
import time
from collections.abc import Callable
from typing import Any


STAGE_FUNCTIONS = (
    ("preprocess", "_preprocess_semantic"),
    ("decode", "_decode_semantic_segmentation"),
    ("decode", "_decode_semantic_fast_main"),
    ("mask_cleanup", "_clean_mask_keep_single_instance"),
    ("centerline", "_extract_centerline_and_intervals"),
    ("centerline", "_extract_fast_main_centerline"),
)


class StageRecorder:
    def __init__(self) -> None:
        self.current: dict[str, float] = {}

    def reset(self) -> None:
        self.current = {}

    def add(self, name: str, elapsed_ms: float) -> None:
        self.current[name] = self.current.get(name, 0.0) + elapsed_ms

    def wrap(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            started_ns = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                self.add(name, (time.perf_counter_ns() - started_ns) / 1_000_000.0)

        return measured


class TimedSession:
    """Delegate model-session calls while timing only ``run``."""

    def __init__(self, session: Any, recorder: StageRecorder) -> None:
        self._session = session
        self._recorder = recorder

    def run(self, *args: Any, **kwargs: Any) -> Any:
        started_ns = time.perf_counter_ns()
        try:
            return self._session.run(*args, **kwargs)
        finally:
            self._recorder.add(
                "npu_inference",
                (time.perf_counter_ns() - started_ns) / 1_000_000.0,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


class TimedNetwork:
    """Split the stai_mpu application-buffer and hardware execution costs."""

    def __init__(self, network: Any, recorder: StageRecorder) -> None:
        self._network = network
        self._recorder = recorder

    def _call(self, stage: str, function: Callable[..., Any], *args: Any) -> Any:
        started_ns = time.perf_counter_ns()
        try:
            return function(*args)
        finally:
            self._recorder.add(stage, (time.perf_counter_ns() - started_ns) / 1_000_000.0)

    def set_input(self, *args: Any) -> Any:
        return self._call("npu_set_input", self._network.set_input, *args)

    def run(self, *args: Any) -> Any:
        return self._call("npu_execute", self._network.run, *args)

    def get_output(self, *args: Any) -> Any:
        return self._call("npu_get_output", self._network.get_output, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._network, name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile road perception stages per frame")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--index", type=int, default=7)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model-npu", default=None)
    parser.add_argument(
        "--road-postprocess-mode",
        choices=["fast-main", "full"],
        default="fast-main",
    )
    parser.add_argument("--no-offset-comp", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="suppress per-frame rows while retaining aggregate statistics",
    )
    parser.add_argument(
        "--cprofile-top",
        type=int,
        default=0,
        help="also print this many aggregate cProfile entries (adds overhead)",
    )
    return parser.parse_args()


def _summary(values: list[float], np: Any) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(array)),
        float(np.percentile(array, 50)),
        float(np.percentile(array, 95)),
        float(np.max(array)),
    )


def main() -> int:
    args = _parse_args()
    if args.frames <= 0 or args.warmup < 0:
        print("ERROR: --frames must be positive and --warmup non-negative", file=sys.stderr)
        return 2

    import cv2
    import numpy as np
    import road_perception as road

    road.configure_model(
        backend="npu",
        npu_model_path=args.model_npu,
        postprocess_mode=args.road_postprocess_mode,
    )
    model_info = road.get_model_io_info()  # Load before installing timing wrappers.
    print(
        "model: "
        f"provider={model_info['provider']} kind={model_info['model_kind']} "
        f"input_size={model_info['input_size']} "
        f"postprocess={model_info['postprocess_mode']}",
        flush=True,
    )

    recorder = StageRecorder()
    for stage_name, function_name in STAGE_FUNCTIONS:
        original = getattr(road, function_name)
        setattr(road, function_name, recorder.wrap(stage_name, original))

    original_get_session = road._get_session
    session, input_name = original_get_session()
    if hasattr(session, "_internal"):
        session._internal = TimedNetwork(session._internal, recorder)
    timed_session = TimedSession(session, recorder)

    def get_timed_session() -> tuple[Any, str]:
        return timed_session, input_name

    road._get_session = get_timed_session

    offset_config = None
    if not args.no_offset_comp:
        offset_config = road.CameraOffsetCompensationConfig(
            enabled=True,
            cam_forward_offset_m=-0.0787,
        )

    cap = cv2.VideoCapture(args.index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {args.index}", file=sys.stderr)
        return 3
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        for _ in range(5):
            cap.read()

        for index in range(args.warmup):
            ok, frame = cap.read()
            if not ok:
                print(f"ERROR: warmup frame {index + 1} capture failed", file=sys.stderr)
                return 4
            recorder.reset()
            road.get_road_perception(frame, offset_comp_config=offset_config)

        profile = cProfile.Profile() if args.cprofile_top > 0 else None
        all_rows: list[dict[str, float]] = []
        road_found = 0

        if not args.summary_only:
            print(f"frame shape: {frame.shape}", flush=True)
            print(
                "frame  capture    total  preprocess      npu (set/run/get)   decode "
                " cleanup centerline    other  found/state",
                flush=True,
            )
        for frame_index in range(1, args.frames + 1):
            capture_started_ns = time.perf_counter_ns()
            ok, frame = cap.read()
            capture_ms = (time.perf_counter_ns() - capture_started_ns) / 1_000_000.0
            if not ok:
                print(f"WARN: frame {frame_index} capture failed", file=sys.stderr)
                continue

            recorder.reset()
            perception_started_ns = time.perf_counter_ns()
            if profile is not None:
                profile.enable()
            result = road.get_road_perception(frame, offset_comp_config=offset_config)
            if profile is not None:
                profile.disable()
            total_ms = (time.perf_counter_ns() - perception_started_ns) / 1_000_000.0

            row = dict(recorder.current)
            row["capture"] = capture_ms
            row["perception_total"] = total_ms
            row["npu_wrapper"] = max(
                0.0,
                row.get("npu_inference", 0.0)
                - row.get("npu_set_input", 0.0)
                - row.get("npu_execute", 0.0)
                - row.get("npu_get_output", 0.0),
            )
            measured_ms = sum(
                row.get(name, 0.0)
                for name in (
                    "preprocess",
                    "npu_inference",
                    "decode",
                    "mask_cleanup",
                    "centerline",
                )
            )
            row["other"] = max(0.0, total_ms - measured_ms)
            all_rows.append(row)
            if result.is_road_found:
                road_found += 1

            if not args.summary_only:
                print(
                    f"{frame_index:5d} "
                    f"{capture_ms:8.1f} {total_ms:8.1f} "
                    f"{row.get('preprocess', 0.0):10.1f} "
                    f"{row.get('npu_inference', 0.0):8.1f} "
                    f"({row.get('npu_set_input', 0.0):3.0f}/"
                    f"{row.get('npu_execute', 0.0):3.0f}/"
                    f"{row.get('npu_get_output', 0.0):3.0f}) "
                    f"{row.get('decode', 0.0):8.1f} "
                    f"{row.get('mask_cleanup', 0.0):8.1f} "
                    f"{row.get('centerline', 0.0):10.1f} "
                    f"{row['other']:8.1f}  "
                    f"{result.is_road_found}/{result.road_state}",
                    flush=True,
                )

        if not all_rows:
            print("ERROR: no frames were profiled", file=sys.stderr)
            return 5

        print("\naggregate stage timing (ms):")
        print("stage                 mean      p50      p95      max   share")
        total_mean = _summary([row["perception_total"] for row in all_rows], np)[0]
        ordered_stages = (
            "capture",
            "preprocess",
            "npu_set_input",
            "npu_execute",
            "npu_get_output",
            "npu_wrapper",
            "npu_inference",
            "decode",
            "mask_cleanup",
            "centerline",
            "other",
            "perception_total",
        )
        for stage in ordered_stages:
            values = [row.get(stage, 0.0) for row in all_rows]
            mean, p50, p95, maximum = _summary(values, np)
            share = mean / total_mean * 100.0 if stage != "capture" else 0.0
            print(
                f"{stage:<20} {mean:8.1f} {p50:8.1f} {p95:8.1f} "
                f"{maximum:8.1f} {share:6.1f}%"
            )
        print(f"road found: {road_found}/{len(all_rows)}")

        if profile is not None:
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(
                args.cprofile_top
            )
            print("\ncProfile aggregate (cumulative time):")
            print(stream.getvalue())
        return 0
    finally:
        cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
