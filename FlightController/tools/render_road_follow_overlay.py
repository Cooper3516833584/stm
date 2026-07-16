"""Render road-follow perception and command overlays onto videos.

This is an offline companion for road_follow_main.py. It reuses the same
road_perception + RoadFollower path so demo videos show the controller's real
selected path and steering intent.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import types
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _setup_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)
    return root


ROOT = _setup_path()

import road_perception  # noqa: E402
from road_perception import CameraWhiteBalanceConfig  # noqa: E402


def _load_solution_module(module_name: str, path: Path):
    """Load a Solutions module without executing FlightController/__init__.py."""
    fc_pkg = sys.modules.setdefault("FlightController", types.ModuleType("FlightController"))
    fc_pkg.__path__ = [str(ROOT / "FlightController")]
    solutions_pkg = sys.modules.setdefault(
        "FlightController.Solutions",
        types.ModuleType("FlightController.Solutions"),
    )
    solutions_pkg.__path__ = [str(ROOT / "FlightController" / "Solutions")]

    full_name = f"FlightController.Solutions.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {full_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


_load_solution_module("ObstacleUtils", ROOT / "FlightController/Solutions/ObstacleUtils.py")
_load_solution_module("Safety", ROOT / "FlightController/Solutions/Safety.py")
_road_follower_module = _load_solution_module(
    "RoadFollower",
    ROOT / "FlightController/Solutions/RoadFollower.py",
)
RoadFollower = _road_follower_module.RoadFollower
RoadFollowerConfig = _road_follower_module.RoadFollowerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay road-follow path perception and direction arrows on videos."
    )
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        help="Input videos. Defaults to temp/video*.mp4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "temp",
        help="Directory for rendered videos.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "FlightController/Solutions/model/road_yolo11n_seg.onnx",
    )
    parser.add_argument("--branch", choices=["auto", "straight", "left", "right"], default="auto")
    parser.add_argument("--output-width", type=int, default=1920)
    parser.add_argument("--perception-width", type=int, default=960)
    parser.add_argument(
        "--perception-every-n",
        type=int,
        default=2,
        help="Run neural perception every N frames and hold overlay in between.",
    )
    parser.add_argument("--suffix", default="_road_overlay")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug limit; 0 renders all frames.")
    parser.add_argument("--wb-enable", action="store_true")
    parser.add_argument("--wb-r", type=float, default=2.78)
    parser.add_argument("--wb-g", type=float, default=1.00)
    parser.add_argument("--wb-b", type=float, default=1.26)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")
    road_perception.configure_model(backend="cpu", cpu_model_path=str(args.model))

    videos = args.videos or sorted((ROOT / "temp").glob("video*.mp4"))
    if not videos:
        raise FileNotFoundError("No input videos found. Pass files or place video*.mp4 in temp.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        render_video(video.resolve(), args)


def render_video(video_path: Path, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    src_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

    out_w, out_h = _scaled_size(src_w, src_h, args.output_width)
    percep_w, percep_h = _scaled_size(src_w, src_h, args.perception_width)
    output_path = args.output_dir / f"{video_path.stem}{args.suffix}.mp4"

    writer = _open_writer(output_path, fps, out_w, out_h)
    follower = RoadFollower(
        RoadFollowerConfig(
            image_width=percep_w,
            branch_preference=args.branch,
        )
    )
    wb_config = CameraWhiteBalanceConfig(
        enabled=bool(args.wb_enable),
        r_gain=args.wb_r,
        g_gain=args.wb_g,
        b_gain=args.wb_b,
    )

    last_perception = None
    last_command = None
    frame_idx = 0
    found_frames = 0
    started = time.perf_counter()
    last_log = started

    print(
        f"[render] {video_path.name}: {src_w}x{src_h}@{fps:.2f} -> "
        f"{out_w}x{out_h}, perception={percep_w}x{percep_h}, frames={total_frames or '?'}",
        flush=True,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            output_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            should_perceive = frame_idx % max(1, args.perception_every_n) == 0 or last_perception is None
            if should_perceive:
                percep_frame = cv2.resize(frame, (percep_w, percep_h), interpolation=cv2.INTER_AREA)
                last_perception = road_perception.get_road_perception(
                    percep_frame,
                    branch_preference=args.branch,
                    previous_branch_label=follower.previous_branch_label(),
                    wb_config=wb_config,
                )
                last_command = follower.update(last_perception, now_s=frame_idx / max(fps, 1e-6))

            if bool(getattr(last_perception, "is_road_found", False)):
                found_frames += 1
            overlay = draw_overlay(
                output_frame,
                last_perception,
                last_command,
                source_size=(percep_w, percep_h),
                frame_idx=frame_idx,
                fps=fps,
            )
            writer.write(overlay)

            frame_idx += 1
            now = time.perf_counter()
            if now - last_log >= 5.0:
                speed = frame_idx / max(now - started, 1e-6)
                progress = f"{frame_idx}/{total_frames}" if total_frames else str(frame_idx)
                print(f"[render] {video_path.name}: {progress} frames, {speed:.1f} fps", flush=True)
                last_log = now
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - started
    print(
        f"[done] {output_path} frames={frame_idx} found={found_frames} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


def draw_overlay(
    frame: np.ndarray,
    perception,
    command,
    *,
    source_size: tuple[int, int],
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    out = frame.copy()
    out_h, out_w = out.shape[:2]
    src_w, src_h = source_size
    sx = out_w / float(src_w)
    sy = out_h / float(src_h)

    mask = getattr(perception, "debug_mask", None)
    if mask is not None and getattr(mask, "size", 0):
        mask_scaled = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        color_layer = np.zeros_like(out)
        color_layer[mask_scaled > 0] = (255, 155, 35)
        out = cv2.addWeighted(out, 1.0, color_layer, 0.30, 0.0)

    branches = list(getattr(perception, "branches", []) or [])
    selected = getattr(perception, "selected_branch", None)
    for idx, branch in enumerate(branches):
        is_selected = _is_selected_branch(branch, selected)
        color = (0, 245, 255) if is_selected else (235, 235, 235)
        thickness = max(4, out_w // 360) if is_selected else max(2, out_w // 700)
        _draw_polyline(out, _scale_points(getattr(branch, "points", []), sx, sy), color, thickness)

    active_branch = selected or (branches[0] if branches else None)
    if active_branch is not None:
        arrow_points = _scale_points(getattr(active_branch, "points", []), sx, sy)
        _draw_path_arrow(out, arrow_points)
    else:
        _draw_search_arrow(out, command)

    _draw_hud(out, perception, command, frame_idx, fps)
    return out


def _draw_path_arrow(out: np.ndarray, points: list[tuple[int, int]]) -> None:
    h, w = out.shape[:2]
    if len(points) >= 2:
        start = points[0]
        end = points[min(len(points) - 1, max(1, len(points) // 3))]
    else:
        start = (w // 2, int(h * 0.88))
        end = (w // 2, int(h * 0.68))

    color = (0, 165, 255)
    thickness = max(8, w // 190)
    cv2.arrowedLine(out, start, end, color, thickness, cv2.LINE_AA, tipLength=0.22)
    cv2.circle(out, start, max(8, w // 160), (0, 245, 255), -1, cv2.LINE_AA)


def _draw_search_arrow(out: np.ndarray, command) -> None:
    h, w = out.shape[:2]
    yaw = float(getattr(command, "yaw_rate_deg_s", 0.0) or 0.0)
    angle_deg = 90.0 + max(-35.0, min(35.0, yaw / 25.0 * 35.0))
    length = int(min(w, h) * 0.22)
    start = (w // 2, int(h * 0.82))
    rad = math.radians(angle_deg)
    end = (int(start[0] + length * math.cos(rad)), int(start[1] - length * math.sin(rad)))
    cv2.arrowedLine(out, start, end, (0, 165, 255), max(8, w // 190), cv2.LINE_AA, tipLength=0.22)


def _draw_hud(out: np.ndarray, perception, command, frame_idx: int, fps: float) -> None:
    h, w = out.shape[:2]
    state = str(getattr(perception, "road_state", "lost"))
    found = bool(getattr(perception, "is_road_found", False))
    branch = getattr(getattr(perception, "selected_branch", None), "label", "none")
    conf = float(getattr(perception, "confidence", 0.0) or 0.0)
    err = float(getattr(perception, "corrected_pixel_error", getattr(perception, "pixel_error", 0.0)) or 0.0)
    angle = float(getattr(perception, "centerline_angle", 90.0) or 90.0)
    vx = float(getattr(command, "vx_cm_s", 0.0) or 0.0)
    yaw = float(getattr(command, "yaw_rate_deg_s", 0.0) or 0.0)
    reason = str(getattr(command, "reason", "no_command"))
    timestamp = frame_idx / max(fps, 1e-6)

    lines = [
        f"road={state} found={int(found)} branch={branch} conf={conf:.2f}",
        f"error={err:+.0f}px angle={angle:.1f}deg cmd_vx={vx:.1f} cmd_yaw={yaw:+.1f}deg/s",
        f"t={timestamp:.1f}s {reason}",
    ]
    x, y = 28, 42
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, min(1.1, w / 1920.0))
    line_h = int(34 * scale)
    pad = int(14 * scale)
    max_text_w = 0
    for line in lines:
        (tw, _), _baseline = cv2.getTextSize(line, font, scale, 2)
        max_text_w = max(max_text_w, tw)
    rect_w = min(w - x * 2, max_text_w + pad * 2)
    rect_h = line_h * len(lines) + pad
    cv2.rectangle(out, (x - pad, y - line_h), (x - pad + rect_w, y - line_h + rect_h), (0, 0, 0), -1)
    cv2.rectangle(out, (x - pad, y - line_h), (x - pad + rect_w, y - line_h + rect_h), (0, 245, 255), 2)
    for i, line in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(out, line, (x, yy), font, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, line, (x, yy), font, scale, (255, 255, 255), 2, cv2.LINE_AA)


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for codec in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {path}")


def _scaled_size(src_w: int, src_h: int, target_w: int) -> tuple[int, int]:
    width = int(max(2, target_w))
    height = int(round(src_h * (width / float(src_w))))
    if height % 2:
        height += 1
    if width % 2:
        width += 1
    return width, height


def _scale_points(points: Iterable[tuple[float, float]], sx: float, sy: float) -> list[tuple[int, int]]:
    return [(int(round(float(x) * sx)), int(round(float(y) * sy))) for x, y, *_ in points]


def _draw_polyline(
    img: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _is_selected_branch(branch, selected) -> bool:
    if selected is None:
        return False
    return branch is selected or getattr(branch, "branch_id", None) == getattr(selected, "branch_id", None)


if __name__ == "__main__":
    main()
