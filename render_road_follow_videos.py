"""Render road-following geometry and direction arrows over the two map videos."""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

if os.name == "nt":
    conda_root = Path(sys.executable).resolve().parents[2]
    os.add_dll_directory(str(conda_root / "Library" / "bin"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO


VIDEO_ROOT = Path(r"C:\Users\TZDEZACR\Desktop\嵌赛\yolo\map")
WEIGHTS = VIDEO_ROOT / "map_try" / "runs" / "road_seg_full" / "weights" / "best.pt"
ROAD_CODE_ROOT = Path(r"C:\Users\TZDEZACR\Desktop\嵌赛\yolo\test\stm")
OUTPUT_SUFFIX = "_road_follow.mp4"

sys.path.insert(0, str(ROAD_CODE_ROOT))
import road_perception as road  # noqa: E402


def make_perception(
    frame: np.ndarray,
    masks: np.ndarray | None,
    confidences: np.ndarray,
    boxes_xyxy: np.ndarray,
) -> tuple[road.RoadPerceptionResult, np.ndarray | None]:
    """Run the project's road geometry code on masks decoded by the supplied PT model."""
    h, w = frame.shape[:2]
    if masks is None or len(masks) == 0:
        return road._lost_result("YOLO found no road mask"), None

    instances: list[road.RoadInstance] = []
    merged_mask = np.zeros((h, w), dtype=np.uint8)
    for index, raw_mask in enumerate(masks):
        mask = cv2.resize(raw_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = road._clean_mask_keep_single_instance(mask)
        if not np.any(mask):
            continue
        x1, y1, x2, y2 = boxes_xyxy[index]
        instance = road.RoadInstance(
            mask=mask,
            score=float(confidences[index]),
            box_xywh=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
            area=0,
            bottom_touch_px=0,
            bottom_cx=0.0,
        )
        road._refresh_instance_geometry(instance, w, h)
        points, _ = road._extract_centerline_and_intervals(mask)
        if len(points) < road.MIN_FIT_PTS:
            continue
        instance.centerline_points = points
        instances.append(instance)
        merged_mask = cv2.bitwise_or(merged_mask, mask)

    if not instances:
        return road._lost_result("no valid road instance after geometry cleanup"), merged_mask

    current = road._select_current_instance(instances, w, h)
    if current is None:
        return road._lost_result("no current road instance selected"), merged_mask

    pixel_error, _ = road._compute_pixel_error(current.centerline_points, w, h)
    angle = road._compute_centerline_angle(current.centerline_points, h)
    path_width = road._compute_path_width(current.centerline_points, h)
    state = road._detect_road_state_from_instances(current, instances, w, h)
    all_points, row_intervals = road._extract_centerline_and_intervals(current.mask)
    fallback_state, _, _ = road._detect_road_state(all_points, row_intervals, h)
    if len(instances) == 1 and fallback_state in {"fork", "intersection"}:
        state = fallback_state

    branches = road._build_branches_from_instances(
        current, instances, w, h, float(np.mean(confidences))
    )
    if len(branches) <= 1 and state in {"fork", "intersection"}:
        branches = road._build_branches(
            current.centerline_points, row_intervals, w, h, float(np.mean(confidences)), state
        )
    branches = road._label_and_number_branches(branches)
    selected, decision = road.choose_branch(branches, preference="auto")
    if selected is not None:
        pixel_error = selected.pixel_error
        angle = selected.centerline_angle
        path_width = selected.path_width_px

    return road.RoadPerceptionResult(
        is_road_found=True,
        road_state=state,
        pixel_error=float(pixel_error),
        centerline_angle=float(angle),
        path_width_px=float(path_width),
        confidence=float(np.mean(confidences)),
        corrected_pixel_error=float(pixel_error),
        branches=branches,
        selected_branch=selected,
        branch_decision=decision,
        debug_msg=f"PT masks={len(instances)} selected={getattr(selected, 'label', 'none')}",
    ), merged_mask


def draw_debug(frame: np.ndarray, mask: np.ndarray | None, result: road.RoadPerceptionResult) -> np.ndarray:
    """Same visualization convention as road_perception._save_debug_image, in memory."""
    image = frame.copy()
    h, w = image.shape[:2]
    if mask is not None and mask.size:
        overlay = np.zeros_like(image)
        overlay[mask > 0] = (255, 0, 0)
        image = cv2.addWeighted(image, 1.0, overlay, 0.35, 0.0)
    cv2.line(image, (w // 2, 0), (w // 2, h - 1), (0, 255, 0), 1)

    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 165, 255)]
    for index, branch in enumerate(result.branches):
        color = colors[index % len(colors)]
        selected = branch is result.selected_branch or (
            result.selected_branch is not None and branch.branch_id == result.selected_branch.branch_id
        )
        road._draw_polyline(image, branch.points, color, 5 if selected else 2)
        if branch.points:
            x, y = branch.points[len(branch.points) // 2]
            label = f"{'SELECTED ' if selected else ''}B{branch.branch_id} {branch.label}"
            cv2.putText(image, label, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    base_x = max(0, min(w - 1, int(round(w / 2 + result.pixel_error))))
    base_y = h - 12
    cv2.circle(image, (base_x, base_y), 6, (0, 255, 255), -1)
    arrow_len = max(35, int(min(w, h) * 0.16))
    angle = math.radians(result.centerline_angle)
    end = (int(round(base_x + arrow_len * math.cos(angle))), int(round(base_y - arrow_len * math.sin(angle))))
    cv2.arrowedLine(image, (base_x, base_y), end, (255, 255, 255), 3, tipLength=0.25)

    info = [
        f"state={result.road_state} found={result.is_road_found}",
        f"error={result.pixel_error:.1f}px angle={result.centerline_angle:.1f}deg",
        f"conf={result.confidence:.2f} branches={len(result.branches)}",
        result.debug_msg[:72],
    ]
    for index, text in enumerate(info):
        pos = (10, 24 + index * 22)
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def render_video(model: YOLO, source: Path, destination: Path) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {destination}")

    started = time.perf_counter()
    previous_label: str | None = None
    try:
        for frame_number in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            prediction = model.predict(frame, device=0, imgsz=640, conf=0.35, half=True, verbose=False)[0]
            masks = prediction.masks.data.detach().cpu().numpy() if prediction.masks is not None else None
            confidences = prediction.boxes.conf.detach().cpu().numpy() if prediction.boxes is not None else np.empty(0)
            boxes = prediction.boxes.xyxy.detach().cpu().numpy() if prediction.boxes is not None else np.empty((0, 4))
            perception, merged_mask = make_perception(frame, masks, confidences, boxes)
            if perception.selected_branch is not None:
                previous_label = perception.selected_branch.label
            writer.write(draw_debug(frame, merged_mask, perception))
            if (frame_number + 1) % 60 == 0 or frame_number + 1 == count:
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(f"{source.name}: {frame_number + 1}/{count} ({(frame_number + 1) / elapsed:.1f} FPS)", flush=True)
    finally:
        capture.release()
        writer.release()
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"Render failed: {destination}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not WEIGHTS.exists():
        raise FileNotFoundError(WEIGHTS)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    model = YOLO(str(WEIGHTS))
    videos = [VIDEO_ROOT / "video_20260716_150550.mp4", VIDEO_ROOT / "video_20260716_150630.mp4"]
    for source in videos:
        render_video(model, source, Path.cwd() / f"{source.stem}{OUTPUT_SUFFIX}")


if __name__ == "__main__":
    main()
