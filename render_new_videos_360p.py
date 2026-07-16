"""Compress the two new 4K map videos to 360p, then render road-follow arrows."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from render_road_follow_videos import VIDEO_ROOT, WEIGHTS, YOLO, render_video


SOURCES = (
    VIDEO_ROOT / "video_20260716_204625.mp4",
    VIDEO_ROOT / "video_20260716_204702.mp4",
)
TARGET_SIZE = (640, 360)


def compress_to_360p(source: Path, destination: Path) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, TARGET_SIZE
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {destination}")

    started = time.perf_counter()
    processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA))
            processed += 1
            if processed % 300 == 0 or processed == total:
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(f"compress {source.name}: {processed}/{total} ({processed / elapsed:.1f} FPS)", flush=True)
    finally:
        capture.release()
        writer.release()
    if processed != total or not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"Compression did not finish: {source}")


def main() -> None:
    model = YOLO(str(WEIGHTS))
    for source in SOURCES:
        compressed = Path.cwd() / f"{source.stem}_360p.mp4"
        rendered = Path.cwd() / f"{source.stem}_360p_road_follow.mp4"
        compress_to_360p(source, compressed)
        render_video(model, compressed, rendered)


if __name__ == "__main__":
    main()
