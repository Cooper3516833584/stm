"""Inspect road-follow controller inputs over the reported 18-19 second interval."""

from __future__ import annotations

import cv2
import numpy as np

import render_road_follow_videos as render


source = render.VIDEO_ROOT / "video_20260716_204702_360p.mp4"
capture = cv2.VideoCapture(str(source))
fps = capture.get(cv2.CAP_PROP_FPS)
start_s = 17.5
capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
model = render.YOLO(str(render.WEIGHTS))
for index in range(int(2.0 * fps)):
    ok, frame = capture.read()
    if not ok:
        break
    prediction = model.predict(
        frame, device=0, imgsz=640, conf=0.35, half=True, verbose=False
    )[0]
    masks = (
        prediction.masks.data.detach().cpu().numpy()
        if prediction.masks is not None
        else None
    )
    confidences = (
        prediction.boxes.conf.detach().cpu().numpy()
        if prediction.boxes is not None
        else np.empty(0)
    )
    boxes = (
        prediction.boxes.xyxy.detach().cpu().numpy()
        if prediction.boxes is not None
        else np.empty((0, 4))
    )
    result, _ = render.make_perception(frame, masks, confidences, boxes)
    error = result.pixel_error
    angle = result.centerline_angle
    controller_error = 0.0 if abs(error) < 20.0 else error
    yaw_rate = max(-25.0, min(25.0, 0.08 * controller_error + 0.4 * (angle - 90.0)))
    if index % 6 == 0:
        print(
            f"{start_s + index / fps:5.2f}s "
            f"state={result.road_state:12s} mode=single-road "
            f"error={error:6.1f} angle={angle:6.1f} yaw={yaw_rate:6.1f}"
        )

capture.release()
