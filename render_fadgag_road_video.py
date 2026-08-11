"""One-off runner: render the road-follow yoloseg overlay for the fadgag video.

Reuses render_road_follow_videos.render_video (same model, same geometry
pipeline, same debug drawing) with a custom source/destination pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Pre-import the local (current) road_perception so that
# render_road_follow_videos' hardcoded sys.path insert of the older
# Desktop/嵌赛/yolo/test/stm copy is bypassed via sys.modules.
import road_perception  # noqa: F401,E402

from ultralytics import YOLO  # noqa: E402

from render_road_follow_videos import render_video  # noqa: E402


SOURCE = Path(r"D:\drone2\video_20260811_160549.mp4")
DESTINATION = SOURCE.with_name(f"{SOURCE.stem}_road_follow.mp4")
WEIGHTS = Path(r"d:/drone2/ObstacleAvoidanceDrone/FlightController/Solutions/model/map.pt")


def main() -> None:
    model = YOLO(str(WEIGHTS))
    render_video(model, SOURCE, DESTINATION)
    print(f"done: {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
