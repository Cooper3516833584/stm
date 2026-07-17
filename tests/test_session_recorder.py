import json

import numpy as np

from FlightController.Solutions.Safety import Command, RadarObstacleField
from FlightController.Solutions.SessionRecorder import SessionRecorder, SessionRecorderConfig


def test_session_recorder_writes_frame_and_radar_snapshot(tmp_path):
    recorder = SessionRecorder(
        SessionRecorderConfig(
            root_dir=str(tmp_path),
            enabled=True,
            mode="test",
            frame_every_n=1,
            radar_every_n=1,
            jpeg_quality=80,
            video_enabled=True,
            video_every_n=1,
            video_fps=10.0,
            metadata={"test_case": "road-diagnostics"},
        )
    )
    try:
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame_path = recorder.record_frame(loop_count=0, now_s=1.0, frame=frame, label="cam")

        field = RadarObstacleField()
        field.update(np.array([[100.0, 0.0], [120.0, 40.0]], dtype=float), now_s=1.0)
        recorder.record_radar(
            loop_count=0,
            now_s=1.0,
            radar_field=field,
            desired=Command(10.0, 0.0, 0.0, 0.0, "desired"),
            safe_command=Command(8.0, 0.0, 0.0, 0.0, "safe"),
            decision_reason="ok",
        )
    finally:
        recorder.close()

    assert frame_path is not None
    assert recorder.session_dir is not None
    assert (recorder.session_dir / "session.json").is_file()
    assert list((recorder.session_dir / "frames").glob("*.jpg"))
    assert (recorder.session_dir / "frames.jsonl").is_file()
    assert (recorder.session_dir / "camera.avi").is_file()
    assert (recorder.session_dir / "camera.avi").stat().st_size > 0
    point_files = list((recorder.session_dir / "radar_points").glob("*.npz"))
    assert point_files

    records = (recorder.session_dir / "radar.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    payload = json.loads(records[0])
    assert payload["point_count"] == 2
    assert payload["desired"]["vy_cm_s"] == 0.0
    assert payload["safe"]["reason"] == "safe"

    frame_records = (recorder.session_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(frame_records) == 1
    frame_payload = json.loads(frame_records[0])
    assert frame_payload["video_written"] is True
    assert frame_payload["video_frame_index"] == 0

    manifest = json.loads((recorder.session_dir / "session.json").read_text(encoding="utf-8"))
    assert manifest["video_frames_written"] == 1
    assert manifest["keyframes_written"] == 1
    assert manifest["metadata"]["test_case"] == "road-diagnostics"
