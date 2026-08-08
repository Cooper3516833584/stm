import time

import numpy as np

from perception_pipeline import CameraFrameDistributor, CameraThread, PerceptionPipeline
import target_perception
from target_perception import PurpleTargetInferenceThread


def test_camera_frame_distributor_publishes_one_frame_to_multiple_subscribers():
    distributor = CameraFrameDistributor()
    first = distributor.subscribe()
    second = distributor.subscribe()
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    distributor.publish(frame, 12.5)

    first_frame, first_ts = first.latest()
    second_frame, second_ts = second.latest()
    assert first is not second
    assert not hasattr(first, "publish")
    assert first_frame is frame
    assert second_frame is frame
    assert first_ts == second_ts == 12.5


def test_camera_thread_exposes_the_same_distribution_buffer_for_legacy_readers():
    camera = CameraThread()
    assert camera.frame_buffer is camera.frame_distributor
    subscription = camera.subscribe_frames()
    assert subscription is not camera.frame_distributor
    assert not hasattr(subscription, "publish")


def test_target_worker_consumes_a_subscription_without_opening_a_camera():
    camera = CameraThread()
    worker = PurpleTargetInferenceThread(camera, poll_interval_s=0.001)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[20:50, 30:70] = (180, 0, 235)  # BGR purple

    worker.start()
    try:
        captured_at = time.monotonic()
        camera.frame_distributor.publish(frame, captured_at)
        deadline = time.monotonic() + 1.0
        result = None
        while result is None and time.monotonic() < deadline:
            result, _age_s, _stale = worker.latest_result()
            time.sleep(0.001)

        assert result is not None
        assert result.found
        assert result.capture_time_s == captured_at
        assert camera._cap is None
    finally:
        worker.stop()


def test_pipeline_disables_only_target_worker_idempotently():
    class _Target:
        def __init__(self):
            self.stop_count = 0

        def stop(self):
            self.stop_count += 1

    pipeline = PerceptionPipeline.__new__(PerceptionPipeline)
    target = _Target()
    pipeline.target = target

    pipeline.disable_target()
    pipeline.disable_target()

    assert target.stop_count == 1
    assert pipeline.target is None


def test_target_worker_is_limited_to_about_ten_hz(monkeypatch):
    monkeypatch.setattr(
        target_perception,
        "find_purple_target_offset",
        lambda *args, **kwargs: None,
    )
    camera = CameraThread()
    worker = PurpleTargetInferenceThread(
        camera,
        max_rate_hz=10.0,
        poll_interval_s=0.001,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    worker.start()
    try:
        started = time.monotonic()
        while time.monotonic() - started < 0.36:
            camera.frame_distributor.publish(frame.copy(), time.monotonic())
            time.sleep(0.01)
    finally:
        worker.stop()

    assert 3 <= worker.detection_count <= 5
