from types import SimpleNamespace

import numpy as np
import pytest

from road_follow_main import _FCTelemetryTracker, _annotate_road_frame, _road_record_extra
from FlightController.Solutions.Safety import Command


class _Field:
    def __init__(self, value):
        self.value = value


def _fc_state(*, yaw, update_count, timestamp):
    return SimpleNamespace(
        yaw=_Field(yaw),
        rol=_Field(1.0),
        pit=_Field(-2.0),
        alt_add=_Field(100),
        vel_x=_Field(5),
        vel_y=_Field(-3),
        vel_z=_Field(0),
        bat=_Field(12.1),
        mode=_Field(2),
        unlock=_Field(True),
        update_count=update_count,
        last_update_monotonic=timestamp,
    )


def test_fc_telemetry_tracker_logs_wrapped_measured_yaw_rate():
    tracker = _FCTelemetryTracker()
    fc = SimpleNamespace(connected=True, state=_fc_state(yaw=179.0, update_count=1, timestamp=10.0))

    first = tracker.update(fc, now_s=10.02)
    fc.state = _fc_state(yaw=-179.0, update_count=2, timestamp=10.1)
    second = tracker.update(fc, now_s=10.12)

    assert first["yaw_rate_deg_s"] is None
    assert second["yaw_rate_deg_s"] == pytest.approx(20.0)
    assert second["telemetry_age_s"] == pytest.approx(0.02)
    assert second["vel_y_cm_s"] == -3.0


def test_road_record_extra_contains_control_latency_geometry_and_fc_state():
    perception = SimpleNamespace(
        road_state="single",
        pixel_error=25.0,
        corrected_pixel_error=20.0,
        centerline_angle=95.0,
        path_width_px=80.0,
        confidence=0.9,
        is_road_found=True,
        debug_msg="ok",
        centerline_points=[(300.0, 430.0), (315.0, 100.0)],
    )

    payload = _road_record_extra(
        perception,
        True,
        controller_diagnostics={"angle_yaw_term_deg_s": -2.0},
        perception_age_s=0.08,
        perception_stale=False,
        frame_age_s=0.02,
        fc_telemetry={"yaw_deg": 42.0},
    )

    assert payload["offset_correction_px"] == -5.0
    assert payload["centerline_point_count"] == 2
    assert payload["centerline_first"] == [300.0, 430.0]
    assert payload["perception_age_s"] == 0.08
    assert payload["controller"]["angle_yaw_term_deg_s"] == -2.0
    assert payload["fc"]["yaw_deg"] == 42.0


def test_diagnostic_video_overlay_keeps_input_unchanged():
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    perception = SimpleNamespace(
        road_state="single",
        is_road_found=True,
        confidence=0.9,
        corrected_pixel_error=20.0,
        centerline_angle=95.0,
        centerline_points=[(180.0, 110.0), (160.0, 40.0)],
    )

    output = _annotate_road_frame(
        frame,
        perception=perception,
        loop_count=3,
        controller_diagnostics={
            "angle_error_deg": -5.0,
            "pixel_yaw_term_deg_s": 0.0,
            "angle_yaw_term_deg_s": -2.0,
            "heading_speed_scale": 1.0,
        },
        safe_command=Command(5.0, -1.0, 0.0, -2.0, "road_follow"),
        fc_telemetry={"yaw_deg": 12.0, "yaw_rate_deg_s": -1.8},
        perception_age_s=0.05,
        perception_stale=False,
    )

    assert output.shape == frame.shape
    assert np.count_nonzero(output) > 0
    assert np.count_nonzero(frame) == 0
