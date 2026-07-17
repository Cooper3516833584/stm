from types import SimpleNamespace

import pytest

from FlightController.Solutions.RoadFollower import RoadFollower, RoadFollowerConfig


def _perception(*, pixel_error=0.0, angle=90.0, confidence=0.9, found=True):
    return SimpleNamespace(
        is_road_found=found,
        confidence=confidence,
        corrected_pixel_error=pixel_error,
        centerline_angle=angle,
        road_state="single",
    )


def test_cross_track_error_uses_lateral_velocity_not_yaw_by_default():
    follower = RoadFollower(RoadFollowerConfig(pixel_deadband_px=0.0))

    command = follower.update(_perception(pixel_error=100.0, angle=90.0), now_s=1.0)

    assert command.yaw_rate_deg_s == 0.0
    assert command.vy_cm_s == pytest.approx(-3.0)
    assert follower.last_diagnostics.pixel_yaw_term_deg_s == 0.0
    assert follower.last_diagnostics.angle_yaw_term_deg_s == 0.0


def test_heading_error_controls_yaw_with_fc_clockwise_positive_convention():
    follower = RoadFollower(RoadFollowerConfig(pixel_deadband_px=0.0))

    road_ahead_left = follower.update(_perception(angle=135.0), now_s=1.0)
    road_ahead_right = follower.update(_perception(angle=45.0), now_s=2.0)

    assert road_ahead_left.yaw_rate_deg_s < 0.0
    assert road_ahead_right.yaw_rate_deg_s > 0.0


def test_large_heading_error_stops_forward_motion_until_realigned():
    follower = RoadFollower(RoadFollowerConfig(heading_stop_deg=70.0))

    command = follower.update(_perception(angle=160.0), now_s=1.0)

    assert command.vx_cm_s == 0.0
    assert command.yaw_rate_deg_s < 0.0
    assert follower.last_diagnostics.heading_speed_scale == 0.0


def test_lost_road_holds_heading_instead_of_blind_search_rotation():
    follower = RoadFollower(RoadFollowerConfig())

    initial = follower.update(None, now_s=10.0)
    timed_out = follower.update(None, now_s=16.0)

    assert initial.yaw_rate_deg_s == 0.0
    assert initial.reason == "road_lost_hold"
    assert timed_out.yaw_rate_deg_s == 0.0
    assert timed_out.reason == "road_lost_timeout"


def test_legacy_pixel_to_yaw_term_remains_explicitly_opt_in():
    follower = RoadFollower(
        RoadFollowerConfig(pixel_deadband_px=0.0, pixel_kp_yaw=0.08, pixel_kp_vy=0.0)
    )

    command = follower.update(_perception(pixel_error=100.0), now_s=1.0)

    assert command.yaw_rate_deg_s == pytest.approx(8.0)
    assert command.vy_cm_s == 0.0
