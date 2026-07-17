from types import SimpleNamespace

import pytest

from FlightController.Solutions.RoadFollower import RoadFollower, RoadFollowerConfig


def _perception(
    *,
    pixel_error=0.0,
    angle=90.0,
    confidence=0.9,
    found=True,
    road_state="single",
):
    return SimpleNamespace(
        is_road_found=found,
        confidence=confidence,
        corrected_pixel_error=pixel_error,
        centerline_angle=angle,
        road_state=road_state,
    )


def test_cross_track_error_uses_lateral_velocity_not_yaw_by_default():
    follower = RoadFollower(RoadFollowerConfig(pixel_deadband_px=0.0))

    command = follower.update(_perception(pixel_error=100.0, angle=90.0), now_s=1.0)

    assert command.yaw_rate_deg_s == 0.0
    assert command.vy_cm_s == pytest.approx(-3.0)
    assert follower.last_diagnostics.pixel_yaw_term_deg_s == 0.0
    assert follower.last_diagnostics.angle_yaw_term_deg_s == 0.0


def test_heading_error_controls_yaw_with_fc_clockwise_positive_convention():
    follower = RoadFollower(
        RoadFollowerConfig(
            pixel_deadband_px=0.0,
            angle_filter_tau_s=0.0,
            angle_filter_max_rate_deg_s=0.0,
        )
    )

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


def test_rough_but_recoverable_centerline_runs_at_half_speed():
    follower = RoadFollower(RoadFollowerConfig(max_vx_cm_s=20.0))

    command = follower.update(_perception(road_state="single_rough"), now_s=1.0)

    assert command.vx_cm_s == 10.0
    assert command.reason == "road_follow:single_rough"


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


def test_single_frame_angle_spike_does_not_reverse_yaw():
    follower = RoadFollower(RoadFollowerConfig(angle_deadband_deg=0.0))

    before = follower.update(_perception(angle=60.0), now_s=1.0)
    spike = follower.update(_perception(angle=120.0), now_s=1.1)

    assert before.yaw_rate_deg_s > 0.0
    assert spike.yaw_rate_deg_s > 0.0
    assert follower.last_diagnostics.raw_centerline_angle_deg == 120.0
    assert follower.last_diagnostics.centerline_angle_deg == pytest.approx(64.5)


def test_sustained_angle_change_eventually_changes_yaw_direction():
    follower = RoadFollower(RoadFollowerConfig(angle_deadband_deg=0.0))
    follower.update(_perception(angle=60.0), now_s=1.0)

    command = None
    for index in range(1, 31):
        command = follower.update(_perception(angle=120.0), now_s=1.0 + index * 0.1)

    assert command is not None
    assert command.yaw_rate_deg_s < 0.0


def test_single_frame_pixel_spike_does_not_reverse_lateral_command():
    follower = RoadFollower(RoadFollowerConfig(pixel_deadband_px=0.0))

    before = follower.update(_perception(pixel_error=100.0), now_s=1.0)
    spike = follower.update(_perception(pixel_error=-100.0), now_s=1.1)

    assert before.vy_cm_s < 0.0
    assert spike.vy_cm_s < 0.0
    assert follower.last_diagnostics.raw_pixel_error_px == -100.0
    assert follower.last_diagnostics.filtered_pixel_error_px == pytest.approx(70.0)
