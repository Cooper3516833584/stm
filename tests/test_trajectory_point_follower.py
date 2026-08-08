import math
from types import SimpleNamespace

import pytest

from FlightController.Solutions.TrajectoryPointFollower import (
    TrajectoryPointFollower,
    TrajectoryPointFollowerConfig,
)


def _perception(
    points,
    *,
    state="single",
    found=True,
    confidence=0.9,
    path_width_px=None,
):
    perception = SimpleNamespace(
        is_road_found=found,
        confidence=confidence,
        road_state=state,
        trajectory_points=points,
        centerline_points=points,
    )
    if path_width_px is not None:
        perception.path_width_px = path_width_px
    return perception


def _follower(**overrides):
    values = {
        "max_vx_cm_s": 10.0,
        "max_vy_cm_s": 8.0,
        "max_yaw_rate_deg_s": 10.0,
        "max_planar_accel_cm_s2": 1_000_000.0,
        "max_yaw_accel_deg_s2": 1_000_000.0,
    }
    values.update(overrides)
    return TrajectoryPointFollower(
        TrajectoryPointFollowerConfig(**values)
    )


def test_reached_nearest_point_advances_to_adaptive_lookahead_and_moves_forward():
    points = [(320.0, float(y)) for y in range(460, 19, -20)]
    follower = _follower()

    command = follower.update(_perception(points), now_s=1.0)

    diagnostics = follower.last_diagnostics
    assert diagnostics.target_reached
    assert diagnostics.target_index == diagnostics.nearest_index + 2
    assert diagnostics.target_distance_px == pytest.approx(40.0)
    assert diagnostics.base_lookahead_px == pytest.approx(24.0)
    assert command.vx_cm_s == pytest.approx(10.0)
    assert command.vy_cm_s == pytest.approx(0.0)
    assert command.yaw_rate_deg_s == pytest.approx(0.0)


def test_offset_target_keeps_forward_speed_and_adds_lateral_correction():
    points = [(400.0, 300.0), (400.0, 240.0), (400.0, 180.0)]
    follower = _follower(min_forward_lookahead_px=0.0)

    command = follower.update(_perception(points), now_s=1.0)

    assert command.vx_cm_s == pytest.approx(10.0)
    assert command.vy_cm_s < 0.0
    assert abs(command.vy_cm_s) <= 8.0
    assert follower.last_diagnostics.target_x_px == 400.0


def test_point_at_camera_height_advances_to_forward_lookahead():
    points = [
        (380.0, 300.0),
        (380.0, 243.0),
        (380.0, 228.0),
        (380.0, 180.0),
    ]
    follower = _follower(min_forward_lookahead_px=12.0)

    command = follower.update(_perception(points), now_s=1.0)

    assert follower.last_diagnostics.target_y_px == 228.0
    assert follower.last_diagnostics.target_advanced_for_lookahead
    assert command.vx_cm_s > 0.0


def test_diagonal_path_moves_toward_point_and_yaws_with_local_tangent():
    points = [
        (250.0, 460.0),
        (270.0, 400.0),
        (290.0, 340.0),
        (310.0, 280.0),
        (330.0, 220.0),
        (350.0, 160.0),
        (370.0, 100.0),
    ]
    follower = _follower()

    command = follower.update(_perception(points), now_s=1.0)

    assert command.vx_cm_s > 0.0
    assert command.vy_cm_s < 0.0
    assert command.yaw_rate_deg_s > 0.0
    assert follower.last_diagnostics.tangent_dx_px > 0.0
    assert follower.last_diagnostics.tangent_dy_px < 0.0


def test_reversed_input_path_is_normalized_to_bottom_to_top():
    points = [(320.0, float(y)) for y in range(20, 461, 20)]
    follower = _follower()

    command = follower.update(_perception(points), now_s=1.0)

    assert command.vx_cm_s > 0.0
    assert follower.last_diagnostics.tangent_dy_px < 0.0


def test_visible_path_end_uses_tangent_to_keep_moving():
    points = [(320.0, 300.0), (320.0, 260.0), (320.0, 240.0)]
    follower = _follower()

    command = follower.update(_perception(points), now_s=1.0)

    assert follower.last_diagnostics.tangent_motion_fallback
    assert command.vx_cm_s == pytest.approx(10.0)
    assert command.vy_cm_s == pytest.approx(0.0)


def test_degraded_fitted_path_keeps_moving_at_reduced_speed():
    points = [(320.0, float(y)) for y in range(460, 19, -20)]
    follower = _follower(degraded_speed_scale=0.75)

    command = follower.update(
        _perception(points, state="single_extrapolated"),
        now_s=1.0,
    )

    assert command.vx_cm_s == pytest.approx(7.5)
    assert follower.last_diagnostics.heading_speed_scale == pytest.approx(0.75)


def test_default_acceleration_limit_ramps_initial_command_without_lowering_cruise_limit():
    points = [(320.0, float(y)) for y in range(460, 19, -20)]
    follower = TrajectoryPointFollower(TrajectoryPointFollowerConfig())

    commands = [
        follower.update(_perception(points), now_s=1.0 + 0.1 * index)
        for index in range(9)
    ]

    assert commands[0].vx_cm_s == pytest.approx(2.4)
    assert commands[-1].vx_cm_s == pytest.approx(20.0)
    assert commands[-1].vy_cm_s == pytest.approx(0.0)


def test_tight_upcoming_curve_slows_while_straight_road_uses_full_speed():
    straight = [(320.0, float(y)) for y in range(460, 19, -20)]
    tight_curve = [
        (320.0, 460.0),
        (320.0, 400.0),
        (320.0, 340.0),
        (320.0, 280.0),
        (320.0, 240.0),
        (320.0, 220.0),
        (325.0, 200.0),
        (340.0, 180.0),
        (365.0, 165.0),
        (395.0, 155.0),
        (430.0, 150.0),
    ]
    straight_follower = TrajectoryPointFollower(
        TrajectoryPointFollowerConfig(
            max_planar_accel_cm_s2=1_000_000.0,
            max_yaw_accel_deg_s2=1_000_000.0,
        )
    )
    curve_follower = TrajectoryPointFollower(
        TrajectoryPointFollowerConfig(
            max_planar_accel_cm_s2=1_000_000.0,
            max_yaw_accel_deg_s2=1_000_000.0,
        )
    )

    straight_command = straight_follower.update(_perception(straight), now_s=1.0)
    curve_command = curve_follower.update(_perception(tight_curve), now_s=1.0)

    assert straight_command.vx_cm_s == pytest.approx(20.0)
    assert straight_follower.last_diagnostics.forward_curvature_deg == pytest.approx(0.0)
    assert curve_follower.last_diagnostics.forward_curvature_deg >= 35.0
    assert curve_follower.last_diagnostics.curve_speed_limit_cm_s == pytest.approx(10.0)
    assert math.hypot(curve_command.vx_cm_s, curve_command.vy_cm_s) == pytest.approx(10.0)


def test_moderate_curvature_interpolates_to_about_fourteen_cm_s():
    follower = TrajectoryPointFollower(TrajectoryPointFollowerConfig())

    assert follower._curve_speed_limit_cm_s(24.2) == pytest.approx(14.0)


def test_high_speed_curve_speed_limits_remain_fast_until_sharp_turns():
    follower = _follower(
        max_vx_cm_s=45.0,
        min_curve_speed_cm_s=28.0,
        curvature_slowdown_start_deg=25.0,
        curvature_full_slowdown_deg=70.0,
    )

    assert follower._curve_speed_limit_cm_s(0.0) == pytest.approx(45.0)
    assert follower._curve_speed_limit_cm_s(45.0) == pytest.approx(37.4444444444)
    assert follower._curve_speed_limit_cm_s(70.0) == pytest.approx(28.0)


def test_corner_lookahead_cap_shortens_only_for_large_curvature():
    follower = _follower(
        min_forward_lookahead_px=30.0,
        max_forward_lookahead_px=130.0,
        corner_lookahead_start_deg=30.0,
        corner_lookahead_full_deg=75.0,
        corner_min_lookahead_px=75.0,
    )

    assert follower._corner_lookahead_cap_px(30.0) == pytest.approx(130.0)
    assert follower._corner_lookahead_cap_px(60.0) == pytest.approx(93.333333)
    assert follower._corner_lookahead_cap_px(75.0) == pytest.approx(75.0)


def test_corner_severity_attacks_immediately_and_releases_slowly():
    follower = _follower(corner_severity_release_tau_s=0.25)

    assert follower._filter_corner_severity(60.0, dt_s=0.1) == pytest.approx(60.0)
    released = follower._filter_corner_severity(0.0, dt_s=0.1)

    assert 0.0 < released < 60.0


def test_signed_preview_turn_reports_direction_and_consistency():
    right_curve = [
        (320.0, 300.0),
        (320.0, 280.0),
        (322.0, 260.0),
        (328.0, 240.0),
        (340.0, 222.0),
        (358.0, 208.0),
        (380.0, 200.0),
    ]
    follower = _follower(tangent_window_points=1)

    turn, consistency = follower._signed_preview_turn_deg(
        right_curve, 0, len(right_curve) - 1
    )
    reverse_turn, _ = follower._signed_preview_turn_deg(
        [(640.0 - x, y) for x, y in right_curve],
        0,
        len(right_curve) - 1,
    )

    assert turn > 0.0
    assert reverse_turn < 0.0
    assert consistency >= 0.60


def test_signed_preview_turn_does_not_read_beyond_final_target_horizon():
    points = [
        (320.0, 320.0),
        (320.0, 300.0),
        (325.0, 280.0),
        (340.0, 260.0),
        (365.0, 245.0),
        (395.0, 238.0),
        (360.0, 225.0),
        (320.0, 205.0),
        (280.0, 180.0),
    ]
    follower = _follower(tangent_window_points=3)

    turn, _ = follower._signed_preview_turn_deg(points, 0, 5)

    assert turn > 0.0


def test_yaw_feedforward_acts_before_feedback_and_is_clamped():
    points = [
        (320.0, 300.0),
        (320.0, 280.0),
        (322.0, 260.0),
        (330.0, 240.0),
        (345.0, 220.0),
        (370.0, 205.0),
        (400.0, 198.0),
    ]
    follower = _follower(
        max_yaw_rate_deg_s=55.0,
        tangent_window_points=1,
        tangent_kp_yaw=0.0,
        curvature_yaw_ff_kp=10.0,
        curvature_yaw_ff_max_deg_s=18.0,
        curvature_yaw_ff_deadband_deg=0.0,
        signed_turn_filter_tau_s=0.0,
    )

    command = follower.update(_perception(points), now_s=1.0)
    diagnostics = follower.last_diagnostics

    assert diagnostics.yaw_feedback_deg_s == pytest.approx(0.0)
    assert abs(diagnostics.yaw_feedforward_deg_s) == pytest.approx(18.0)
    assert abs(command.yaw_rate_deg_s) > abs(diagnostics.yaw_feedback_deg_s)
    assert abs(command.yaw_rate_deg_s) <= 55.0


def test_signed_turn_filter_reverses_quickly_while_corner_severity_releases_slowly():
    follower = _follower(
        signed_turn_filter_tau_s=0.08,
        corner_severity_release_tau_s=0.25,
    )
    follower._filtered_signed_preview_turn_deg = 60.0
    follower._corner_severity_deg = 60.0

    filtered_turn = follower._filter_angle(
        -60.0,
        follower._filtered_signed_preview_turn_deg,
        tau_s=follower.config.signed_turn_filter_tau_s,
        max_rate_per_s=1_000_000.0,
        dt_s=1.0 / 12.0,
    )
    corner_severity = follower._filter_corner_severity(0.0, dt_s=1.0 / 12.0)

    assert filtered_turn < 0.0
    assert corner_severity > 40.0


def test_production_style_straight_road_keeps_full_speed_without_recovery():
    points = [(320.0, float(y)) for y in range(460, 19, -10)]
    follower = _follower(
        max_vx_cm_s=45.0,
        max_vy_cm_s=16.0,
        normal_max_vy_cm_s=12.0,
        max_yaw_rate_deg_s=55.0,
        min_forward_lookahead_px=30.0,
        max_forward_lookahead_px=130.0,
        curvature_yaw_ff_kp=0.30,
        curvature_yaw_ff_max_deg_s=18.0,
        edge_recovery_start_ratio=0.55,
        edge_recovery_full_ratio=0.90,
        edge_recovery_max_vy_cm_s=16.0,
    )

    command = follower.update(
        _perception(points, path_width_px=200.0), now_s=1.0
    )
    diagnostics = follower.last_diagnostics

    assert command.vx_cm_s == pytest.approx(45.0)
    assert command.vy_cm_s == pytest.approx(0.0)
    assert command.yaw_rate_deg_s == pytest.approx(0.0)
    assert diagnostics.yaw_feedforward_deg_s == pytest.approx(0.0)
    assert diagnostics.edge_recovery_blend == pytest.approx(0.0)
    assert diagnostics.corner_lookahead_cap_px == pytest.approx(130.0)


def test_new_curve_speed_profile_keeps_sharp_turns_above_34_cm_s():
    follower = _follower(
        max_vx_cm_s=45.0,
        curvature_slowdown_start_deg=35.0,
        curvature_full_slowdown_deg=80.0,
        min_curve_speed_cm_s=34.0,
    )

    assert follower._curve_speed_limit_cm_s(0.0) == pytest.approx(45.0)
    assert follower._curve_speed_limit_cm_s(35.0) == pytest.approx(45.0)
    assert follower._curve_speed_limit_cm_s(55.0) == pytest.approx(40.111111)
    assert follower._curve_speed_limit_cm_s(80.0) == pytest.approx(34.0)
    assert follower._curve_speed_limit_cm_s(90.0) == pytest.approx(34.0)


def test_high_yaw_response_tracks_tangent_and_clamps_at_limit():
    diagonal = [(320.0, 460.0), (350.0, 408.0), (380.0, 356.0)]
    follower = _follower(
        tangent_kp_yaw=0.9,
        max_yaw_rate_deg_s=40.0,
        max_yaw_accel_deg_s2=1_000_000.0,
        tangent_filter_tau_s=0.0,
        tangent_filter_max_rate_deg_s=1_000_000.0,
        tangent_deadband_deg=0.0,
    )

    command = follower.update(_perception(diagonal), now_s=1.0)

    expected = 0.9 * follower.last_diagnostics.raw_tangent_error_deg
    assert command.yaw_rate_deg_s == pytest.approx(expected)

    sharp = [(320.0, 460.0), (500.0, 408.0), (620.0, 356.0)]
    clamped = follower.update(_perception(sharp), now_s=1.1)

    assert abs(clamped.yaw_rate_deg_s) == pytest.approx(40.0)


def test_speed_and_measured_latency_expand_forward_lookahead():
    points = [(320.0, float(y)) for y in range(460, 19, -5)]
    latency_follower = TrajectoryPointFollower(
        TrajectoryPointFollowerConfig(
            max_planar_accel_cm_s2=1_000_000.0,
            latency_compensation_s=0.134,
        )
    )
    no_latency_follower = TrajectoryPointFollower(
        TrajectoryPointFollowerConfig(
            max_planar_accel_cm_s2=1_000_000.0,
            latency_compensation_s=0.0,
        )
    )
    perception = _perception(points, path_width_px=200.0)

    latency_follower.update(perception, now_s=1.0)
    latency_follower.update(perception, now_s=1.1)
    no_latency_follower.update(perception, now_s=1.0)
    no_latency_follower.update(perception, now_s=1.1)

    latency = latency_follower.last_diagnostics
    no_latency = no_latency_follower.last_diagnostics
    assert latency.current_planar_speed_cm_s == pytest.approx(20.0)
    assert latency.base_lookahead_px == pytest.approx(48.0)
    assert latency.latency_prediction_px == pytest.approx(10.72)
    assert latency.effective_lookahead_px == pytest.approx(58.72)
    assert latency.target_index > no_latency.target_index


def test_small_lateral_error_is_ignored_but_larger_error_is_corrected():
    small_error = [(326.0, 300.0), (326.0, 240.0), (326.0, 180.0)]
    large_error = [(340.0, 300.0), (340.0, 240.0), (340.0, 180.0)]
    small_follower = _follower(min_forward_lookahead_px=0.0)
    large_follower = _follower(min_forward_lookahead_px=0.0)

    small_command = small_follower.update(_perception(small_error), now_s=1.0)
    large_command = large_follower.update(_perception(large_error), now_s=1.0)

    assert small_follower.last_diagnostics.used_pixel_error_px == 0.0
    assert small_command.vy_cm_s == pytest.approx(0.0)
    assert abs(large_follower.last_diagnostics.used_pixel_error_px) == pytest.approx(12.0)
    assert large_command.vy_cm_s < 0.0


def test_forward_priority_does_not_trade_vx_for_lateral_error():
    points = [(440.0, 300.0), (440.0, 240.0), (440.0, 180.0)]
    follower = _follower(
        max_vx_cm_s=45.0,
        max_vy_cm_s=12.0,
        min_forward_lookahead_px=30.0,
        lateral_deadband_px=24.0,
        lateral_kp_cm_s_per_px=0.10,
    )

    command = follower.update(_perception(points), now_s=1.0)

    assert command.vx_cm_s == pytest.approx(45.0)
    assert command.vy_cm_s < 0.0
    assert abs(command.vy_cm_s) <= 12.0


def test_lateral_deadband_keeps_forward_motion_without_lateral_correction():
    points = [(344.0, 300.0), (344.0, 240.0), (344.0, 180.0)]
    follower = _follower(
        min_forward_lookahead_px=30.0,
        lateral_deadband_px=24.0,
    )

    command = follower.update(_perception(points), now_s=1.0)

    assert command.vx_cm_s > 0.0
    assert command.vy_cm_s == pytest.approx(0.0)


def test_centerline_x_at_camera_y_interpolates_and_falls_back_safely():
    follower = _follower()

    assert follower._centerline_x_at_camera_y(
        [(300.0, 300.0), (340.0, 200.0)], 240.0, 0, 320.0
    ) == pytest.approx(324.0)
    assert follower._centerline_x_at_camera_y(
        [(300.0, 300.0), (340.0, 280.0)], 240.0, 0, 320.0
    ) == pytest.approx(340.0)
    assert follower._centerline_x_at_camera_y(
        [(300.0, 240.0), (340.0, 240.0)], 240.0, 0, 320.0
    ) == pytest.approx(320.0)
    assert follower._centerline_x_at_camera_y([], 240.0, 0, 320.0) is None


def test_centerline_x_at_camera_y_prefers_crossing_near_nearest_index():
    points = [
        (300.0, 300.0),
        (300.0, 240.0),
        (340.0, 200.0),
        (500.0, 180.0),
        (520.0, 220.0),
        (500.0, 260.0),
        (460.0, 300.0),
    ]
    follower = _follower()

    result = follower._centerline_x_at_camera_y(points, 240.0, 1, 320.0)

    assert result == pytest.approx(300.0)


def test_centerline_x_at_camera_y_can_select_far_crossing_when_nearest_index_moves():
    points = [
        (300.0, 300.0),
        (300.0, 240.0),
        (340.0, 200.0),
        (500.0, 180.0),
        (520.0, 220.0),
        (500.0, 260.0),
        (460.0, 300.0),
    ]
    follower = _follower()

    result = follower._centerline_x_at_camera_y(points, 240.0, 4, 320.0)

    assert result == pytest.approx(510.0)


@pytest.mark.parametrize(
    ("cross_track_px", "expected_ratio"),
    [(0.0, 0.0), (50.0, 0.5), (80.0, 0.8), (100.0, 1.0)],
)
def test_edge_ratio_uses_current_cross_track_and_measured_width(
    cross_track_px, expected_ratio
):
    x = 320.0 + cross_track_px
    points = [(x, 300.0), (x, 240.0), (x, 180.0)]
    follower = _follower(min_forward_lookahead_px=30.0)

    follower.update(_perception(points, path_width_px=200.0), now_s=1.0)

    assert follower.last_diagnostics.edge_ratio == pytest.approx(expected_ratio)


def test_edge_recovery_blends_to_larger_vy_and_inward_yaw():
    points = [(420.0, 300.0), (420.0, 240.0), (420.0, 180.0)]
    follower = _follower(
        max_vy_cm_s=16.0,
        normal_max_vy_cm_s=12.0,
        min_forward_lookahead_px=30.0,
        lateral_deadband_px=24.0,
        edge_recovery_start_ratio=0.55,
        edge_recovery_full_ratio=0.90,
        edge_recovery_lateral_kp=0.22,
        edge_recovery_max_vy_cm_s=16.0,
        edge_yaw_start_ratio=0.75,
        edge_yaw_full_ratio=0.95,
        edge_yaw_max_deg_s=8.0,
    )

    command = follower.update(
        _perception(points, path_width_px=200.0), now_s=1.0
    )
    diagnostics = follower.last_diagnostics

    assert diagnostics.edge_recovery_blend == pytest.approx(1.0)
    assert abs(diagnostics.normal_vy_cm_s) <= 12.0
    assert abs(diagnostics.recovery_vy_cm_s) == pytest.approx(16.0)
    assert 12.0 < abs(command.vy_cm_s) <= 16.0
    assert 0.0 < diagnostics.edge_yaw_bias_deg_s <= 8.0


def test_edge_yaw_reverses_with_cross_track_and_requires_path_width():
    config = dict(
        min_forward_lookahead_px=30.0,
        edge_yaw_start_ratio=0.75,
        edge_yaw_full_ratio=0.95,
        edge_yaw_max_deg_s=8.0,
    )
    right = _follower(**config)
    left = _follower(**config)
    no_width = _follower(**config)

    right.update(
        _perception([(420.0, 300.0), (420.0, 240.0), (420.0, 180.0)], path_width_px=200.0),
        now_s=1.0,
    )
    left.update(
        _perception([(220.0, 300.0), (220.0, 240.0), (220.0, 180.0)], path_width_px=200.0),
        now_s=1.0,
    )
    no_width.update(
        _perception([(420.0, 300.0), (420.0, 240.0), (420.0, 180.0)]),
        now_s=1.0,
    )

    assert right.last_diagnostics.edge_yaw_bias_deg_s > 0.0
    assert left.last_diagnostics.edge_yaw_bias_deg_s < 0.0
    assert no_width.last_diagnostics.edge_ratio is None
    assert no_width.last_diagnostics.edge_recovery_blend == 0.0
    assert no_width.last_diagnostics.edge_yaw_bias_deg_s == 0.0


def test_edge_speed_cap_is_smooth_and_never_raises_curve_limit():
    follower = _follower(
        edge_speed_slow_start_ratio=0.90,
        edge_emergency_ratio=0.95,
        edge_emergency_vx_cap_cm_s=38.0,
    )

    assert follower._edge_speed_cap_cm_s(45.0, 0.89) == pytest.approx(45.0)
    assert 38.0 < follower._edge_speed_cap_cm_s(45.0, 0.925) < 45.0
    assert follower._edge_speed_cap_cm_s(45.0, 0.95) == pytest.approx(38.0)
    assert follower._edge_speed_cap_cm_s(34.0, 1.0) == pytest.approx(34.0)


def test_planar_acceleration_and_deceleration_use_independent_rates():
    accelerated = TrajectoryPointFollower._limit_planar_acceleration(
        45.0, 0.0, 0.0, 0.0,
        max_accel_cm_s2=55.0,
        max_decel_cm_s2=120.0,
        dt_s=0.1,
    )
    decelerated = TrajectoryPointFollower._limit_planar_acceleration(
        34.0, 0.0, 45.0, 0.0,
        max_accel_cm_s2=55.0,
        max_decel_cm_s2=120.0,
        dt_s=0.1,
    )

    assert accelerated[0] == pytest.approx(5.5)
    assert accelerated[4] == pytest.approx(55.0)
    assert not accelerated[5]
    assert decelerated[0] == pytest.approx(34.0)
    assert decelerated[4] == pytest.approx(120.0)
    assert not decelerated[5]


def test_planar_deceleration_limiter_clamps_large_speed_drop():
    result = TrajectoryPointFollower._limit_planar_acceleration(
        20.0,
        0.0,
        45.0,
        0.0,
        max_accel_cm_s2=55.0,
        max_decel_cm_s2=120.0,
        dt_s=0.1,
    )

    assert result[0] == pytest.approx(33.0)
    assert result[1] == pytest.approx(0.0)
    assert result[2]
    assert result[4] == pytest.approx(120.0)
    assert result[5]


def test_planar_acceleration_limit_brakes_before_direction_reversal():
    right = [(400.0, 300.0), (400.0, 240.0), (400.0, 180.0)]
    left = [(240.0, 300.0), (240.0, 240.0), (240.0, 180.0)]
    follower = _follower(
        max_planar_accel_cm_s2=16.0,
        target_filter_tau_s=0.0,
        target_filter_max_rate_px_s=1_000_000.0,
    )
    previous = None
    for index in range(8):
        previous = follower.update(_perception(right), now_s=1.0 + 0.1 * index)

    command = follower.update(_perception(left), now_s=1.8)
    delta = ((command.vx_cm_s - previous.vx_cm_s) ** 2 + (command.vy_cm_s - previous.vy_cm_s) ** 2) ** 0.5

    assert delta == pytest.approx(1.6)
    assert previous.vy_cm_s < 0.0
    assert command.vy_cm_s < 0.0
    assert follower.last_diagnostics.planar_accel_limited


def test_lost_road_stops_immediately_and_reacquisition_ramps_from_zero():
    points = [(320.0, float(y)) for y in range(460, 19, -20)]
    follower = _follower(max_planar_accel_cm_s2=16.0)
    follower.update(_perception(points), now_s=1.0)

    stopped = follower.update(None, now_s=1.1)
    restarted = follower.update(_perception(points), now_s=1.2)

    assert stopped.vx_cm_s == 0.0
    assert stopped.vy_cm_s == 0.0
    assert restarted.vx_cm_s == pytest.approx(1.6)


def test_short_road_loss_uses_entry_command_grace_and_reacquires_tracking():
    points = [(320.0, float(y)) for y in range(460, 19, -20)]
    follower = _follower(
        max_vx_cm_s=45.0,
        lost_grace_s=0.18,
        lost_grace_vx_scale=0.80,
    )
    follower.update(_perception(points), now_s=1.0)

    grace = follower.update(None, now_s=1.1)
    recovered = follower.update(_perception(points), now_s=1.2)

    assert grace.vx_cm_s == pytest.approx(36.0)
    assert follower.last_diagnostics.state == "tracking"
    assert recovered.vx_cm_s > 0.0


def test_road_loss_stops_after_grace_expires():
    points = [(320.0, float(y)) for y in range(460, 19, -20)]
    follower = _follower(max_vx_cm_s=45.0, lost_grace_s=0.18)
    follower.update(_perception(points), now_s=1.0)
    follower.update(None, now_s=1.1)

    stopped = follower.update(None, now_s=1.29)

    assert stopped.vx_cm_s == 0.0
    assert stopped.vy_cm_s == 0.0
    assert stopped.yaw_rate_deg_s == 0.0
    assert follower.last_diagnostics.state == "road_lost_hold"


@pytest.mark.parametrize(
    "perception",
    [None, _perception([], found=False), _perception([(320.0, 240.0)])],
)
def test_missing_or_unsupported_trajectory_holds_position(perception):
    follower = _follower()

    command = follower.update(perception, now_s=1.0)

    assert command.vx_cm_s == 0.0
    assert command.vy_cm_s == 0.0
    assert command.yaw_rate_deg_s == 0.0
    assert command.reason == "trajectory_road_lost_hold"
