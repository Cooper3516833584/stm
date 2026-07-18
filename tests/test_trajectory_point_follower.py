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
        "max_planar_decel_cm_s2": 1_000_000.0,
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


def test_offset_target_moves_directly_sideways_with_camera_mapping():
    points = [(400.0, 300.0), (400.0, 240.0), (400.0, 180.0)]
    follower = _follower(min_forward_lookahead_px=0.0)

    command = follower.update(_perception(points), now_s=1.0)

    assert command.vx_cm_s == pytest.approx(0.0)
    assert command.vy_cm_s == pytest.approx(-8.0)
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
    assert curve_follower.last_diagnostics.curve_speed_limit_cm_s == pytest.approx(8.0)
    assert curve_follower.last_diagnostics.turn_active
    assert curve_follower.last_diagnostics.turn_recovery_active
    assert math.hypot(curve_command.vx_cm_s, curve_command.vy_cm_s) == pytest.approx(4.0)


def test_moderate_curvature_interpolates_to_about_thirteen_cm_s():
    follower = TrajectoryPointFollower(TrajectoryPointFollowerConfig())

    assert follower._curve_speed_limit_cm_s(24.2) == pytest.approx(12.8)


def test_signed_curvature_feedforward_turns_toward_each_curve_direction():
    right_curve = [
        (320.0, 460.0),
        (320.0, 400.0),
        (320.0, 340.0),
        (320.0, 280.0),
        (320.0, 240.0),
        (325.0, 215.0),
        (340.0, 190.0),
        (365.0, 170.0),
        (400.0, 155.0),
        (440.0, 150.0),
    ]
    left_curve = [(640.0 - x, y) for x, y in right_curve]
    right_follower = _follower()
    left_follower = _follower()

    right = right_follower.update(
        _perception(right_curve, path_width_px=220.0),
        now_s=1.0,
    )
    left = left_follower.update(
        _perception(left_curve, path_width_px=220.0),
        now_s=1.0,
    )

    assert right_follower.last_diagnostics.raw_signed_curvature_deg > 0.0
    assert left_follower.last_diagnostics.raw_signed_curvature_deg < 0.0
    assert right_follower.last_diagnostics.curvature_feedforward_deg_s > 0.0
    assert left_follower.last_diagnostics.curvature_feedforward_deg_s < 0.0
    assert right.yaw_rate_deg_s >= 6.0
    assert left.yaw_rate_deg_s <= -6.0


def test_turn_state_holds_curve_speed_until_all_errors_stay_clear():
    curve = [
        (320.0, 460.0),
        (320.0, 360.0),
        (320.0, 260.0),
        (320.0, 240.0),
        (350.0, 200.0),
        (410.0, 170.0),
        (470.0, 165.0),
    ]
    straight = [(320.0, float(y)) for y in range(460, 19, -20)]
    follower = _follower(
        curvature_filter_tau_s=0.0,
        tangent_filter_tau_s=0.0,
        tangent_filter_max_rate_deg_s=1_000_000.0,
        target_filter_tau_s=0.0,
        turn_exit_hold_s=0.5,
    )

    follower.update(_perception(curve), now_s=1.0)
    first_clear = follower.update(_perception(straight), now_s=1.1)
    still_holding = follower.update(_perception(straight), now_s=1.4)
    released = follower.update(_perception(straight), now_s=1.61)

    assert math.hypot(first_clear.vx_cm_s, first_clear.vy_cm_s) == pytest.approx(8.0)
    assert math.hypot(still_holding.vx_cm_s, still_holding.vy_cm_s) == pytest.approx(8.0)
    assert not follower.last_diagnostics.turn_active
    assert released.vx_cm_s == pytest.approx(10.0)


def test_large_unfinished_turn_enters_slow_yaw_priority_recovery():
    sharp_diagonal = [
        (240.0, 320.0),
        (280.0, 280.0),
        (320.0, 240.0),
        (380.0, 190.0),
        (450.0, 150.0),
        (520.0, 130.0),
    ]
    follower = _follower()

    command = follower.update(
        _perception(sharp_diagonal, path_width_px=220.0),
        now_s=1.0,
    )
    diagnostics = follower.last_diagnostics

    assert diagnostics.turn_active
    assert diagnostics.turn_recovery_active
    assert diagnostics.active_speed_limit_cm_s == pytest.approx(4.0)
    assert math.hypot(command.vx_cm_s, command.vy_cm_s) == pytest.approx(4.0)
    assert abs(command.yaw_rate_deg_s) >= 8.0


def test_curvature_collapse_does_not_restore_speed_while_turn_is_still_offset():
    left_curve = [
        (320.0, 460.0),
        (320.0, 400.0),
        (320.0, 340.0),
        (320.0, 280.0),
        (320.0, 240.0),
        (315.0, 215.0),
        (300.0, 190.0),
        (275.0, 170.0),
        (240.0, 155.0),
        (200.0, 150.0),
    ]
    offset_straight = [
        (400.0, 320.0),
        (400.0, 260.0),
        (400.0, 240.0),
        (400.0, 180.0),
        (400.0, 120.0),
    ]
    follower = _follower(
        target_filter_tau_s=0.0,
        target_filter_max_rate_px_s=1_000_000.0,
        tangent_filter_max_rate_deg_s=1_000_000.0,
    )

    turning = follower.update(
        _perception(left_curve, path_width_px=220.0),
        now_s=1.0,
    )
    unfinished = follower.update(
        _perception(offset_straight, path_width_px=220.0),
        now_s=1.1,
    )
    diagnostics = follower.last_diagnostics

    assert turning.yaw_rate_deg_s < 0.0
    assert diagnostics.raw_signed_curvature_deg == pytest.approx(0.0)
    assert diagnostics.filtered_signed_curvature_deg < 0.0
    assert diagnostics.turn_active
    assert diagnostics.turn_recovery_active
    assert diagnostics.active_speed_limit_cm_s == pytest.approx(4.0)
    assert math.hypot(unfinished.vx_cm_s, unfinished.vy_cm_s) == pytest.approx(4.0)
    assert unfinished.yaw_rate_deg_s <= -8.0


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


def test_planar_acceleration_limit_brakes_before_direction_reversal():
    right = [(400.0, 300.0), (400.0, 240.0), (400.0, 180.0)]
    left = [(240.0, 300.0), (240.0, 240.0), (240.0, 180.0)]
    follower = _follower(
        max_planar_accel_cm_s2=16.0,
        max_planar_decel_cm_s2=48.0,
        target_filter_tau_s=0.0,
        target_filter_max_rate_px_s=1_000_000.0,
    )
    previous = None
    for index in range(8):
        previous = follower.update(_perception(right), now_s=1.0 + 0.1 * index)

    command = follower.update(_perception(left), now_s=1.8)
    delta = ((command.vx_cm_s - previous.vx_cm_s) ** 2 + (command.vy_cm_s - previous.vy_cm_s) ** 2) ** 0.5

    assert delta == pytest.approx(4.8)
    assert previous.vy_cm_s < 0.0
    assert command.vy_cm_s < 0.0
    assert follower.last_diagnostics.planar_accel_limited
    assert follower.last_diagnostics.planar_braking


def test_lost_road_stops_immediately_and_reacquisition_ramps_from_zero():
    points = [(320.0, float(y)) for y in range(460, 19, -20)]
    follower = _follower(max_planar_accel_cm_s2=16.0)
    follower.update(_perception(points), now_s=1.0)

    stopped = follower.update(None, now_s=1.1)
    restarted = follower.update(_perception(points), now_s=1.2)

    assert stopped.vx_cm_s == 0.0
    assert stopped.vy_cm_s == 0.0
    assert restarted.vx_cm_s == pytest.approx(1.6)


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
