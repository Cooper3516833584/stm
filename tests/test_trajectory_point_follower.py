from types import SimpleNamespace

import pytest

from FlightController.Solutions.TrajectoryPointFollower import (
    TrajectoryPointFollower,
    TrajectoryPointFollowerConfig,
)


def _perception(points, *, state="single", found=True, confidence=0.9):
    return SimpleNamespace(
        is_road_found=found,
        confidence=confidence,
        road_state=state,
        trajectory_points=points,
        centerline_points=points,
    )


def _follower(**overrides):
    return TrajectoryPointFollower(
        TrajectoryPointFollowerConfig(
            max_vx_cm_s=10.0,
            max_vy_cm_s=8.0,
            max_yaw_rate_deg_s=10.0,
            **overrides,
        )
    )


def test_reached_nearest_point_advances_to_next_point_and_moves_forward():
    points = [(320.0, float(y)) for y in range(460, 19, -20)]
    follower = _follower()

    command = follower.update(_perception(points), now_s=1.0)

    diagnostics = follower.last_diagnostics
    assert diagnostics.target_reached
    assert diagnostics.target_index == diagnostics.nearest_index + 1
    assert diagnostics.target_distance_px == pytest.approx(20.0)
    assert command.vx_cm_s == pytest.approx(10.0)
    assert command.vy_cm_s == pytest.approx(0.0)
    assert command.yaw_rate_deg_s == pytest.approx(0.0)


def test_offset_target_moves_directly_sideways_with_camera_mapping():
    points = [(400.0, 300.0), (400.0, 240.0), (400.0, 180.0)]
    follower = _follower()

    command = follower.update(_perception(points), now_s=1.0)

    assert command.vx_cm_s == pytest.approx(0.0)
    assert command.vy_cm_s == pytest.approx(-8.0)
    assert follower.last_diagnostics.target_x_px == 400.0


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
