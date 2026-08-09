"""Planner state, ownership and encounter-freezing tests."""

from types import SimpleNamespace

import numpy as np

from FlightController.Solutions.Safety import Command, RadarObstacleField
from experiments.road_contour_bypass.planner import (
    ContourBypassState,
    ContourTrajectoryBypassPlanner,
)


VISUAL = Command(12.0, 1.0, 0.0, 2.0, "visual")
ROAD = SimpleNamespace(
    is_road_found=True,
    confidence=0.9,
    corrected_pixel_error=0.0,
)


def _field(points) -> RadarObstacleField:
    return RadarObstacleField().update(np.asarray(points, dtype=float).reshape(-1, 2), 1.0)


def _obstacle(y=20.0):
    return np.array(
        [[100.0, y], [104.0, y + 2.0], [108.0, y - 2.0], [111.0, y + 1.0]]
    )


def _update(planner, field, now=1.0, fresh=True):
    return planner.update(
        visual_desired=VISUAL,
        perception=ROAD,
        radar_field=field,
        radar_fresh=fresh,
        now_s=now,
        dt_s=0.1,
    )


def _planned():
    planner = ContourTrajectoryBypassPlanner()
    field = _field(_obstacle())
    _update(planner, field, 1.0)
    command = _update(planner, field, 1.1)
    assert planner.state == ContourBypassState.FOLLOW_BYPASS
    return planner, command


def test_no_obstacle_returns_visual_command():
    planner = ContourTrajectoryBypassPlanner()
    assert _update(planner, _field([])) == VISUAL


def test_obstacle_requires_two_frames():
    planner = ContourTrajectoryBypassPlanner()
    field = _field(_obstacle())
    assert _update(planner, field, 1.0) == VISUAL
    assert planner.state == ContourBypassState.ACQUIRE

    command = _update(planner, field, 1.1)
    assert planner.state == ContourBypassState.FOLLOW_BYPASS
    assert command.reason == "frozen_contour_bypass"


def test_bypass_does_not_zero_forward_speed():
    _planner, command = _planned()
    assert command.vx_cm_s > 0.0
    assert command.vy_cm_s != 0.0


def test_yaw_held_before_fov_exit():
    planner, command = _planned()
    assert abs(command.yaw_rate_deg_s) <= 5.0


def test_generated_path_is_frozen_across_new_point_cloud_frames():
    planner, _command = _planned()
    original_path = planner.path_samples.copy()
    original_controls = planner.path_control_points
    original_side = planner.locked_bypass_side

    for index in range(8):
        jittered_or_new = _field(_obstacle(y=-35.0) + np.array((index * 2.0, index)))
        _update(planner, jittered_or_new, 1.2 + index * 0.1)

    assert planner.plan_generation_count == 1
    assert planner.path_control_points == original_controls
    assert planner.locked_bypass_side == original_side
    np.testing.assert_array_equal(planner.path_samples, original_path)
    assert not planner.path_samples.flags.writeable


def test_collision_retry_pushes_path_outward_and_validates_clearance():
    planner, _command = _planned()
    diagnostics = planner.diagnostics()
    assert diagnostics["plan_retry_count"] > 0
    assert diagnostics["path_min_obstacle_clearance_cm"] >= 85.0


def test_random_dropout_does_not_trigger_fov_exit():
    planner, _command = _planned()
    _update(planner, _field([]), 1.2)
    assert planner.state == ContourBypassState.FOLLOW_BYPASS
    assert not planner.diagnostics()["fov_exit_confirmed"]


def test_expected_fov_exit_requires_edge_then_three_missing_frames():
    planner, _command = _planned()
    for bearing_deg in (30.0, 50.0, 68.0):
        radius = 100.0
        radians = np.radians(bearing_deg)
        center = np.array((radius * np.cos(radians), radius * np.sin(radians)))
        edge = _field(center + np.array([[0, 0], [2, 1], [-2, -1], [1, -2]]))
        _update(planner, edge, 1.2 + bearing_deg / 1000.0)
    assert planner.diagnostics()["fov_exit_armed"]

    for index in range(3):
        _update(planner, _field([]), 1.4 + index * 0.1)
    assert planner.state == ContourBypassState.RETURN_TO_ROAD
    assert planner.diagnostics()["fov_exit_confirmed"]
