from types import SimpleNamespace

import numpy as np
import pytest

from FlightController.Solutions.Safety import Command, RadarObstacleField
from experiments.visual_radar_bypass.circular_tube_bypass import (
    CircularBypassState,
    CircularTubeBypassConfig,
    CircularTubeBypassPlanner,
)


def _perception(error=100.0, **overrides):
    values = {
        "is_road_found": True,
        "confidence": 0.9,
        "corrected_pixel_error": error,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _desired(yaw=10.0):
    return Command(12.0, 1.0, 0.0, yaw, "trajectory_point_follow:single")


def _field(points, now_s=1.0):
    return RadarObstacleField().update(np.asarray(points, dtype=float), now_s)


def _right_tube():
    return _field(
        [[99.0, -39.0], [100.0, -40.0], [101.0, -41.0], [102.0, -42.0]]
    )


def _activate(planner, field=None):
    field = field or _right_tube()
    first = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=field,
        now_s=1.0,
    )
    second = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=field,
        now_s=1.1,
    )
    return first, second


def test_default_inflated_radius_is_tube_plus_safety_radius():
    config = CircularTubeBypassConfig()

    assert config.tube_radius_cm == 15.0
    assert config.safety_radius_cm == 75.0
    assert config.orbit_radius_cm == 90.0


def test_radar_arc_is_directly_fitted_and_then_inflated():
    angles = np.radians(np.linspace(135.0, 225.0, 9))
    center = np.asarray([100.0, -40.0])
    radius = 8.0
    points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    planner = CircularTubeBypassPlanner()

    _activate(planner, _field(points))
    diagnostics = planner.diagnostics()

    assert diagnostics["circle_fit_used"]
    assert diagnostics["fitted_tube_radius_cm"] == pytest.approx(radius)
    assert diagnostics["circle_fit_rms_cm"] == pytest.approx(0.0, abs=1e-9)
    assert diagnostics["orbit_radius_cm"] == pytest.approx(radius + 75.0)


def test_right_tube_selects_forward_left_circle_tangent():
    planner = CircularTubeBypassPlanner()

    first, orbit = _activate(planner)

    assert first == _desired()
    assert planner.state == CircularBypassState.ORBIT_LEFT
    assert planner.active_bypass_side == 1
    assert orbit.vx_cm_s > 0.0
    assert orbit.vy_cm_s > 0.0
    assert abs(orbit.yaw_rate_deg_s) < 8.0
    assert "circular_tube:orbit_left" in orbit.reason


def test_left_tube_selects_forward_right_circle_tangent():
    planner = CircularTubeBypassPlanner()
    left = _field(
        [[99.0, 39.0], [100.0, 40.0], [101.0, 41.0], [102.0, 42.0]]
    )

    _, orbit = _activate(planner, left)

    assert planner.state == CircularBypassState.ORBIT_RIGHT
    assert planner.active_bypass_side == -1
    assert orbit.vx_cm_s > 0.0
    assert orbit.vy_cm_s < 0.0


def test_visual_error_below_50_ends_orbit_after_minimum_guard():
    planner = CircularTubeBypassPlanner()
    field = _right_tube()
    _activate(planner, field)

    still_orbit = planner.update(
        desired=_desired(),
        perception=_perception(error=49.9),
        radar_field=field,
        now_s=1.9,
    )
    returning = planner.update(
        desired=_desired(),
        perception=_perception(error=49.9),
        radar_field=field,
        now_s=2.2,
    )

    assert planner.state == CircularBypassState.RETURN_VISUAL
    assert "orbit_left" in still_orbit.reason
    assert "return_visual" in returning.reason
    assert abs(returning.yaw_rate_deg_s) < 8.0
    assert planner.diagnostics()["exit_reason"] == "visual_error"


def test_return_blend_clamps_visual_yaw_strictly_below_eight():
    config = CircularTubeBypassConfig(return_blend_s=2.0)
    planner = CircularTubeBypassPlanner(config)
    field = _right_tube()
    _activate(planner, field)
    planner.update(
        desired=_desired(yaw=20.0),
        perception=_perception(error=0.0),
        radar_field=field,
        now_s=2.2,
    )

    returning = planner.update(
        desired=_desired(yaw=20.0),
        perception=_perception(error=0.0),
        radar_field=field,
        now_s=3.2,
    )

    assert planner.state == CircularBypassState.RETURN_VISUAL
    assert abs(returning.yaw_rate_deg_s) <= 7.0
    assert abs(returning.yaw_rate_deg_s) < 8.0
    assert abs(returning.as_fc_tuple()[3]) < 8


def test_initially_small_visual_error_does_not_abort_avoidance_immediately():
    planner = CircularTubeBypassPlanner()
    field = _right_tube()
    planner.update(
        desired=_desired(),
        perception=_perception(error=20.0),
        radar_field=field,
        now_s=1.0,
    )
    planner.update(
        desired=_desired(),
        perception=_perception(error=20.0),
        radar_field=field,
        now_s=1.1,
    )

    output = planner.update(
        desired=_desired(),
        perception=_perception(error=20.0),
        radar_field=field,
        now_s=2.2,
    )

    assert planner.state == CircularBypassState.ORBIT_LEFT
    assert "orbit_left" in output.reason
    assert not planner.diagnostics()["visual_return_armed"]


def test_configured_arc_completion_enters_visual_return():
    planner = CircularTubeBypassPlanner(
        CircularTubeBypassConfig(
            target_arc_deg=1.0,
            min_orbit_before_visual_return_s=99.0,
        )
    )
    field = _right_tube()
    _activate(planner, field)

    planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=field,
        now_s=1.2,
    )
    output = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=field,
        now_s=1.3,
    )

    assert planner.state == CircularBypassState.RETURN_VISUAL
    assert "return_visual" in output.reason
    assert planner.diagnostics()["exit_reason"] == "arc_complete"


def test_no_obstacle_keeps_visual_command_unchanged():
    planner = CircularTubeBypassPlanner()

    output = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=_field([]),
        now_s=1.0,
    )

    assert output == _desired()
    assert planner.state == CircularBypassState.NORMAL


def test_custom_radii_must_produce_requested_circle():
    config = CircularTubeBypassConfig(tube_radius_cm=8.0, safety_radius_cm=62.0)

    assert config.orbit_radius_cm == pytest.approx(70.0)
