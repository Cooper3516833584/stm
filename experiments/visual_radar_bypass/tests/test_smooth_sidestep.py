from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from FlightController.Solutions.Safety import (
    Command,
    RadarFieldConfig,
    RadarObstacleField,
)
from experiments.visual_radar_bypass.smooth_sidestep import (
    SmoothSidestepConfig,
    SmoothSidestepPlanner,
    SmoothSidestepState,
)


def _perception(**overrides):
    values = {
        "is_road_found": True,
        "confidence": 0.9,
        "corrected_pixel_error": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _field(points):
    field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=300.0,
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=75.0,
        )
    )
    return field.update(np.asarray(points, dtype=float), now_s=1.0)


RIGHT_OBSTACLE = _field(
    [[79.0, -39.0], [80.0, -40.0], [81.0, -41.0], [82.0, -42.0]]
)
LEFT_OBSTACLE = _field(
    [[79.0, 39.0], [80.0, 40.0], [81.0, 41.0], [82.0, 42.0]]
)
EMPTY = _field([])


def _desired(vy=-6.0):
    return Command(14.0, vy, 0.0, 2.0, "trajectory_point_follow:single")


def _update(planner, field, now_s, desired=None, perception=None):
    return planner.update(
        desired=desired or _desired(),
        perception=perception or _perception(),
        radar_field=field,
        now_s=now_s,
    )


def _activate_right(planner):
    first = _update(planner, RIGHT_OBSTACLE, 1.0)
    second = _update(planner, RIGHT_OBSTACLE, 1.1)
    return first, second


def test_right_obstacle_locks_left_without_an_instant_command_reversal():
    planner = SmoothSidestepPlanner()

    first, second = _activate_right(planner)

    assert first == _desired()
    assert planner.state == SmoothSidestepState.SHIFT_LEFT
    assert planner.active_bypass_side == 1
    assert planner.target_y_cm == 90.0
    assert _desired().vy_cm_s < second.vy_cm_s < 10.0
    assert second.yaw_rate_deg_s == _desired().yaw_rate_deg_s


def test_ramp_in_is_monotonic_and_reaches_configured_lateral_speed():
    planner = SmoothSidestepPlanner()
    _activate_right(planner)
    outputs = []

    for step in range(2, 12):
        outputs.append(_update(planner, RIGHT_OBSTACLE, 1.0 + step * 0.1))

    lateral = [command.vy_cm_s for command in outputs]
    assert lateral == sorted(lateral)
    assert lateral[-1] == 10.0
    assert outputs[-1].vx_cm_s == 8.0


def test_clear_hold_prevents_short_radar_dropout_from_returning_to_vision():
    planner = SmoothSidestepPlanner()
    _activate_right(planner)
    for step in range(2, 12):
        _update(planner, RIGHT_OBSTACLE, 1.0 + step * 0.1)

    held = _update(planner, EMPTY, 3.0)

    assert planner.state == SmoothSidestepState.SHIFT_LEFT
    assert held.vy_cm_s == 10.0


def test_blend_back_changes_smoothly_then_returns_exact_visual_command():
    planner = SmoothSidestepPlanner()
    _activate_right(planner)
    for step in range(2, 12):
        _update(planner, RIGHT_OBSTACLE, 1.0 + step * 0.1)

    outputs = []
    for step in range(22, 72):
        outputs.append(_update(planner, EMPTY, 1.0 + step * 0.1))

    non_visual = [command for command in outputs if command != _desired()]
    assert non_visual
    assert non_visual[0].vy_cm_s == 10.0
    assert all(
        later.vy_cm_s <= earlier.vy_cm_s
        for earlier, later in zip(non_visual, non_visual[1:])
    )
    assert outputs[-1] == _desired()
    assert planner.state == SmoothSidestepState.NORMAL


def test_reappearance_during_blend_keeps_original_locked_side():
    planner = SmoothSidestepPlanner()
    _activate_right(planner)
    for step in range(2, 12):
        _update(planner, RIGHT_OBSTACLE, 1.0 + step * 0.1)
    for step in range(22, 39):
        _update(planner, EMPTY, 1.0 + step * 0.1)
    assert planner.state == SmoothSidestepState.BLEND_BACK

    output = _update(planner, LEFT_OBSTACLE, 5.0)

    assert planner.state == SmoothSidestepState.SHIFT_LEFT
    assert planner.active_bypass_side == 1
    assert output.vy_cm_s > _desired().vy_cm_s


def test_three_second_dropout_cannot_start_a_second_encounter():
    planner = SmoothSidestepPlanner()
    _activate_right(planner)
    for step in range(2, 12):
        _update(planner, RIGHT_OBSTACLE, 1.0 + step * 0.1)

    for step in range(22, 51):
        _update(planner, EMPTY, 1.0 + step * 0.1)
    resumed = _update(planner, RIGHT_OBSTACLE, 6.1)

    assert planner.state == SmoothSidestepState.SHIFT_LEFT
    assert planner.active_bypass_side == 1
    assert resumed.vy_cm_s > _desired().vy_cm_s


def test_left_obstacle_selects_right_once():
    planner = SmoothSidestepPlanner()
    _update(planner, LEFT_OBSTACLE, 1.0)
    output = _update(planner, LEFT_OBSTACLE, 1.1, desired=_desired(vy=6.0))

    assert planner.state == SmoothSidestepState.SHIFT_RIGHT
    assert planner.active_bypass_side == -1
    assert output.vy_cm_s < 6.0


def test_points_outside_simple_rectangular_gate_do_not_trigger():
    planner = SmoothSidestepPlanner()
    outside = _field(
        [[5.0, -20.0], [100.0, -80.0], [181.0, -20.0], [200.0, 0.0]]
    )

    for now_s in (1.0, 1.1, 1.2):
        output = _update(planner, outside, now_s)

    assert output == _desired()
    assert planner.state == SmoothSidestepState.NORMAL


def test_confirmed_sidestep_continues_when_visual_road_is_temporarily_lost():
    planner = SmoothSidestepPlanner()
    _activate_right(planner)
    hold = Command.zero("trajectory_road_lost_hold")

    output = _update(
        planner,
        RIGHT_OBSTACLE,
        1.2,
        desired=hold,
        perception=_perception(is_road_found=False),
    )

    assert planner.state == SmoothSidestepState.SHIFT_LEFT
    assert output.vy_cm_s > 0.0
    assert output.vx_cm_s == 0.0


def test_maximum_sidestep_time_stops_instead_of_exceeding_activity_range():
    config = replace(SmoothSidestepConfig(), max_sidestep_s=1.0)
    planner = SmoothSidestepPlanner(config)
    _activate_right(planner)

    output = _update(planner, RIGHT_OBSTACLE, 2.2)

    assert output.vx_cm_s == 0.0
    assert output.vy_cm_s == 0.0
    assert planner.state == SmoothSidestepState.TIMEOUT_STOP
    assert "smooth_sidestep_timeout_stop" in output.reason


def test_diagnostics_expose_locked_side_and_smooth_weight():
    planner = SmoothSidestepPlanner()
    _activate_right(planner)

    diagnostics = planner.diagnostics()

    assert diagnostics["planner"] == "smooth_sidestep"
    assert diagnostics["active_bypass_side"] == "left"
    assert 0.0 < diagnostics["blend_alpha"] < 1.0
    assert diagnostics["obstacle_side"] == "right"
