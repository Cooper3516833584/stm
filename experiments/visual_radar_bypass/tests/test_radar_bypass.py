from types import SimpleNamespace

import numpy as np

from FlightController.Solutions.Safety import (
    Command,
    RadarFieldConfig,
    RadarObstacleField,
)
from experiments.visual_radar_bypass.radar_bypass import (
    ObstacleBypassPlanner,
    ObstacleBypassState,
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


def _desired():
    return Command(10.0, 2.0, 0.0, 0.0, "trajectory_point_follow:single")


def _activate(planner, field):
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


def test_left_obstacle_cluster_automatically_selects_right_bypass():
    planner = ObstacleBypassPlanner()
    field = _field(
        [[99.0, 38.0], [100.0, 40.0], [101.0, 41.0], [102.0, 42.0]]
    )

    first, second = _activate(planner, field)

    assert first == _desired()
    assert planner.state == ObstacleBypassState.BYPASS_RIGHT
    assert planner.target_y_cm == -90.0
    assert planner.active_bypass_side == -1
    assert second.vy_cm_s == -8.0
    assert second.yaw_rate_deg_s == 0.0
    assert "tube_obstacle_sidestep:right" in second.reason
    assert planner.diagnostics()["obstacle_side"] == "left"


def test_right_obstacle_cluster_automatically_selects_left_bypass():
    planner = ObstacleBypassPlanner()
    field = _field(
        [[99.0, -38.0], [100.0, -40.0], [101.0, -41.0], [102.0, -42.0]]
    )

    _, output = _activate(planner, field)

    assert planner.state == ObstacleBypassState.BYPASS_LEFT
    assert planner.target_y_cm == 90.0
    assert planner.active_bypass_side == 1
    assert output.vy_cm_s == 8.0
    assert output.yaw_rate_deg_s == 0.0
    assert "tube_obstacle_sidestep:left" in output.reason
    assert planner.diagnostics()["obstacle_side"] == "right"


def test_center_obstacle_uses_deterministic_default_right_bypass():
    planner = ObstacleBypassPlanner()
    field = _field(
        [[99.0, -2.0], [100.0, 0.0], [101.0, 1.0], [102.0, 2.0]]
    )

    _, output = _activate(planner, field)

    assert planner.state == ObstacleBypassState.BYPASS_RIGHT
    assert planner.target_y_cm == -90.0
    assert output.vy_cm_s == -8.0
    assert planner.diagnostics()["obstacle_side"] == "center"


def test_dense_physical_cluster_wins_over_diffuse_bilateral_returns():
    planner = ObstacleBypassPlanner()
    field = _field(
        [
            [95.0, -42.0],
            [96.0, -41.0],
            [97.0, -40.0],
            [98.0, -39.0],
            [99.0, -38.0],
            [70.0, 15.0],
            [110.0, 30.0],
            [150.0, 55.0],
            [175.0, 70.0],
        ]
    )

    _, output = _activate(planner, field)

    assert planner.state == ObstacleBypassState.BYPASS_LEFT
    assert output.vy_cm_s == 8.0
    assert planner.diagnostics()["cluster_point_count"] >= 3
    assert planner.diagnostics()["obstacle_side"] == "right"


def test_opposite_cluster_during_locked_encounter_keeps_direction():
    planner = ObstacleBypassPlanner()
    left = _field(
        [[99.0, 38.0], [100.0, 40.0], [101.0, 41.0], [102.0, 42.0]]
    )
    _activate(planner, left)
    right_noise = _field(
        [[99.0, -38.0], [100.0, -40.0], [101.0, -41.0], [102.0, -42.0]]
    )

    output = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=right_noise,
        now_s=1.2,
    )

    assert planner.state == ObstacleBypassState.BYPASS_RIGHT
    assert planner.active_bypass_side == -1
    assert output.vy_cm_s == -8.0
    assert "tube_obstacle_sidestep:right" in output.reason


def test_small_cluster_motion_keeps_stable_target_on_selected_side():
    planner = ObstacleBypassPlanner()
    initial = _field(
        [[99.0, 38.0], [100.0, 40.0], [101.0, 41.0], [102.0, 42.0]]
    )
    _activate(planner, initial)
    shifted = _field(
        [[100.0, 39.0], [101.0, 40.0], [102.0, 42.0], [103.0, 43.0]]
    )

    output = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=shifted,
        now_s=1.2,
    )

    assert planner.state == ObstacleBypassState.BYPASS_RIGHT
    assert planner.target_y_cm == -90.0
    assert output.vy_cm_s == -8.0


def test_no_cluster_on_other_road_sections_preserves_visual_command():
    planner = ObstacleBypassPlanner()
    field = _field([])
    desired = _desired()

    for now_s in (1.0, 1.1, 1.2):
        output = planner.update(
            desired=desired,
            perception=_perception(),
            radar_field=field,
            now_s=now_s,
        )

    assert output == desired
    assert planner.state == ObstacleBypassState.NORMAL


def test_points_outside_bilateral_75cm_envelope_do_not_trigger():
    planner = ObstacleBypassPlanner()
    field = _field(
        [[98.0, 80.0], [99.0, 81.0], [100.0, 82.0], [101.0, 83.0]]
    )

    _, output = _activate(planner, field)

    assert output == _desired()
    assert planner.state == ObstacleBypassState.NORMAL


def test_lost_visual_road_keeps_confirmed_radar_sidestep():
    planner = ObstacleBypassPlanner()
    field = _field(
        [[99.0, -38.0], [100.0, -40.0], [101.0, -41.0], [102.0, -42.0]]
    )
    _activate(planner, field)
    hold = Command.zero("trajectory_road_lost_hold")

    output = planner.update(
        desired=hold,
        perception=_perception(is_road_found=False),
        radar_field=field,
        now_s=1.2,
    )

    assert output.vx_cm_s == 0.0
    assert output.vy_cm_s == 8.0
    assert planner.state == ObstacleBypassState.BYPASS_LEFT


def test_close_obstacle_below_40cm_remains_detected():
    planner = ObstacleBypassPlanner()
    field = _field(
        [[29.0, -38.0], [30.0, -40.0], [31.0, -41.0], [32.0, -42.0]]
    )

    _, output = _activate(planner, field)

    assert planner.state == ObstacleBypassState.BYPASS_LEFT
    assert output.vy_cm_s == 8.0


def test_sidestep_releases_directly_back_to_visual_command():
    planner = ObstacleBypassPlanner()
    obstacle = _field(
        [[99.0, -38.0], [100.0, -40.0], [101.0, -41.0], [102.0, -42.0]]
    )
    empty = _field([])
    _activate(planner, obstacle)

    held = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=empty,
        now_s=1.2,
    )
    released = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=empty,
        now_s=2.0,
    )

    assert held.vy_cm_s == 8.0
    assert released == _desired()
    assert planner.state == ObstacleBypassState.NORMAL
