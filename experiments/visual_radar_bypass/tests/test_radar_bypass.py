from types import SimpleNamespace

import numpy as np

from FlightController.Solutions.Safety import (
    Command,
    RadarFieldConfig,
    RadarObstacleField,
)
from experiments.visual_radar_bypass.radar_bypass import (
    LeftTreeBypassPlanner,
    LeftTreeBypassState,
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


def test_realistic_left_tree_at_40cm_selects_right_40cm_target():
    planner = LeftTreeBypassPlanner()
    field = _field([[100.0, 40.0]])

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

    assert first == _desired()
    assert planner.state == LeftTreeBypassState.BYPASS_RIGHT
    assert planner.target_y_cm == -40.0
    assert second.vx_cm_s == 10.0
    assert second.vy_cm_s == 0.0
    assert second.yaw_rate_deg_s > 0.0
    assert "left_tree_bypass:right" in second.reason


def test_verified_right_side_ghost_returns_do_not_block_local_right_bypass():
    planner = LeftTreeBypassPlanner()
    field = _field(
        [
            [100.0, 40.0],
            [90.0, -70.0],
            [130.0, -30.0],
        ]
    )

    planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=field,
        now_s=1.0,
    )
    output = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=field,
        now_s=1.1,
    )

    assert len(field.points_body_cm) == 3
    assert planner.state == LeftTreeBypassState.BYPASS_RIGHT
    assert planner.target_y_cm == -40.0
    assert output.yaw_rate_deg_s > 0.0


def test_left_point_outside_75cm_intrusion_envelope_does_not_trigger():
    planner = LeftTreeBypassPlanner()
    field = _field([[100.0, 80.0]])

    desired = _desired()
    for now_s in (1.0, 1.1, 1.2):
        output = planner.update(
            desired=desired,
            perception=_perception(),
            radar_field=field,
            now_s=now_s,
        )

    assert output == desired
    assert planner.state == LeftTreeBypassState.NORMAL


def test_lost_visual_road_resets_bypass_and_holds_visual_command():
    planner = LeftTreeBypassPlanner()
    field = _field([[100.0, 40.0]])
    planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=field,
        now_s=1.0,
    )
    planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=field,
        now_s=1.1,
    )
    hold = Command.zero("trajectory_road_lost_hold")

    output = planner.update(
        desired=hold,
        perception=_perception(is_road_found=False),
        radar_field=field,
        now_s=1.2,
    )

    assert output == hold
    assert planner.state == LeftTreeBypassState.NORMAL
