from argparse import Namespace

import numpy as np

from FlightController.Solutions.RoadObstacleBypassPlanner import (
    BYPASS_ACTIVITY_HALF_WIDTH_CM,
    OBSTACLE_SAFETY_DISTANCE_CM,
    ROAD_HALF_WIDTH_CM,
    ROAD_WIDTH_CM,
    RoadBypassConfig,
    RoadBypassState,
    RoadObstacleBypassPlanner,
)
from FlightController.Solutions.Safety import Command, RadarFieldConfig, RadarObstacleField


def _perception(**overrides):
    values = {
        "is_road_found": True,
        "confidence": 0.8,
        "corrected_pixel_error": 80.0,
    }
    values.update(overrides)
    return Namespace(**values)


def _field(points):
    field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=300.0,
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=80.0,
        )
    )
    field.update(np.asarray(points, dtype=float), now_s=0.0)
    return field


def _desired():
    return Command(25.0, 0.0, 0.0, 2.0, "road_follow")


def _planner(**overrides):
    # Behavioral bypass tests use an explicitly wide synthetic road.  The
    # production default is the measured 50cm road, which intentionally has
    # no lateral bypass room after the 25cm body/edge margin.
    values = {
        "enabled": True,
        "road_half_width_cm": 120.0,
        "intrusion_half_width_cm": 80.0,
        "bypass_activity_half_width_cm": 90.0,
    }
    values.update(overrides)
    return RoadObstacleBypassPlanner(RoadBypassConfig(**values))


def test_default_road_geometry_uses_50cm_full_width():
    config = RoadBypassConfig()

    assert ROAD_WIDTH_CM == 50.0
    assert ROAD_HALF_WIDTH_CM == 25.0
    assert config.road_half_width_cm == 25.0
    assert config.intrusion_half_width_cm == 25.0
    assert config.effective_intrusion_half_width_cm == 75.0
    assert OBSTACLE_SAFETY_DISTANCE_CM == 75.0
    assert BYPASS_ACTIVITY_HALF_WIDTH_CM == 90.0
    assert config.bypass_activity_half_width_cm == 90.0
    assert config.road_half_width_cm - config.road_edge_margin_cm == 0.0


def test_default_50cm_demo_uses_clear_side_for_center_obstacle():
    planner = RoadObstacleBypassPlanner(
        RoadBypassConfig(enabled=True, activate_frames=1)
    )
    output = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=_field([[100.0, 0.0]]),
        now_s=0.0,
    )

    assert planner.state in {RoadBypassState.BYPASS_LEFT, RoadBypassState.BYPASS_RIGHT}
    assert planner.last_target_y_cm is not None
    assert abs(planner.last_target_y_cm) > RoadBypassConfig().bypass_clearance_cm
    assert output.vx_cm_s <= RoadBypassConfig().bypass_speed_cm_s
    assert output.vy_cm_s == 0.0
    assert "road_bypass:" in output.reason


def test_obstacle_outside_road_but_inside_safety_envelope_triggers_bypass():
    planner = RoadObstacleBypassPlanner(
        RoadBypassConfig(enabled=True, activate_frames=1)
    )
    output = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=_field([[100.0, 60.0]]),
        now_s=0.0,
    )

    assert planner.state == RoadBypassState.BYPASS_RIGHT
    assert planner.last_target_y_cm is not None and planner.last_target_y_cm < 0.0
    assert output.yaw_rate_deg_s > _desired().yaw_rate_deg_s


def test_obstacle_outside_safety_envelope_does_not_trigger_bypass():
    planner = RoadObstacleBypassPlanner(
        RoadBypassConfig(enabled=True, activate_frames=1)
    )
    desired = _desired()
    output = planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=_field([[100.0, 80.0]]),
        now_s=0.0,
    )

    assert output == desired
    assert planner.state == RoadBypassState.NORMAL


def test_known_clear_right_ignores_clear_side_returns_and_turns_right():
    planner = RoadObstacleBypassPlanner(
        RoadBypassConfig(
            enabled=True,
            activate_frames=1,
            known_clear_side="right",
        )
    )
    desired = _desired()
    output = planner.update(
        desired=desired,
        perception=_perception(),
        # Positive Y is the real left/tree side.  Negative-Y points emulate
        # the lower-radar/ground returns observed on the verified-clear side.
        radar_field=_field([[100.0, 15.0], [90.0, -70.0], [130.0, -30.0]]),
        now_s=0.0,
    )

    assert planner.state == RoadBypassState.BYPASS_RIGHT
    assert planner.last_target_y_cm == -70.0
    assert output.yaw_rate_deg_s > desired.yaw_rate_deg_s


def test_known_clear_left_is_symmetric():
    planner = RoadObstacleBypassPlanner(
        RoadBypassConfig(
            enabled=True,
            activate_frames=1,
            known_clear_side="left",
        )
    )
    desired = _desired()
    output = planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=_field([[100.0, -15.0], [90.0, 70.0], [130.0, 30.0]]),
        now_s=0.0,
    )

    assert planner.state == RoadBypassState.BYPASS_LEFT
    assert planner.last_target_y_cm == 70.0
    assert output.yaw_rate_deg_s < desired.yaw_rate_deg_s


def test_auto_mode_keeps_conservative_no_gap_behavior_for_two_sided_points():
    planner = RoadObstacleBypassPlanner(
        RoadBypassConfig(enabled=True, activate_frames=1, known_clear_side=None)
    )
    output = planner.update(
        desired=_desired(),
        perception=_perception(),
        radar_field=_field([[100.0, -70.0], [100.0, 0.0], [100.0, 70.0]]),
        now_s=0.0,
    )

    assert planner.state == RoadBypassState.NORMAL
    assert planner.last_target_y_cm is None
    assert "road_bypass_no_gap" in output.reason


def test_disabled_does_not_modify_command():
    desired = _desired()
    planner = RoadObstacleBypassPlanner(RoadBypassConfig(enabled=False))

    output = planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=_field([[100.0, 20.0]]),
        now_s=0.0,
    )

    assert output == desired


def test_unusable_road_does_not_modify_command_and_resets():
    desired = _desired()
    planner = _planner(activate_frames=1)
    planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=_field([[100.0, 30.0], [120.0, 60.0]]),
        now_s=0.0,
    )
    assert planner.state != RoadBypassState.NORMAL

    output = planner.update(
        desired=desired,
        perception=_perception(is_road_found=False),
        radar_field=_field([[100.0, 30.0]]),
        now_s=0.1,
    )

    assert output == desired
    assert planner.state == RoadBypassState.NORMAL
    assert planner.last_target_y_cm is None


def test_single_intrusion_frame_does_not_trigger():
    desired = _desired()
    planner = _planner(activate_frames=2)

    output = planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=_field([[100.0, 20.0], [120.0, 30.0]]),
        now_s=0.0,
    )

    assert output == desired
    assert planner.state == RoadBypassState.NORMAL


def test_consecutive_intrusions_trigger_bypass():
    desired = _desired()
    planner = _planner(activate_frames=2, bypass_speed_cm_s=12.0)
    field = _field([[100.0, 20.0], [120.0, 30.0]])

    planner.update(desired=desired, perception=_perception(), radar_field=field, now_s=0.0)
    output = planner.update(desired=desired, perception=_perception(), radar_field=field, now_s=0.1)

    assert planner.state in {RoadBypassState.BYPASS_LEFT, RoadBypassState.BYPASS_RIGHT}
    assert output.vx_cm_s <= 12.0
    assert output.vy_cm_s == 0.0
    assert "road_bypass" in output.reason


def test_left_side_blocked_selects_right_bypass():
    desired = _desired()
    planner = _planner(activate_frames=1)

    output = planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=_field([[100.0, 30.0], [120.0, 60.0]]),
        now_s=0.0,
    )

    assert planner.last_target_y_cm is not None
    assert planner.last_target_y_cm < 0.0
    assert planner.state == RoadBypassState.BYPASS_RIGHT
    assert output.yaw_rate_deg_s > desired.yaw_rate_deg_s


def test_right_side_blocked_selects_left_bypass():
    desired = _desired()
    planner = _planner(activate_frames=1)

    output = planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=_field([[100.0, -30.0], [120.0, -60.0]]),
        now_s=0.0,
    )

    assert planner.last_target_y_cm is not None
    assert planner.last_target_y_cm > 0.0
    assert planner.state == RoadBypassState.BYPASS_LEFT
    assert output.yaw_rate_deg_s < desired.yaw_rate_deg_s


def test_obstacle_release_enters_return_center():
    desired = _desired()
    planner = _planner(activate_frames=1, release_s=0.5)
    planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=_field([[100.0, 30.0], [120.0, 60.0]]),
        now_s=0.0,
    )

    output = planner.update(
        desired=desired,
        perception=_perception(corrected_pixel_error=80.0),
        radar_field=_field([]),
        now_s=0.6,
    )

    assert planner.state == RoadBypassState.RETURN_CENTER
    assert "road_bypass_return" in output.reason
    assert output.vx_cm_s <= RoadBypassConfig().bypass_speed_cm_s * 1.2


def test_return_center_restores_normal_when_centered():
    desired = _desired()
    planner = _planner(activate_frames=1, release_s=0.5)
    planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=_field([[100.0, 30.0], [120.0, 60.0]]),
        now_s=0.0,
    )
    planner.update(
        desired=desired,
        perception=_perception(corrected_pixel_error=80.0),
        radar_field=_field([]),
        now_s=0.6,
    )

    output = planner.update(
        desired=desired,
        perception=_perception(corrected_pixel_error=0.0),
        radar_field=_field([]),
        now_s=0.7,
    )

    assert output == desired
    assert planner.state == RoadBypassState.NORMAL
