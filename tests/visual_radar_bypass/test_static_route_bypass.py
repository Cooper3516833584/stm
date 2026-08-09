"""Tests for the frozen static-route bypass planner."""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from FlightController.Solutions.Safety import Command, RadarFieldConfig, RadarObstacleField
from experiments.visual_radar_bypass.static_route_bypass import (
    STATIC_ROUTE_PROFILE_NAME,
    STATIC_ROUTE_PROFILE_STATUS,
    StaticRouteBypassConfig,
    StaticRouteBypassPlanner,
    StaticRouteBypassState,
)
from experiments.visual_radar_bypass.replay_radar_session import replay


def _perception(**overrides):
    values = {"is_road_found": True, "confidence": 0.9}
    values.update(overrides)
    return SimpleNamespace(**values)


def _desired(vy=-6.0, yaw=3.0):
    return Command(14.0, vy, 0.0, yaw, "trajectory_point_follow:single")


def _field(points, now=1.0):
    field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=300.0,
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=75.0,
        )
    )
    return field.update(np.asarray(points, dtype=float), now)


def _cluster(x, y):
    return [[x - 1.5, y - 1.5], [x - 0.5, y - 0.5], [x + 0.5, y + 0.5], [x + 1.5, y + 1.5]]


def _update(planner, field, now, desired=None, perception=None):
    return planner.update(
        desired=desired or _desired(),
        perception=perception or _perception(),
        radar_field=field,
        now_s=now,
    )


def _activate_right(planner):
    field = _field(_cluster(100.0, -40.0))
    _update(planner, field, 1.0)
    return _update(planner, field, 1.1)


def test_default_policy_is_front_180_and_sixty_percent_forward_speed():
    config = StaticRouteBypassConfig()

    assert config.front_fov_deg == 180.0
    assert config.half_fov_deg == 90.0
    assert config.avoidance_forward_ratio == 0.60
    assert config.avoidance_vx_cm_s == pytest.approx(8.4)


def test_current_normal_activation_uses_10_to_300_x_and_100_cm_body_radius():
    config = replace(
        StaticRouteBypassConfig(),
        normal_activation_radius_cm=100.0,
    )
    planner = StaticRouteBypassPlanner(config)

    _update(planner, _field(_cluster(110.0, 0.0)), 1.0)
    _update(planner, _field(_cluster(110.0, 0.0)), 1.1)
    assert planner.state == StaticRouteBypassState.NORMAL

    near_low_x = _field(_cluster(11.0, 40.0))
    _update(planner, near_low_x, 1.2)
    _update(planner, near_low_x, 1.3)
    assert planner.state == StaticRouteBypassState.DIVERGE_RIGHT


def test_target_guidance_override_allows_bypass_without_road_and_suppresses_yaw():
    planner = StaticRouteBypassPlanner()
    field = _field(_cluster(100.0, -40.0))
    target = Command(13.2, 0.0, 0.0, 12.0, "purple_target:approach")

    assert planner.has_obstacle_conflict(field)
    assert planner.state == StaticRouteBypassState.NORMAL

    first = planner.update(
        desired=target,
        perception=None,
        radar_field=field,
        now_s=1.0,
        guidance_usable=True,
        preserve_guidance_yaw=False,
    )
    active = planner.update(
        desired=target,
        perception=None,
        radar_field=field,
        now_s=1.1,
        guidance_usable=True,
        preserve_guidance_yaw=False,
    )
    active = planner.update(
        desired=target,
        perception=None,
        radar_field=field,
        now_s=1.2,
        guidance_usable=True,
        preserve_guidance_yaw=False,
    )

    assert first == target
    assert planner.state != StaticRouteBypassState.NORMAL
    assert active.vy_cm_s > 0.0
    assert active.yaw_rate_deg_s == 0.0


def test_flight_validated_v1_profile_defaults_are_frozen():
    config = StaticRouteBypassConfig()

    assert STATIC_ROUTE_PROFILE_NAME == "static-route-flight-v1"
    assert STATIC_ROUTE_PROFILE_STATUS == "FROZEN_FLIGHT_VALIDATED"
    assert (
        config.front_fov_deg,
        config.avoidance_vx_cm_s,
        config.target_surface_clearance_cm,
        config.reshift_surface_clearance_cm,
        config.max_outward_vy_cm_s,
        config.association_radius_cm,
        config.edge_arm_deg,
        config.clearance_run_s,
        config.rear_margin_cm,
        config.translation_credit_ratio,
        config.track_lost_hold_s,
        config.max_encounter_s,
    ) == (
        180.0,
        8.4,
        85.0,
        75.0,
        8.0,
        50.0,
        80.0,
        None,
        20.0,
        0.70,
        1.0,
        40.0,
    )


def test_faster_future_profile_can_override_speed_without_mutating_v1():
    v1 = StaticRouteBypassConfig()
    future = replace(v1, avoidance_forward_ratio=0.75, max_outward_vy_cm_s=9.0)

    assert v1.avoidance_vx_cm_s == pytest.approx(8.4)
    assert v1.max_outward_vy_cm_s == 8.0
    assert future.avoidance_vx_cm_s == pytest.approx(10.5)
    assert future.max_outward_vy_cm_s == 9.0
    assert StaticRouteBypassConfig() == v1


def test_right_front_obstacle_locks_left_and_ignores_visual_lateral_command():
    first = StaticRouteBypassPlanner()
    second = StaticRouteBypassPlanner()
    _activate_right(first)
    _activate_right(second)

    left_visual = _update(first, _field(_cluster(100.0, -40.0)), 1.3, desired=_desired(vy=-10.0, yaw=4.0))
    right_visual = _update(second, _field(_cluster(100.0, -40.0)), 1.3, desired=_desired(vy=10.0, yaw=4.0))

    assert first.state == StaticRouteBypassState.DIVERGE_LEFT
    assert first.active_bypass_side == 1
    assert left_visual.vx_cm_s == pytest.approx(8.4)
    assert left_visual.vy_cm_s == pytest.approx(right_visual.vy_cm_s)
    assert left_visual.vy_cm_s > 0.0
    assert left_visual.yaw_rate_deg_s == 4.0


def test_path_tangent_yaw_is_preserved_but_path_lateral_is_suppressed():
    planner = StaticRouteBypassPlanner()
    _activate_right(planner)

    command = _update(planner, _field(_cluster(100.0, -40.0)), 1.4, desired=_desired(vy=-9.0, yaw=-7.0))

    assert command.yaw_rate_deg_s == -7.0
    assert command.vy_cm_s > 0.0
    assert planner.diagnostics()["visual_lateral_suppressed"]


def test_tracking_accepts_points_between_old_75_and_new_90_degree_edges():
    planner = StaticRouteBypassPlanner()
    angle = math.radians(85.0)
    points = _cluster(100.0 * math.cos(angle), -100.0 * math.sin(angle))
    planner._predicted_center = np.asarray(
        [100.0 * math.cos(angle), -100.0 * math.sin(angle)], dtype=float
    )

    observation = planner._observe(_field(points), tracking=True)

    assert observation is not None
    assert abs(observation.bearing_deg) > 80.0
    assert abs(observation.bearing_deg) <= 90.0


def test_points_behind_the_front_half_plane_are_not_used_for_tracking():
    planner = StaticRouteBypassPlanner()
    observation = planner._observe(_field(_cluster(-10.0, -100.0)), tracking=True)
    assert observation is None


def test_target_clearance_moves_to_forward_pass_without_side_switch():
    planner = StaticRouteBypassPlanner()
    _activate_right(planner)
    clear = _field(_cluster(70.0, -90.0))
    _update(planner, _field(_cluster(90.0, -65.0)), 1.2)
    _update(planner, _field(_cluster(80.0, -80.0)), 1.25)

    for index in range(3):
        command = _update(planner, clear, 1.3 + index * 0.1)

    assert planner.state == StaticRouteBypassState.PASS_FORWARD_LEFT
    assert planner.active_bypass_side == 1
    assert command.vx_cm_s == pytest.approx(8.4)
    assert command.vy_cm_s >= 0.0
    decayed = _update(planner, clear, 2.6)
    assert decayed.vy_cm_s == 0.0


def test_unexpected_central_dropout_stops_instead_of_claiming_passage():
    planner = StaticRouteBypassPlanner()
    _activate_right(planner)

    command = _update(planner, _field([]), 1.3)

    assert planner.state == StaticRouteBypassState.TRACK_LOST_HOLD
    assert command.vx_cm_s == 0.0
    assert command.vy_cm_s == 0.0


def test_path_loss_stops_active_bypass_and_recovers_same_side():
    planner = StaticRouteBypassPlanner()
    _activate_right(planner)
    obstacle = _field(_cluster(100.0, -40.0))

    stopped = _update(
        planner,
        obstacle,
        1.3,
        desired=Command.zero("trajectory_road_lost_hold"),
        perception=_perception(is_road_found=False),
    )
    _update(planner, obstacle, 1.4)
    resumed = _update(planner, obstacle, 1.5)

    assert stopped.vx_cm_s == 0.0
    assert stopped.vy_cm_s == 0.0
    assert planner.active_bypass_side == 1
    assert resumed.vy_cm_s > 0.0


def test_command_odometry_uses_xy_and_yaw_only_when_applied():
    planner = StaticRouteBypassPlanner()
    _activate_right(planner)
    before = np.asarray([
        planner.diagnostics()["predicted_center_x_cm"],
        planner.diagnostics()["predicted_center_y_cm"],
    ])

    planner.report_applied_command(Command(8.4, 2.0, 0.0, 10.0), 1.0, False)
    unchanged = np.asarray([
        planner.diagnostics()["predicted_center_x_cm"],
        planner.diagnostics()["predicted_center_y_cm"],
    ])
    planner.report_applied_command(Command(8.4, 2.0, 0.0, 10.0), 1.0, True)
    changed = np.asarray([
        planner.diagnostics()["predicted_center_x_cm"],
        planner.diagnostics()["predicted_center_y_cm"],
    ])

    assert np.allclose(before, unchanged)
    assert not np.allclose(before, changed)
    assert planner.diagnostics()["credited_translation_cm"] > 0.0
    assert planner.diagnostics()["credited_yaw_deg"] == pytest.approx(5.0)


def test_expected_90_degree_exit_requires_rear_margin_before_visual_blend():
    config = replace(
        StaticRouteBypassConfig(),
        target_surface_clearance_cm=70.0,
        reshift_surface_clearance_cm=60.0,
        rear_margin_cm=5.0,
        translation_credit_ratio=1.0,
        blend_back_s=0.2,
    )
    planner = StaticRouteBypassPlanner(config)
    _activate_right(planner)
    _update(planner, _field(_cluster(90.0, -60.0)), 1.2)
    for index, point in enumerate(((65.0, -75.0), (45.0, -80.0), (30.0, -80.0))):
        _update(planner, _field(_cluster(*point)), 1.3 + index * 0.1)
    assert planner.state == StaticRouteBypassState.PASS_FORWARD_LEFT

    _update(planner, _field(_cluster(12.0, -90.0)), 1.7)
    _update(planner, _field(_cluster(6.0, -90.0)), 1.8)
    assert planner.state == StaticRouteBypassState.SIDE_PASS_CONFIRM

    for index in range(3):
        _update(planner, _field([]), 1.9 + index * 0.1)
    assert planner.state == StaticRouteBypassState.CLEARANCE_RUN

    # Visual asks to return right, but it remains suppressed until the known
    # tube radius and rear margin are behind the current path-normal plane.
    visual_return_seen_early = False
    now = 2.2
    for _ in range(80):
        command = _update(planner, _field([]), now, desired=_desired(vy=-10.0))
        if planner.state not in {StaticRouteBypassState.BLEND_BACK, StaticRouteBypassState.NORMAL}:
            visual_return_seen_early |= command.vy_cm_s < 0.0
        planner.report_applied_command(command, 0.1, True)
        now += 0.1
        if planner.state == StaticRouteBypassState.BLEND_BACK:
            break

    assert not visual_return_seen_early
    assert planner.state == StaticRouteBypassState.BLEND_BACK
    assert planner.diagnostics()["predicted_center_x_cm"] + config.tube_radius_cm + config.rear_margin_cm <= 0.0


def test_timed_clearance_uses_applied_forward_time_and_ignores_distant_background():
    config = replace(
        StaticRouteBypassConfig(),
        target_surface_clearance_cm=70.0,
        reshift_surface_clearance_cm=60.0,
        clearance_run_s=1.5,
        blend_back_s=0.2,
    )
    planner = StaticRouteBypassPlanner(config)
    _activate_right(planner)
    for index, point in enumerate(
        ((90.0, -60.0), (65.0, -75.0), (45.0, -80.0), (12.0, -90.0), (6.0, -90.0))
    ):
        _update(planner, _field(_cluster(*point)), 1.2 + index * 0.1)
    assert planner.state == StaticRouteBypassState.SIDE_PASS_CONFIRM
    for index in range(3):
        _update(planner, _field([]), 1.7 + index * 0.1)
    assert planner.state == StaticRouteBypassState.CLEARANCE_RUN
    assert planner.diagnostics()["predicted_center_x_cm"] is None

    # Stopped or rejected frames do not count toward the continuous 1.5 s.
    planner.report_applied_command(Command.zero("safety_stop"), 0.5, True)
    planner.report_applied_command(Command(8.4, 0.0, 0.0, 0.0), 0.5, False)
    assert planner.diagnostics()["clearance_forward_s"] == 0.0

    # This return keeps the legacy ±75 cm forward corridor non-empty, but it
    # is beyond the 180 cm route-acquisition lookahead and must not block the
    # simplified clearance timer.
    distant_background = _field(_cluster(220.0, 0.0))
    now = 2.0
    for _ in range(15):
        command = _update(planner, distant_background, now)
        assert planner.state == StaticRouteBypassState.CLEARANCE_RUN
        planner.report_applied_command(command, 0.1, True)
        now += 0.1

    assert not planner.diagnostics()["front_corridor_clear"]
    assert planner.diagnostics()["clearance_forward_s"] == pytest.approx(1.5)
    stopped = _update(planner, distant_background, now)
    assert planner.state == StaticRouteBypassState.WAIT_VISUAL
    assert planner.transition_reason == "clearance_forward_time_complete"
    assert stopped.vx_cm_s == 0.0
    _update(planner, distant_background, now + 0.1)
    assert planner.state == StaticRouteBypassState.BLEND_BACK


def test_timed_clearance_restarts_for_a_confirmed_new_route_obstacle():
    config = replace(
        StaticRouteBypassConfig(),
        target_surface_clearance_cm=70.0,
        reshift_surface_clearance_cm=60.0,
        clearance_run_s=1.5,
        clearance_reacquire_radius_cm=80.0,
    )
    planner = StaticRouteBypassPlanner(config)
    _activate_right(planner)
    for index, point in enumerate(
        ((90.0, -60.0), (65.0, -75.0), (45.0, -80.0), (12.0, -90.0), (6.0, -90.0))
    ):
        _update(planner, _field(_cluster(*point)), 1.2 + index * 0.1)
    for index in range(3):
        _update(planner, _field([]), 1.7 + index * 0.1)
    assert planner.state == StaticRouteBypassState.CLEARANCE_RUN
    first_encounter = planner.encounter_id

    now = 2.0
    for _ in range(12):
        command = _update(planner, _field([]), now)
        planner.report_applied_command(command, 0.1, True)
        now += 0.1
    assert planner.diagnostics()["clearance_forward_s"] == pytest.approx(1.2)

    outside_radius = _field(_cluster(85.0, 0.0))
    for _ in range(2):
        command = _update(planner, outside_radius, now)
        planner.report_applied_command(command, 0.1, True)
        now += 0.1
    assert planner.diagnostics()["clearance_forward_s"] == pytest.approx(1.4)

    new_obstacle = _field(_cluster(70.0, 0.0))
    first = _update(planner, new_obstacle, now)
    assert planner.state == StaticRouteBypassState.CLEARANCE_RUN
    assert planner.diagnostics()["clearance_forward_s"] == 0.0
    planner.report_applied_command(first, 0.1, True)
    _update(planner, new_obstacle, now + 0.1)

    assert planner.encounter_id == first_encounter + 1
    assert planner.state == StaticRouteBypassState.DIVERGE_RIGHT
    assert planner.diagnostics()["predicted_center_x_cm"] is not None


def test_expected_edge_exit_does_not_switch_to_opposite_background_cluster():
    """Regression for the 2026-08-06 flight's static_model_mismatch stop."""
    config = replace(
        StaticRouteBypassConfig(),
        target_surface_clearance_cm=70.0,
        reshift_surface_clearance_cm=60.0,
    )
    planner = StaticRouteBypassPlanner(config)
    left_obstacle = _field(_cluster(100.0, 40.0))
    _update(planner, left_obstacle, 1.0)
    _update(planner, left_obstacle, 1.1)
    assert planner.state == StaticRouteBypassState.DIVERGE_RIGHT

    for index, point in enumerate(
        ((90.0, 60.0), (70.0, 75.0), (45.0, 85.0), (20.0, 95.0), (12.0, 100.0))
    ):
        _update(planner, _field(_cluster(*point)), 1.2 + index * 0.1)
    assert planner.state == StaticRouteBypassState.SIDE_PASS_CONFIRM
    assert planner.active_bypass_side == -1

    # The tracked tube has just left through +90 degrees.  A dense cluster on
    # the opposite side must not be adopted as the same physical tube.
    opposite_background = _field(_cluster(20.0, -210.0))
    for index in range(3):
        command = _update(planner, opposite_background, 1.8 + index * 0.1)

    diagnostics = planner.diagnostics()
    assert planner.state == StaticRouteBypassState.CLEARANCE_RUN
    assert planner.active_bypass_side == -1
    assert diagnostics["association_status"] == "prediction_gate_miss"
    assert diagnostics["nearest_candidate_to_prediction_cm"] > config.association_radius_cm
    assert diagnostics["static_model_bad_count"] == 0
    assert command.vx_cm_s == pytest.approx(8.4)


def test_diverge_accepts_valid_80_degree_edge_before_nominal_clearance():
    """Regression for the 2026-08-06 left-side hover at about +84 degrees."""
    planner = StaticRouteBypassPlanner()
    left_obstacle = _field(_cluster(100.0, 40.0))
    _update(planner, left_obstacle, 1.0)
    _update(planner, left_obstacle, 1.1)
    assert planner.state == StaticRouteBypassState.DIVERGE_RIGHT

    # Clearance remains below the nominal 85 cm target while the same static
    # tube moves monotonically to the locked-side 80--90 degree edge.
    for index, point in enumerate(((75.0, 65.0), (45.0, 78.0), (9.0, 82.0))):
        command = _update(planner, _field(_cluster(*point)), 1.2 + index * 0.1)

    diagnostics = planner.diagnostics()
    assert diagnostics["obstacle_surface_clearance_cm"] < planner.config.target_surface_clearance_cm
    assert planner.state == StaticRouteBypassState.SIDE_PASS_CONFIRM
    assert planner.active_bypass_side == -1
    assert planner.transition_reason == "edge_reached_with_hysteresis_clearance"
    assert command.vx_cm_s == pytest.approx(8.4)
    assert command.vy_cm_s <= 0.0

    for index in range(3):
        command = _update(planner, _field([]), 1.6 + index * 0.1)

    assert planner.state == StaticRouteBypassState.CLEARANCE_RUN
    assert command.vx_cm_s == pytest.approx(8.4)


def test_timeout_is_latched_stop():
    planner = StaticRouteBypassPlanner(replace(StaticRouteBypassConfig(), max_encounter_s=0.5))
    _activate_right(planner)
    command = _update(planner, _field(_cluster(100.0, -40.0)), 1.7)

    assert planner.state == StaticRouteBypassState.TIMEOUT_STOP
    assert command.vx_cm_s == 0.0
    assert command.vy_cm_s == 0.0


def test_none_timeout_keeps_encounter_active():
    planner = StaticRouteBypassPlanner(replace(StaticRouteBypassConfig(), max_encounter_s=None))
    _activate_right(planner)
    command = _update(planner, _field(_cluster(100.0, -40.0)), 1000.0)

    assert planner.state != StaticRouteBypassState.TIMEOUT_STOP
    assert command.vx_cm_s > 0.0


def test_recorded_replay_never_credits_unexecuted_motion(tmp_path):
    points_dir = tmp_path / "radar_points"
    points_dir.mkdir()
    points_path = points_dir / "radar_000000_0000000000000.npz"
    np.savez_compressed(
        points_path,
        points_body_cm=np.asarray(_cluster(100.0, -40.0)),
        raw_points_body_cm=np.asarray(_cluster(100.0, -40.0)),
    )
    (tmp_path / "radar.jsonl").write_text(
        '{"points_file": "radar_points/radar_000000_0000000000000.npz"}\n',
        encoding="utf-8",
    )

    result = replay(tmp_path)

    assert result["frames"] == 1
    assert result["front_180_point_total"] == 4
    assert result["command_progress_applied"] == 0.0
