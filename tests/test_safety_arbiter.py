import math
from types import SimpleNamespace

import numpy as np
import pytest

from FlightController.Solutions.Safety import (
    Command,
    FlightHealth,
    FlightStatus,
    RadarFieldConfig,
    RadarObstacleField,
    SafetyArbiter,
    SafetyConfig,
    flight_status_from_fc,
)


def _healthy_flight():
    return FlightStatus(connected=True, mode=2, unlocked=True, roll_deg=0.0, pitch_deg=0.0)


def test_flight_status_uses_alt_add_for_mission_height():
    value = lambda item: SimpleNamespace(value=item)
    fc = SimpleNamespace(
        connected=True,
        state=SimpleNamespace(
            mode=value(2),
            unlock=value(True),
            bat=value(12.0),
            rol=value(0.0),
            pit=value(0.0),
            alt_add=value(63),
        ),
    )

    assert flight_status_from_fc(fc).alt_cm == 63


def _radar_field(points):
    return RadarObstacleField(
        RadarFieldConfig(
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=50.0,
            side_corridor_x_half_cm=25.0,
        )
    ).update(np.asarray(points, dtype=float), now_s=1.0)


def _filter(command, points):
    return SafetyArbiter(SafetyConfig(require_unlocked=True)).filter(
        command,
        flight=_healthy_flight(),
        radar_connected=True,
        radar_age_s=0.0,
        radar_field=_radar_field(points),
        enable_flight=True,
    )


def test_fc_not_connected_stops():
    arbiter = SafetyArbiter()
    decision = arbiter.evaluate(
        Command(10, 0, 0, 0, "test"),
        FlightHealth(fc_connected=False, fc_mode=2, radar_fresh=True),
    )
    assert decision.hard_stop
    assert not decision.allowed
    assert decision.command.as_fc_tuple() == (0, 0, 0, 0)
    assert decision.reason == "fc_not_connected"


def test_not_hold_pos_mode_stops():
    arbiter = SafetyArbiter()
    decision = arbiter.evaluate(
        Command(10, 0, 0, 0, "test"),
        FlightHealth(fc_connected=True, fc_mode=1, radar_fresh=True),
    )
    assert decision.hard_stop
    assert decision.reason == "not_hold_pos_mode"
    assert decision.command.as_fc_tuple() == (0, 0, 0, 0)


def test_stale_radar_stops():
    arbiter = SafetyArbiter()
    decision = arbiter.evaluate(
        Command(10, 0, 0, 0, "test"),
        FlightHealth(fc_connected=True, fc_mode=2, radar_fresh=False),
    )
    assert decision.hard_stop
    assert decision.reason == "radar_not_fresh"
    assert decision.command.as_fc_tuple() == (0, 0, 0, 0)


def test_locked_fc_stops_when_unlock_is_required():
    arbiter = SafetyArbiter(SafetyConfig(require_unlocked=True))
    decision = arbiter.evaluate(
        Command(10, 2, 0, 3, "road_follow"),
        FlightHealth(fc_connected=True, fc_mode=2, unlock=False, radar_fresh=True),
    )

    assert decision.hard_stop
    assert not decision.allowed
    assert decision.command == Command.zero("safety_stop:fc_locked")
    assert decision.reason == "fc_locked"


def test_large_attitude_stops():
    arbiter = SafetyArbiter()
    decision = arbiter.evaluate(
        Command(10, 0, 0, 0, "test"),
        FlightHealth(fc_connected=True, fc_mode=2, radar_fresh=True, roll_deg=30.0),
    )
    assert decision.hard_stop
    assert decision.reason == "roll_too_large"
    assert decision.command.as_fc_tuple() == (0, 0, 0, 0)


def test_normal_state_clamps_velocity():
    arbiter = SafetyArbiter(SafetyConfig(max_vx_cm_s=35, max_vy_cm_s=25, max_vz_cm_s=20, max_yaw_rate_deg_s=30))
    decision = arbiter.evaluate(
        Command(80, -40, 25, 60, "test"),
        FlightHealth(fc_connected=True, fc_mode=2, radar_fresh=True),
    )
    assert decision.allowed
    assert not decision.hard_stop
    assert decision.reason == "ok+clamped"
    assert decision.command.as_fc_tuple() == (35, -25, 20, 30)


def test_usb_battery_zero_is_ok_when_threshold_disabled():
    arbiter = SafetyArbiter(SafetyConfig(min_battery_v=None))
    decision = arbiter.evaluate(
        Command(10, 0, 0, 0, "test"),
        FlightHealth(fc_connected=True, fc_mode=2, radar_fresh=True, battery_v=0.0),
    )
    assert decision.allowed
    assert not decision.hard_stop
    assert decision.reason == "ok"


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_command_is_rejected_before_send(bad_value):
    decision = SafetyArbiter().evaluate(
        Command(bad_value, 1.0, 0.0, 0.0, "bad_sensor_math"),
        FlightHealth(fc_connected=True, fc_mode=2, radar_fresh=True),
    )

    assert decision.hard_stop
    assert decision.reason == "invalid_command"
    assert decision.command.as_fc_tuple() == (0, 0, 0, 0)


@pytest.mark.parametrize("field_name", ["roll_deg", "pitch_deg"])
def test_nonfinite_attitude_is_a_hard_stop(field_name):
    health = FlightHealth(fc_connected=True, fc_mode=2, radar_fresh=True)
    setattr(health, field_name, math.nan)

    decision = SafetyArbiter().evaluate(Command(1.0), health)

    assert decision.hard_stop
    assert decision.reason == "invalid_attitude"


def test_required_battery_measurement_must_be_available_and_finite():
    arbiter = SafetyArbiter(SafetyConfig(min_battery_v=10.5))

    missing = arbiter.evaluate(
        Command(1.0),
        FlightHealth(fc_connected=True, fc_mode=2, radar_fresh=True, battery_v=None),
    )
    invalid = arbiter.evaluate(
        Command(1.0),
        FlightHealth(fc_connected=True, fc_mode=2, radar_fresh=True, battery_v=math.nan),
    )

    assert missing.reason == "battery_unavailable"
    assert invalid.reason == "battery_unavailable"
    assert missing.hard_stop and invalid.hard_stop


def test_far_rear_near_centerline_return_does_not_block_right_motion():
    result = _filter(Command(0.0, -8.0, 0.0, 0.0, "right_bypass"), [[-250.0, -0.1]])

    assert result.right_side_clearance_cm is None
    assert "right_side_blocked" not in result.reasons
    assert result.command.vy_cm_s == -8.0


def test_forward_near_centerline_return_is_not_misclassified_as_side_obstacle():
    result = _filter(Command(0.0, -8.0, 0.0, 0.0, "right_bypass"), [[40.0, -1.0]])

    assert result.right_side_clearance_cm is None
    assert "right_side_blocked" not in result.reasons
    assert result.command.vy_cm_s == -8.0


def test_genuine_right_swept_corridor_obstacle_blocks_only_right_motion():
    right = _filter(Command(0.0, -8.0, 0.0, 0.0, "right_bypass"), [[0.0, -40.0]])
    left = _filter(Command(0.0, 8.0, 0.0, 0.0, "escape_left"), [[0.0, -40.0]])

    assert right.right_side_clearance_cm == 40.0
    assert right.command.vy_cm_s == 0.0
    assert right.reasons == ["right_side_blocked"]
    assert left.command.vy_cm_s == 8.0
    assert left.reasons == []


def test_side_corridor_uses_50_cm_body_length_not_100_cm():
    inside = _filter(Command(0.0, -8.0, 0.0, 0.0), [[25.0, -40.0]])
    outside = _filter(Command(0.0, -8.0, 0.0, 0.0), [[25.1, -40.0]])

    assert inside.command.vy_cm_s == 0.0
    assert inside.reasons == ["right_side_blocked"]
    assert outside.command.vy_cm_s == -8.0
    assert outside.reasons == []


def test_front_and_side_obstacles_are_arbitrated_together():
    result = _filter(
        Command(8.4, -8.0, 0.0, 2.0, "static_route"),
        [[60.0, 0.0], [0.0, -40.0]],
    )

    assert result.state == "OBSTACLE_STOP"
    assert result.reasons == ["front_obstacle_stop", "right_side_blocked"]
    assert result.command.vx_cm_s == 0.0
    assert result.command.vy_cm_s == 0.0
    assert result.command.yaw_rate_deg_s == 2.0


def test_front_stop_preserves_safe_lateral_escape_command():
    result = _filter(Command(8.4, 8.0, 0.0, 0.0, "escape_left"), [[60.0, 0.0]])

    assert result.state == "OBSTACLE_STOP"
    assert result.reasons == ["front_obstacle_stop"]
    assert result.command.vx_cm_s == 0.0
    assert result.command.vy_cm_s == 8.0


def test_nonfinite_radar_points_are_removed_before_safety_geometry():
    field = _radar_field([[math.nan, -40.0], [0.0, math.inf], [0.0, -40.0]])

    assert field.points_body_cm.shape == (1, 2)
    assert field.side_clearance_cm("right") == 40.0
