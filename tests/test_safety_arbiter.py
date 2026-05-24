from FlightController.Solutions.Safety import Command, FlightHealth, SafetyArbiter, SafetyConfig


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

