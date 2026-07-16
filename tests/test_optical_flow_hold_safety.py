import struct
from types import SimpleNamespace

from FlightController.Base import Byte_Var, FC_State_Struct
from FlightController.tools.test_optical_flow_hold import (
    TAKEOFF_COMMAND,
    _ConsecutiveRangeGuard,
    _median,
    _takeoff_evidence,
    parse_args,
)


class _Value:
    def __init__(self, value, *, raw_value=None):
        self.value = value
        self.raw_value = value if raw_value is None else raw_value


def _fake_fc(*, fused=10, add=10, vz=0, flight_state=1, command=(0, 0, 0)):
    cid, cmd_0, cmd_1 = command
    return SimpleNamespace(
        state=SimpleNamespace(
            alt_fused=_Value(fused),
            alt_add=_Value(add),
            vel_z=_Value(vz),
            unlock=_Value(bool(flight_state), raw_value=flight_state),
            cid=_Value(cid),
            cmd_0=_Value(cmd_0),
            cmd_1=_Value(cmd_1),
        )
    )


def test_bool_byte_var_preserves_raw_flight_state():
    value = Byte_Var("u8", bool)

    value.update_value_with_mul(2)

    assert value.value is True
    assert value.raw_value == 2


def test_dependency_free_median_supports_even_and_odd_samples():
    assert _median([3, 1, 2]) == 2
    assert _median([4, 1, 3, 2]) == 2.5


def test_fc_state_records_update_metadata_and_raw_payload():
    state = FC_State_Struct()
    values = (0, 0, 0, 10, 2487, 0, 0, 0, 0, 0, 1229, 3, 1, 16, 0, 5)
    payload = struct.pack(state._fmt_string, *values)

    state.update_from_bytes(payload)

    assert state.update_count == 1
    assert state.last_update_monotonic > 0
    assert state.last_raw_bytes == payload
    assert state.alt_add.value == 2487
    assert state.unlock.value is True
    assert state.unlock.raw_value == 1


def test_height_guard_requires_consecutive_distinct_observations():
    guard = _ConsecutiveRangeGuard(maximum=160, confirm_frames=3)

    assert guard.observe(2487) is None
    assert guard.observe(10) is None
    assert guard.observe(2487) is None
    assert guard.observe(2487) is None
    assert guard.observe(2487) == "high"


def test_takeoff_command_alone_is_not_physical_takeoff_evidence():
    fc = _fake_fc(command=TAKEOFF_COMMAND)

    evidence = _takeoff_evidence(fc, baseline_fused_cm=10)

    assert evidence == {
        "command": True,
        "airborne_state": False,
        "height_rise": False,
        "vertical_speed": False,
    }


def test_takeoff_evidence_accepts_fused_height_rise():
    fc = _fake_fc(fused=19, command=TAKEOFF_COMMAND)

    evidence = _takeoff_evidence(fc, baseline_fused_cm=10)

    assert evidence["command"] is True
    assert evidence["height_rise"] is True


def test_new_safety_defaults_are_enabled():
    args = parse_args([])

    assert args.post_unlock_delay_s == 2.0
    assert args.takeoff_start_timeout_s == 8.0
    assert args.height_outlier_confirm_frames == 3
