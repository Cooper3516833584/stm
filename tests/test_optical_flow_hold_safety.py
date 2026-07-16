import csv
import inspect
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from FlightController.Application import FC_Application
from FlightController.Base import Byte_Var, FC_State_Struct
from FlightController.Protocal import FC_Protocol
from FlightController.tools.test_optical_flow_hold import (
    TAKEOFF_COMMAND,
    _ConsecutiveRangeGuard,
    _DiagnosticLogger,
    _hold_with_fc_optical_flow,
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


def test_diagnostic_logger_records_full_state_and_derived_position_velocity(tmp_path):
    log_path = tmp_path / "optical-flow.csv"
    diagnostic = _DiagnosticLogger(log_path)
    state = FC_State_Struct()
    first_values = (100, -200, 300, 184, 9, 0, 0, 0, 0, 0, 1230, 2, 1, 0, 0, 0)
    second_values = (100, -200, 300, 184, 9, 4, 3, 0, 4, 3, 1230, 2, 1, 0, 0, 0)
    base_time = diagnostic._start_monotonic + 0.1

    diagnostic.set_phase("hold")
    diagnostic.set_origin(0, 0)
    state.update_from_bytes(struct.pack(state._fmt_string, *first_values))
    state.last_update_monotonic = base_time
    diagnostic.capture_state(state)
    state.update_from_bytes(struct.pack(state._fmt_string, *second_values))
    state.last_update_monotonic = base_time + 1.0
    diagnostic.capture_state(state)
    diagnostic.close()

    with log_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    state_rows = [row for row in rows if row["record_type"] == "state"]

    assert len(state_rows) == 2
    latest = state_rows[-1]
    assert latest["phase"] == "hold"
    assert latest["raw_state_hex"]
    assert float(latest["alt_fused_cm"]) == 184
    assert float(latest["alt_add_cm"]) == 9
    assert float(latest["alt_fused_minus_add_cm"]) == 175
    assert float(latest["offset_x_cm"]) == 4
    assert float(latest["offset_y_cm"]) == 3
    assert float(latest["drift_cm"]) == 5
    assert float(latest["dpos_window_vel_x_cm_s"]) == 4
    assert float(latest["dpos_window_vel_y_cm_s"]) == 3
    assert float(latest["dpos_window_vel_xy_cm_s"]) == 5
    assert float(latest["dpos_window_minus_fused_vel_x_cm_s"]) == 0
    assert float(latest["dpos_window_minus_fused_vel_y_cm_s"]) == 0
    assert diagnostic.dropped_rows == 0
    assert diagnostic.capture_errors == 0


def test_height_guard_requires_consecutive_distinct_observations():
    guard = _ConsecutiveRangeGuard(maximum=160, confirm_frames=3)

    assert guard.observe(2487) is None
    assert guard.observe(10) is None
    assert guard.observe(2487) is None
    assert guard.observe(2487) is None
    assert guard.observe(2487) == "high"


def test_takeoff_command_alone_is_not_physical_takeoff_evidence():
    fc = _fake_fc(command=TAKEOFF_COMMAND)

    evidence = _takeoff_evidence(fc, baseline_add_cm=10)

    assert evidence == {
        "command": True,
        "airborne_state": False,
        "height_rise": False,
        "vertical_speed": False,
    }


def test_takeoff_evidence_accepts_alt_add_height_rise():
    fc = _fake_fc(add=19, command=TAKEOFF_COMMAND)

    evidence = _takeoff_evidence(fc, baseline_add_cm=10)

    assert evidence["command"] is True
    assert evidence["height_rise"] is True


def test_takeoff_evidence_ignores_disabled_fused_height_rise():
    fc = _fake_fc(fused=500, add=10, command=TAKEOFF_COMMAND)

    evidence = _takeoff_evidence(fc, baseline_add_cm=10)

    assert evidence["height_rise"] is False


def test_application_rejects_disabled_fused_height_source():
    with pytest.raises(ValueError, match="ALT_FU"):
        FC_Application.set_height(SimpleNamespace(), 0, 100, 20)


def test_application_height_control_uses_alt_add_source_one():
    calls = []
    fake = SimpleNamespace(
        state=SimpleNamespace(
            alt_add=_Value(10),
            update_event=SimpleNamespace(clear=lambda: None, wait=lambda: True),
        ),
        _action_log=lambda *args: None,
        go_up=lambda distance, speed: calls.append(("up", distance, speed)),
        go_down=lambda distance, speed: calls.append(("down", distance, speed)),
    )

    FC_Application.set_height(fake, 1, 100, 20)

    assert calls == [("up", 90, 20)]


def test_integrated_position_control_apis_are_disabled():
    with pytest.raises(RuntimeError, match="pos_x/pos_y"):
        FC_Application.reset_position_prediction(SimpleNamespace())

    with pytest.raises(RuntimeError, match="pos_x/pos_y"):
        FC_Protocol.set_target_position(SimpleNamespace(), 100, 200)


def test_hold_monitor_does_not_use_integrated_xy_for_safety_decisions():
    source = inspect.getsource(_hold_with_fc_optical_flow)

    assert "max_drift_cm" not in source
    assert "光流定点漂移超过限制" not in source


def test_runtime_control_code_does_not_read_fc_integrated_position():
    repository_root = Path(__file__).resolve().parents[1]
    flight_controller_root = repository_root / "FlightController"
    allowed_diagnostic_readers = {
        flight_controller_root / "Base.py",
        flight_controller_root / "tools" / "test_optical_flow_hold.py",
    }
    violations = []
    for path in flight_controller_root.rglob("*.py"):
        if path in allowed_diagnostic_readers:
            continue
        text = path.read_text(encoding="utf-8")
        if ".state.pos_x" in text or ".state.pos_y" in text:
            violations.append(str(path.relative_to(repository_root)))

    assert violations == []


def test_runtime_code_does_not_read_disabled_alt_fused():
    repository_root = Path(__file__).resolve().parents[1]
    violations = []
    for path in repository_root.rglob("*.py"):
        if path == repository_root / "FlightController" / "Base.py":
            continue
        if "tests" in path.parts:
            continue
        if ".alt_fused" in path.read_text(encoding="utf-8"):
            violations.append(str(path.relative_to(repository_root)))

    assert violations == []


def test_new_safety_defaults_are_enabled():
    args = parse_args([])

    assert args.post_unlock_delay_s == 2.0
    assert args.takeoff_start_timeout_s == 8.0
    assert args.height_outlier_confirm_frames == 3
    assert args.max_drift_cm is None


def test_diagnostic_log_path_can_be_overridden():
    args = parse_args(["--diagnostic-log", "custom-flight.csv"])

    assert args.diagnostic_log == Path("custom-flight.csv")
