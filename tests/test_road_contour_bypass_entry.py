"""Static command-chain and direct-send integration tests."""

from pathlib import Path

import pytest

from FlightController.Solutions.Safety import Command
from experiments.road_contour_bypass import main


class _FC:
    def __init__(self):
        self.calls = []

    def send_realtime_control_data(self, *values):
        self.calls.append(values)


def _model(tmp_path: Path) -> str:
    model = tmp_path / "model.nb"
    model.write_bytes(b"test")
    return str(model)


def test_new_entry_has_no_environmental_command_gate_or_guarded_sender():
    source = Path(main.__file__).read_text(encoding="utf-8")
    forbidden = ("SafetyArbiter(", "SafetyConfig(", "arbiter.filter(", "send_command_safely(")
    assert not any(token in source for token in forbidden)
    assert source.count("fc.send_realtime_control_data(") == 1


def test_direct_send_clamps_rounds_and_sends_once():
    fc = _FC()
    applied = main.send_direct_command(fc, Command(99.4, -30.2, 0.2, 40.0), dry_run=False)

    assert fc.calls == [(22, -12, 0, 18)]
    assert applied.as_fc_tuple() == (22, -12, 0, 18)


def test_direct_send_dry_run_never_touches_fc():
    fc = _FC()
    main.send_direct_command(fc, Command(10.0, 2.0, 0.0, 1.0), dry_run=True)
    assert fc.calls == []


def test_direct_send_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite"):
        main.send_direct_command(_FC(), Command(float("nan"), 0.0, 0.0, 0.0), False)


def test_target_config_uses_current_planar_controller_fields():
    visual = main.build_experimental_visual_config()

    target = main.build_experimental_target_config(visual)

    assert target.high_planar_speed_cm_s == visual.max_vx_cm_s * 0.60
    assert target.camera_width_px == visual.camera_width
    assert target.offset_filter_tau_s == visual.target_filter_tau_s
    assert target.offset_filter_max_rate_px_s == visual.target_filter_max_rate_px_s
    assert target.max_planar_accel_cm_s2 == visual.max_planar_accel_cm_s2


def test_real_flight_requires_new_dedicated_confirmation(tmp_path):
    model = _model(tmp_path)
    with pytest.raises(ValueError, match="confirm-road-contour"):
        main.validate_args(
            main.parse_args(["--model-npu", model, "--enable-flight", "--auto-takeoff"])
        )
    args = main.parse_args(
        [
            "--model-npu",
            model,
            "--enable-flight",
            "--auto-takeoff",
            "--confirm-road-contour-bypass-flight-test",
        ]
    )
    main.validate_args(args)


def test_hc14_preflight_defaults_and_validation(tmp_path):
    args = main.parse_args(["--model-npu", _model(tmp_path)])
    main.validate_args(args)
    assert args.hc14_port is None
    assert args.hc14_baudrate is None
    assert args.hc14_connect_timeout_s == 5.0

    invalid = main.parse_args(
        ["--model-npu", _model(tmp_path), "--hc14-connect-timeout-s", "0"]
    )
    with pytest.raises(ValueError, match="hc14-connect-timeout"):
        main.validate_args(invalid)
