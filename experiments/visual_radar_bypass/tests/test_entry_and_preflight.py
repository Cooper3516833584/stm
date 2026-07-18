from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.visual_radar_bypass import main
from experiments.visual_radar_bypass.flight_runtime import (
    wait_for_radars,
    wait_for_visual_road,
)
from experiments.visual_radar_bypass.visual_guidance import FrozenVisualConfig


def _model(tmp_path: Path) -> str:
    path = tmp_path / "model.nb"
    path.write_bytes(b"test-only-placeholder")
    return str(path)


def test_default_entry_is_real_sensor_dry_run(tmp_path):
    args = main.parse_args(["--model-npu", _model(tmp_path)])
    main.validate_args(args)

    assert not args.enable_flight
    assert not args.auto_takeoff
    assert not args.confirm_visual_radar_flight_test
    assert args.camera_index == 7
    assert args.upper_port == "/dev/ttySTM4"
    assert args.lower_port == "/dev/ttySTM9"
    assert not hasattr(args, "synthetic_radar")


def test_real_flight_requires_both_takeoff_and_explicit_confirmation(tmp_path):
    model = _model(tmp_path)
    with pytest.raises(ValueError, match="--auto-takeoff"):
        main.validate_args(
            main.parse_args(["--model-npu", model, "--enable-flight"])
        )
    with pytest.raises(ValueError, match="confirm-visual-radar"):
        main.validate_args(
            main.parse_args(
                ["--model-npu", model, "--enable-flight", "--auto-takeoff"]
            )
        )


def test_complete_real_flight_command_is_accepted(tmp_path):
    args = main.parse_args(
        [
            "--model-npu",
            _model(tmp_path),
            "--enable-flight",
            "--auto-takeoff",
            "--confirm-visual-radar-flight-test",
            "--takeoff-height-cm",
            "100",
            "--duration-s",
            "60",
        ]
    )

    main.validate_args(args)
    assert args.enable_flight
    assert not args.no_record


@pytest.mark.parametrize(
    "unsafe",
    [
        ["--no-record"],
        ["--takeoff-height-cm", "101"],
        ["--duration-s", "121"],
    ],
)
def test_real_flight_rejects_unsafe_overrides(tmp_path, unsafe):
    base = [
        "--model-npu",
        _model(tmp_path),
        "--enable-flight",
        "--auto-takeoff",
        "--confirm-visual-radar-flight-test",
    ]
    with pytest.raises(ValueError):
        main.validate_args(main.parse_args([*base, *unsafe]))


def test_visual_snapshot_matches_existing_trajectory_defaults():
    config = FrozenVisualConfig()

    assert config.postprocess_mode == "fast-main"
    assert config.max_vx_cm_s == 10.0
    assert config.max_vy_cm_s == 8.0
    assert config.max_yaw_rate_deg_s == 10.0
    assert config.reach_radius_px == 20.0
    assert config.min_forward_lookahead_px == 12.0


class _ReadyRadars:
    connected = True

    def is_fresh(self, max_age_s):
        return max_age_s == 0.5


class _MissingRadars:
    connected = False

    def is_fresh(self, max_age_s):
        return False


def test_physical_radar_preflight_requires_fresh_streams():
    wait_for_radars(_ReadyRadars(), timeout_s=0.0, max_age_s=0.5)
    with pytest.raises(RuntimeError, match="physical radars"):
        wait_for_radars(_MissingRadars(), timeout_s=0.0, max_age_s=0.5)


class _VisualGuidance:
    def __init__(self, usable):
        self.pipeline = SimpleNamespace(camera_ok=usable)
        self._usable = usable

    def latest_perception(self):
        result = SimpleNamespace(
            is_road_found=self._usable,
            confidence=0.9 if self._usable else 0.0,
            trajectory_points=[(320.0, 460.0), (320.0, 200.0)]
            if self._usable
            else [],
        )
        return result, 0.0, not self._usable


def test_visual_preflight_requires_real_usable_road_frames():
    wait_for_visual_road(
        _VisualGuidance(True),
        timeout_s=0.0,
        consecutive_frames=1,
    )
    with pytest.raises(RuntimeError, match="camera/NPU"):
        wait_for_visual_road(
            _VisualGuidance(False),
            timeout_s=0.0,
            consecutive_frames=1,
        )
