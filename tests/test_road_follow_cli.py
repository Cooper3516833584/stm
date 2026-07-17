from types import SimpleNamespace

import pytest

import road_follow_main


def test_road_follow_disables_radar_by_default():
    args = road_follow_main.parse_args([])

    assert args.no_radar is True


def test_auto_takeoff_remains_valid_in_camera_only_mode():
    args = road_follow_main.parse_args(["--enable-flight", "--auto-takeoff"])

    road_follow_main._validate_flight_args(args)
    assert args.no_radar is True
    assert args.min_takeoff_battery_v == 10.5
    assert args.takeoff_low_battery_confirm_frames == 3


def test_road_follow_can_explicitly_enable_radar():
    args = road_follow_main.parse_args(["--enable-radar"])

    assert args.no_radar is False


def test_legacy_no_radar_flag_remains_supported():
    args = road_follow_main.parse_args(["--no-radar"])

    assert args.no_radar is True


def test_radar_bypass_requires_radar_opt_in():
    args = road_follow_main.parse_args(["--road-bypass-enable"])

    with pytest.raises(ValueError, match="requires --enable-radar"):
        road_follow_main._validate_flight_args(args)


def test_auto_takeoff_rejects_sustained_loaded_battery_sag(monkeypatch):
    class _Field:
        def __init__(self, value):
            self.value = value

    class _FakeFC:
        PROGRAM_MODE = 3
        HOLD_POS_MODE = 2

        def __init__(self):
            self.connected = True
            self.state = SimpleNamespace(
                mode=_Field(2),
                unlock=_Field(False),
                bat=_Field(11.4),
                alt_add=_Field(0.0),
            )
            self.takeoff_requested = False

        def set_flight_mode(self, mode):
            self.state.mode.value = mode

        def unlock(self):
            self.state.unlock.value = True

        def take_off(self, _height_cm):
            self.takeoff_requested = True
            self.state.bat.value = 10.4

        def stablize(self):
            pass

    clock = iter(index * 0.1 for index in range(1000))
    monkeypatch.setattr(road_follow_main.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(road_follow_main.time, "sleep", lambda _seconds: None)
    args = road_follow_main.parse_args(["--enable-flight", "--auto-takeoff"])
    fc = _FakeFC()

    with pytest.raises(RuntimeError, match="stayed too low during takeoff"):
        road_follow_main._auto_takeoff(fc, args)

    assert fc.takeoff_requested
