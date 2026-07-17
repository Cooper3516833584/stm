import pytest

import road_follow_main


def test_road_follow_disables_radar_by_default():
    args = road_follow_main.parse_args([])

    assert args.no_radar is True


def test_auto_takeoff_remains_valid_in_camera_only_mode():
    args = road_follow_main.parse_args(["--enable-flight", "--auto-takeoff"])

    road_follow_main._validate_flight_args(args)
    assert args.no_radar is True


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
