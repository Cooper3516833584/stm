import pytest

import road_follow_main
import road_obstacle_flight_test


def _args(extra=None):
    parsed = road_follow_main.parse_args(
        road_obstacle_flight_test.build_argv(extra or [])
    )
    return road_follow_main._normalize_args(parsed)


def test_flight_test_entry_selects_synthetic_trajectory_and_radar_bypass():
    args = _args()

    assert args.road_controller == "trajectory-point"
    assert args.obstacle_flight_test
    assert not args.obstacle_test
    assert not args.no_radar
    assert args.road_bypass_enable
    assert args.max_vx_cm_s == 10.0
    assert args.max_vy_cm_s == 8.0
    assert args.max_yaw_rate_deg_s == 10.0
    assert args.takeoff_height_cm == 100


@pytest.mark.parametrize(
    "provided",
    [
        [],
        ["--enable-flight"],
        ["--enable-flight", "--auto-takeoff"],
    ],
)
def test_flight_test_refuses_to_arm_without_all_confirmations(provided):
    with pytest.raises(ValueError, match="--obstacle-flight-test requires"):
        road_follow_main._validate_flight_args(_args(provided))


def test_flight_test_accepts_complete_explicit_confirmation():
    args = _args(
        [
            "--enable-flight",
            "--auto-takeoff",
            "--confirm-obstacle-flight-test",
        ]
    )

    road_follow_main._validate_flight_args(args)
    assert args.enable_flight
    assert args.auto_takeoff
    assert args.confirm_obstacle_flight_test
    assert not args.no_fc
    assert not args.dry_run
    assert not args.no_record


@pytest.mark.parametrize(
    "unsafe",
    [
        ["--enable-flight", "--auto-takeoff", "--confirm-obstacle-flight-test", "--no-fc"],
        ["--enable-flight", "--auto-takeoff", "--confirm-obstacle-flight-test", "--dry-run"],
        ["--enable-flight", "--auto-takeoff", "--confirm-obstacle-flight-test", "--no-record"],
        ["--enable-flight", "--auto-takeoff", "--confirm-obstacle-flight-test", "--max-vx-cm-s", "11"],
        ["--enable-flight", "--auto-takeoff", "--confirm-obstacle-flight-test", "--takeoff-height-cm", "101"],
        ["--enable-flight", "--auto-takeoff", "--confirm-obstacle-flight-test", "--road-bypass-known-clear-side", "auto"],
    ],
)
def test_flight_test_rejects_unsafe_overrides(unsafe):
    with pytest.raises(ValueError):
        road_follow_main._validate_flight_args(_args(unsafe))


@pytest.mark.parametrize(
    "unsafe_geometry",
    [
        ["--corridor-half-width-cm", "74"],
        ["--road-half-width-cm", "30"],
        ["--road-bypass-activity-half-width-cm", "100"],
        ["--road-bypass-clearance-cm", "70"],
        ["--road-bypass-yaw-sign", "1"],
        ["--road-bypass-activate-frames", "1"],
    ],
)
def test_flight_test_locks_verified_bypass_geometry(unsafe_geometry):
    full_confirmation = [
        "--enable-flight",
        "--auto-takeoff",
        "--confirm-obstacle-flight-test",
    ]
    with pytest.raises(ValueError):
        road_follow_main._validate_flight_args(
            _args([*full_confirmation, *unsafe_geometry])
        )


class _RadarReady:
    connected = True

    def is_fresh(self, max_age_s):
        return max_age_s == 0.5


class _RadarNotReady:
    connected = False

    def is_fresh(self, max_age_s):
        return False


def test_flight_test_radar_preflight_accepts_two_fresh_radars():
    road_follow_main._wait_for_multi_radar_ready(
        _RadarReady(),
        timeout_s=0.0,
        max_age_s=0.5,
    )


def test_flight_test_radar_preflight_refuses_missing_or_stale_radars():
    with pytest.raises(RuntimeError, match="before unlock"):
        road_follow_main._wait_for_multi_radar_ready(
            _RadarNotReady(),
            timeout_s=0.0,
            max_age_s=0.5,
        )


def test_static_and_flight_test_modes_are_mutually_exclusive():
    args = _args(["--obstacle-test"])

    with pytest.raises(ValueError, match="mutually exclusive"):
        road_follow_main._validate_flight_args(args)
