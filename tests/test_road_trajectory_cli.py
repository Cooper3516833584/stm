import road_follow_main
import road_trajectory_main
import numpy as np
import pytest

from FlightController.Solutions.RoadObstacleBypassPlanner import (
    RoadBypassConfig,
    RoadBypassState,
    RoadObstacleBypassPlanner,
)
from FlightController.Solutions.Safety import RadarFieldConfig, RadarObstacleField
from FlightController.Solutions.TrajectoryPointFollower import (
    TrajectoryPointFollower,
    TrajectoryPointFollowerConfig,
)


def test_trajectory_entry_selects_adaptive_fast_point_controller_defaults():
    args = road_follow_main.parse_args(road_trajectory_main.build_argv([]))

    assert args.road_controller == "trajectory-point"
    assert args.road_instance_selection == "highest-confidence"
    assert args.loop_hz == 12.0
    assert args.trajectory_reach_radius_px == 30.0
    assert args.trajectory_min_forward_lookahead_px == 24.0
    assert args.trajectory_max_forward_lookahead_px == 64.0
    assert args.trajectory_lookahead_speed_gain_px_per_cm_s == 1.2
    assert args.trajectory_latency_compensation_s == 0.134
    assert args.trajectory_lateral_deadband_px == 8.0
    assert args.trajectory_max_planar_accel_cm_s2 == 24.0
    assert args.trajectory_max_yaw_accel_deg_s2 == 20.0
    assert args.trajectory_min_curve_speed_cm_s == 10.0
    assert args.max_vx_cm_s == 20.0
    assert args.max_vy_cm_s == 12.0
    assert args.max_yaw_rate_deg_s == 10.0
    assert args.no_radar is True
    assert args.enable_flight is True
    assert args.auto_takeoff is True
    assert args.require_model is True
    assert args.takeoff_height_cm == road_trajectory_main.TAKEOFF_HEIGHT_CM


def test_explicit_trajectory_cli_values_override_program_defaults():
    args = road_follow_main.parse_args(
        road_trajectory_main.build_argv(
            [
                "--max-vx-cm-s",
                "6",
                "--trajectory-reach-radius-px",
                "24",
                "--road-instance-selection",
                "geometry",
            ]
        )
    )

    assert args.max_vx_cm_s == 6.0
    assert args.trajectory_reach_radius_px == 24.0
    assert args.road_instance_selection == "geometry"


def test_plain_trajectory_invocation_is_valid_production_auto_flight():
    args = road_follow_main.parse_args(road_trajectory_main.build_argv([]))
    args = road_follow_main._normalize_args(args)
    road_follow_main._validate_flight_args(args)

    assert args.road_controller == "trajectory-point"
    assert not args.obstacle_test
    assert not args.obstacle_flight_test
    assert args.no_radar
    assert not args.road_bypass_enable
    assert args.enable_flight
    assert args.auto_takeoff
    assert args.require_model
    assert not args.dry_run
    assert not args.no_fc


def test_takeoff_height_constant_and_explicit_cli_value_control_default(monkeypatch):
    monkeypatch.setattr(road_trajectory_main, "TAKEOFF_HEIGHT_CM", 120)
    default_args = road_follow_main.parse_args(road_trajectory_main.build_argv([]))
    override_args = road_follow_main.parse_args(
        road_trajectory_main.build_argv(["--takeoff-height-cm", "140"])
    )

    assert default_args.takeoff_height_cm == 120
    assert override_args.takeoff_height_cm == 140


@pytest.mark.parametrize(
    "safe_mode",
    [
        "--dry-run",
        "--no-fc",
        "--connect-fc",
        "--obstacle-test",
        "--obstacle-flight-test",
    ],
)
def test_explicit_safety_modes_do_not_inherit_auto_flight_defaults(safe_mode):
    args = road_follow_main.parse_args(road_trajectory_main.build_argv([safe_mode]))

    assert not args.enable_flight
    assert not args.auto_takeoff


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            [
                "--trajectory-min-forward-lookahead-px",
                "50",
                "--trajectory-max-forward-lookahead-px",
                "40",
            ],
            "max-forward-lookahead",
        ),
        (["--trajectory-physical-road-width-cm", "0"], "physical-road-width"),
        (
            [
                "--trajectory-curvature-slowdown-start-deg",
                "20",
                "--trajectory-curvature-full-slowdown-deg",
                "20",
            ],
            "curvature-full-slowdown",
        ),
    ],
)
def test_invalid_adaptive_trajectory_parameters_are_rejected(options, message):
    args = road_follow_main.parse_args(road_trajectory_main.build_argv(options))

    with pytest.raises(ValueError, match=message):
        road_follow_main._validate_flight_args(args)


def test_obstacle_test_forces_radar_bypass_and_no_fc_output():
    args = road_follow_main.parse_args(
        road_trajectory_main.build_argv(["--obstacle-test"])
    )
    args = road_follow_main._normalize_args(args)
    road_follow_main._validate_flight_args(args)

    assert args.road_controller == "trajectory-point"
    assert args.obstacle_test
    assert args.no_radar is False
    assert args.road_bypass_enable
    assert args.no_fc
    assert args.dry_run
    assert args.no_record


@pytest.mark.parametrize(
    "unsafe_option",
    ["--enable-flight", "--auto-takeoff", "--connect-fc"],
)
def test_obstacle_test_rejects_fc_and_flight_options(unsafe_option):
    args = road_follow_main.parse_args(
        road_trajectory_main.build_argv(["--obstacle-test", unsafe_option])
    )
    args = road_follow_main._normalize_args(args)

    with pytest.raises(ValueError, match="--obstacle-test forbids"):
        road_follow_main._validate_flight_args(args)


def test_obstacle_test_pipeline_outputs_centered_straight_trajectory():
    pipeline = road_follow_main._StraightRoadTestPipeline(640, 480)
    perception, age_s, stale = pipeline.latest_perception()

    assert perception.is_road_found
    assert perception.centerline_angle == 90.0
    assert perception.pixel_error == 0.0
    assert perception.corrected_pixel_error == 0.0
    assert perception.confidence == 1.0
    assert all(point[0] == 320.0 for point in perception.trajectory_points)
    assert age_s == 0.0
    assert stale is False
    assert pipeline.camera_ok
    assert pipeline.latest_frame() == (None, 0.0)

    follower = TrajectoryPointFollower(
        TrajectoryPointFollowerConfig(
            max_planar_accel_cm_s2=1_000_000.0,
            max_yaw_accel_deg_s2=1_000_000.0,
        )
    )
    desired = follower.update(perception, now_s=1.0)
    assert desired.vx_cm_s > 0.0
    assert desired.vy_cm_s == pytest.approx(0.0)
    assert desired.yaw_rate_deg_s == pytest.approx(0.0)


def test_obstacle_test_straight_command_is_redirected_around_left_tree():
    args = road_follow_main.parse_args(
        road_trajectory_main.build_argv(["--obstacle-test"])
    )
    args = road_follow_main._normalize_args(args)
    perception = road_follow_main._StraightRoadTestPipeline(
        args.camera_width,
        args.camera_height,
    ).latest_perception()[0]
    follower = TrajectoryPointFollower(
        TrajectoryPointFollowerConfig(
            image_width=args.camera_width,
            image_height=args.camera_height,
            max_vx_cm_s=args.max_vx_cm_s,
            max_vy_cm_s=args.max_vy_cm_s,
            max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
            max_planar_accel_cm_s2=1_000_000.0,
            max_yaw_accel_deg_s2=1_000_000.0,
        )
    )
    desired = follower.update(perception, now_s=1.0)
    field = RadarObstacleField(
        RadarFieldConfig(forward_corridor_half_width_cm=args.corridor_half_width_cm)
    )
    field.update(
        np.asarray([[100.0, 15.0], [90.0, -70.0]], dtype=float),
        now_s=1.0,
    )
    planner = RoadObstacleBypassPlanner(
        RoadBypassConfig(
            enabled=True,
            road_half_width_cm=args.road_half_width_cm,
            bypass_activity_half_width_cm=args.road_bypass_activity_half_width_cm,
            known_clear_side=args.road_bypass_known_clear_side,
            bypass_clearance_cm=args.road_bypass_clearance_cm,
            bypass_yaw_sign=args.road_bypass_yaw_sign,
            max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
            activate_frames=args.road_bypass_activate_frames,
        )
    )

    planner.update(
        desired=desired,
        perception=perception,
        radar_field=field,
        now_s=1.0,
    )
    planned = planner.update(
        desired=desired,
        perception=perception,
        radar_field=field,
        now_s=1.1,
    )

    assert planner.state == RoadBypassState.BYPASS_RIGHT
    assert planner.last_target_y_cm == -70.0
    assert planned.vx_cm_s > 0.0
    assert planned.vy_cm_s == 0.0
    assert planned.yaw_rate_deg_s > 0.0
    assert "road_bypass:bypass_right" in planned.reason
