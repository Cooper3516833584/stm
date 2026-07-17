import road_follow_main
import road_trajectory_main


def test_trajectory_entry_selects_point_controller_and_conservative_limits():
    args = road_follow_main.parse_args(road_trajectory_main.build_argv([]))

    assert args.road_controller == "trajectory-point"
    assert args.trajectory_reach_radius_px == 20.0
    assert args.max_vx_cm_s == 10.0
    assert args.max_vy_cm_s == 8.0
    assert args.max_yaw_rate_deg_s == 10.0
    assert args.no_radar is True


def test_explicit_trajectory_cli_values_override_program_defaults():
    args = road_follow_main.parse_args(
        road_trajectory_main.build_argv(
            ["--max-vx-cm-s", "6", "--trajectory-reach-radius-px", "24"]
        )
    )

    assert args.max_vx_cm_s == 6.0
    assert args.trajectory_reach_radius_px == 24.0
