import math
from argparse import Namespace

import numpy as np

import goal_nav_main
from FlightController.Solutions.RelativeGoalNavigator import RelativeGoalConfig, RelativeGoalNavigator
from FlightController.Solutions.Safety import RadarFieldConfig, RadarObstacleField


def _field(points):
    field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=300.0,
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=80.0,
        )
    )
    field.update(np.asarray(points, dtype=float), now_s=0.0)
    return field


def _nav(**kwargs):
    defaults = {
        "goal_x_cm": 200.0,
        "goal_y_cm": 0.0,
        "cruise_speed_cm_s": 20.0,
        "yaw_rate_limit_deg_s": 25.0,
        "yaw_kp": 0.5,
        "obstacle_clearance_cm": 80.0,
        "clearance_release_cm": 90.0,
        "scan_fov_deg": 150.0,
        "candidate_edge_margin_deg": 10.0,
        "candidate_step_deg": 5.0,
        "lookahead_cm": 220.0,
        "avoid_begin_distance_cm": 150.0,
        "align_start_deg": 10.0,
        "align_stop_deg": 3.0,
        "min_turn_yaw_rate_deg_s": 6.0,
    }
    defaults.update(kwargs)
    cfg = RelativeGoalConfig(**defaults)
    return RelativeGoalNavigator(cfg)


def test_empty_field_moves_forward_only():
    cmd = _nav().update(_field([]))
    assert cmd.vx_cm_s > 0
    assert cmd.vy_cm_s == 0
    assert cmd.vz_cm_s == 0
    assert cmd.yaw_rate_deg_s == 0
    assert "forward_clear" in cmd.reason


def test_goal_direction_requires_yaw_before_forward():
    nav = _nav(goal_x_cm=200.0, goal_y_cm=100.0)
    cmd = nav.update(_field([]))
    assert cmd.vx_cm_s == 0
    assert cmd.vy_cm_s == 0
    assert abs(cmd.yaw_rate_deg_s) > 0
    assert "turn_to_dir" in cmd.reason


def test_front_obstacle_inside_80_never_moves_forward():
    cmd = _nav().update(_field([[60.0, 0.0]]))
    assert cmd.vx_cm_s == 0
    assert cmd.vy_cm_s == 0
    assert cmd.vz_cm_s == 0
    assert "blocked" in cmd.reason or "turn" in cmd.reason


def test_front_obstacle_at_120_triggers_turn_before_forward():
    cmd = _nav().update(_field([[120.0, 0.0]]))
    assert cmd.vx_cm_s == 0
    assert cmd.vy_cm_s == 0
    assert abs(cmd.yaw_rate_deg_s) > 0
    assert "turn_to_dir" in cmd.reason or "blocked_turn" in cmd.reason


def test_obstacle_outside_front_150_deg_is_ignored():
    # 100 degrees is outside +/-75 degrees.
    angle = math.radians(100.0)
    point = [100.0 * math.cos(angle), 100.0 * math.sin(angle)]
    cmd = _nav().update(_field([point]))
    assert cmd.vx_cm_s > 0
    assert cmd.vy_cm_s == 0
    assert cmd.yaw_rate_deg_s == 0


def test_never_outputs_sideways_velocity_across_cases():
    nav = _nav()
    cases = [
        [],
        [[120.0, 0.0]],
        [[100.0, 50.0]],
        [[100.0, -50.0]],
        [[60.0, 0.0], [90.0, 40.0], [90.0, -40.0]],
    ]
    for points in cases:
        cmd = nav.update(_field(points))
        assert cmd.vy_cm_s == 0


def test_no_radar_forces_dry_run_even_with_enable_flight():
    args = Namespace(dry_run=False, no_fc=False, no_radar=True, enable_flight=True)
    assert goal_nav_main._is_actual_dry_run(args)


def test_candidate_directions_keep_margin_inside_scan_edges():
    nav = _nav(scan_fov_deg=150.0, candidate_edge_margin_deg=10.0)
    angles = nav._candidate_angles_deg()
    assert min(angles) >= -65.0
    assert max(angles) <= 65.0
    assert -75.0 not in angles
    assert 75.0 not in angles


def test_front_scan_keeps_tube_edge_points_beyond_lookahead_radius():
    nav = _nav(lookahead_cm=220.0, obstacle_clearance_cm=80.0)
    point = [219.0, 80.0]
    assert math.hypot(*point) > 220.0

    points = nav._front_scan_points(_field([point]))
    assert points.shape == (1, 2)
    assert np.allclose(points[0], point)

    evaluated = nav._evaluate_direction(0.0, 0.0, _field([point]))
    assert evaluated.tube_clearance_cm == 219.0


def test_blocked_clearance_requires_release_before_forward():
    nav = _nav(goal_x_cm=200.0, goal_y_cm=0.0)

    blocked = nav.update(_field([[79.0, 0.0]]))
    assert blocked.vx_cm_s == 0
    assert "blocked" in blocked.reason or "turn" in blocked.reason

    held = nav.update(_field([[85.0, 0.0]]))
    assert held.vx_cm_s == 0
    assert held.vy_cm_s == 0
    assert "blocked_hold" in held.reason

    released = nav.update(_field([[91.0, 0.0]]))
    assert released.vx_cm_s > 0
    assert released.vy_cm_s == 0
    assert released.yaw_rate_deg_s == 0
    assert "forward_release" in released.reason
