"""Geometry tests for the independent contour-bypass task."""

import numpy as np

from experiments.road_contour_bypass.geometry import (
    GeometryConfig,
    build_inflated_occupancy,
    choose_bypass_side,
    extract_external_contours,
    path_is_collision_free,
    path_minimum_clearance_cm,
    sample_quintic_bezier,
)


def test_inflated_single_point_radius():
    cfg = GeometryConfig()
    occupancy = build_inflated_occupancy(np.array([[100.0, 0.0]]), cfg)
    contours = extract_external_contours(occupancy)

    assert len(contours) == 1
    radii = np.linalg.norm(contours[0] - np.array((100.0, 0.0)), axis=1)
    assert abs(float(np.max(radii)) - cfg.inflation_radius_cm) <= cfg.grid_resolution_cm


def test_multiple_points_form_union():
    occupancy = build_inflated_occupancy(
        np.array([[100.0, -8.0], [105.0, 0.0], [110.0, 8.0]])
    )
    assert len(extract_external_contours(occupancy)) == 1


def test_left_obstacle_selects_right():
    assert choose_bypass_side(np.array([[100.0, 20.0], [105.0, 25.0]])) == -1


def test_right_obstacle_selects_left():
    assert choose_bypass_side(np.array([[100.0, -20.0], [105.0, -25.0]])) == +1


def test_center_obstacle_defaults_right():
    assert choose_bypass_side(np.array([[100.0, -1.0], [105.0, 1.0]])) == -1


def test_bezier_collision_check_covers_all_121_samples():
    obstacle = np.array([[100.0, 0.0]])
    occupancy = build_inflated_occupancy(obstacle)
    colliding = sample_quintic_bezier(
        np.array([[0, 0], [30, 0], [70, 0], [130, 0], [180, 0], [220, 0]], dtype=float),
        121,
    )
    clear = sample_quintic_bezier(
        np.array([[0, 0], [20, -80], [50, -190], [145, -190], [190, -80], [230, 0]], dtype=float),
        121,
    )

    assert not path_is_collision_free(colliding, occupancy)
    assert path_is_collision_free(clear, occupancy)
    assert path_minimum_clearance_cm(clear, obstacle) >= 85.0
