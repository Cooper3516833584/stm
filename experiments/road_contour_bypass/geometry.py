"""Geometry for inflated radar contours and quintic Bezier bypass paths.

Coordinates follow the flight-controller convention used by the repository:
body/encounter +X is forward and +Y is left.  This module is deliberately
stateless; encounter-level freezing is owned by :mod:`planner`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    grid_resolution_cm: float = 2.5
    grid_x_min_cm: float = -30.0
    grid_x_max_cm: float = 280.0
    grid_y_min_cm: float = -200.0
    grid_y_max_cm: float = 200.0
    inflation_radius_cm: float = 85.0
    trajectory_extra_margin_cm: float = 5.0
    envelope_bin_cm: float = 5.0
    bezier_sample_count: int = 121


@dataclass(frozen=True)
class InflatedOccupancy:
    image: np.ndarray
    config: GeometryConfig
    obstacle_points_cm: np.ndarray

    def contains(self, points_cm: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        points = np.asarray(points_cm, dtype=float).reshape(-1, 2)
        if not len(points):
            return np.zeros(0, dtype=bool)
        cfg = self.config
        rows = np.rint((points[:, 0] - cfg.grid_x_min_cm) / cfg.grid_resolution_cm).astype(int)
        cols = np.rint((points[:, 1] - cfg.grid_y_min_cm) / cfg.grid_resolution_cm).astype(int)
        valid = (
            (rows >= 0)
            & (rows < self.image.shape[0])
            & (cols >= 0)
            & (cols < self.image.shape[1])
        )
        occupied = np.zeros(len(points), dtype=bool)
        occupied[valid] = self.image[rows[valid], cols[valid]] != 0
        return occupied


def _normalized_points(points: np.ndarray | Iterable[Sequence[float]]) -> np.ndarray:
    result = np.asarray(points, dtype=float)
    if result.size == 0:
        return np.empty((0, 2), dtype=float)
    result = result.reshape(-1, 2)
    return result[np.all(np.isfinite(result), axis=1)]


def build_inflated_occupancy(
    obstacle_points_cm: np.ndarray | Iterable[Sequence[float]],
    config: GeometryConfig | None = None,
) -> InflatedOccupancy:
    """Rasterize the union of one inflation circle per selected radar point."""
    cfg = config or GeometryConfig()
    if cfg.grid_resolution_cm <= 0.0 or cfg.inflation_radius_cm <= 0.0:
        raise ValueError("grid resolution and inflation radius must be positive")
    rows = int(math.ceil((cfg.grid_x_max_cm - cfg.grid_x_min_cm) / cfg.grid_resolution_cm)) + 1
    cols = int(math.ceil((cfg.grid_y_max_cm - cfg.grid_y_min_cm) / cfg.grid_resolution_cm)) + 1
    image = np.zeros((rows, cols), dtype=np.uint8)
    points = _normalized_points(obstacle_points_cm)
    radius_px = int(math.ceil(cfg.inflation_radius_cm / cfg.grid_resolution_cm))
    for x_cm, y_cm in points:
        row = int(round((x_cm - cfg.grid_x_min_cm) / cfg.grid_resolution_cm))
        col = int(round((y_cm - cfg.grid_y_min_cm) / cfg.grid_resolution_cm))
        cv2.circle(image, (col, row), radius_px, 255, thickness=-1)
    return InflatedOccupancy(image=image, config=cfg, obstacle_points_cm=points.copy())


def extract_external_contours(occupancy: InflatedOccupancy) -> list[np.ndarray]:
    """Return external occupancy contours as ordered ``(x_cm, y_cm)`` arrays."""
    contours, _ = cv2.findContours(
        occupancy.image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    cfg = occupancy.config
    result: list[np.ndarray] = []
    for contour in contours:
        pixels = contour.reshape(-1, 2).astype(float)
        world = np.column_stack(
            (
                cfg.grid_x_min_cm + pixels[:, 1] * cfg.grid_resolution_cm,
                cfg.grid_y_min_cm + pixels[:, 0] * cfg.grid_resolution_cm,
            )
        )
        result.append(world)
    return result


def select_cluster_contour(
    contours: Sequence[np.ndarray],
    cluster_points_cm: np.ndarray | Iterable[Sequence[float]],
) -> np.ndarray:
    """Select the external contour closest to the selected cluster centroid."""
    if not contours:
        return np.empty((0, 2), dtype=float)
    points = _normalized_points(cluster_points_cm)
    centroid = np.mean(points, axis=0) if len(points) else np.zeros(2, dtype=float)
    return min(
        (np.asarray(contour, dtype=float).reshape(-1, 2) for contour in contours),
        key=lambda contour: float(np.min(np.linalg.norm(contour - centroid, axis=1))),
    ).copy()


def choose_bypass_side(
    cluster_points_cm: np.ndarray | Iterable[Sequence[float]],
    *,
    side_deadband_cm: float = 8.0,
    occupancy: InflatedOccupancy | None = None,
) -> int:
    """Choose +1/left or -1/right and default deterministically to right."""
    points = _normalized_points(cluster_points_cm)
    centroid_y = float(np.mean(points[:, 1])) if len(points) else 0.0
    if centroid_y > side_deadband_cm:
        return -1
    if centroid_y < -side_deadband_cm:
        return +1
    if occupancy is not None:
        cfg = occupancy.config
        occupied = occupancy.image != 0
        center_col = int(round((0.0 - cfg.grid_y_min_cm) / cfg.grid_resolution_cm))
        center_col = int(np.clip(center_col, 0, occupied.shape[1] - 1))
        # Free grid area is a stable tie-breaker for central obstacles.  Equal
        # space deliberately returns right to match the competition layout.
        left_free = int(np.count_nonzero(~occupied[:, center_col + 1 :]))
        right_free = int(np.count_nonzero(~occupied[:, :center_col]))
        if left_free > right_free:
            return +1
        if right_free > left_free:
            return -1
    return -1


def extract_side_envelope(
    contour_cm: np.ndarray | Iterable[Sequence[float]],
    bypass_side: int,
    *,
    bin_cm: float = 5.0,
) -> np.ndarray:
    """Extract the outer side envelope without forcing the contour into y=f(x)."""
    if bypass_side not in (-1, +1):
        raise ValueError("bypass_side must be +1 (left) or -1 (right)")
    points = _normalized_points(contour_cm)
    if not len(points):
        return points
    if bin_cm <= 0.0:
        raise ValueError("bin_cm must be positive")
    bins = np.floor(points[:, 0] / bin_cm).astype(int)
    selected: list[np.ndarray] = []
    for bin_id in np.unique(bins):
        candidates = points[bins == bin_id]
        index = int(np.argmax(candidates[:, 1]) if bypass_side > 0 else np.argmin(candidates[:, 1]))
        selected.append(candidates[index])
    envelope = np.asarray(selected, dtype=float)
    return envelope[np.argsort(envelope[:, 0])]


def quintic_bezier(control_points: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    controls = np.asarray(control_points, dtype=float).reshape(6, 2)
    values = np.asarray(t, dtype=float).reshape(-1)
    one_minus = 1.0 - values
    basis = np.column_stack(
        (
            one_minus**5,
            5.0 * one_minus**4 * values,
            10.0 * one_minus**3 * values**2,
            10.0 * one_minus**2 * values**3,
            5.0 * one_minus * values**4,
            values**5,
        )
    )
    result = basis @ controls
    return result[0] if np.asarray(t).ndim == 0 else result


def sample_quintic_bezier(control_points: np.ndarray, sample_count: int = 121) -> np.ndarray:
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    return quintic_bezier(control_points, np.linspace(0.0, 1.0, sample_count))


def build_quintic_control_points(
    envelope_cm: np.ndarray,
    cluster_points_cm: np.ndarray,
    bypass_side: int,
    *,
    extra_margin_cm: float = 5.0,
    outward_retry_cm: float = 0.0,
) -> np.ndarray:
    """Build P0..P5 from front/rear portions of the selected outer envelope."""
    envelope = _normalized_points(envelope_cm)
    cluster = _normalized_points(cluster_points_cm)
    if len(envelope) < 2 or not len(cluster):
        raise ValueError("a non-empty cluster and envelope are required")
    x_low, x_high = float(np.min(envelope[:, 0])), float(np.max(envelope[:, 0]))
    span = max(1.0, x_high - x_low)

    def outer_in_window(low: float, high: float) -> np.ndarray:
        candidates = envelope[
            (envelope[:, 0] >= x_low + low * span)
            & (envelope[:, 0] <= x_low + high * span)
        ]
        if not len(candidates):
            candidates = envelope
        index = int(np.argmax(candidates[:, 1]) if bypass_side > 0 else np.argmin(candidates[:, 1]))
        return candidates[index].copy()

    p2 = outer_in_window(0.30, 0.45)
    p3 = outer_in_window(0.70, 0.85)
    outward = bypass_side * (extra_margin_cm + max(0.0, outward_retry_cm))
    p2[1] += outward
    p3[1] += outward
    # Keep both middle controls on the locked outer side.  This produces a
    # continuous diagonal motion and gives collision retries a monotonic knob.
    far_y = p2[1] if bypass_side * p2[1] >= bypass_side * p3[1] else p3[1]
    # A Bezier control point is generally not a point on the curve.  With the
    # two middle controls sharing the same lateral value, their combined peak
    # basis weight is about 0.7.  Compensate for that attenuation so an
    # envelope point outside the 85 cm contour yields a curve outside it too;
    # collision sampling below remains the final authority.
    far_y *= 1.5
    p2[1] = far_y
    p3[1] = far_y
    obstacle_rear_x = float(np.max(cluster[:, 0]))
    p4 = np.array((obstacle_rear_x + 70.0, 0.35 * far_y), dtype=float)
    p5 = np.array((obstacle_rear_x + 120.0, 0.0), dtype=float)
    return np.vstack((np.array((0.0, 0.0)), np.array((30.0, 0.0)), p2, p3, p4, p5))


def path_minimum_clearance_cm(path_cm: np.ndarray, obstacle_points_cm: np.ndarray) -> float:
    path = _normalized_points(path_cm)
    obstacles = _normalized_points(obstacle_points_cm)
    if not len(path) or not len(obstacles):
        return float("inf")
    distances = np.linalg.norm(path[:, None, :] - obstacles[None, :, :], axis=2)
    return float(np.min(distances))


def path_is_collision_free(
    path_cm: np.ndarray,
    occupancy: InflatedOccupancy,
    *,
    minimum_clearance_cm: float | None = None,
) -> bool:
    path = _normalized_points(path_cm)
    if not len(path) or bool(np.any(occupancy.contains(path))):
        return False
    required = (
        occupancy.config.inflation_radius_cm
        if minimum_clearance_cm is None
        else float(minimum_clearance_cm)
    )
    return path_minimum_clearance_cm(path, occupancy.obstacle_points_cm) + 1e-6 >= required


def render_plan_debug(
    output_path: str | Path,
    *,
    occupancy: InflatedOccupancy,
    contour_cm: np.ndarray,
    control_points_cm: np.ndarray,
    path_cm: np.ndarray,
    bypass_side: int,
    minimum_clearance_cm: float,
) -> Path:
    """Save a top-down plan image containing raw cloud, contour and path."""
    image = cv2.cvtColor(occupancy.image, cv2.COLOR_GRAY2BGR)
    image[occupancy.image != 0] = (55, 55, 120)
    cfg = occupancy.config

    def px(point: Sequence[float]) -> tuple[int, int]:
        x_cm, y_cm = point
        return (
            int(round((y_cm - cfg.grid_y_min_cm) / cfg.grid_resolution_cm)),
            int(round((x_cm - cfg.grid_x_min_cm) / cfg.grid_resolution_cm)),
        )

    contour = _normalized_points(contour_cm)
    if len(contour) > 1:
        cv2.polylines(image, [np.asarray([px(p) for p in contour])], True, (0, 180, 255), 1)
    path = _normalized_points(path_cm)
    if len(path) > 1:
        cv2.polylines(image, [np.asarray([px(p) for p in path])], False, (0, 255, 0), 2)
    for point in occupancy.obstacle_points_cm:
        cv2.circle(image, px(point), 2, (255, 80, 30), -1)
    for index, point in enumerate(_normalized_points(control_points_cm)):
        cv2.circle(image, px(point), 3, (255, 255, 0), -1)
        cv2.putText(image, f"P{index}", px(point), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.circle(image, px((0.0, 0.0)), 4, (255, 0, 255), -1)
    cv2.putText(
        image,
        f"side={'left' if bypass_side > 0 else 'right'} min_clear={minimum_clearance_cm:.1f}cm",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"failed to write debug plan: {destination}")
    return destination


__all__ = [
    "GeometryConfig",
    "InflatedOccupancy",
    "build_inflated_occupancy",
    "extract_external_contours",
    "select_cluster_contour",
    "choose_bypass_side",
    "extract_side_envelope",
    "quintic_bezier",
    "sample_quintic_bezier",
    "build_quintic_control_points",
    "path_minimum_clearance_cm",
    "path_is_collision_free",
    "render_plan_debug",
]
