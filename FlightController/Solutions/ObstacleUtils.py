from __future__ import annotations

import numpy as np


def mask_body_reflection(
    points_body_cm: np.ndarray,
    *,
    x_half_cm: float = 25.0,
    y_half_cm: float = 25.0,
) -> np.ndarray:
    points = np.asarray(points_body_cm, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    points = points.reshape(-1, 2)
    mask = ~((np.abs(points[:, 0]) < x_half_cm) & (np.abs(points[:, 1]) < y_half_cm))
    return points[mask]


def select_forward_corridor(
    points_body_cm: np.ndarray,
    *,
    min_x_cm: float = 10.0,
    half_width_cm: float = 50.0,
) -> np.ndarray:
    points = np.asarray(points_body_cm, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    points = points.reshape(-1, 2)
    return points[(points[:, 0] > min_x_cm) & (np.abs(points[:, 1]) < half_width_cm)]


__all__ = ["mask_body_reflection", "select_forward_corridor"]
