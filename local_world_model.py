"""Local obstacle world model shared by both autonomy modes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autonomy_context import Obstacle
from obstacle_classifier import ObstacleClassifier


@dataclass
class LocalWorldModelConfig:
    max_distance_cm: float = 300.0
    body_x_half_cm: float = 25.0
    body_y_half_cm: float = 25.0
    obstacle_ttl_s: float = 1.0
    forward_corridor_half_width_cm: float = 50.0
    min_obstacle_distance_cm: float = 10.0


@dataclass
class LocalWorldSnapshot:
    points_body_cm: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    filtered_points_body_cm: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    obstacles: list[Obstacle] = field(default_factory=list)
    updated_s: float = 0.0


class LocalWorldModel:
    def __init__(
        self,
        config: LocalWorldModelConfig | None = None,
        classifier: ObstacleClassifier | None = None,
    ):
        self.config = config or LocalWorldModelConfig()
        self.classifier = classifier or ObstacleClassifier()
        self.snapshot = LocalWorldSnapshot()

    def update_from_radar_points(self, points_body_cm: np.ndarray, now_s: float) -> LocalWorldSnapshot:
        points = np.asarray(points_body_cm, dtype=float)
        if points.size == 0:
            points = np.empty((0, 2), dtype=float)
        else:
            points = points.reshape(-1, 2)
            points = self._within_range(points)

        filtered = self._remove_body_reflections(points)
        obstacles = self.classifier.classify_points(filtered, now_s)
        self.snapshot = LocalWorldSnapshot(
            points_body_cm=points,
            filtered_points_body_cm=filtered,
            obstacles=obstacles,
            updated_s=now_s,
        )
        return self.snapshot

    def radar_age_s(self, now_s: float) -> float | None:
        if self.snapshot.updated_s <= 0.0:
            return None
        return max(0.0, now_s - self.snapshot.updated_s)

    def nearest_forward_obstacle_cm(self) -> float | None:
        points = self.snapshot.filtered_points_body_cm
        if points.size == 0:
            return None
        cfg = self.config
        forward = points[
            (points[:, 0] > cfg.min_obstacle_distance_cm)
            & (np.abs(points[:, 1]) <= cfg.forward_corridor_half_width_cm)
        ]
        if forward.size == 0:
            return None
        return float(np.min(forward[:, 0]))

    def sector_clearance_cm(self, angle_deg: float, sector_half_width_deg: float = 12.0) -> float | None:
        points = self.snapshot.filtered_points_body_cm
        if points.size == 0:
            return None
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        delta = np.abs(_wrap_deg(angles - angle_deg))
        selected = points[delta <= sector_half_width_deg]
        if selected.size == 0:
            return None
        return float(np.min(np.linalg.norm(selected, axis=1)))

    def _within_range(self, points: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(points, axis=1)
        return points[distance <= self.config.max_distance_cm]

    def _remove_body_reflections(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        cfg = self.config
        body_mask = (np.abs(points[:, 0]) < cfg.body_x_half_cm) & (
            np.abs(points[:, 1]) < cfg.body_y_half_cm
        )
        return points[~body_mask]


def _wrap_deg(values: np.ndarray | float) -> np.ndarray | float:
    return (values + 180.0) % 360.0 - 180.0

