"""Placeholder radar obstacle clustering and geometric classification.

The D500 radar gives geometry, not rich semantics. This module classifies
radar clusters into conservative geometry labels such as wall, pole, or block.
Camera-based semantic association can be added later without changing the
safety layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autonomy_context import Obstacle


@dataclass
class ObstacleClassifierConfig:
    cluster_distance_cm: float = 35.0
    max_cluster_points: int = 350
    min_points_per_cluster: int = 3


class ObstacleClassifier:
    def __init__(self, config: ObstacleClassifierConfig | None = None):
        self.config = config or ObstacleClassifierConfig()

    def classify_points(self, points_body_cm: np.ndarray, now_s: float) -> list[Obstacle]:
        points = np.asarray(points_body_cm, dtype=float)
        if points.size == 0:
            return []
        points = points.reshape(-1, 2)
        points = self._limit_points(points)

        clusters = self._cluster_points(points)
        obstacles: list[Obstacle] = []
        for cluster in clusters:
            if len(cluster) < self.config.min_points_per_cluster:
                continue
            obstacles.append(self._cluster_to_obstacle(cluster, now_s))
        return obstacles

    def _limit_points(self, points: np.ndarray) -> np.ndarray:
        max_points = max(1, int(self.config.max_cluster_points))
        if len(points) <= max_points:
            return points
        step = max(1, len(points) // max_points)
        return points[::step][:max_points]

    def _cluster_points(self, points: np.ndarray) -> list[np.ndarray]:
        # Simple O(n^2) placeholder clustering. Good enough for dry-run and
        # small downsampled point sets; replace with grid/DBSCAN if needed.
        unused = set(range(len(points)))
        clusters: list[np.ndarray] = []
        threshold = float(self.config.cluster_distance_cm)

        while unused:
            seed = unused.pop()
            cluster_ids = [seed]
            frontier = [seed]
            while frontier:
                idx = frontier.pop()
                if not unused:
                    break
                candidates = np.array(list(unused), dtype=int)
                d = np.linalg.norm(points[candidates] - points[idx], axis=1)
                near_ids = candidates[d <= threshold].tolist()
                for near in near_ids:
                    unused.remove(near)
                    frontier.append(near)
                    cluster_ids.append(near)
            clusters.append(points[cluster_ids])
        return clusters

    def _cluster_to_obstacle(self, cluster: np.ndarray, now_s: float) -> Obstacle:
        min_xy = cluster.min(axis=0)
        max_xy = cluster.max(axis=0)
        center = cluster.mean(axis=0)
        width = float(max_xy[1] - min_xy[1])
        depth = float(max_xy[0] - min_xy[0])
        distance = float(np.linalg.norm(center))
        kind, confidence = self._classify_geometry(width, depth, len(cluster))
        return Obstacle(
            x_cm=float(center[0]),
            y_cm=float(center[1]),
            distance_cm=distance,
            width_cm=width,
            depth_cm=depth,
            kind=kind,
            confidence=confidence,
            last_seen_s=now_s,
        )

    @staticmethod
    def _classify_geometry(width_cm: float, depth_cm: float, points_count: int) -> tuple[str, float]:
        if width_cm > 120.0 and depth_cm < 45.0:
            return "wall", 0.65
        if width_cm > 70.0:
            return "wide_block", 0.55
        if width_cm < 30.0 and depth_cm < 30.0 and points_count >= 4:
            return "pole", 0.5
        if points_count < 4:
            return "sparse_unknown", 0.25
        return "unknown_block", 0.4

