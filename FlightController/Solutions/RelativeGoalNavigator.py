"""Relative body-frame goal navigation demo.

This is intentionally not a global navigator. Until a pose provider is added,
goals are interpreted in the current body frame or the program can run in a
plain forward-test mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .Safety import Command, RadarObstacleField


@dataclass
class RelativeGoalConfig:
    goal_x_cm: float = 200.0
    goal_y_cm: float = 0.0
    forward_test: bool = False
    cruise_speed_cm_s: float = 20.0
    yaw_rate_limit_deg_s: float = 25.0
    yaw_kp: float = 0.5
    arrive_distance_cm: float = 30.0
    candidate_min_deg: float = -90.0
    candidate_max_deg: float = 90.0
    candidate_step_deg: float = 10.0
    obstacle_clearance_cm: float = 120.0
    allow_sideways_velocity: bool = False


class RelativeGoalNavigator:
    def __init__(self, config: RelativeGoalConfig | None = None):
        self.config = config or RelativeGoalConfig()

    def update(self, obstacles_body_cm=None, now_s: float | None = None) -> Command:
        # This is a relative-direction demo, not a global navigation solution.
        _ = now_s
        cfg = self.config
        if cfg.forward_test:
            return Command(cfg.cruise_speed_cm_s, 0.0, 0.0, 0.0, "forward_test")

        goal_distance = math.hypot(cfg.goal_x_cm, cfg.goal_y_cm)
        if goal_distance <= cfg.arrive_distance_cm:
            return Command.zero("relative_goal_reached")

        goal_angle = math.degrees(math.atan2(cfg.goal_y_cm, cfg.goal_x_cm))
        radar_field = _as_radar_field(obstacles_body_cm)
        selected_angle = self._select_direction(goal_angle, radar_field)
        yaw_rate = max(
            -cfg.yaw_rate_limit_deg_s,
            min(cfg.yaw_rate_limit_deg_s, selected_angle * cfg.yaw_kp),
        )

        if cfg.allow_sideways_velocity:
            rad = math.radians(selected_angle)
            vx = cfg.cruise_speed_cm_s * math.cos(rad)
            vy = cfg.cruise_speed_cm_s * math.sin(rad)
        else:
            vx = cfg.cruise_speed_cm_s
            vy = 0.0

        return Command(vx, vy, 0.0, yaw_rate, f"relative_goal_dir_{selected_angle:.0f}")

    def _select_direction(self, goal_angle: float, radar_field: RadarObstacleField) -> float:
        cfg = self.config
        best_angle = 0.0
        best_cost = float("inf")
        angle = cfg.candidate_min_deg
        while angle <= cfg.candidate_max_deg + 1e-6:
            cost = abs(_wrap_deg(angle - goal_angle))
            clearance = radar_field.sector_clearance_cm(angle)
            if clearance is not None and clearance < cfg.obstacle_clearance_cm:
                cost += (cfg.obstacle_clearance_cm - clearance) * 2.0
            if cost < best_cost:
                best_angle = angle
                best_cost = cost
            angle += cfg.candidate_step_deg
        return best_angle


def _wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _as_radar_field(obstacles_body_cm) -> RadarObstacleField:
    if isinstance(obstacles_body_cm, RadarObstacleField):
        return obstacles_body_cm
    field = RadarObstacleField()
    points = np.empty((0, 2), dtype=float) if obstacles_body_cm is None else obstacles_body_cm
    field.update(points, now_s=0.0)
    return field


__all__ = ["RelativeGoalConfig", "RelativeGoalNavigator"]
