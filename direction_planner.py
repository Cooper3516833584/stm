"""Simple candidate-direction planner for relative goal navigation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from autonomy_command import VelocityCommand
from local_world_model import LocalWorldModel


@dataclass
class DirectionPlannerConfig:
    cruise_speed_cm_s: float = 20.0
    candidate_min_deg: float = -90.0
    candidate_max_deg: float = 90.0
    candidate_step_deg: float = 10.0
    obstacle_clearance_cm: float = 120.0
    yaw_gain: float = 0.7
    max_yaw_rate_deg_s: float = 25.0
    allow_sideways_velocity: bool = False


class DirectionPlanner:
    """Picks a near-term safe direction toward a body-frame goal.

    This is a placeholder local planner. It does not require pose estimation:
    the goal is interpreted in the current body frame.
    """

    def __init__(self, config: DirectionPlannerConfig | None = None):
        self.config = config or DirectionPlannerConfig()

    def plan_to_body_goal(self, goal_x_cm: float, goal_y_cm: float, world: LocalWorldModel) -> VelocityCommand:
        cfg = self.config
        goal_distance = math.hypot(goal_x_cm, goal_y_cm)
        if goal_distance < 30.0:
            return VelocityCommand.zero("goal_reached_placeholder")

        goal_angle = math.degrees(math.atan2(goal_y_cm, goal_x_cm))
        best_angle = self._select_candidate_angle(goal_angle, world)
        yaw_rate = max(
            -cfg.max_yaw_rate_deg_s,
            min(cfg.max_yaw_rate_deg_s, best_angle * cfg.yaw_gain),
        )

        if cfg.allow_sideways_velocity:
            rad = math.radians(best_angle)
            vx = cfg.cruise_speed_cm_s * math.cos(rad)
            vy = cfg.cruise_speed_cm_s * math.sin(rad)
        else:
            vx = cfg.cruise_speed_cm_s
            vy = 0.0

        return VelocityCommand(vx, vy, 0.0, yaw_rate, f"goal_nav_dir_{best_angle:.0f}")

    def plan_forward_test(self) -> VelocityCommand:
        return VelocityCommand(self.config.cruise_speed_cm_s, 0.0, 0.0, 0.0, "forward_test")

    def _select_candidate_angle(self, goal_angle: float, world: LocalWorldModel) -> float:
        cfg = self.config
        best_angle = 0.0
        best_cost = float("inf")
        angle = cfg.candidate_min_deg
        while angle <= cfg.candidate_max_deg + 1e-6:
            clearance = world.sector_clearance_cm(angle)
            cost = abs(_wrap_deg(angle - goal_angle))
            if clearance is not None and clearance < cfg.obstacle_clearance_cm:
                cost += (cfg.obstacle_clearance_cm - clearance) * 2.0
            if cost < best_cost:
                best_cost = cost
                best_angle = angle
            angle += cfg.candidate_step_deg
        return best_angle


def _wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0

