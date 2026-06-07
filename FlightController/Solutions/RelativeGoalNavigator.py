"""Front-only relative body-frame obstacle avoidance navigator.

This module is used by goal_nav_main.py only. It intentionally implements a
local radar-only avoidance demo, not a global navigator. Motion policy:

1. scan only the front 150 degrees by software filtering the fused radar cloud;
2. choose a safe direction inside the scan FOV, with a margin at each edge;
3. rotate in place first when the safe direction is not aligned with the nose;
4. move only along body +x after alignment;
5. never output lateral velocity.
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

    # Front-only radar planning window.
    scan_fov_deg: float = 150.0
    candidate_edge_margin_deg: float = 10.0
    candidate_step_deg: float = 5.0
    min_obstacle_distance_cm: float = 10.0

    # Safety and lookahead distances.
    obstacle_clearance_cm: float = 80.0
    clearance_release_cm: float = 90.0
    lookahead_cm: float = 220.0
    avoid_begin_distance_cm: float = 150.0

    # Turn-then-forward state machine hysteresis.
    align_start_deg: float = 10.0
    align_stop_deg: float = 3.0
    min_turn_yaw_rate_deg_s: float = 6.0

    # Cost tuning.
    clearance_cost_weight: float = 120.0
    switch_cost_weight: float = 0.15
    switch_cost_margin: float = 5.0
    default_avoid_sign: float = 1.0  # +1 means prefer left when left/right are exactly tied.

    # Forward speed shaping.
    min_forward_speed_cm_s: float = 8.0

    # Kept only for backward-compatible construction through wrappers.
    # This navigator always forces vy=0 regardless of this value.
    allow_sideways_velocity: bool = False


@dataclass(frozen=True)
class _DirectionEval:
    angle_deg: float
    allowed: bool
    tube_clearance_cm: float
    cost: float


class RelativeGoalNavigator:
    def __init__(self, config: RelativeGoalConfig | None = None):
        self.config = config or RelativeGoalConfig()
        self._last_selected_angle_deg: float | None = None
        self._turning: bool = False
        self._blocked: bool = False

    def update(self, obstacles_body_cm=None, now_s: float | None = None) -> Command:
        _ = now_s
        cfg = self.config
        radar_field = _as_radar_field(obstacles_body_cm)

        if cfg.forward_test:
            # Forward test is still front-only and never outputs lateral velocity.
            front = self._evaluate_direction(0.0, 0.0, radar_field)
            if not front.allowed:
                self._blocked = True
            speed = self._speed_from_clearance(front.tube_clearance_cm) if self._can_move_forward(front) else 0.0
            return Command(speed, 0.0, 0.0, 0.0, f"forward_test_clear_{front.tube_clearance_cm:.0f}")

        goal_distance = math.hypot(cfg.goal_x_cm, cfg.goal_y_cm)
        if goal_distance <= cfg.arrive_distance_cm:
            self._turning = False
            return Command.zero("relative_goal_reached")

        half_fov = self._scan_half_fov_deg()
        raw_goal_angle = math.degrees(math.atan2(cfg.goal_y_cm, cfg.goal_x_cm))
        goal_angle = _clip(_wrap_deg(raw_goal_angle), -half_fov, half_fov)

        front_release = self._evaluate_direction(0.0, goal_angle, radar_field)
        if (
            self._blocked
            and abs(goal_angle) <= cfg.align_stop_deg
            and front_release.tube_clearance_cm >= cfg.clearance_release_cm
        ):
            self._blocked = False
            self._turning = False
            speed = self._speed_from_clearance(front_release.tube_clearance_cm)
            return Command(
                speed,
                0.0,
                0.0,
                0.0,
                f"forward_release_{front_release.tube_clearance_cm:.0f}_dir_0",
            )

        selected = self._select_direction(goal_angle, radar_field)
        self._last_selected_angle_deg = selected.angle_deg

        if not selected.allowed:
            self._turning = True
            self._blocked = True
            yaw_rate = self._yaw_command(selected.angle_deg)
            return Command(
                0.0,
                0.0,
                0.0,
                yaw_rate,
                f"blocked_turn_dir_{selected.angle_deg:.0f}_clear_{selected.tube_clearance_cm:.0f}",
            )

        if self._should_turn(selected.angle_deg):
            yaw_rate = self._yaw_command(selected.angle_deg)
            return Command(
                0.0,
                0.0,
                0.0,
                yaw_rate,
                f"turn_to_dir_{selected.angle_deg:.0f}_clear_{selected.tube_clearance_cm:.0f}",
            )

        # Once aligned, command only body +x. Re-check the actual 0-degree tube,
        # because the vehicle does not fly along selected_angle; it flies forward.
        front = self._evaluate_direction(0.0, goal_angle, radar_field)
        if not front.allowed:
            self._turning = True
            self._blocked = True
            yaw_rate = self._yaw_command(selected.angle_deg)
            return Command(
                0.0,
                0.0,
                0.0,
                yaw_rate,
                f"front_blocked_turn_dir_{selected.angle_deg:.0f}_front_{front.tube_clearance_cm:.0f}",
            )

        if not self._can_move_forward(front):
            return Command(
                0.0,
                0.0,
                0.0,
                0.0,
                f"blocked_hold_front_{front.tube_clearance_cm:.0f}_release_{cfg.clearance_release_cm:.0f}",
            )

        self._turning = False
        speed = self._speed_from_clearance(front.tube_clearance_cm)
        reason_prefix = "forward_release" if self._blocked else "forward_clear"
        self._blocked = False
        return Command(
            speed,
            0.0,
            0.0,
            0.0,
            f"{reason_prefix}_{front.tube_clearance_cm:.0f}_dir_{selected.angle_deg:.0f}",
        )

    def _select_direction(self, goal_angle: float, radar_field: RadarObstacleField) -> _DirectionEval:
        cfg = self.config
        evaluations = [
            self._evaluate_direction(angle, goal_angle, radar_field)
            for angle in self._candidate_angles_deg()
        ]
        allowed = [item for item in evaluations if item.allowed]

        if not allowed:
            # No translationally safe direction exists. Choose the widest opening for yaw-only search.
            return max(
                evaluations,
                key=lambda item: (
                    item.tube_clearance_cm,
                    -abs(_wrap_deg(item.angle_deg - goal_angle)),
                    -self._default_side_tiebreak(item.angle_deg),
                ),
            )

        best = min(
            allowed,
            key=lambda item: (
                item.cost,
                abs(_wrap_deg(item.angle_deg - goal_angle)),
                self._default_side_tiebreak(item.angle_deg),
            ),
        )

        if self._last_selected_angle_deg is not None:
            previous_like = min(
                allowed,
                key=lambda item: abs(_wrap_deg(item.angle_deg - self._last_selected_angle_deg)),
            )
            if previous_like.cost <= best.cost + cfg.switch_cost_margin:
                best = previous_like

        return best

    def _evaluate_direction(
        self,
        angle_deg: float,
        goal_angle: float,
        radar_field: RadarObstacleField,
    ) -> _DirectionEval:
        cfg = self.config
        points = self._front_scan_points(radar_field)

        if points.size == 0:
            tube_clearance = cfg.lookahead_cm
        else:
            theta = math.radians(angle_deg)
            unit = np.array([math.cos(theta), math.sin(theta)], dtype=float)
            normal = np.array([-math.sin(theta), math.cos(theta)], dtype=float)

            along = points @ unit
            lateral = np.abs(points @ normal)
            in_tube = (
                (along > cfg.min_obstacle_distance_cm)
                & (along <= cfg.lookahead_cm)
                & (lateral <= cfg.obstacle_clearance_cm)
            )
            tube_clearance = float(np.min(along[in_tube])) if np.any(in_tube) else cfg.lookahead_cm

        allowed = tube_clearance > cfg.obstacle_clearance_cm
        heading_cost = abs(_wrap_deg(angle_deg - goal_angle))
        clearance_cost = self._clearance_cost(tube_clearance)
        switch_cost = 0.0
        if self._last_selected_angle_deg is not None:
            switch_cost = abs(_wrap_deg(angle_deg - self._last_selected_angle_deg)) * cfg.switch_cost_weight

        cost = heading_cost + clearance_cost + switch_cost
        return _DirectionEval(angle_deg, allowed, tube_clearance, cost)

    def _front_scan_points(self, radar_field: RadarObstacleField) -> np.ndarray:
        cfg = self.config
        points = radar_field.points_body_cm
        if points.size == 0:
            return np.empty((0, 2), dtype=float)

        half_fov = self._scan_half_fov_deg()
        scan_distance_limit = math.hypot(cfg.lookahead_cm, cfg.obstacle_clearance_cm)
        distances = np.linalg.norm(points, axis=1)
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        mask = (
            (distances > cfg.min_obstacle_distance_cm)
            & (distances <= scan_distance_limit)
            & (np.abs(_wrap_deg_array(angles)) <= half_fov)
        )
        return points[mask]

    def _clearance_cost(self, tube_clearance_cm: float) -> float:
        cfg = self.config
        if tube_clearance_cm >= cfg.avoid_begin_distance_cm:
            return 0.0
        denom = max(1.0, cfg.avoid_begin_distance_cm - cfg.obstacle_clearance_cm)
        ratio = (cfg.avoid_begin_distance_cm - tube_clearance_cm) / denom
        ratio = max(0.0, min(1.0, ratio))
        return ratio * cfg.clearance_cost_weight

    def _speed_from_clearance(self, front_clearance_cm: float) -> float:
        cfg = self.config
        if front_clearance_cm <= cfg.obstacle_clearance_cm:
            return 0.0
        if front_clearance_cm >= cfg.avoid_begin_distance_cm:
            return cfg.cruise_speed_cm_s
        denom = max(1.0, cfg.avoid_begin_distance_cm - cfg.obstacle_clearance_cm)
        ratio = (front_clearance_cm - cfg.obstacle_clearance_cm) / denom
        ratio = max(0.0, min(1.0, ratio))
        return cfg.min_forward_speed_cm_s + ratio * (cfg.cruise_speed_cm_s - cfg.min_forward_speed_cm_s)

    def _should_turn(self, selected_angle_deg: float) -> bool:
        cfg = self.config
        abs_angle = abs(selected_angle_deg)
        if self._turning:
            return abs_angle > cfg.align_stop_deg
        should_turn = abs_angle > cfg.align_start_deg
        self._turning = should_turn
        return should_turn

    def _can_move_forward(self, front: _DirectionEval) -> bool:
        cfg = self.config
        if front.tube_clearance_cm <= cfg.obstacle_clearance_cm:
            self._blocked = True
            return False
        if self._blocked and front.tube_clearance_cm < cfg.clearance_release_cm:
            return False
        return True

    def _yaw_command(self, angle_deg: float) -> float:
        cfg = self.config
        if abs(angle_deg) <= cfg.align_stop_deg:
            return 0.0
        yaw = _clip(
            angle_deg * cfg.yaw_kp,
            -cfg.yaw_rate_limit_deg_s,
            cfg.yaw_rate_limit_deg_s,
        )
        if 0.0 < abs(yaw) < cfg.min_turn_yaw_rate_deg_s:
            yaw = math.copysign(cfg.min_turn_yaw_rate_deg_s, yaw)
        return yaw

    def _candidate_angles_deg(self) -> list[float]:
        cfg = self.config
        half_fov = self._candidate_half_fov_deg()
        step = max(1.0, abs(cfg.candidate_step_deg))
        values: list[float] = []
        angle = -half_fov
        while angle <= half_fov + 1e-6:
            values.append(round(angle, 6))
            angle += step
        if not any(abs(v) < 1e-6 for v in values):
            values.append(0.0)
            values.sort()
        return values

    def _scan_half_fov_deg(self) -> float:
        return max(1.0, min(90.0, abs(self.config.scan_fov_deg) * 0.5))

    def _candidate_half_fov_deg(self) -> float:
        return max(0.0, self._scan_half_fov_deg() - abs(self.config.candidate_edge_margin_deg))

    def _default_side_tiebreak(self, angle_deg: float) -> float:
        # In min() keys, larger value is worse, so return 0 for preferred side.
        if abs(angle_deg) < 1e-6:
            return 1.0
        preferred_sign = 1.0 if self.config.default_avoid_sign >= 0.0 else -1.0
        return 0.0 if math.copysign(1.0, angle_deg) == preferred_sign else 1.0


def _wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _wrap_deg_array(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _as_radar_field(obstacles_body_cm) -> RadarObstacleField:
    if isinstance(obstacles_body_cm, RadarObstacleField):
        return obstacles_body_cm
    field = RadarObstacleField()
    points = np.empty((0, 2), dtype=float) if obstacles_body_cm is None else obstacles_body_cm
    field.update(points, now_s=0.0)
    return field


__all__ = ["RelativeGoalConfig", "RelativeGoalNavigator"]
