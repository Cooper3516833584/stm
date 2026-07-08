"""Front-only relative body-frame obstacle avoidance navigator.

This module is used by goal_nav_main.py only. It intentionally implements a
local radar-only forward-intent avoidance demo, not a global navigator. Motion policy:

1. scan only the front 150 degrees by software filtering the fused radar cloud;
2. choose a safe direction inside the scan FOV, with a margin at each edge;
3. rotate in place first when the safe direction is not aligned with the nose;
4. move only along body +x after alignment;
5. never output lateral velocity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .Safety import Command, RadarObstacleField


@dataclass
class RelativeGoalConfig:
    goal_x_cm: float = 200.0
    goal_y_cm: float = 0.0
    forward_test: bool = False
    continuous_forward: bool = True
    stop_when_no_path: bool = True

    cruise_speed_cm_s: float = 20.0
    yaw_rate_limit_deg_s: float = 25.0
    yaw_kp: float = 0.5
    arrive_distance_cm: float = 30.0

    # Front-only radar planning window.
    scan_fov_deg: float = 150.0
    candidate_edge_margin_deg: float = 10.0
    candidate_step_deg: float = 2.0
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

    # Gap / polar occupancy planning.
    gap_min_width_deg: float = 8.0
    obstacle_angle_padding_deg: float = 2.0
    gap_center_cost_weight: float = 0.35
    gap_width_cost_weight: float = 0.20
    turn_cost_weight: float = 0.05

    # Direction lock / anti-oscillation.
    direction_lock_margin: float = 12.0
    active_gap_release_margin_deg: float = 4.0

    # Dynamic clearance used only by the planner.
    reaction_time_s: float = 0.35
    brake_margin_cm: float = 10.0
    max_dynamic_clearance_cm: float = 120.0

    # Forward speed shaping.
    min_forward_speed_cm_s: float = 8.0
    clearance_speed_gain: float = 0.25

    # Point history and noise filtering used only by the planner.
    point_history_s: float = 0.40
    min_cluster_points: int = 1
    cluster_angle_window_deg: float = 4.0
    cluster_distance_window_cm: float = 35.0

    # Kept only for backward-compatible construction through wrappers.
    # This navigator always forces vy=0 regardless of this value.
    allow_sideways_velocity: bool = False


@dataclass(frozen=True)
class _DirectionEval:
    angle_deg: float
    allowed: bool
    tube_clearance_cm: float
    cost: float
    gap_center_deg: float = 0.0
    gap_width_deg: float = 0.0
    gap_min_clearance_cm: float = 0.0
    gap_index: int = -1


@dataclass(frozen=True)
class _Gap:
    index: int
    start_deg: float
    end_deg: float
    center_deg: float
    width_deg: float
    min_clearance_cm: float


class RelativeGoalNavigator:
    def __init__(self, config: RelativeGoalConfig | None = None):
        self.config = config or RelativeGoalConfig()
        self._last_selected_angle_deg: float | None = None
        self._turning: bool = False
        self._blocked: bool = False
        self._point_history = deque()
        self._active_gap_index: int | None = None
        self._active_gap_sign: int | None = None

    def update(self, obstacles_body_cm=None, now_s: float | None = None) -> Command:
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
        if not cfg.continuous_forward and goal_distance <= cfg.arrive_distance_cm:
            self._turning = False
            return Command.zero("relative_goal_reached")

        half_fov = self._scan_half_fov_deg()
        raw_goal_angle = (
            0.0
            if goal_distance <= 1e-6
            else math.degrees(math.atan2(cfg.goal_y_cm, cfg.goal_x_cm))
        )
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
        if (
            self._blocked
            and abs(goal_angle) <= cfg.align_stop_deg
            and front_release.tube_clearance_cm > cfg.obstacle_clearance_cm
            and front_release.tube_clearance_cm < cfg.clearance_release_cm
        ):
            return Command(
                0.0,
                0.0,
                0.0,
                0.0,
                f"blocked_hold_front_{front_release.tube_clearance_cm:.0f}_release_{cfg.clearance_release_cm:.0f}",
            )

        selected = self._select_direction(goal_angle, radar_field, now_s=now_s)
        self._last_selected_angle_deg = selected.angle_deg

        if not selected.allowed:
            self._blocked = True
            if cfg.stop_when_no_path:
                self._turning = False
                return Command.zero(
                    f"blocked_no_path_dir_{selected.angle_deg:.0f}_clear_{selected.tube_clearance_cm:.0f}_gap_{selected.gap_width_deg:.0f}"
                )
            self._turning = True
            search_angle = self._no_path_search_angle(selected.angle_deg)
            yaw_rate = self._yaw_command(search_angle)
            return Command(
                0.0,
                0.0,
                0.0,
                yaw_rate,
                f"blocked_turn_dir_{search_angle:.0f}_clear_{selected.tube_clearance_cm:.0f}",
            )

        if self._should_turn(selected.angle_deg):
            yaw_rate = self._yaw_command(selected.angle_deg)
            return Command(
                0.0,
                0.0,
                0.0,
                yaw_rate,
                f"turn_to_dir_{selected.angle_deg:.0f}_clear_{selected.tube_clearance_cm:.0f}_gap_{selected.gap_width_deg:.0f}",
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
            f"{reason_prefix}_{front.tube_clearance_cm:.0f}_dir_{selected.angle_deg:.0f}_gap_{selected.gap_width_deg:.0f}",
        )

    def _select_direction(
        self,
        goal_angle: float,
        radar_field: RadarObstacleField,
        now_s: float | None = None,
    ) -> _DirectionEval:
        cfg = self.config
        candidate_angles = self._candidate_angles_deg()
        planning_points = self._planning_points(radar_field, now_s)
        angles, blocked, nearest = self._build_polar_occupancy(planning_points, candidate_angles)
        gaps = self._extract_gaps(angles, blocked, nearest)

        if not gaps:
            self._active_gap_index = None
            self._active_gap_sign = None
            if len(angles) > 0:
                best_idx = int(np.argmax(nearest))
                return _DirectionEval(
                    angle_deg=float(angles[best_idx]),
                    allowed=False,
                    tube_clearance_cm=float(nearest[best_idx]),
                    cost=float("inf"),
                )
            return _DirectionEval(0.0, False, 0.0, float("inf"))

        evals: list[_DirectionEval] = []
        for gap in gaps:
            in_gap = (angles >= gap.start_deg) & (angles <= gap.end_deg) & (~blocked)
            for angle in angles[in_gap]:
                angle_float = float(angle)
                base = self._evaluate_direction(
                    angle_float,
                    goal_angle,
                    radar_field,
                    points=planning_points,
                )
                if not base.allowed:
                    continue

                heading_cost = abs(_wrap_deg(angle_float - goal_angle))
                center_cost = abs(_wrap_deg(angle_float - gap.center_deg)) * cfg.gap_center_cost_weight
                switch_cost = 0.0
                if self._last_selected_angle_deg is not None:
                    switch_cost = (
                        abs(_wrap_deg(angle_float - self._last_selected_angle_deg))
                        * cfg.switch_cost_weight
                    )
                turn_cost = abs(angle_float) * cfg.turn_cost_weight
                width_bonus = gap.width_deg * cfg.gap_width_cost_weight
                cost = (
                    heading_cost
                    + center_cost
                    + self._clearance_cost(base.tube_clearance_cm)
                    + switch_cost
                    + turn_cost
                    - width_bonus
                )
                evals.append(
                    _DirectionEval(
                        angle_deg=angle_float,
                        allowed=True,
                        tube_clearance_cm=base.tube_clearance_cm,
                        cost=cost,
                        gap_center_deg=gap.center_deg,
                        gap_width_deg=gap.width_deg,
                        gap_min_clearance_cm=gap.min_clearance_cm,
                        gap_index=gap.index,
                    )
                )

        if not evals:
            self._active_gap_index = None
            self._active_gap_sign = None
            best_gap = max(gaps, key=lambda item: (item.min_clearance_cm, item.width_deg))
            return _DirectionEval(
                angle_deg=best_gap.center_deg,
                allowed=False,
                tube_clearance_cm=best_gap.min_clearance_cm,
                cost=float("inf"),
                gap_center_deg=best_gap.center_deg,
                gap_width_deg=best_gap.width_deg,
                gap_min_clearance_cm=best_gap.min_clearance_cm,
                gap_index=best_gap.index,
            )

        best = min(
            evals,
            key=lambda item: (
                item.cost,
                abs(_wrap_deg(item.angle_deg - goal_angle)),
                self._default_side_tiebreak(item.angle_deg),
            ),
        )

        if self._active_gap_index is not None and best.gap_index != self._active_gap_index:
            active_gap_evals = [item for item in evals if item.gap_index == self._active_gap_index]
            if active_gap_evals:
                active_best = min(
                    active_gap_evals,
                    key=lambda item: (
                        item.cost,
                        abs(_wrap_deg(item.angle_deg - goal_angle)),
                        self._default_side_tiebreak(item.angle_deg),
                    ),
                )
                if active_best.cost <= best.cost + cfg.active_gap_release_margin_deg:
                    best = active_best

        if self._last_selected_angle_deg is not None:
            previous_like = min(
                evals,
                key=lambda item: abs(_wrap_deg(item.angle_deg - self._last_selected_angle_deg)),
            )
            if previous_like.cost <= best.cost + cfg.direction_lock_margin:
                best = previous_like

        self._active_gap_index = best.gap_index
        if abs(best.angle_deg) > cfg.align_stop_deg:
            self._active_gap_sign = 1 if best.angle_deg > 0.0 else -1
        else:
            self._active_gap_sign = None

        return best

    def _evaluate_direction(
        self,
        angle_deg: float,
        goal_angle: float,
        radar_field: RadarObstacleField,
        points: np.ndarray | None = None,
    ) -> _DirectionEval:
        cfg = self.config
        if points is None:
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
                & (lateral <= self._dynamic_clearance_cm())
            )
            tube_clearance = float(np.min(along[in_tube])) if np.any(in_tube) else cfg.lookahead_cm

        allowed = tube_clearance > cfg.obstacle_clearance_cm
        heading_cost = abs(_wrap_deg(angle_deg - goal_angle))
        clearance_cost = self._clearance_cost(tube_clearance)
        switch_cost = 0.0
        if self._last_selected_angle_deg is not None:
            switch_cost = abs(_wrap_deg(angle_deg - self._last_selected_angle_deg)) * cfg.switch_cost_weight

        cost = heading_cost + clearance_cost + switch_cost
        return _DirectionEval(float(angle_deg), allowed, tube_clearance, cost)

    def _front_scan_points(self, radar_field: RadarObstacleField) -> np.ndarray:
        cfg = self.config
        points = radar_field.points_body_cm
        if points.size == 0:
            return np.empty((0, 2), dtype=float)

        half_fov = self._scan_half_fov_deg()
        scan_distance_limit = math.hypot(cfg.lookahead_cm, self._dynamic_clearance_cm())
        distances = np.linalg.norm(points, axis=1)
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        mask = (
            (distances > cfg.min_obstacle_distance_cm)
            & (distances <= scan_distance_limit)
            & (np.abs(_wrap_deg_array(angles)) <= half_fov)
        )
        return points[mask]

    def _planning_points(self, radar_field: RadarObstacleField, now_s: float | None) -> np.ndarray:
        current = self._front_scan_points(radar_field)
        cfg = self.config

        if now_s is None or cfg.point_history_s <= 0.0:
            return self._filter_isolated_points(current)

        self._point_history.append((float(now_s), current.copy()))
        cutoff = float(now_s) - cfg.point_history_s
        while self._point_history and self._point_history[0][0] < cutoff:
            self._point_history.popleft()

        arrays = [pts for _, pts in self._point_history if pts.size > 0]
        if not arrays:
            return np.empty((0, 2), dtype=float)

        merged = np.vstack(arrays)
        return self._filter_isolated_points(merged)

    def _filter_isolated_points(self, points: np.ndarray) -> np.ndarray:
        cfg = self.config
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        if cfg.min_cluster_points <= 1:
            return points
        if len(points) < cfg.min_cluster_points:
            return np.empty((0, 2), dtype=float)

        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        distances = np.linalg.norm(points, axis=1)
        keep = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            dtheta = np.abs(_wrap_deg_array(angles - angles[i]))
            dr = np.abs(distances - distances[i])
            neighbors = (
                (dtheta <= cfg.cluster_angle_window_deg)
                & (dr <= cfg.cluster_distance_window_cm)
            )
            if int(np.count_nonzero(neighbors)) >= cfg.min_cluster_points:
                keep[i] = True

        return points[keep]

    def _dynamic_clearance_cm(self) -> float:
        cfg = self.config
        dynamic = (
            cfg.obstacle_clearance_cm
            + max(0.0, cfg.cruise_speed_cm_s) * max(0.0, cfg.reaction_time_s)
            + max(0.0, cfg.brake_margin_cm)
        )
        return min(max(cfg.obstacle_clearance_cm, dynamic), cfg.max_dynamic_clearance_cm)

    def _build_polar_occupancy(
        self,
        points: np.ndarray,
        candidate_angles: list[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        angles = np.asarray(candidate_angles, dtype=float)
        blocked = np.zeros(len(angles), dtype=bool)
        nearest = np.full(len(angles), cfg.lookahead_cm, dtype=float)

        if points.size == 0 or len(angles) == 0:
            return angles, blocked, nearest

        safe_radius = self._dynamic_clearance_cm()
        distances = np.linalg.norm(points, axis=1)
        point_angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))

        for r, theta in zip(distances, point_angles):
            if r <= cfg.min_obstacle_distance_cm:
                continue
            if r > math.hypot(cfg.lookahead_cm, safe_radius):
                continue

            expand_deg = math.degrees(math.atan2(safe_radius, max(float(r), 1.0)))
            expand_deg += max(0.0, cfg.obstacle_angle_padding_deg)

            diff = np.abs(_wrap_deg_array(angles - theta))
            affected = diff <= expand_deg
            if np.any(affected):
                blocked[affected] = True
                nearest[affected] = np.minimum(nearest[affected], float(r))

        return angles, blocked, nearest

    def _extract_gaps(
        self,
        angles: np.ndarray,
        blocked: np.ndarray,
        nearest: np.ndarray,
    ) -> list[_Gap]:
        cfg = self.config
        gaps: list[_Gap] = []
        if len(angles) == 0:
            return gaps

        i = 0
        gap_index = 0
        while i < len(angles):
            if blocked[i]:
                i += 1
                continue

            start = i
            while i + 1 < len(angles) and not blocked[i + 1]:
                i += 1
            end = i

            start_deg = float(angles[start])
            end_deg = float(angles[end])
            width_deg = end_deg - start_deg
            center_deg = 0.5 * (start_deg + end_deg)
            min_clearance = (
                float(np.min(nearest[start : end + 1]))
                if end >= start
                else cfg.lookahead_cm
            )

            if width_deg >= cfg.gap_min_width_deg:
                gaps.append(
                    _Gap(
                        index=gap_index,
                        start_deg=start_deg,
                        end_deg=end_deg,
                        center_deg=center_deg,
                        width_deg=width_deg,
                        min_clearance_cm=min_clearance,
                    )
                )
                gap_index += 1
            i += 1

        return gaps

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
        safe_excess = max(0.0, front_clearance_cm - cfg.obstacle_clearance_cm)
        clearance_cap = cfg.clearance_speed_gain * safe_excess
        if front_clearance_cm >= cfg.avoid_begin_distance_cm:
            return min(cfg.cruise_speed_cm_s, max(cfg.min_forward_speed_cm_s, clearance_cap))
        denom = max(1.0, cfg.avoid_begin_distance_cm - cfg.obstacle_clearance_cm)
        ratio = (front_clearance_cm - cfg.obstacle_clearance_cm) / denom
        ratio = max(0.0, min(1.0, ratio))
        shaped = cfg.min_forward_speed_cm_s + ratio * (cfg.cruise_speed_cm_s - cfg.min_forward_speed_cm_s)
        return min(cfg.cruise_speed_cm_s, shaped, max(0.0, clearance_cap))

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

    def _no_path_search_angle(self, selected_angle_deg: float) -> float:
        cfg = self.config
        if abs(selected_angle_deg) > cfg.align_stop_deg:
            return selected_angle_deg

        half_fov = self._candidate_half_fov_deg()
        if half_fov <= cfg.align_stop_deg:
            return selected_angle_deg

        if (
            self._last_selected_angle_deg is not None
            and abs(self._last_selected_angle_deg) > cfg.align_stop_deg
        ):
            sign = math.copysign(1.0, self._last_selected_angle_deg)
        else:
            sign = 1.0 if cfg.default_avoid_sign >= 0.0 else -1.0
        search_mag = min(half_fov, max(cfg.align_start_deg, 30.0))
        return sign * search_mag

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
