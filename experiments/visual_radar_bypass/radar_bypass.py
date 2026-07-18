"""Side-agnostic physical-obstacle radar bypass for the isolated experiment.

Coordinate convention: body +X is forward and body +Y is left.  No obstacle
position is configured or injected.  The planner finds the densest real radar
cluster inside the bilateral intrusion envelope, infers its side per encounter,
and guides the aircraft toward the opposite side.  The map guarantee that the
tubular test obstacle has no neighboring obstacles permits local cluster-based
planning; the global SafetyArbiter still receives the complete physical point
cloud.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math

import numpy as np

from FlightController.Solutions.Safety import Command, RadarObstacleField


class ObstacleBypassState(str, Enum):
    NORMAL = "normal"
    BYPASS_LEFT = "bypass_left"
    BYPASS_RIGHT = "bypass_right"
    RETURN_CENTER = "return_center"


@dataclass(frozen=True)
class ObstacleBypassConfig:
    road_half_width_cm: float = 25.0
    intrusion_half_width_cm: float = 75.0
    activity_half_width_cm: float = 90.0
    clearance_cm: float = 75.0
    min_x_cm: float = 40.0
    lookahead_cm: float = 180.0
    lateral_step_cm: float = 10.0
    guide_distance_cm: float = 150.0
    bypass_speed_cm_s: float = 10.0
    yaw_kp: float = 0.75
    max_yaw_bias_deg_s: float = 10.0
    max_yaw_rate_deg_s: float = 10.0
    activate_frames: int = 2
    release_s: float = 0.5
    min_confidence: float = 0.4
    return_pixel_deadband_px: float = 35.0

    # Real tubular obstacle extraction.  A peak grid cell is expanded to a
    # small neighborhood so an obstacle crossing cell boundaries stays whole.
    cluster_grid_cm: float = 10.0
    cluster_radius_x_cm: float = 20.0
    cluster_radius_y_cm: float = 15.0
    min_cluster_points: int = 3
    side_deadband_cm: float = 5.0
    center_obstacle_default_bypass_side: str = "right"
    w_center: float = 1.0
    w_switch: float = 0.25

    @property
    def effective_intrusion_half_width_cm(self) -> float:
        return max(
            abs(float(self.road_half_width_cm)),
            abs(float(self.clearance_cm)),
            abs(float(self.intrusion_half_width_cm)),
        )


@dataclass(frozen=True)
class ObstacleCluster:
    points: np.ndarray
    center_x_cm: float
    center_y_cm: float
    obstacle_side: int  # +1 left, -1 right, 0 centre


class ObstacleBypassPlanner:
    """Infer obstacle side from each physical point-cloud encounter."""

    def __init__(self, config: ObstacleBypassConfig | None = None) -> None:
        self.config = config or ObstacleBypassConfig()
        self.state = ObstacleBypassState.NORMAL
        self._intrusion_count = 0
        self._last_intrusion_s: float | None = None
        self._target_y_cm: float | None = None
        self._active_bypass_side: int | None = None
        self._cluster_point_count = 0
        self._obstacle_center_x_cm: float | None = None
        self._obstacle_center_y_cm: float | None = None
        self._obstacle_side: int | None = None

    @property
    def target_y_cm(self) -> float | None:
        return self._target_y_cm

    @property
    def active_bypass_side(self) -> int | None:
        return self._active_bypass_side

    def update(
        self,
        *,
        desired: Command,
        perception,
        radar_field: RadarObstacleField,
        now_s: float,
    ) -> Command:
        if not self._road_usable(perception):
            self.reset()
            return desired

        cluster = self._detect_obstacle_cluster(self._points(radar_field))
        self._update_cluster_diagnostics(cluster)
        has_obstacle = cluster is not None
        if has_obstacle:
            self._intrusion_count += 1
            self._last_intrusion_s = float(now_s)
        else:
            self._intrusion_count = 0

        if self.state == ObstacleBypassState.NORMAL:
            if self._intrusion_count < max(1, int(self.config.activate_frames)):
                return desired
            assert cluster is not None
            bypass_side = self._opposite_bypass_side(cluster.obstacle_side)
            target_y = self._choose_target(cluster.points, bypass_side)
            if target_y is None:
                return self._no_gap_command(desired)
            self._set_bypass(target_y, bypass_side)
            return self._bypass_command(desired, target_y)

        if self.state in {
            ObstacleBypassState.BYPASS_LEFT,
            ObstacleBypassState.BYPASS_RIGHT,
        }:
            recently_blocked = (
                self._last_intrusion_s is not None
                and now_s - self._last_intrusion_s <= self.config.release_s
            )
            if has_obstacle:
                # Keep the selected bypass side for the current encounter so
                # one noisy frame cannot reverse the turn direction.
                bypass_side = self._active_bypass_side or self._opposite_bypass_side(
                    cluster.obstacle_side
                )
                target_y = self._choose_target(cluster.points, bypass_side)
                if target_y is None:
                    return self._no_gap_command(desired)
                self._set_bypass(target_y, bypass_side)
                return self._bypass_command(desired, target_y)
            if recently_blocked and self._target_y_cm is not None:
                return self._bypass_command(desired, self._target_y_cm)
            self.state = ObstacleBypassState.RETURN_CENTER
            self._target_y_cm = None
            self._active_bypass_side = None
            return self._return_command(desired)

        if has_obstacle:
            bypass_side = self._opposite_bypass_side(cluster.obstacle_side)
            target_y = self._choose_target(cluster.points, bypass_side)
            if target_y is None:
                return self._no_gap_command(desired)
            self._set_bypass(target_y, bypass_side)
            return self._bypass_command(desired, target_y)

        pixel_error = abs(
            float(getattr(perception, "corrected_pixel_error", 0.0))
        )
        if pixel_error <= self.config.return_pixel_deadband_px:
            self.reset()
            return desired
        return self._return_command(desired)

    def reset(self) -> None:
        self.state = ObstacleBypassState.NORMAL
        self._intrusion_count = 0
        self._last_intrusion_s = None
        self._target_y_cm = None
        self._active_bypass_side = None
        self._cluster_point_count = 0
        self._obstacle_center_x_cm = None
        self._obstacle_center_y_cm = None
        self._obstacle_side = None

    def diagnostics(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "target_y_cm": self._target_y_cm,
            "active_bypass_side": _side_name(self._active_bypass_side),
            "intrusion_count": self._intrusion_count,
            "cluster_point_count": self._cluster_point_count,
            "obstacle_center_x_cm": self._obstacle_center_x_cm,
            "obstacle_center_y_cm": self._obstacle_center_y_cm,
            "obstacle_side": _side_name(self._obstacle_side),
            "config": asdict(self.config),
        }

    def _road_usable(self, perception) -> bool:
        return bool(
            perception is not None
            and getattr(perception, "is_road_found", False)
            and float(getattr(perception, "confidence", 0.0))
            >= self.config.min_confidence
        )

    @staticmethod
    def _points(radar_field: RadarObstacleField) -> np.ndarray:
        points = np.asarray(
            getattr(radar_field, "points_body_cm", np.empty((0, 2))),
            dtype=float,
        )
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        return points.reshape(-1, 2)

    def _bilateral_intrusion(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        cfg = self.config
        return points[
            (points[:, 0] >= cfg.min_x_cm)
            & (points[:, 0] <= cfg.lookahead_cm)
            & (np.abs(points[:, 1]) <= cfg.effective_intrusion_half_width_cm)
        ]

    def _detect_obstacle_cluster(
        self,
        points: np.ndarray,
    ) -> ObstacleCluster | None:
        intrusion = self._bilateral_intrusion(points)
        cfg = self.config
        if len(intrusion) < max(1, int(cfg.min_cluster_points)):
            return None

        grid = max(1.0, float(cfg.cluster_grid_cm))
        cells: dict[tuple[int, int], list[int]] = {}
        for index, (x_cm, y_cm) in enumerate(intrusion):
            key = (math.floor(float(x_cm) / grid), math.floor(float(y_cm) / grid))
            cells.setdefault(key, []).append(index)

        def peak_rank(indices: list[int]) -> tuple[float, float, float]:
            cell_points = intrusion[indices]
            return (
                -float(len(indices)),
                float(np.median(cell_points[:, 0])),
                abs(float(np.median(cell_points[:, 1]))),
            )

        peak_indices = min(cells.values(), key=peak_rank)
        peak_points = intrusion[peak_indices]
        seed_x = float(np.median(peak_points[:, 0]))
        seed_y = float(np.median(peak_points[:, 1]))
        cluster_points = intrusion[
            (np.abs(intrusion[:, 0] - seed_x) <= cfg.cluster_radius_x_cm)
            & (np.abs(intrusion[:, 1] - seed_y) <= cfg.cluster_radius_y_cm)
        ]
        if len(cluster_points) < max(1, int(cfg.min_cluster_points)):
            return None

        center_x = float(np.median(cluster_points[:, 0]))
        center_y = float(np.median(cluster_points[:, 1]))
        if center_y > cfg.side_deadband_cm:
            obstacle_side = 1
        elif center_y < -cfg.side_deadband_cm:
            obstacle_side = -1
        else:
            obstacle_side = 0
        return ObstacleCluster(
            points=cluster_points,
            center_x_cm=center_x,
            center_y_cm=center_y,
            obstacle_side=obstacle_side,
        )

    def _opposite_bypass_side(self, obstacle_side: int) -> int:
        if obstacle_side > 0:
            return -1
        if obstacle_side < 0:
            return 1
        return (
            1
            if self.config.center_obstacle_default_bypass_side == "left"
            else -1
        )

    def _choose_target(
        self,
        obstacle_points: np.ndarray,
        bypass_side: int,
    ) -> float | None:
        cfg = self.config
        step = max(1.0, float(cfg.lateral_step_cm))
        if bypass_side > 0:
            candidates = np.arange(0.0, cfg.activity_half_width_cm + 1e-6, step)
        else:
            candidates = np.arange(-cfg.activity_half_width_cm, 0.0 + 1e-6, step)
        safe: list[tuple[float, float]] = []
        for target_y in candidates:
            target = float(target_y)
            if obstacle_points.size and np.any(
                np.abs(obstacle_points[:, 1] - target) <= cfg.clearance_cm
            ):
                continue
            cost = cfg.w_center * abs(target)
            if self._target_y_cm is not None:
                cost += cfg.w_switch * abs(target - self._target_y_cm)
            safe.append((cost, target))
        if not safe:
            return None
        return min(safe, key=lambda item: item[0])[1]

    def _set_bypass(self, target_y_cm: float, bypass_side: int) -> None:
        self._target_y_cm = float(target_y_cm)
        self._active_bypass_side = 1 if bypass_side > 0 else -1
        self.state = (
            ObstacleBypassState.BYPASS_LEFT
            if bypass_side > 0
            else ObstacleBypassState.BYPASS_RIGHT
        )

    def _update_cluster_diagnostics(
        self,
        cluster: ObstacleCluster | None,
    ) -> None:
        if cluster is None:
            self._cluster_point_count = 0
            self._obstacle_center_x_cm = None
            self._obstacle_center_y_cm = None
            self._obstacle_side = None
            return
        self._cluster_point_count = int(len(cluster.points))
        self._obstacle_center_x_cm = cluster.center_x_cm
        self._obstacle_center_y_cm = cluster.center_y_cm
        self._obstacle_side = cluster.obstacle_side

    def _bypass_command(self, desired: Command, target_y_cm: float) -> Command:
        cfg = self.config
        angle_deg = math.degrees(
            math.atan2(float(target_y_cm), max(1.0, cfg.guide_distance_cm))
        )
        # FC yaw>0 is clockwise/right.  Body +Y target is left, hence -angle.
        yaw_bias = _clamp(
            -cfg.yaw_kp * angle_deg,
            -cfg.max_yaw_bias_deg_s,
            cfg.max_yaw_bias_deg_s,
        )
        yaw = _clamp(
            desired.yaw_rate_deg_s + yaw_bias,
            -cfg.max_yaw_rate_deg_s,
            cfg.max_yaw_rate_deg_s,
        )
        side = "left" if target_y_cm > 0.0 else "right"
        return Command(
            min(desired.vx_cm_s, cfg.bypass_speed_cm_s),
            0.0,
            desired.vz_cm_s,
            yaw,
            _append_reason(
                desired.reason,
                f"tube_obstacle_bypass:{side}:y={target_y_cm:.0f}",
            ),
        )

    def _return_command(self, desired: Command) -> Command:
        return Command(
            min(desired.vx_cm_s, self.config.bypass_speed_cm_s),
            desired.vy_cm_s,
            desired.vz_cm_s,
            desired.yaw_rate_deg_s,
            _append_reason(desired.reason, "tube_obstacle_return_visual"),
        )

    def _no_gap_command(self, desired: Command) -> Command:
        return Command(
            0.0,
            0.0,
            desired.vz_cm_s,
            0.0,
            _append_reason(desired.reason, "tube_obstacle_no_gap_stop"),
        )


def _side_name(side: int | None) -> str | None:
    if side is None:
        return None
    if side > 0:
        return "left"
    if side < 0:
        return "right"
    return "center"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _append_reason(reason: str, suffix: str) -> str:
    return f"{reason}+{suffix}" if reason else suffix
