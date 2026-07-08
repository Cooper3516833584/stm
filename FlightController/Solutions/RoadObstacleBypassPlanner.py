"""Road-center obstacle bypass helper for the visual road-following mission.

This module does not replace RoadFollower and does not replace SafetyArbiter.
It only applies a small low-speed yaw correction when radar points indicate
that branches/vines intrude into the road-center safety corridor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .Safety import Command, RadarObstacleField


class RoadBypassState(str, Enum):
    NORMAL = "normal"
    BYPASS_LEFT = "bypass_left"
    BYPASS_RIGHT = "bypass_right"
    RETURN_CENTER = "return_center"


@dataclass
class RoadBypassConfig:
    enabled: bool = False

    road_half_width_cm: float = 120.0
    road_edge_margin_cm: float = 25.0

    min_x_cm: float = 40.0
    lookahead_cm: float = 180.0
    intrusion_half_width_cm: float = 80.0
    bypass_clearance_cm: float = 75.0

    lateral_step_cm: float = 10.0
    guide_distance_cm: float = 150.0

    bypass_speed_cm_s: float = 12.0
    bypass_yaw_kp: float = 0.75
    max_bypass_yaw_bias_deg_s: float = 15.0
    max_yaw_rate_deg_s: float = 25.0
    bypass_yaw_sign: float = 1.0

    activate_frames: int = 2
    release_s: float = 0.5

    min_confidence: float = 0.4
    return_pixel_deadband_px: float = 35.0

    w_center: float = 1.0
    w_switch: float = 0.25
    side_switch_penalty: float = 40.0


class RoadObstacleBypassPlanner:
    def __init__(self, config: RoadBypassConfig | None = None):
        self.config = config or RoadBypassConfig()
        self.state = RoadBypassState.NORMAL
        self._intrusion_count = 0
        self._last_intrusion_s: float | None = None
        self._last_target_y_cm: float | None = None
        self._active_side: int | None = None

    @property
    def last_target_y_cm(self) -> float | None:
        return self._last_target_y_cm

    @property
    def active_side(self) -> int | None:
        return self._active_side

    def update(
        self,
        *,
        desired: Command,
        perception,
        radar_field: RadarObstacleField,
        now_s: float,
    ) -> Command:
        cfg = self.config
        if not cfg.enabled:
            return desired

        if not self._road_usable(perception):
            self._reset()
            return desired

        points = self._points_array(radar_field)
        intrusion = self._intrusion_points(points)
        has_intrusion = bool(intrusion.size > 0)

        if has_intrusion:
            self._intrusion_count += 1
            self._last_intrusion_s = now_s
        else:
            self._intrusion_count = 0

        if self.state == RoadBypassState.NORMAL:
            if self._intrusion_count < max(1, int(cfg.activate_frames)):
                return desired

            target_y = self._choose_bypass_target(points)
            if target_y is None:
                return self._make_no_gap_command(desired)

            self._set_bypass_state(target_y)
            return self._make_bypass_command(desired, target_y)

        if self.state in {RoadBypassState.BYPASS_LEFT, RoadBypassState.BYPASS_RIGHT}:
            recently_blocked = (
                self._last_intrusion_s is not None
                and now_s - self._last_intrusion_s <= max(0.0, cfg.release_s)
            )

            if not recently_blocked:
                self.state = RoadBypassState.RETURN_CENTER
                self._last_target_y_cm = None
                self._active_side = None
                return self._make_return_command(desired)

            target_y = self._choose_bypass_target(points)
            if target_y is None:
                return self._make_no_gap_command(desired)

            self._set_bypass_state(target_y)
            return self._make_bypass_command(desired, target_y)

        if self.state == RoadBypassState.RETURN_CENTER:
            if has_intrusion:
                target_y = self._choose_bypass_target(points)
                if target_y is None:
                    return self._make_no_gap_command(desired)
                self._set_bypass_state(target_y)
                return self._make_bypass_command(desired, target_y)

            pixel_error = abs(float(getattr(perception, "corrected_pixel_error", 0.0)))
            if pixel_error <= cfg.return_pixel_deadband_px:
                self._reset()
                return desired

            return self._make_return_command(desired)

        self._reset()
        return desired

    def _road_usable(self, perception) -> bool:
        if perception is None:
            return False
        if not bool(getattr(perception, "is_road_found", False)):
            return False
        if float(getattr(perception, "confidence", 0.0)) < self.config.min_confidence:
            return False
        return True

    def _points_array(self, radar_field: RadarObstacleField) -> np.ndarray:
        points = np.asarray(getattr(radar_field, "points_body_cm", np.empty((0, 2))), dtype=float)
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        return points.reshape(-1, 2)

    def _intrusion_points(self, points: np.ndarray) -> np.ndarray:
        cfg = self.config
        if points.size == 0:
            return np.empty((0, 2), dtype=float)

        return points[
            (points[:, 0] >= cfg.min_x_cm)
            & (points[:, 0] <= cfg.lookahead_cm)
            & (np.abs(points[:, 1]) <= cfg.intrusion_half_width_cm)
        ]

    def _choose_bypass_target(self, points: np.ndarray) -> float | None:
        cfg = self.config
        y_min = -cfg.road_half_width_cm + cfg.road_edge_margin_cm
        y_max = cfg.road_half_width_cm - cfg.road_edge_margin_cm
        if y_min > y_max:
            return None

        step = max(1.0, float(cfg.lateral_step_cm))
        candidates = np.arange(y_min, y_max + 1e-6, step)

        if points.size:
            ahead = points[
                (points[:, 0] >= cfg.min_x_cm)
                & (points[:, 0] <= cfg.lookahead_cm)
            ]
        else:
            ahead = np.empty((0, 2), dtype=float)

        safe: list[tuple[float, float]] = []
        for target_y in candidates:
            target = float(target_y)
            if self._candidate_blocked(ahead, target):
                continue

            cost = cfg.w_center * abs(target)

            if self._last_target_y_cm is not None:
                cost += cfg.w_switch * abs(target - self._last_target_y_cm)

            side = _sign(target)
            if self._active_side is not None and side != 0 and side != self._active_side:
                cost += cfg.side_switch_penalty

            safe.append((cost, target))

        if not safe:
            return None
        return min(safe, key=lambda item: item[0])[1]

    def _candidate_blocked(self, ahead: np.ndarray, target_y_cm: float) -> bool:
        if ahead.size == 0:
            return False
        lateral_dist = np.abs(ahead[:, 1] - target_y_cm)
        return bool(np.any(lateral_dist < self.config.bypass_clearance_cm))

    def _set_bypass_state(self, target_y_cm: float) -> None:
        self._last_target_y_cm = float(target_y_cm)
        side = _sign(target_y_cm)
        if side > 0:
            self.state = RoadBypassState.BYPASS_LEFT
            self._active_side = 1
        elif side < 0:
            self.state = RoadBypassState.BYPASS_RIGHT
            self._active_side = -1
        else:
            self.state = RoadBypassState.NORMAL
            self._active_side = None

    def _make_bypass_command(self, desired: Command, target_y_cm: float) -> Command:
        cfg = self.config
        guide = max(1.0, float(cfg.guide_distance_cm))
        bypass_angle_deg = math.degrees(math.atan2(float(target_y_cm), guide))
        yaw_bias = cfg.bypass_yaw_sign * cfg.bypass_yaw_kp * bypass_angle_deg
        yaw_bias = _clamp(
            yaw_bias,
            -cfg.max_bypass_yaw_bias_deg_s,
            cfg.max_bypass_yaw_bias_deg_s,
        )
        yaw_rate = _clamp(
            desired.yaw_rate_deg_s + yaw_bias,
            -cfg.max_yaw_rate_deg_s,
            cfg.max_yaw_rate_deg_s,
        )
        vx = min(desired.vx_cm_s, cfg.bypass_speed_cm_s)
        return Command(
            vx,
            0.0,
            desired.vz_cm_s,
            yaw_rate,
            _append_reason(desired.reason, f"road_bypass:{self.state.value}:y={target_y_cm:.0f}"),
        )

    def _make_return_command(self, desired: Command) -> Command:
        cfg = self.config
        vx = min(desired.vx_cm_s, cfg.bypass_speed_cm_s * 1.2)
        return Command(
            vx,
            0.0,
            desired.vz_cm_s,
            desired.yaw_rate_deg_s,
            _append_reason(desired.reason, "road_bypass_return"),
        )

    def _make_no_gap_command(self, desired: Command) -> Command:
        cfg = self.config
        return Command(
            min(desired.vx_cm_s, cfg.bypass_speed_cm_s),
            0.0,
            desired.vz_cm_s,
            desired.yaw_rate_deg_s,
            _append_reason(desired.reason, "road_bypass_no_gap"),
        )

    def _reset(self) -> None:
        self.state = RoadBypassState.NORMAL
        self._intrusion_count = 0
        self._last_intrusion_s = None
        self._last_target_y_cm = None
        self._active_side = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _sign(value: float) -> int:
    if value > 1.0:
        return 1
    if value < -1.0:
        return -1
    return 0


def _append_reason(reason: str, suffix: str) -> str:
    return f"{reason}+{suffix}" if reason else suffix


__all__ = ["RoadBypassConfig", "RoadBypassState", "RoadObstacleBypassPlanner"]
