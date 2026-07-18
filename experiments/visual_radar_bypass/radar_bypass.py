"""Isolated radar bypass planner for one tree on the aircraft's left.

Coordinate convention: body +X is forward and body +Y is left.  The test map
places the tree around Y=+40 cm from road centre and guarantees that the right
side is clear.  Only this file should normally change while tuning the radar
bypass experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math

import numpy as np

from FlightController.Solutions.Safety import Command, RadarObstacleField


class LeftTreeBypassState(str, Enum):
    NORMAL = "normal"
    BYPASS_RIGHT = "bypass_right"
    RETURN_CENTER = "return_center"


@dataclass(frozen=True)
class LeftTreeBypassConfig:
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


class LeftTreeBypassPlanner:
    """Add a right-turn bias while a verified left-side tree intrudes."""

    def __init__(self, config: LeftTreeBypassConfig | None = None) -> None:
        self.config = config or LeftTreeBypassConfig()
        self.state = LeftTreeBypassState.NORMAL
        self._intrusion_count = 0
        self._last_intrusion_s: float | None = None
        self._target_y_cm: float | None = None
        self._intrusion_point_count = 0

    @property
    def target_y_cm(self) -> float | None:
        return self._target_y_cm

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

        points = self._points(radar_field)
        intrusion = self._left_intrusion(points)
        self._intrusion_point_count = int(len(intrusion))
        has_intrusion = bool(intrusion.size)
        if has_intrusion:
            self._intrusion_count += 1
            self._last_intrusion_s = float(now_s)
        else:
            self._intrusion_count = 0

        if self.state == LeftTreeBypassState.NORMAL:
            if self._intrusion_count < max(1, int(self.config.activate_frames)):
                return desired
            target_y = self._choose_right_target(points)
            if target_y is None:
                return self._no_gap_command(desired)
            self.state = LeftTreeBypassState.BYPASS_RIGHT
            self._target_y_cm = target_y
            return self._bypass_command(desired, target_y)

        if self.state == LeftTreeBypassState.BYPASS_RIGHT:
            recently_blocked = (
                self._last_intrusion_s is not None
                and now_s - self._last_intrusion_s <= self.config.release_s
            )
            if recently_blocked:
                target_y = self._choose_right_target(points)
                if target_y is None:
                    return self._no_gap_command(desired)
                self._target_y_cm = target_y
                return self._bypass_command(desired, target_y)
            self.state = LeftTreeBypassState.RETURN_CENTER
            self._target_y_cm = None
            return self._return_command(desired)

        if has_intrusion:
            target_y = self._choose_right_target(points)
            if target_y is None:
                return self._no_gap_command(desired)
            self.state = LeftTreeBypassState.BYPASS_RIGHT
            self._target_y_cm = target_y
            return self._bypass_command(desired, target_y)

        pixel_error = abs(
            float(getattr(perception, "corrected_pixel_error", 0.0))
        )
        if pixel_error <= self.config.return_pixel_deadband_px:
            self.reset()
            return desired
        return self._return_command(desired)

    def reset(self) -> None:
        self.state = LeftTreeBypassState.NORMAL
        self._intrusion_count = 0
        self._last_intrusion_s = None
        self._target_y_cm = None
        self._intrusion_point_count = 0

    def diagnostics(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "target_y_cm": self._target_y_cm,
            "intrusion_count": self._intrusion_count,
            "intrusion_point_count": self._intrusion_point_count,
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

    def _left_intrusion(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        cfg = self.config
        return points[
            (points[:, 0] >= cfg.min_x_cm)
            & (points[:, 0] <= cfg.lookahead_cm)
            & (points[:, 1] >= 0.0)
            & (points[:, 1] <= cfg.intrusion_half_width_cm)
        ]

    def _choose_right_target(self, points: np.ndarray) -> float | None:
        cfg = self.config
        # The test map explicitly verifies the right side as clear.  Ignore
        # negative-Y ground/ghost returns here, but leave them in the global
        # SafetyArbiter's full radar field.
        left_ahead = points[
            (points[:, 0] >= cfg.min_x_cm)
            & (points[:, 0] <= cfg.lookahead_cm)
            & (points[:, 1] >= 0.0)
        ] if points.size else np.empty((0, 2), dtype=float)
        candidates = np.arange(
            -cfg.activity_half_width_cm,
            0.0 + 1e-6,
            max(1.0, cfg.lateral_step_cm),
        )
        safe: list[float] = []
        for target in candidates:
            if left_ahead.size and np.any(
                np.abs(left_ahead[:, 1] - float(target)) <= cfg.clearance_cm
            ):
                continue
            safe.append(float(target))
        if not safe:
            return None
        # Prefer the smallest deviation from road centre that strictly keeps
        # the verified 75cm lateral clearance.
        return min(safe, key=abs)

    def _bypass_command(self, desired: Command, target_y_cm: float) -> Command:
        cfg = self.config
        angle_deg = math.degrees(
            math.atan2(float(target_y_cm), max(1.0, cfg.guide_distance_cm))
        )
        # target_y<0 means right.  FC yaw>0 is clockwise/right.
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
        return Command(
            min(desired.vx_cm_s, cfg.bypass_speed_cm_s),
            0.0,
            desired.vz_cm_s,
            yaw,
            _append_reason(
                desired.reason,
                f"left_tree_bypass:right:y={target_y_cm:.0f}",
            ),
        )

    def _return_command(self, desired: Command) -> Command:
        return Command(
            min(desired.vx_cm_s, self.config.bypass_speed_cm_s),
            desired.vy_cm_s,
            desired.vz_cm_s,
            desired.yaw_rate_deg_s,
            _append_reason(desired.reason, "left_tree_return_visual"),
        )

    def _no_gap_command(self, desired: Command) -> Command:
        return Command(
            0.0,
            0.0,
            desired.vz_cm_s,
            max(0.0, desired.yaw_rate_deg_s),
            _append_reason(desired.reason, "left_tree_no_gap_stop"),
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _append_reason(reason: str, suffix: str) -> str:
    return f"{reason}+{suffix}" if reason else suffix
