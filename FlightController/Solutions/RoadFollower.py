"""Road-following mission controller."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .Safety import Command


@dataclass
class RoadFollowerConfig:
    image_width: int = 640
    max_vx_cm_s: float = 25.0
    max_vy_cm_s: float = 5.0
    search_yaw_rate_deg_s: float = 0.0
    max_yaw_rate_deg_s: float = 25.0
    # A multicopter can correct cross-track error with lateral velocity.  Keep
    # the legacy pixel-to-yaw term configurable, but disabled by default so a
    # large lateral error cannot turn the nose perpendicular to the road.
    pixel_kp_yaw: float = 0.0
    pixel_kp_vy: float = 0.03
    angle_kp_yaw: float = 0.25
    pixel_deadband_px: float = 20.0
    angle_deadband_deg: float = 3.0
    pixel_filter_tau_s: float = 0.35
    angle_filter_tau_s: float = 0.35
    pixel_filter_max_rate_px_s: float = 300.0
    angle_filter_max_rate_deg_s: float = 45.0
    target_centerline_angle_deg: float = 90.0
    heading_slowdown_start_deg: float = 30.0
    heading_stop_deg: float = 70.0
    lost_timeout_s: float = 5.0
    min_confidence: float = 0.35
    yaw_sign: float = 1.0
    lateral_sign: float = -1.0
    ambiguous_speed_scale: float = 0.5


@dataclass(frozen=True)
class RoadFollowerDiagnostics:
    state: str
    raw_pixel_error_px: float | None = None
    filtered_pixel_error_px: float | None = None
    used_pixel_error_px: float | None = None
    raw_centerline_angle_deg: float | None = None
    centerline_angle_deg: float | None = None
    angle_error_deg: float | None = None
    pixel_yaw_term_deg_s: float = 0.0
    angle_yaw_term_deg_s: float = 0.0
    unclamped_yaw_rate_deg_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    unclamped_vy_cm_s: float = 0.0
    vy_cm_s: float = 0.0
    heading_speed_scale: float = 0.0
    lost_elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "state": self.state,
            "raw_pixel_error_px": self.raw_pixel_error_px,
            "filtered_pixel_error_px": self.filtered_pixel_error_px,
            "used_pixel_error_px": self.used_pixel_error_px,
            "raw_centerline_angle_deg": self.raw_centerline_angle_deg,
            "centerline_angle_deg": self.centerline_angle_deg,
            "angle_error_deg": self.angle_error_deg,
            "pixel_yaw_term_deg_s": self.pixel_yaw_term_deg_s,
            "angle_yaw_term_deg_s": self.angle_yaw_term_deg_s,
            "unclamped_yaw_rate_deg_s": self.unclamped_yaw_rate_deg_s,
            "yaw_rate_deg_s": self.yaw_rate_deg_s,
            "unclamped_vy_cm_s": self.unclamped_vy_cm_s,
            "vy_cm_s": self.vy_cm_s,
            "heading_speed_scale": self.heading_speed_scale,
            "lost_elapsed_s": self.lost_elapsed_s,
        }


class RoadFollower:
    """Converts road_perception output into a desired velocity command."""

    def __init__(self, config: RoadFollowerConfig | None = None):
        self.config = config or RoadFollowerConfig()
        self._lost_since_s: float | None = None
        self._last_update_s: float | None = None
        self._filtered_pixel_error_px: float | None = None
        self._filtered_angle_deg: float | None = None
        self.last_diagnostics = RoadFollowerDiagnostics(state="not_started")

    def update(self, perception, now_s: float) -> Command:
        if not self._road_is_usable(perception):
            return self._lost_command(now_s)

        self._lost_since_s = None
        dt_s = self._observation_dt(now_s)
        raw_pixel_error = float(getattr(perception, "corrected_pixel_error", 0.0))
        raw_angle = float(getattr(perception, "centerline_angle", 90.0))
        pixel_error = self._filter_observation(
            raw_pixel_error,
            previous=self._filtered_pixel_error_px,
            tau_s=self.config.pixel_filter_tau_s,
            max_rate_per_s=self.config.pixel_filter_max_rate_px_s,
            dt_s=dt_s,
        )
        angle = self._filter_observation(
            raw_angle,
            previous=self._filtered_angle_deg,
            tau_s=self.config.angle_filter_tau_s,
            max_rate_per_s=self.config.angle_filter_max_rate_deg_s,
            dt_s=dt_s,
        )
        self._filtered_pixel_error_px = pixel_error
        self._filtered_angle_deg = angle
        used_pixel_error = self._deadband(pixel_error, self.config.pixel_deadband_px)
        angle_error = self._deadband(
            self.config.target_centerline_angle_deg - angle,
            self.config.angle_deadband_deg,
        )
        pixel_yaw_term = self.config.pixel_kp_yaw * used_pixel_error
        angle_yaw_term = self.config.angle_kp_yaw * angle_error
        unclamped_yaw_rate = self.config.yaw_sign * (pixel_yaw_term + angle_yaw_term)
        yaw_rate = _clamp(
            unclamped_yaw_rate,
            -self.config.max_yaw_rate_deg_s,
            self.config.max_yaw_rate_deg_s,
        )
        unclamped_vy = self.config.lateral_sign * self.config.pixel_kp_vy * used_pixel_error
        vy = _clamp(unclamped_vy, -self.config.max_vy_cm_s, self.config.max_vy_cm_s)
        heading_speed_scale = self._heading_speed_scale(angle_error)

        road_state = str(getattr(perception, "road_state", "unknown"))
        vx = self.config.max_vx_cm_s * heading_speed_scale
        if road_state in {"ambiguous", "single_rough", "single_extrapolated"}:
            vx *= self.config.ambiguous_speed_scale

        self.last_diagnostics = RoadFollowerDiagnostics(
            state="tracking",
            raw_pixel_error_px=raw_pixel_error,
            filtered_pixel_error_px=pixel_error,
            used_pixel_error_px=used_pixel_error,
            raw_centerline_angle_deg=raw_angle,
            centerline_angle_deg=angle,
            angle_error_deg=angle_error,
            pixel_yaw_term_deg_s=pixel_yaw_term,
            angle_yaw_term_deg_s=angle_yaw_term,
            unclamped_yaw_rate_deg_s=unclamped_yaw_rate,
            yaw_rate_deg_s=yaw_rate,
            unclamped_vy_cm_s=unclamped_vy,
            vy_cm_s=vy,
            heading_speed_scale=heading_speed_scale,
        )
        reason = f"road_follow:{road_state}"
        return Command(vx, vy, 0.0, yaw_rate, reason)

    def _lost_command(self, now_s: float) -> Command:
        if self._lost_since_s is None:
            self._lost_since_s = now_s
        self._last_update_s = now_s
        lost_elapsed_s = now_s - self._lost_since_s
        if lost_elapsed_s >= self.config.lost_timeout_s:
            self._filtered_pixel_error_px = None
            self._filtered_angle_deg = None
            self.last_diagnostics = RoadFollowerDiagnostics(
                state="lost_timeout",
                lost_elapsed_s=lost_elapsed_s,
            )
            return Command.zero("road_lost_timeout")
        yaw_rate = _clamp(
            self.config.search_yaw_rate_deg_s,
            -self.config.max_yaw_rate_deg_s,
            self.config.max_yaw_rate_deg_s,
        )
        reason = "road_lost_hold" if yaw_rate == 0.0 else "road_search"
        self.last_diagnostics = RoadFollowerDiagnostics(
            state=reason,
            yaw_rate_deg_s=yaw_rate,
            lost_elapsed_s=lost_elapsed_s,
        )
        return Command(0.0, 0.0, 0.0, yaw_rate, reason)

    def _road_is_usable(self, perception) -> bool:
        if perception is None:
            return False
        if not bool(getattr(perception, "is_road_found", False)):
            return False
        return float(getattr(perception, "confidence", 0.0)) >= self.config.min_confidence

    def _compute_yaw_rate(self, pixel_error: float, centerline_angle_deg: float) -> float:
        pixel_error = self._deadband(pixel_error, self.config.pixel_deadband_px)
        # Image coordinates have +X to the right and +Y downward.  The road
        # geometry reports angles below 90° when its forward direction points
        # toward image-right.  FC API yaw is clockwise/right-positive, so the
        # heading term must use the opposite sign from (angle - 90).
        angle_error = self._deadband(
            self.config.target_centerline_angle_deg - float(centerline_angle_deg),
            self.config.angle_deadband_deg,
        )
        yaw_rate = (
            self.config.pixel_kp_yaw * pixel_error
            + self.config.angle_kp_yaw * angle_error
        )
        yaw_rate *= self.config.yaw_sign
        return _clamp(
            yaw_rate,
            -self.config.max_yaw_rate_deg_s,
            self.config.max_yaw_rate_deg_s,
        )

    def _heading_speed_scale(self, angle_error_deg: float) -> float:
        error = abs(float(angle_error_deg))
        start = max(0.0, float(self.config.heading_slowdown_start_deg))
        stop = max(start + 1e-6, float(self.config.heading_stop_deg))
        if error <= start:
            return 1.0
        if error >= stop:
            return 0.0
        return 1.0 - (error - start) / (stop - start)

    def _observation_dt(self, now_s: float) -> float:
        previous_s = self._last_update_s
        self._last_update_s = float(now_s)
        if previous_s is None:
            return 0.1
        return _clamp(float(now_s) - previous_s, 0.02, 0.5)

    @staticmethod
    def _filter_observation(
        raw_value: float,
        *,
        previous: float | None,
        tau_s: float,
        max_rate_per_s: float,
        dt_s: float,
    ) -> float:
        raw_value = float(raw_value)
        if previous is None:
            return raw_value

        previous = float(previous)
        tau_s = max(0.0, float(tau_s))
        if tau_s == 0.0:
            target = raw_value
        else:
            alpha = 1.0 - math.exp(-max(0.0, float(dt_s)) / tau_s)
            target = previous + alpha * (raw_value - previous)

        max_rate_per_s = max(0.0, float(max_rate_per_s))
        if max_rate_per_s == 0.0:
            return target
        max_change = max_rate_per_s * max(0.0, float(dt_s))
        return previous + _clamp(target - previous, -max_change, max_change)

    @staticmethod
    def _deadband(value: float, deadband: float) -> float:
        value = float(value)
        return 0.0 if abs(value) < max(0.0, float(deadband)) else value


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


__all__ = ["RoadFollower", "RoadFollowerConfig", "RoadFollowerDiagnostics"]
