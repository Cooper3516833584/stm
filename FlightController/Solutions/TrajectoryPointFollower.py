"""Image-space trajectory point follower for a downward-facing road camera."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

from .Safety import Command


Point = tuple[float, float]


@dataclass
class TrajectoryPointFollowerConfig:
    image_width: int = 640
    image_height: int = 480
    max_vx_cm_s: float = 10.0
    max_vy_cm_s: float = 8.0
    max_yaw_rate_deg_s: float = 10.0
    reach_radius_px: float = 20.0
    min_forward_lookahead_px: float = 12.0
    tangent_window_points: int = 3
    min_confidence: float = 0.35
    tangent_kp_yaw: float = 0.25
    tangent_deadband_deg: float = 3.0
    yaw_sign: float = 1.0
    lateral_sign: float = -1.0
    target_filter_tau_s: float = 0.20
    tangent_filter_tau_s: float = 0.25
    target_filter_max_rate_px_s: float = 600.0
    tangent_filter_max_rate_deg_s: float = 90.0
    max_planar_accel_cm_s2: float = 16.0
    max_yaw_accel_deg_s2: float = 20.0
    degraded_speed_scale: float = 0.75


@dataclass(frozen=True)
class TrajectoryPointFollowerDiagnostics:
    controller_mode: str = "trajectory_point"
    state: str = "not_started"
    path_point_count: int = 0
    nearest_index: int | None = None
    target_index: int | None = None
    camera_center_x_px: float | None = None
    camera_center_y_px: float | None = None
    target_x_px: float | None = None
    target_y_px: float | None = None
    target_distance_px: float | None = None
    target_reached: bool = False
    target_advanced_for_lookahead: bool = False
    tangent_dx_px: float | None = None
    tangent_dy_px: float | None = None
    raw_forward_error_px: float | None = None
    filtered_forward_error_px: float | None = None
    raw_lateral_error_px: float | None = None
    filtered_lateral_error_px: float | None = None
    raw_tangent_error_deg: float | None = None
    tangent_error_deg: float | None = None
    raw_centerline_angle_deg: float | None = None
    centerline_angle_deg: float | None = None
    angle_error_deg: float | None = None
    raw_pixel_error_px: float | None = None
    filtered_pixel_error_px: float | None = None
    used_pixel_error_px: float | None = None
    pixel_yaw_term_deg_s: float = 0.0
    angle_yaw_term_deg_s: float = 0.0
    unclamped_yaw_rate_deg_s: float = 0.0
    clamped_yaw_rate_deg_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    yaw_accel_limited: bool = False
    unclamped_vx_cm_s: float = 0.0
    unclamped_vy_cm_s: float = 0.0
    vy_cm_s: float = 0.0
    vx_cm_s: float = 0.0
    planar_accel_limited: bool = False
    planar_command_delta_cm_s: float = 0.0
    heading_speed_scale: float = 0.0
    tangent_motion_fallback: bool = False
    lost_elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class TrajectoryPointFollower:
    """Chase the closest forward trajectory point while aligning its tangent."""

    def __init__(self, config: TrajectoryPointFollowerConfig | None = None):
        self.config = config or TrajectoryPointFollowerConfig()
        self._last_update_s: float | None = None
        self._lost_since_s: float | None = None
        self._filtered_forward_px: float | None = None
        self._filtered_lateral_px: float | None = None
        self._filtered_tangent_error_deg: float | None = None
        self._limited_vx_cm_s = 0.0
        self._limited_vy_cm_s = 0.0
        self._limited_yaw_rate_deg_s = 0.0
        self.last_diagnostics = TrajectoryPointFollowerDiagnostics()

    def update(self, perception, now_s: float) -> Command:
        points = self._usable_trajectory(perception)
        if len(points) < 2:
            return self._lost_command(now_s)

        self._lost_since_s = None
        dt_s = self._observation_dt(now_s)
        center_x = float(self.config.image_width) / 2.0
        center_y = float(self.config.image_height) / 2.0

        nearest_index = min(
            range(len(points)),
            key=lambda index: _distance_sq(points[index], (center_x, center_y)),
        )
        nearest_distance = math.sqrt(
            _distance_sq(points[nearest_index], (center_x, center_y))
        )
        target_index = nearest_index
        target_advanced_for_lookahead = False
        while target_index + 1 < len(points):
            point_x, point_y = points[target_index]
            target_distance = math.sqrt(
                _distance_sq((point_x, point_y), (center_x, center_y))
            )
            has_forward_lookahead = (
                center_y - point_y >= float(self.config.min_forward_lookahead_px)
            )
            if (
                target_distance >= float(self.config.reach_radius_px)
                and has_forward_lookahead
            ):
                break
            if target_distance >= float(self.config.reach_radius_px):
                target_advanced_for_lookahead = True
            target_index += 1

        target_x, target_y = points[target_index]
        target_distance = math.hypot(target_x - center_x, target_y - center_y)
        tangent_dx, tangent_dy = self._local_forward_tangent(points, target_index)

        raw_forward_px = center_y - target_y
        raw_lateral_px = target_x - center_x
        tangent_motion_fallback = False
        if target_index == len(points) - 1 and target_distance < self.config.reach_radius_px:
            # The visible path ends at the camera centre.  Keep moving along
            # its final tangent until the next frame reveals more road.
            raw_forward_px = -tangent_dy
            raw_lateral_px = tangent_dx
            tangent_motion_fallback = True

        filtered_forward_px = self._filter_scalar(
            raw_forward_px,
            self._filtered_forward_px,
            tau_s=self.config.target_filter_tau_s,
            max_rate_per_s=self.config.target_filter_max_rate_px_s,
            dt_s=dt_s,
        )
        filtered_lateral_px = self._filter_scalar(
            raw_lateral_px,
            self._filtered_lateral_px,
            tau_s=self.config.target_filter_tau_s,
            max_rate_per_s=self.config.target_filter_max_rate_px_s,
            dt_s=dt_s,
        )
        self._filtered_forward_px = filtered_forward_px
        self._filtered_lateral_px = filtered_lateral_px

        raw_tangent_error_deg = math.degrees(math.atan2(tangent_dx, -tangent_dy))
        tangent_error_deg = self._filter_angle(
            raw_tangent_error_deg,
            self._filtered_tangent_error_deg,
            tau_s=self.config.tangent_filter_tau_s,
            max_rate_per_s=self.config.tangent_filter_max_rate_deg_s,
            dt_s=dt_s,
        )
        self._filtered_tangent_error_deg = tangent_error_deg
        used_tangent_error_deg = _deadband(
            tangent_error_deg,
            self.config.tangent_deadband_deg,
        )
        unclamped_yaw_rate = (
            self.config.yaw_sign
            * self.config.tangent_kp_yaw
            * used_tangent_error_deg
        )
        clamped_yaw_rate = _clamp(
            unclamped_yaw_rate,
            -self.config.max_yaw_rate_deg_s,
            self.config.max_yaw_rate_deg_s,
        )
        yaw_rate, yaw_accel_limited = self._limit_scalar_rate(
            clamped_yaw_rate,
            self._limited_yaw_rate_deg_s,
            max_rate_per_s=self.config.max_yaw_accel_deg_s2,
            dt_s=dt_s,
        )
        self._limited_yaw_rate_deg_s = yaw_rate

        requested_vx, requested_vy = self._directional_velocity(
            filtered_forward_px,
            filtered_lateral_px,
        )
        road_state = str(getattr(perception, "road_state", "unknown"))
        speed_scale = (
            self.config.degraded_speed_scale
            if road_state in {"single_rough", "single_extrapolated"}
            else 1.0
        )
        requested_vx *= speed_scale
        requested_vy *= speed_scale
        vx, vy, planar_accel_limited, planar_command_delta = (
            self._limit_planar_acceleration(
                requested_vx,
                requested_vy,
                self._limited_vx_cm_s,
                self._limited_vy_cm_s,
                max_accel_cm_s2=self.config.max_planar_accel_cm_s2,
                dt_s=dt_s,
            )
        )
        self._limited_vx_cm_s = vx
        self._limited_vy_cm_s = vy

        self.last_diagnostics = TrajectoryPointFollowerDiagnostics(
            state="tracking",
            path_point_count=len(points),
            nearest_index=nearest_index,
            target_index=target_index,
            camera_center_x_px=center_x,
            camera_center_y_px=center_y,
            target_x_px=target_x,
            target_y_px=target_y,
            target_distance_px=target_distance,
            target_reached=nearest_distance < self.config.reach_radius_px,
            target_advanced_for_lookahead=target_advanced_for_lookahead,
            tangent_dx_px=tangent_dx,
            tangent_dy_px=tangent_dy,
            raw_forward_error_px=raw_forward_px,
            filtered_forward_error_px=filtered_forward_px,
            raw_lateral_error_px=raw_lateral_px,
            filtered_lateral_error_px=filtered_lateral_px,
            raw_tangent_error_deg=raw_tangent_error_deg,
            tangent_error_deg=tangent_error_deg,
            raw_centerline_angle_deg=90.0 - raw_tangent_error_deg,
            centerline_angle_deg=90.0 - tangent_error_deg,
            angle_error_deg=used_tangent_error_deg,
            raw_pixel_error_px=raw_lateral_px,
            filtered_pixel_error_px=filtered_lateral_px,
            used_pixel_error_px=filtered_lateral_px,
            angle_yaw_term_deg_s=unclamped_yaw_rate,
            unclamped_yaw_rate_deg_s=unclamped_yaw_rate,
            clamped_yaw_rate_deg_s=clamped_yaw_rate,
            yaw_rate_deg_s=yaw_rate,
            yaw_accel_limited=yaw_accel_limited,
            unclamped_vx_cm_s=requested_vx,
            unclamped_vy_cm_s=requested_vy,
            vy_cm_s=vy,
            vx_cm_s=vx,
            planar_accel_limited=planar_accel_limited,
            planar_command_delta_cm_s=planar_command_delta,
            heading_speed_scale=speed_scale,
            tangent_motion_fallback=tangent_motion_fallback,
        )
        return Command(
            vx,
            vy,
            0.0,
            yaw_rate,
            f"trajectory_point_follow:{road_state}",
        )

    def _usable_trajectory(self, perception) -> list[Point]:
        if perception is None:
            return []
        if not bool(getattr(perception, "is_road_found", False)):
            return []
        if float(getattr(perception, "confidence", 0.0)) < self.config.min_confidence:
            return []

        raw_points: Iterable[object] | None = getattr(
            perception,
            "trajectory_points",
            None,
        )
        if raw_points is None or _is_empty(raw_points):
            raw_points = getattr(perception, "centerline_points", None)
        if raw_points is None:
            raw_points = []
        points: list[Point] = []
        for point in raw_points:
            try:
                x, y = float(point[0]), float(point[1])  # type: ignore[index]
            except (IndexError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        if len(points) >= 2 and points[0][1] < points[-1][1]:
            points.reverse()
        return points

    def _local_forward_tangent(
        self,
        points: list[Point],
        target_index: int,
    ) -> Point:
        window = max(1, int(self.config.tangent_window_points))
        first = max(0, target_index - window)
        last = min(len(points) - 1, target_index + window)
        if first == last:
            first = max(0, last - 1)
            last = min(len(points) - 1, first + 1)
        dx = points[last][0] - points[first][0]
        dy = points[last][1] - points[first][1]
        if math.hypot(dx, dy) < 1e-6:
            return 0.0, -1.0
        if dy > 0.0:
            return -dx, -dy
        return dx, dy

    def _directional_velocity(self, forward_px: float, lateral_px: float) -> Point:
        magnitude = math.hypot(forward_px, lateral_px)
        if magnitude < 1e-6:
            return 0.0, 0.0

        unit_x = forward_px / magnitude
        unit_y = self.config.lateral_sign * lateral_px / magnitude
        scale_limits = [
            max(abs(self.config.max_vx_cm_s), abs(self.config.max_vy_cm_s)),
        ]
        if abs(unit_x) > 1e-9:
            scale_limits.append(abs(self.config.max_vx_cm_s) / abs(unit_x))
        if abs(unit_y) > 1e-9:
            scale_limits.append(abs(self.config.max_vy_cm_s) / abs(unit_y))
        scale = min(scale_limits)
        return unit_x * scale, unit_y * scale

    @staticmethod
    def _limit_planar_acceleration(
        requested_vx: float,
        requested_vy: float,
        previous_vx: float,
        previous_vy: float,
        *,
        max_accel_cm_s2: float,
        dt_s: float,
    ) -> tuple[float, float, bool, float]:
        delta_vx = float(requested_vx) - float(previous_vx)
        delta_vy = float(requested_vy) - float(previous_vy)
        requested_delta = math.hypot(delta_vx, delta_vy)
        if requested_delta < 1e-9 or max_accel_cm_s2 <= 0.0:
            return float(requested_vx), float(requested_vy), False, requested_delta

        # Do not let one delayed loop consume an arbitrarily large slew budget.
        max_delta = float(max_accel_cm_s2) * min(float(dt_s), 0.25)
        if requested_delta <= max_delta:
            return float(requested_vx), float(requested_vy), False, requested_delta
        scale = max_delta / requested_delta
        return (
            float(previous_vx) + delta_vx * scale,
            float(previous_vy) + delta_vy * scale,
            True,
            max_delta,
        )

    @staticmethod
    def _limit_scalar_rate(
        requested: float,
        previous: float,
        *,
        max_rate_per_s: float,
        dt_s: float,
    ) -> tuple[float, bool]:
        delta = float(requested) - float(previous)
        if max_rate_per_s <= 0.0:
            return float(requested), False
        max_delta = float(max_rate_per_s) * min(float(dt_s), 0.25)
        limited_delta = _clamp(delta, -max_delta, max_delta)
        return float(previous) + limited_delta, abs(limited_delta - delta) > 1e-9

    def _observation_dt(self, now_s: float) -> float:
        if self._last_update_s is None:
            dt_s = 0.1
        else:
            dt_s = _clamp(float(now_s) - self._last_update_s, 0.01, 1.0)
        self._last_update_s = float(now_s)
        return dt_s

    @staticmethod
    def _filter_scalar(
        value: float,
        previous: float | None,
        *,
        tau_s: float,
        max_rate_per_s: float,
        dt_s: float,
    ) -> float:
        if previous is None:
            return float(value)
        delta = float(value) - previous
        max_delta = max(0.0, float(max_rate_per_s)) * dt_s
        delta = _clamp(delta, -max_delta, max_delta)
        alpha = dt_s / (max(0.0, float(tau_s)) + dt_s)
        return previous + alpha * delta

    @staticmethod
    def _filter_angle(
        value: float,
        previous: float | None,
        *,
        tau_s: float,
        max_rate_per_s: float,
        dt_s: float,
    ) -> float:
        value = _wrap_angle_deg(value)
        if previous is None:
            return value
        delta = _wrap_angle_deg(value - previous)
        max_delta = max(0.0, float(max_rate_per_s)) * dt_s
        delta = _clamp(delta, -max_delta, max_delta)
        alpha = dt_s / (max(0.0, float(tau_s)) + dt_s)
        return _wrap_angle_deg(previous + alpha * delta)

    def _lost_command(self, now_s: float) -> Command:
        if self._lost_since_s is None:
            self._lost_since_s = float(now_s)
        lost_elapsed_s = max(0.0, float(now_s) - self._lost_since_s)
        self._last_update_s = float(now_s)
        self._filtered_forward_px = None
        self._filtered_lateral_px = None
        self._filtered_tangent_error_deg = None
        self._limited_vx_cm_s = 0.0
        self._limited_vy_cm_s = 0.0
        self._limited_yaw_rate_deg_s = 0.0
        self.last_diagnostics = TrajectoryPointFollowerDiagnostics(
            state="road_lost_hold",
            lost_elapsed_s=lost_elapsed_s,
        )
        return Command.zero("trajectory_road_lost_hold")


def _distance_sq(first: Point, second: Point) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _is_empty(values: Iterable[object]) -> bool:
    try:
        return len(values) == 0  # type: ignore[arg-type]
    except TypeError:
        return False


def _deadband(value: float, deadband: float) -> float:
    deadband = max(0.0, float(deadband))
    if abs(value) <= deadband:
        return 0.0
    return math.copysign(abs(value) - deadband, value)


def _wrap_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
