"""Image-space trajectory point follower for a downward-facing road camera."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable

from .Safety import Command


Point = tuple[float, float]


@dataclass
class TrajectoryPointFollowerConfig:
    image_width: int = 640
    image_height: int = 480
    max_vx_cm_s: float = 20.0
    max_vy_cm_s: float = 12.0
    max_yaw_rate_deg_s: float = 10.0
    reach_radius_px: float = 30.0
    min_forward_lookahead_px: float = 24.0
    max_forward_lookahead_px: float = 64.0
    lookahead_speed_gain_px_per_cm_s: float = 1.2
    latency_compensation_s: float = 0.134
    physical_road_width_cm: float = 50.0
    max_latency_prediction_px: float = 16.0
    tangent_window_points: int = 5
    min_confidence: float = 0.35
    tangent_kp_yaw: float = 0.25
    tangent_deadband_deg: float = 3.0
    lateral_deadband_px: float = 8.0
    lateral_kp_cm_s_per_px: float = 0.10
    normal_max_vy_cm_s: float = 12.0
    curvature_yaw_ff_kp: float = 0.0
    curvature_yaw_ff_max_deg_s: float = 0.0
    curvature_yaw_ff_deadband_deg: float = 6.0
    signed_turn_filter_tau_s: float = 0.08
    corner_lookahead_start_deg: float = 30.0
    corner_lookahead_full_deg: float = 75.0
    corner_min_lookahead_px: float = 75.0
    corner_severity_release_tau_s: float = 0.25
    edge_recovery_start_ratio: float = 1.0
    edge_recovery_full_ratio: float = 1.0
    edge_recovery_lateral_kp: float = 0.22
    edge_recovery_max_vy_cm_s: float = 0.0
    edge_yaw_start_ratio: float = 1.0
    edge_yaw_full_ratio: float = 1.0
    edge_yaw_max_deg_s: float = 0.0
    edge_speed_slow_start_ratio: float = 1.0
    edge_emergency_ratio: float = 1.0
    edge_emergency_vx_cap_cm_s: float = 0.0
    yaw_sign: float = 1.0
    lateral_sign: float = -1.0
    target_filter_tau_s: float = 0.15
    tangent_filter_tau_s: float = 0.20
    target_filter_max_rate_px_s: float = 600.0
    tangent_filter_max_rate_deg_s: float = 90.0
    max_planar_accel_cm_s2: float = 24.0
    max_planar_decel_cm_s2: float = 24.0
    max_yaw_accel_deg_s2: float = 20.0
    degraded_speed_scale: float = 0.85
    curvature_slowdown_start_deg: float = 8.0
    curvature_full_slowdown_deg: float = 35.0
    min_curve_speed_cm_s: float = 10.0
    lost_grace_s: float = 0.0
    lost_grace_vx_scale: float = 0.80
    lost_grace_vy_scale: float = 0.50
    lost_grace_yaw_scale: float = 0.70


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
    current_planar_speed_cm_s: float = 0.0
    base_lookahead_px: float = 0.0
    latency_prediction_px: float = 0.0
    effective_lookahead_px: float = 0.0
    provisional_lookahead_px: float = 0.0
    corner_lookahead_cap_px: float = 0.0
    target_path_distance_px: float = 0.0
    path_width_px: float | None = None
    tangent_dx_px: float | None = None
    tangent_dy_px: float | None = None
    forward_curvature_deg: float = 0.0
    raw_corner_severity_deg: float = 0.0
    corner_severity_deg: float = 0.0
    signed_preview_turn_deg: float = 0.0
    filtered_signed_preview_turn_deg: float = 0.0
    turn_consistency: float = 0.0
    curve_speed_limit_cm_s: float = 0.0
    curvature_speed_scale: float = 1.0
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
    yaw_feedback_deg_s: float = 0.0
    yaw_feedforward_deg_s: float = 0.0
    edge_yaw_bias_deg_s: float = 0.0
    unclamped_yaw_rate_deg_s: float = 0.0
    clamped_yaw_rate_deg_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    yaw_accel_limited: bool = False
    unclamped_vx_cm_s: float = 0.0
    unclamped_vy_cm_s: float = 0.0
    normal_vy_cm_s: float = 0.0
    recovery_vy_cm_s: float = 0.0
    vy_cm_s: float = 0.0
    vx_cm_s: float = 0.0
    planar_accel_limited: bool = False
    planar_decel_limited: bool = False
    planar_rate_limit_cm_s2: float = 0.0
    planar_command_delta_cm_s: float = 0.0
    heading_speed_scale: float = 0.0
    tangent_motion_fallback: bool = False
    lost_elapsed_s: float = 0.0
    centerline_x_at_camera_y_px: float | None = None
    current_cross_track_px: float | None = None
    road_half_width_px: float | None = None
    edge_ratio: float | None = None
    edge_recovery_blend: float = 0.0
    edge_speed_cap_cm_s: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class TrajectoryPointFollower:
    """Chase the closest forward trajectory point while aligning its tangent."""

    def __init__(self, config: TrajectoryPointFollowerConfig | None = None):
        self.config = config or TrajectoryPointFollowerConfig()
        self._last_update_s: float | None = None
        self._lost_since_s: float | None = None
        self._lost_entry_command: tuple[float, float, float] | None = None
        self._filtered_forward_px: float | None = None
        self._filtered_lateral_px: float | None = None
        self._filtered_tangent_error_deg: float | None = None
        self._filtered_signed_preview_turn_deg: float | None = None
        self._corner_severity_deg = 0.0
        self._limited_vx_cm_s = 0.0
        self._limited_vy_cm_s = 0.0
        self._limited_yaw_rate_deg_s = 0.0
        self.last_diagnostics = TrajectoryPointFollowerDiagnostics()

    def update(
        self,
        perception,
        now_s: float,
        *,
        allow_lost_grace: bool = True,
    ) -> Command:
        points = self._usable_trajectory(perception)
        if len(points) < 2:
            return self._lost_command(now_s, allow_grace=allow_lost_grace)

        self._lost_since_s = None
        self._lost_entry_command = None
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
        current_planar_speed = math.hypot(
            self._limited_vx_cm_s,
            self._limited_vy_cm_s,
        )
        path_width_px = _finite_or_none(getattr(perception, "path_width_px", None))
        base_lookahead_px = self._adaptive_lookahead_px(current_planar_speed)
        latency_prediction_px = self._latency_prediction_px(
            current_planar_speed,
            path_width_px,
        )
        provisional_lookahead_px = min(
            max(
                float(self.config.min_forward_lookahead_px),
                base_lookahead_px + latency_prediction_px,
            ),
            max(
                float(self.config.min_forward_lookahead_px),
                float(self.config.max_forward_lookahead_px),
            ),
        )
        provisional_target_index, _ = self._select_forward_target(
            points,
            nearest_index=nearest_index,
            center_y=center_y,
            lookahead_px=provisional_lookahead_px,
        )
        raw_corner_severity_deg = self._forward_curvature_deg(
            points,
            nearest_index,
            provisional_target_index,
        )
        corner_severity_deg = self._filter_corner_severity(
            raw_corner_severity_deg,
            dt_s=dt_s,
        )
        corner_lookahead_cap_px = self._corner_lookahead_cap_px(
            corner_severity_deg
        )
        effective_lookahead_px = min(
            provisional_lookahead_px,
            corner_lookahead_cap_px,
        )
        target_index, target_path_distance_px = self._select_forward_target(
            points,
            nearest_index=nearest_index,
            center_y=center_y,
            lookahead_px=effective_lookahead_px,
        )
        target_advanced_for_lookahead = target_index > nearest_index

        target_x, target_y = points[target_index]
        target_distance = math.hypot(target_x - center_x, target_y - center_y)
        tangent_dx, tangent_dy = self._local_forward_tangent(points, target_index)
        signed_preview_turn_deg, turn_consistency = self._signed_preview_turn_deg(
            points,
            nearest_index,
            target_index,
        )
        filtered_signed_preview_turn_deg = self._filter_angle(
            signed_preview_turn_deg,
            self._filtered_signed_preview_turn_deg,
            tau_s=self.config.signed_turn_filter_tau_s,
            max_rate_per_s=1_000_000.0,
            dt_s=dt_s,
        )
        self._filtered_signed_preview_turn_deg = filtered_signed_preview_turn_deg

        curve_speed_limit_cm_s = self._curve_speed_limit_cm_s(corner_severity_deg)
        maximum_cruise_speed = max(0.0, float(self.config.max_vx_cm_s))
        curvature_speed_scale = (
            curve_speed_limit_cm_s / maximum_cruise_speed
            if maximum_cruise_speed > 1e-9
            else 0.0
        )

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
        used_lateral_px = _deadband(
            filtered_lateral_px,
            self.config.lateral_deadband_px,
        )

        _, normal_vy = self._directional_velocity(
            filtered_forward_px,
            used_lateral_px,
            speed_limit_cm_s=curve_speed_limit_cm_s,
        )
        centerline_x_at_camera_y_px = self._centerline_x_at_camera_y(
            points,
            center_y,
            nearest_index,
            center_x,
        )
        current_cross_track_px = (
            centerline_x_at_camera_y_px - center_x
            if centerline_x_at_camera_y_px is not None
            else None
        )
        road_half_width_px = (
            path_width_px * 0.5
            if path_width_px is not None and path_width_px > 0.0
            else None
        )
        edge_ratio = (
            abs(current_cross_track_px) / max(road_half_width_px, 1.0)
            if current_cross_track_px is not None and road_half_width_px is not None
            else None
        )
        edge_recovery_blend = self._ratio_blend(
            edge_ratio,
            self.config.edge_recovery_start_ratio,
            self.config.edge_recovery_full_ratio,
        )
        used_cross_track_px = _deadband(
            current_cross_track_px or 0.0,
            self.config.lateral_deadband_px,
        )
        recovery_limit = min(
            max(0.0, float(self.config.max_vy_cm_s)),
            max(0.0, float(self.config.edge_recovery_max_vy_cm_s)),
        )
        recovery_vy = _clamp(
            float(self.config.lateral_sign)
            * float(self.config.edge_recovery_lateral_kp)
            * used_cross_track_px,
            -recovery_limit,
            recovery_limit,
        )
        requested_vy = (
            (1.0 - edge_recovery_blend) * normal_vy
            + edge_recovery_blend * recovery_vy
        )
        global_max_vy = max(0.0, float(self.config.max_vy_cm_s))
        requested_vy = _clamp(requested_vy, -global_max_vy, global_max_vy)

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
        yaw_feedback_deg_s = (
            self.config.yaw_sign
            * self.config.tangent_kp_yaw
            * used_tangent_error_deg
        )
        used_signed_turn_deg = _deadband(
            filtered_signed_preview_turn_deg,
            self.config.curvature_yaw_ff_deadband_deg,
        )
        yaw_feedforward_limit = max(
            0.0, float(self.config.curvature_yaw_ff_max_deg_s)
        )
        yaw_feedforward_deg_s = _clamp(
            float(self.config.yaw_sign)
            * float(self.config.curvature_yaw_ff_kp)
            * used_signed_turn_deg,
            -yaw_feedforward_limit,
            yaw_feedforward_limit,
        )
        edge_yaw_blend = self._ratio_blend(
            edge_ratio,
            self.config.edge_yaw_start_ratio,
            self.config.edge_yaw_full_ratio,
        )
        edge_yaw_bias_deg_s = (
            float(self.config.yaw_sign)
            * _sign(current_cross_track_px or 0.0)
            * max(0.0, float(self.config.edge_yaw_max_deg_s))
            * edge_yaw_blend
        )
        unclamped_yaw_rate = (
            yaw_feedback_deg_s
            + yaw_feedforward_deg_s
            + edge_yaw_bias_deg_s
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

        edge_speed_cap_cm_s = self._edge_speed_cap_cm_s(
            curve_speed_limit_cm_s,
            edge_ratio,
        )
        requested_vx, _ = self._directional_velocity(
            filtered_forward_px,
            used_lateral_px,
            speed_limit_cm_s=edge_speed_cap_cm_s,
        )
        road_state = str(getattr(perception, "road_state", "unknown"))
        speed_scale = (
            self.config.degraded_speed_scale
            if road_state in {"single_rough", "single_extrapolated"}
            else 1.0
        )
        requested_vx *= speed_scale
        requested_vy *= speed_scale
        (
            vx,
            vy,
            planar_accel_limited,
            planar_command_delta,
            planar_rate_limit_cm_s2,
            planar_decel_limited,
        ) = (
            self._limit_planar_acceleration(
                requested_vx,
                requested_vy,
                self._limited_vx_cm_s,
                self._limited_vy_cm_s,
                max_accel_cm_s2=self.config.max_planar_accel_cm_s2,
                max_decel_cm_s2=self.config.max_planar_decel_cm_s2,
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
            current_planar_speed_cm_s=current_planar_speed,
            base_lookahead_px=base_lookahead_px,
            latency_prediction_px=latency_prediction_px,
            provisional_lookahead_px=provisional_lookahead_px,
            corner_lookahead_cap_px=corner_lookahead_cap_px,
            effective_lookahead_px=effective_lookahead_px,
            target_path_distance_px=target_path_distance_px,
            path_width_px=path_width_px,
            tangent_dx_px=tangent_dx,
            tangent_dy_px=tangent_dy,
            forward_curvature_deg=raw_corner_severity_deg,
            raw_corner_severity_deg=raw_corner_severity_deg,
            corner_severity_deg=corner_severity_deg,
            signed_preview_turn_deg=signed_preview_turn_deg,
            filtered_signed_preview_turn_deg=filtered_signed_preview_turn_deg,
            turn_consistency=turn_consistency,
            curve_speed_limit_cm_s=curve_speed_limit_cm_s,
            curvature_speed_scale=curvature_speed_scale,
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
            used_pixel_error_px=used_lateral_px,
            angle_yaw_term_deg_s=yaw_feedback_deg_s,
            yaw_feedback_deg_s=yaw_feedback_deg_s,
            yaw_feedforward_deg_s=yaw_feedforward_deg_s,
            edge_yaw_bias_deg_s=edge_yaw_bias_deg_s,
            unclamped_yaw_rate_deg_s=unclamped_yaw_rate,
            clamped_yaw_rate_deg_s=clamped_yaw_rate,
            yaw_rate_deg_s=yaw_rate,
            yaw_accel_limited=yaw_accel_limited,
            unclamped_vx_cm_s=requested_vx,
            unclamped_vy_cm_s=requested_vy,
            normal_vy_cm_s=normal_vy,
            recovery_vy_cm_s=recovery_vy,
            vy_cm_s=vy,
            vx_cm_s=vx,
            planar_accel_limited=planar_accel_limited,
            planar_decel_limited=planar_decel_limited,
            planar_rate_limit_cm_s2=planar_rate_limit_cm_s2,
            planar_command_delta_cm_s=planar_command_delta,
            heading_speed_scale=speed_scale,
            tangent_motion_fallback=tangent_motion_fallback,
            centerline_x_at_camera_y_px=centerline_x_at_camera_y_px,
            current_cross_track_px=current_cross_track_px,
            road_half_width_px=road_half_width_px,
            edge_ratio=edge_ratio,
            edge_recovery_blend=edge_recovery_blend,
            edge_speed_cap_cm_s=edge_speed_cap_cm_s,
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

    def _local_forward_tangent_bounded(
        self,
        points: list[Point],
        index: int,
        *,
        min_index: int,
        max_index: int,
    ) -> Point:
        if not points:
            return 0.0, -1.0

        min_index = max(0, min(int(min_index), len(points) - 1))
        max_index = max(min_index, min(int(max_index), len(points) - 1))
        index = max(min_index, min(int(index), max_index))
        window = max(1, int(self.config.tangent_window_points))
        first = max(min_index, index - window)
        last = min(max_index, index + window)
        if first == last:
            if last < max_index:
                last += 1
            elif first > min_index:
                first -= 1

        dx = points[last][0] - points[first][0]
        dy = points[last][1] - points[first][1]
        if math.hypot(dx, dy) < 1e-6:
            return 0.0, -1.0
        if dy > 0.0:
            return -dx, -dy
        return dx, dy

    def _adaptive_lookahead_px(self, current_speed_cm_s: float) -> float:
        minimum = max(0.0, float(self.config.min_forward_lookahead_px))
        maximum = max(minimum, float(self.config.max_forward_lookahead_px))
        requested = minimum + (
            max(0.0, float(current_speed_cm_s))
            * max(0.0, float(self.config.lookahead_speed_gain_px_per_cm_s))
        )
        return _clamp(requested, minimum, maximum)

    def _latency_prediction_px(
        self,
        current_speed_cm_s: float,
        path_width_px: float | None,
    ) -> float:
        if path_width_px is None or path_width_px <= 0.0:
            return 0.0
        road_width_cm = float(self.config.physical_road_width_cm)
        if road_width_cm <= 0.0:
            return 0.0
        pixels_per_cm = path_width_px / road_width_cm
        predicted_px = (
            max(0.0, float(current_speed_cm_s))
            * max(0.0, float(self.config.latency_compensation_s))
            * pixels_per_cm
        )
        return _clamp(
            predicted_px,
            0.0,
            max(0.0, float(self.config.max_latency_prediction_px)),
        )

    def _select_forward_target(
        self,
        points: list[Point],
        *,
        nearest_index: int,
        center_y: float,
        lookahead_px: float,
    ) -> tuple[int, float]:
        """Choose the farthest centreline point inside the adaptive arc horizon."""

        target_index = nearest_index
        path_distance_px = 0.0
        for index in range(nearest_index + 1, len(points)):
            step_px = math.sqrt(_distance_sq(points[index - 1], points[index]))
            candidate_distance_px = path_distance_px + step_px
            if candidate_distance_px > lookahead_px:
                break
            target_index = index
            path_distance_px = candidate_distance_px

        minimum_forward_px = max(0.0, float(self.config.min_forward_lookahead_px))
        while (
            target_index + 1 < len(points)
            and center_y - points[target_index][1] < minimum_forward_px
        ):
            next_index = target_index + 1
            path_distance_px += math.sqrt(
                _distance_sq(points[target_index], points[next_index])
            )
            target_index = next_index
        return target_index, path_distance_px

    def _forward_curvature_deg(
        self,
        points: list[Point],
        nearest_index: int,
        target_index: int,
    ) -> float:
        """Estimate upcoming heading change without penalising a straight diagonal road."""

        window = max(1, int(self.config.tangent_window_points))
        last_probe = min(
            len(points) - 1,
            max(target_index, nearest_index + 1) + 2 * window,
        )
        span = max(1, last_probe - nearest_index)
        probe_indices = sorted(
            {
                nearest_index,
                nearest_index + span // 3,
                nearest_index + (2 * span) // 3,
                target_index,
                last_probe,
            }
        )
        headings = []
        for index in probe_indices:
            tangent_dx, tangent_dy = self._local_forward_tangent(points, index)
            headings.append(math.degrees(math.atan2(tangent_dx, -tangent_dy)))
        return max(
            (
                abs(_wrap_angle_deg(second - first))
                for position, first in enumerate(headings)
                for second in headings[position + 1 :]
            ),
            default=0.0,
        )

    def _signed_preview_turn_deg(
        self,
        points: list[Point],
        nearest_index: int,
        target_index: int,
    ) -> tuple[float, float]:
        span = max(1, target_index - nearest_index)
        probe_indices = sorted(
            {
                min(len(points) - 1, nearest_index + round(span * fraction))
                for fraction in (0.0, 0.25, 0.50, 0.75, 1.0)
            }
        )
        headings = []
        for index in probe_indices:
            tangent_dx, tangent_dy = self._local_forward_tangent_bounded(
                points,
                index,
                min_index=nearest_index,
                max_index=target_index,
            )
            headings.append(math.degrees(math.atan2(tangent_dx, -tangent_dy)))
        deltas = [
            _wrap_angle_deg(second - first)
            for first, second in zip(headings, headings[1:])
        ]
        nonzero_deltas = [delta for delta in deltas if abs(delta) >= 2.0]
        if not nonzero_deltas:
            return 0.0, 1.0

        median_delta = float(statistics.median(nonzero_deltas))
        dominant_sign = _sign(median_delta)
        consistent_count = sum(
            1 for delta in nonzero_deltas if _sign(delta) == dominant_sign
        )
        consistency = consistent_count / len(nonzero_deltas)
        signed_turn = median_delta * len(nonzero_deltas)
        if consistency < 0.60:
            signed_turn *= 0.5
        return _clamp(signed_turn, -90.0, 90.0), consistency

    def _filter_corner_severity(self, raw_severity_deg: float, *, dt_s: float) -> float:
        raw = max(0.0, float(raw_severity_deg))
        previous = max(0.0, float(self._corner_severity_deg))
        if raw >= previous:
            filtered = raw
        else:
            tau_s = max(0.0, float(self.config.corner_severity_release_tau_s))
            alpha = float(dt_s) / (tau_s + float(dt_s))
            filtered = previous + alpha * (raw - previous)
        self._corner_severity_deg = filtered
        return filtered

    def _corner_lookahead_cap_px(self, corner_severity_deg: float) -> float:
        normal_max = max(
            float(self.config.min_forward_lookahead_px),
            float(self.config.max_forward_lookahead_px),
        )
        minimum = _clamp(
            float(self.config.corner_min_lookahead_px),
            float(self.config.min_forward_lookahead_px),
            normal_max,
        )
        start = max(0.0, float(self.config.corner_lookahead_start_deg))
        full = max(start + 1e-6, float(self.config.corner_lookahead_full_deg))
        severity = max(0.0, float(corner_severity_deg))
        if severity <= start:
            return normal_max
        if severity >= full:
            return minimum
        ratio = (severity - start) / (full - start)
        return normal_max + ratio * (minimum - normal_max)

    @staticmethod
    def _centerline_x_at_camera_y(
        points: list[Point],
        center_y: float,
        nearest_index: int,
        center_x: float,
    ) -> float | None:
        if not points:
            return None
        nearest_index = int(_clamp(nearest_index, 0, len(points) - 1))
        crossings: list[tuple[int, float, float]] = []
        for index, (first, second) in enumerate(zip(points, points[1:])):
            x1, y1 = first
            x2, y2 = second
            if abs(y2 - y1) < 1e-9:
                if abs(center_y - y1) < 1e-9:
                    x = 0.5 * (x1 + x2)
                    distance_to_nearest = min(
                        abs(index - nearest_index),
                        abs(index + 1 - nearest_index),
                    )
                    crossings.append(
                        (distance_to_nearest, abs(x - center_x), x)
                    )
                continue
            if min(y1, y2) <= center_y <= max(y1, y2):
                ratio = (center_y - y1) / (y2 - y1)
                x = x1 + ratio * (x2 - x1)
                distance_to_nearest = min(
                    abs(index - nearest_index),
                    abs(index + 1 - nearest_index),
                )
                crossings.append((distance_to_nearest, abs(x - center_x), x))
        if crossings:
            return min(crossings, key=lambda item: (item[0], item[1]))[2]
        return min(points, key=lambda point: abs(point[1] - center_y))[0]

    @staticmethod
    def _ratio_blend(value: float | None, start: float, full: float) -> float:
        if value is None:
            return 0.0
        start_value = float(start)
        full_value = float(full)
        if full_value <= start_value:
            return 1.0 if float(value) >= full_value else 0.0
        return _smoothstep01((float(value) - start_value) / (full_value - start_value))

    def _edge_speed_cap_cm_s(
        self,
        curve_speed_limit_cm_s: float,
        edge_ratio: float | None,
    ) -> float:
        curve_limit = max(0.0, float(curve_speed_limit_cm_s))
        emergency_cap = max(0.0, float(self.config.edge_emergency_vx_cap_cm_s))
        if edge_ratio is None or emergency_cap <= 0.0 or emergency_cap >= curve_limit:
            return curve_limit
        blend = self._ratio_blend(
            edge_ratio,
            self.config.edge_speed_slow_start_ratio,
            self.config.edge_emergency_ratio,
        )
        return curve_limit + blend * (emergency_cap - curve_limit)

    def _curve_speed_limit_cm_s(self, curvature_deg: float) -> float:
        maximum = max(0.0, float(self.config.max_vx_cm_s))
        minimum = _clamp(float(self.config.min_curve_speed_cm_s), 0.0, maximum)
        slowdown_start = max(0.0, float(self.config.curvature_slowdown_start_deg))
        full_slowdown = max(
            slowdown_start + 1e-6,
            float(self.config.curvature_full_slowdown_deg),
        )
        curvature = max(0.0, float(curvature_deg))
        if curvature <= slowdown_start:
            return maximum
        if curvature >= full_slowdown:
            return minimum
        ratio = (curvature - slowdown_start) / (full_slowdown - slowdown_start)
        return maximum + ratio * (minimum - maximum)

    def _directional_velocity(
        self,
        forward_px: float,
        lateral_px: float,
        *,
        speed_limit_cm_s: float | None = None,
    ) -> Point:
        if abs(forward_px) < 1e-6 and abs(lateral_px) < 1e-6:
            return 0.0, 0.0

        maximum_vx = max(0.0, float(self.config.max_vx_cm_s))
        vx_limit = (
            maximum_vx
            if speed_limit_cm_s is None
            else _clamp(float(speed_limit_cm_s), 0.0, maximum_vx)
        )

        # Lateral correction does not consume the forward cruise budget.
        vx = vx_limit if forward_px >= 0.0 else 0.0
        vy = (
            float(self.config.lateral_sign)
            * float(self.config.lateral_kp_cm_s_per_px)
            * float(lateral_px)
        )
        max_vy = min(
            max(0.0, float(self.config.max_vy_cm_s)),
            max(0.0, float(self.config.normal_max_vy_cm_s)),
        )
        return vx, _clamp(vy, -max_vy, max_vy)

    @staticmethod
    def _limit_planar_acceleration(
        requested_vx: float,
        requested_vy: float,
        previous_vx: float,
        previous_vy: float,
        *,
        max_accel_cm_s2: float,
        max_decel_cm_s2: float,
        dt_s: float,
    ) -> tuple[float, float, bool, float, float, bool]:
        delta_vx = float(requested_vx) - float(previous_vx)
        delta_vy = float(requested_vy) - float(previous_vy)
        requested_delta = math.hypot(delta_vx, delta_vy)
        previous_speed = math.hypot(previous_vx, previous_vy)
        requested_speed = math.hypot(requested_vx, requested_vy)
        decelerating = requested_speed < previous_speed
        rate_limit = float(max_decel_cm_s2 if decelerating else max_accel_cm_s2)
        if requested_delta < 1e-9 or rate_limit <= 0.0:
            return (
                float(requested_vx),
                float(requested_vy),
                False,
                requested_delta,
                rate_limit,
                False,
            )

        # Do not let one delayed loop consume an arbitrarily large slew budget.
        max_delta = rate_limit * min(float(dt_s), 0.25)
        if requested_delta <= max_delta:
            return (
                float(requested_vx),
                float(requested_vy),
                False,
                requested_delta,
                rate_limit,
                False,
            )
        scale = max_delta / requested_delta
        return (
            float(previous_vx) + delta_vx * scale,
            float(previous_vy) + delta_vy * scale,
            True,
            max_delta,
            rate_limit,
            decelerating,
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

    def _lost_command(self, now_s: float, *, allow_grace: bool = True) -> Command:
        if self._lost_since_s is None:
            self._lost_since_s = float(now_s)
            self._lost_entry_command = (
                self._limited_vx_cm_s,
                self._limited_vy_cm_s,
                self._limited_yaw_rate_deg_s,
            )
        lost_elapsed_s = max(0.0, float(now_s) - self._lost_since_s)
        self._last_update_s = float(now_s)

        grace_s = max(0.0, float(self.config.lost_grace_s))
        if (
            allow_grace
            and grace_s > 0.0
            and lost_elapsed_s <= grace_s
            and self._lost_entry_command is not None
        ):
            entry_vx, entry_vy, entry_yaw = self._lost_entry_command
            vx = entry_vx * _clamp(float(self.config.lost_grace_vx_scale), 0.0, 1.0)
            vy = entry_vy * _clamp(float(self.config.lost_grace_vy_scale), 0.0, 1.0)
            yaw_rate = entry_yaw * _clamp(
                float(self.config.lost_grace_yaw_scale), 0.0, 1.0
            )
            self._limited_vx_cm_s = vx
            self._limited_vy_cm_s = vy
            self._limited_yaw_rate_deg_s = yaw_rate
            self.last_diagnostics = TrajectoryPointFollowerDiagnostics(
                state="road_lost_grace",
                lost_elapsed_s=lost_elapsed_s,
                vx_cm_s=vx,
                vy_cm_s=vy,
                yaw_rate_deg_s=yaw_rate,
            )
            return Command(vx, vy, 0.0, yaw_rate, "trajectory_road_lost_grace")

        self._filtered_forward_px = None
        self._filtered_lateral_px = None
        self._filtered_tangent_error_deg = None
        self._filtered_signed_preview_turn_deg = None
        self._corner_severity_deg = 0.0
        self._limited_vx_cm_s = 0.0
        self._limited_vy_cm_s = 0.0
        self._limited_yaw_rate_deg_s = 0.0
        self._lost_entry_command = None
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


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _deadband(value: float, deadband: float) -> float:
    deadband = max(0.0, float(deadband))
    if abs(value) <= deadband:
        return 0.0
    return math.copysign(abs(value) - deadband, value)


def _wrap_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _smoothstep01(value: float) -> float:
    x = _clamp(float(value), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0
