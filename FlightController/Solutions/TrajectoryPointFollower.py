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
    yaw_sign: float = 1.0
    lateral_sign: float = -1.0
    target_filter_tau_s: float = 0.15
    tangent_filter_tau_s: float = 0.20
    target_filter_max_rate_px_s: float = 600.0
    tangent_filter_max_rate_deg_s: float = 90.0
    max_planar_accel_cm_s2: float = 24.0
    max_planar_decel_cm_s2: float = 48.0
    max_yaw_accel_deg_s2: float = 20.0
    degraded_speed_scale: float = 0.85
    curvature_slowdown_start_deg: float = 8.0
    curvature_full_slowdown_deg: float = 35.0
    min_curve_speed_cm_s: float = 8.0
    curvature_filter_tau_s: float = 0.30
    curvature_feedforward_gain: float = 1.0
    turn_enter_curvature_deg: float = 22.0
    turn_exit_curvature_deg: float = 10.0
    turn_enter_heading_deg: float = 18.0
    turn_exit_heading_deg: float = 8.0
    turn_exit_lateral_px: float = 30.0
    turn_exit_hold_s: float = 0.50
    turn_speed_cm_s: float = 8.0
    turn_max_lateral_cm_s: float = 6.0
    turn_tangent_kp_yaw: float = 0.40
    turn_min_yaw_rate_deg_s: float = 6.0
    recovery_heading_deg: float = 35.0
    recovery_lateral_px: float = 70.0
    recovery_target_distance_px: float = 90.0
    recovery_speed_cm_s: float = 4.0
    recovery_yaw_rate_deg_s: float = 8.0


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
    target_path_distance_px: float = 0.0
    path_width_px: float | None = None
    tangent_dx_px: float | None = None
    tangent_dy_px: float | None = None
    raw_signed_curvature_deg: float = 0.0
    filtered_signed_curvature_deg: float = 0.0
    forward_curvature_deg: float = 0.0
    curvature_arc_px: float = 0.0
    curvature_feedforward_deg_s: float = 0.0
    curve_speed_limit_cm_s: float = 0.0
    curvature_speed_scale: float = 1.0
    turn_active: bool = False
    turn_recovery_active: bool = False
    turn_clear_elapsed_s: float = 0.0
    active_speed_limit_cm_s: float = 0.0
    active_lateral_limit_cm_s: float = 0.0
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
    planar_braking: bool = False
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
        self._filtered_signed_curvature_deg: float | None = None
        self._limited_vx_cm_s = 0.0
        self._limited_vy_cm_s = 0.0
        self._limited_yaw_rate_deg_s = 0.0
        self._turn_active = False
        self._turn_clear_since_s: float | None = None
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
        effective_lookahead_px = min(
            max(
                float(self.config.min_forward_lookahead_px),
                base_lookahead_px + latency_prediction_px,
            ),
            max(
                float(self.config.min_forward_lookahead_px),
                float(self.config.max_forward_lookahead_px),
            ),
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
        tangent_dx, tangent_dy = self._forward_tangent(points, target_index)
        raw_signed_curvature_deg, curvature_arc_px = self._forward_curve_geometry(
            points,
            nearest_index,
            target_index,
        )
        filtered_signed_curvature_deg = self._filter_scalar(
            raw_signed_curvature_deg,
            self._filtered_signed_curvature_deg,
            tau_s=self.config.curvature_filter_tau_s,
            max_rate_per_s=1_000.0,
            dt_s=dt_s,
        )
        self._filtered_signed_curvature_deg = filtered_signed_curvature_deg
        forward_curvature_deg = abs(filtered_signed_curvature_deg)
        curve_speed_limit_cm_s = self._curve_speed_limit_cm_s(forward_curvature_deg)
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
        turn_clear_elapsed_s = self._update_turn_state(
            now_s=now_s,
            curvature_deg=forward_curvature_deg,
            heading_error_deg=used_tangent_error_deg,
            lateral_error_px=raw_lateral_px,
        )
        turn_recovery_active = self._turn_active and (
            abs(used_tangent_error_deg) >= self.config.recovery_heading_deg
            or (
                abs(raw_lateral_px) >= self.config.recovery_lateral_px
                and target_distance >= self.config.recovery_target_distance_px
            )
        )
        heading_kp = (
            self.config.turn_tangent_kp_yaw
            if self._turn_active
            else self.config.tangent_kp_yaw
        )
        heading_yaw_term = (
            self.config.yaw_sign * heading_kp * used_tangent_error_deg
        )
        curvature_feedforward_deg_s = self.config.yaw_sign * (
            self._curvature_feedforward_deg_s(
                filtered_signed_curvature_deg,
                curvature_arc_px,
                path_width_px,
                current_planar_speed,
                curve_speed_limit_cm_s,
            )
        )
        unclamped_yaw_rate = heading_yaw_term + curvature_feedforward_deg_s
        if self._turn_active and (
            turn_recovery_active
            or abs(used_tangent_error_deg) >= self.config.turn_exit_heading_deg
            or forward_curvature_deg >= self.config.turn_exit_curvature_deg
        ):
            minimum_yaw_rate = (
                self.config.recovery_yaw_rate_deg_s
                if turn_recovery_active
                else self.config.turn_min_yaw_rate_deg_s
            )
            turn_direction = _first_nonzero_sign(
                unclamped_yaw_rate,
                self.config.yaw_sign * used_tangent_error_deg,
                self.config.yaw_sign * filtered_signed_curvature_deg,
            )
            if turn_direction != 0.0 and abs(unclamped_yaw_rate) < minimum_yaw_rate:
                unclamped_yaw_rate = turn_direction * minimum_yaw_rate
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

        active_speed_limit_cm_s = curve_speed_limit_cm_s
        active_lateral_limit_cm_s = max(0.0, float(self.config.max_vy_cm_s))
        if self._turn_active:
            active_speed_limit_cm_s = min(
                active_speed_limit_cm_s,
                max(0.0, float(self.config.turn_speed_cm_s)),
            )
            active_lateral_limit_cm_s = min(
                active_lateral_limit_cm_s,
                max(0.0, float(self.config.turn_max_lateral_cm_s)),
            )
        if turn_recovery_active:
            active_speed_limit_cm_s = min(
                active_speed_limit_cm_s,
                max(0.0, float(self.config.recovery_speed_cm_s)),
            )
            active_lateral_limit_cm_s = min(
                active_lateral_limit_cm_s,
                max(0.0, float(self.config.recovery_speed_cm_s)),
            )

        requested_vx, requested_vy = self._directional_velocity(
            filtered_forward_px,
            used_lateral_px,
            speed_limit_cm_s=active_speed_limit_cm_s,
            max_lateral_cm_s=active_lateral_limit_cm_s,
        )
        road_state = str(getattr(perception, "road_state", "unknown"))
        speed_scale = (
            self.config.degraded_speed_scale
            if road_state in {"single_rough", "single_extrapolated"}
            else 1.0
        )
        requested_vx *= speed_scale
        requested_vy *= speed_scale
        vx, vy, planar_accel_limited, planar_braking, planar_command_delta = (
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
            effective_lookahead_px=effective_lookahead_px,
            target_path_distance_px=target_path_distance_px,
            path_width_px=path_width_px,
            tangent_dx_px=tangent_dx,
            tangent_dy_px=tangent_dy,
            raw_signed_curvature_deg=raw_signed_curvature_deg,
            filtered_signed_curvature_deg=filtered_signed_curvature_deg,
            forward_curvature_deg=forward_curvature_deg,
            curvature_arc_px=curvature_arc_px,
            curvature_feedforward_deg_s=curvature_feedforward_deg_s,
            curve_speed_limit_cm_s=curve_speed_limit_cm_s,
            curvature_speed_scale=curvature_speed_scale,
            turn_active=self._turn_active,
            turn_recovery_active=turn_recovery_active,
            turn_clear_elapsed_s=turn_clear_elapsed_s,
            active_speed_limit_cm_s=active_speed_limit_cm_s,
            active_lateral_limit_cm_s=active_lateral_limit_cm_s,
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
            angle_yaw_term_deg_s=heading_yaw_term,
            unclamped_yaw_rate_deg_s=unclamped_yaw_rate,
            clamped_yaw_rate_deg_s=clamped_yaw_rate,
            yaw_rate_deg_s=yaw_rate,
            yaw_accel_limited=yaw_accel_limited,
            unclamped_vx_cm_s=requested_vx,
            unclamped_vy_cm_s=requested_vy,
            vy_cm_s=vy,
            vx_cm_s=vx,
            planar_accel_limited=planar_accel_limited,
            planar_braking=planar_braking,
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

    def _forward_tangent(
        self,
        points: list[Point],
        target_index: int,
    ) -> Point:
        """Estimate the target heading from points ahead, without averaging the turn away."""

        window = max(1, int(self.config.tangent_window_points))
        first = min(max(0, target_index), len(points) - 1)
        last = min(len(points) - 1, first + window)
        if first == last:
            return self._local_forward_tangent(points, target_index)
        dx = points[last][0] - points[first][0]
        dy = points[last][1] - points[first][1]
        if math.hypot(dx, dy) < 1e-6:
            return self._local_forward_tangent(points, target_index)
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
        signed_curvature_deg, _ = self._forward_curve_geometry(
            points,
            nearest_index,
            target_index,
        )
        return abs(signed_curvature_deg)

    def _forward_curve_geometry(
        self,
        points: list[Point],
        nearest_index: int,
        target_index: int,
    ) -> tuple[float, float]:
        """Return the largest signed upcoming heading change and its path arc."""

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
        headings: list[tuple[int, float]] = []
        for index in probe_indices:
            tangent_dx, tangent_dy = self._local_forward_tangent(points, index)
            headings.append(
                (index, math.degrees(math.atan2(tangent_dx, -tangent_dy)))
            )

        best_delta = 0.0
        best_first, reference_heading = headings[0]
        best_last = last_probe
        for second_index, second_heading in headings[1:]:
            delta = _wrap_angle_deg(second_heading - reference_heading)
            if abs(delta) > abs(best_delta):
                best_delta = delta
                best_last = second_index

        arc_px = sum(
            math.sqrt(_distance_sq(points[index - 1], points[index]))
            for index in range(best_first + 1, best_last + 1)
        )
        if arc_px < 1e-6:
            arc_px = sum(
                math.sqrt(_distance_sq(points[index - 1], points[index]))
                for index in range(nearest_index + 1, last_probe + 1)
            )
        return best_delta, arc_px

    def _curvature_feedforward_deg_s(
        self,
        signed_curvature_deg: float,
        curvature_arc_px: float,
        path_width_px: float | None,
        current_planar_speed_cm_s: float,
        curve_speed_limit_cm_s: float,
    ) -> float:
        if (
            path_width_px is None
            or path_width_px <= 0.0
            or curvature_arc_px <= 1e-6
            or abs(signed_curvature_deg) < 1e-6
        ):
            return 0.0
        road_width_cm = max(1e-6, float(self.config.physical_road_width_cm))
        pixels_per_cm = path_width_px / road_width_cm
        arc_cm = curvature_arc_px / max(1e-6, pixels_per_cm)
        feedforward_speed = min(
            max(0.0, float(curve_speed_limit_cm_s)),
            max(
                max(0.0, float(current_planar_speed_cm_s)),
                max(0.0, float(self.config.min_curve_speed_cm_s)),
            ),
        )
        return (
            float(self.config.curvature_feedforward_gain)
            * feedforward_speed
            * float(signed_curvature_deg)
            / max(1e-6, arc_cm)
        )

    def _update_turn_state(
        self,
        *,
        now_s: float,
        curvature_deg: float,
        heading_error_deg: float,
        lateral_error_px: float,
    ) -> float:
        if not self._turn_active and (
            abs(curvature_deg) >= self.config.turn_enter_curvature_deg
            or abs(heading_error_deg) >= self.config.turn_enter_heading_deg
        ):
            self._turn_active = True
            self._turn_clear_since_s = None

        if not self._turn_active:
            self._turn_clear_since_s = None
            return 0.0

        turn_is_clear = (
            abs(curvature_deg) <= self.config.turn_exit_curvature_deg
            and abs(heading_error_deg) <= self.config.turn_exit_heading_deg
            and abs(lateral_error_px) <= self.config.turn_exit_lateral_px
        )
        if not turn_is_clear:
            self._turn_clear_since_s = None
            return 0.0

        if self._turn_clear_since_s is None:
            self._turn_clear_since_s = float(now_s)
        clear_elapsed_s = max(0.0, float(now_s) - self._turn_clear_since_s)
        if clear_elapsed_s >= max(0.0, float(self.config.turn_exit_hold_s)):
            self._turn_active = False
            self._turn_clear_since_s = None
        return clear_elapsed_s

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
        max_lateral_cm_s: float | None = None,
    ) -> Point:
        magnitude = math.hypot(forward_px, lateral_px)
        if magnitude < 1e-6:
            return 0.0, 0.0

        unit_x = forward_px / magnitude
        unit_y = self.config.lateral_sign * lateral_px / magnitude
        speed_limit = (
            max(abs(self.config.max_vx_cm_s), abs(self.config.max_vy_cm_s))
            if speed_limit_cm_s is None
            else max(0.0, float(speed_limit_cm_s))
        )
        scale_limits = [speed_limit]
        if abs(unit_x) > 1e-9:
            scale_limits.append(abs(self.config.max_vx_cm_s) / abs(unit_x))
        lateral_limit = (
            abs(self.config.max_vy_cm_s)
            if max_lateral_cm_s is None
            else max(0.0, float(max_lateral_cm_s))
        )
        if abs(unit_y) > 1e-9:
            scale_limits.append(lateral_limit / abs(unit_y))
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
        max_decel_cm_s2: float,
        dt_s: float,
    ) -> tuple[float, float, bool, bool, float]:
        delta_vx = float(requested_vx) - float(previous_vx)
        delta_vy = float(requested_vy) - float(previous_vy)
        requested_delta = math.hypot(delta_vx, delta_vy)
        previous_speed = math.hypot(previous_vx, previous_vy)
        requested_speed = math.hypot(requested_vx, requested_vy)
        direction_dot = (
            float(requested_vx) * float(previous_vx)
            + float(requested_vy) * float(previous_vy)
        )
        braking = (
            requested_speed < previous_speed - 1e-9
            or (previous_speed > 1e-9 and direction_dot <= 0.0)
        )
        rate_limit = max_decel_cm_s2 if braking else max_accel_cm_s2
        if requested_delta < 1e-9 or rate_limit <= 0.0:
            return (
                float(requested_vx),
                float(requested_vy),
                False,
                braking,
                requested_delta,
            )

        # Do not let one delayed loop consume an arbitrarily large slew budget.
        max_delta = float(rate_limit) * min(float(dt_s), 0.25)
        if requested_delta <= max_delta:
            return (
                float(requested_vx),
                float(requested_vy),
                False,
                braking,
                requested_delta,
            )
        scale = max_delta / requested_delta
        return (
            float(previous_vx) + delta_vx * scale,
            float(previous_vy) + delta_vy * scale,
            True,
            braking,
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
        self._filtered_signed_curvature_deg = None
        self._limited_vx_cm_s = 0.0
        self._limited_vy_cm_s = 0.0
        self._limited_yaw_rate_deg_s = 0.0
        self._turn_active = False
        self._turn_clear_since_s = None
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


def _first_nonzero_sign(*values: float) -> float:
    for value in values:
        if abs(float(value)) > 1e-9:
            return math.copysign(1.0, float(value))
    return 0.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
