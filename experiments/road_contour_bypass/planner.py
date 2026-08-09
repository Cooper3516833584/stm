"""Encounter-frozen inflated-contour trajectory bypass planner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import time
from typing import Any

from loguru import logger
import numpy as np

from FlightController.Solutions.Safety import Command, RadarObstacleField

from .geometry import (
    GeometryConfig,
    InflatedOccupancy,
    build_inflated_occupancy,
    build_quintic_control_points,
    choose_bypass_side,
    extract_external_contours,
    extract_side_envelope,
    path_is_collision_free,
    path_minimum_clearance_cm,
    render_plan_debug,
    sample_quintic_bezier,
    select_cluster_contour,
)
from .path_tracker import FrozenPathTracker, PathTrackerConfig, PathTrackingResult


class ContourBypassState(Enum):
    NORMAL = "normal"
    ACQUIRE = "acquire"
    PLAN = "plan"
    FOLLOW_BYPASS = "follow_bypass"
    FOV_EXIT_CONFIRM = "fov_exit_confirm"
    RETURN_TO_ROAD = "return_to_road"
    REJOIN_BLEND = "rejoin_blend"
    WAIT_VISUAL = "wait_visual"
    PLAN_FAILED = "plan_failed"


@dataclass(frozen=True)
class ContourBypassConfig:
    radar_fov_deg: float = 150.0
    radar_min_x_cm: float = 10.0
    radar_max_range_cm: float = 300.0

    activation_radius_cm: float = 120.0
    activation_frames: int = 2
    activation_corridor_half_width_cm: float = 80.0

    safety_radius_cm: float = 75.0
    planning_margin_cm: float = 10.0
    inflation_radius_cm: float = 85.0
    trajectory_extra_margin_cm: float = 5.0

    min_cluster_points: int = 3
    cluster_grid_cm: float = 10.0
    cluster_radius_x_cm: float = 25.0
    cluster_radius_y_cm: float = 20.0
    side_deadband_cm: float = 8.0

    grid_resolution_cm: float = 2.5
    grid_x_min_cm: float = -30.0
    grid_x_max_cm: float = 280.0
    grid_y_min_cm: float = -200.0
    grid_y_max_cm: float = 200.0
    envelope_bin_cm: float = 5.0

    bezier_sample_count: int = 121
    plan_retry_step_cm: float = 10.0
    max_plan_retries: int = 5

    bypass_nominal_speed_cm_s: float = 20.0
    bypass_max_speed_cm_s: float = 22.0
    bypass_min_speed_cm_s: float = 14.0
    max_vy_cm_s: float = 12.0
    max_planar_accel_cm_s2: float = 36.0
    max_planar_decel_cm_s2: float = 60.0
    path_lookahead_cm: float = 25.0

    yaw_hold_kp: float = 0.6
    yaw_hold_max_rate_deg_s: float = 5.0
    fov_exit_arm_deg: float = 65.0
    fov_exit_missing_frames: int = 3
    obstacle_association_bearing_deg: float = 30.0

    return_yaw_kp: float = 0.6
    return_max_yaw_rate_deg_s: float = 8.0
    max_yaw_accel_deg_s2: float = 40.0

    path_complete_progress: float = 0.95
    path_complete_distance_cm: float = 22.0
    rejoin_min_confidence: float = 0.5
    rejoin_pixel_deadband_px: float = 35.0
    rejoin_confirm_frames: int = 3
    rejoin_blend_s: float = 0.7


class ContourTrajectoryBypassPlanner:
    """Plan exactly once per obstacle encounter, then follow the frozen result."""

    def __init__(
        self,
        config: ContourBypassConfig | None = None,
        *,
        debug_plan_dir: str | Path | None = None,
    ) -> None:
        self.config = config or ContourBypassConfig()
        self.state = ContourBypassState.NORMAL
        self.previous_state = ContourBypassState.NORMAL
        self.transition_reason = "initialized"
        self.encounter_id = 0
        self.encounter_frame: tuple[float, float, float] | None = None
        self.locked_bypass_side: int | None = None
        self.path_samples: np.ndarray | None = None
        self.path_control_points: tuple[tuple[float, float], ...] | None = None
        self.frozen_cluster_points: np.ndarray | None = None
        self._frozen_occupancy: InflatedOccupancy | None = None
        self._frozen_contour: np.ndarray | None = None
        self._debug_plan_dir = Path(debug_plan_dir) if debug_plan_dir else None
        self._tracker = FrozenPathTracker(self._tracker_config())
        self._acquire_count = 0
        self._rejoin_confirm_count = 0
        self._blend_start_s = 0.0
        self._blend_from = Command.zero("rejoin_blend_start")
        self._last_bypass_command = Command.zero("not_active")
        self._last_yaw_rate_deg_s = 0.0
        self._last_tracking = self._tracker.last_result
        self._last_obstacle_bearing: float | None = None
        self._fov_exit_armed = False
        self._fov_missing_count = 0
        self._fov_exit_confirmed = False
        self._fov_exit_progress = 0.0
        self._plan_retry_count = 0
        self._path_min_clearance_cm: float | None = None
        self._plan_generation_count = 0
        self._planner_elapsed_us = 0.0
        self._cluster_point_count = 0
        self._cluster_centroid = np.array((math.nan, math.nan), dtype=float)
        self._local_x_cm = 0.0
        self._local_y_cm = 0.0
        self._local_yaw_deg = 0.0
        self._road_found = False
        self._road_confidence = 0.0
        self._road_pixel_error = math.inf

    @property
    def active(self) -> bool:
        return self.state not in (ContourBypassState.NORMAL, ContourBypassState.ACQUIRE)

    @property
    def plan_generation_count(self) -> int:
        """Number of paths generated; useful for proving encounter-level freezing."""
        return self._plan_generation_count

    def update(
        self,
        *,
        visual_desired: Command,
        perception: Any,
        radar_field: RadarObstacleField,
        radar_fresh: bool,
        now_s: float,
        dt_s: float,
    ) -> Command:
        started_ns = time.perf_counter_ns()
        try:
            self._read_road(perception)
            if self.state in (ContourBypassState.NORMAL, ContourBypassState.ACQUIRE):
                return self._update_activation(
                    visual_desired=visual_desired,
                    radar_field=radar_field,
                    radar_fresh=radar_fresh,
                    now_s=now_s,
                    dt_s=dt_s,
                )
            if self.state == ContourBypassState.PLAN_FAILED:
                if self._activation_cluster(radar_field, radar_fresh) is None:
                    self._reset_encounter("failed_obstacle_cleared")
                    return visual_desired
                return Command.zero("contour_plan_failed")
            if self.state == ContourBypassState.REJOIN_BLEND:
                return self._update_blend(visual_desired, now_s)
            if self.state == ContourBypassState.WAIT_VISUAL:
                if self._road_rejoin_confirmed():
                    self._begin_blend(now_s, self._last_bypass_command, "visual_reacquired")
                    return self._update_blend(visual_desired, now_s)
                return Command.zero("wait_visual_road")

            self._observe_fov_exit(radar_field, radar_fresh)
            tracking = self._tracker.update(
                local_x_cm=self._local_x_cm,
                local_y_cm=self._local_y_cm,
                local_yaw_deg=self._local_yaw_deg,
                dt_s=dt_s,
            )
            self._last_tracking = tracking
            yaw_rate = self._yaw_command(tracking, dt_s)
            command = Command(
                tracking.command.vx_cm_s,
                tracking.command.vy_cm_s,
                0.0,
                yaw_rate,
                (
                    "frozen_contour_return"
                    if self.state == ContourBypassState.RETURN_TO_ROAD
                    else "frozen_contour_bypass"
                ),
            )
            self._last_bypass_command = command
            if self.state == ContourBypassState.RETURN_TO_ROAD and tracking.complete:
                if self._road_rejoin_confirmed():
                    self._begin_blend(now_s, command, "path_complete_visual_confirmed")
                    return self._update_blend(visual_desired, now_s)
                self._transition(ContourBypassState.WAIT_VISUAL, "path_complete_visual_unavailable")
                self._last_bypass_command = Command.zero("path_complete")
                return self._last_bypass_command
            if self.state == ContourBypassState.RETURN_TO_ROAD:
                self._road_rejoin_confirmed()
            return command
        finally:
            self._planner_elapsed_us = (time.perf_counter_ns() - started_ns) / 1000.0

    def report_applied_command(self, command: Command, dt_s: float) -> None:
        """Dead-reckon only from the final command actually selected by main."""
        if self.encounter_frame is None or self.state == ContourBypassState.NORMAL:
            return
        dt = max(0.0, min(0.5, float(dt_s)))
        if dt <= 0.0:
            return
        yaw_rad = math.radians(self._local_yaw_deg)
        vx_world = math.cos(yaw_rad) * command.vx_cm_s - math.sin(yaw_rad) * command.vy_cm_s
        vy_world = math.sin(yaw_rad) * command.vx_cm_s + math.cos(yaw_rad) * command.vy_cm_s
        self._local_x_cm += vx_world * dt
        self._local_y_cm += vy_world * dt
        self._local_yaw_deg = _wrap_deg(
            self._local_yaw_deg + command.yaw_rate_deg_s * dt
        )

    def diagnostics(self) -> dict[str, Any]:
        tracking = self._last_tracking
        controls = (
            [list(point) for point in self.path_control_points]
            if self.path_control_points is not None
            else None
        )
        return {
            "state": self.state.value,
            "previous_state": self.previous_state.value,
            "transition_reason": self.transition_reason,
            "encounter_id": self.encounter_id,
            "bypass_side": self.locked_bypass_side,
            "bypass_side_name": (
                "left" if self.locked_bypass_side == 1 else "right" if self.locked_bypass_side == -1 else None
            ),
            "cluster_point_count": self._cluster_point_count,
            "cluster_centroid_x": _finite_or_none(self._cluster_centroid[0]),
            "cluster_centroid_y": _finite_or_none(self._cluster_centroid[1]),
            "control_points": controls,
            "path_frozen": self.path_samples is not None,
            "plan_generation_count": self._plan_generation_count,
            "path_progress": tracking.progress,
            "nearest_path_index": tracking.nearest_index,
            "local_x": self._local_x_cm,
            "local_y": self._local_y_cm,
            "local_yaw": self._local_yaw_deg,
            "current_target_x": tracking.target_x_cm,
            "current_target_y": tracking.target_y_cm,
            "path_speed": tracking.speed_cm_s,
            "command_vx": self._last_bypass_command.vx_cm_s,
            "command_vy": self._last_bypass_command.vy_cm_s,
            "command_yaw": self._last_bypass_command.yaw_rate_deg_s,
            "last_obstacle_bearing": self._last_obstacle_bearing,
            "fov_exit_armed": self._fov_exit_armed,
            "fov_missing_count": self._fov_missing_count,
            "fov_exit_confirmed": self._fov_exit_confirmed,
            "road_found": self._road_found,
            "road_confidence": self._road_confidence,
            "road_pixel_error": _finite_or_none(self._road_pixel_error),
            "rejoin_confirm_count": self._rejoin_confirm_count,
            "plan_retry_count": self._plan_retry_count,
            "path_min_obstacle_clearance_cm": self._path_min_clearance_cm,
            "planner_elapsed_us": self._planner_elapsed_us,
            "safety_arbiter_enabled": False,
        }

    def save_debug_plan(self, output_path: str | Path) -> Path:
        if (
            self._frozen_occupancy is None
            or self._frozen_contour is None
            or self.path_samples is None
            or self.path_control_points is None
            or self.locked_bypass_side is None
        ):
            raise RuntimeError("no successful frozen bypass plan is available")
        return render_plan_debug(
            output_path,
            occupancy=self._frozen_occupancy,
            contour_cm=self._frozen_contour,
            control_points_cm=np.asarray(self.path_control_points, dtype=float),
            path_cm=self.path_samples,
            bypass_side=self.locked_bypass_side,
            minimum_clearance_cm=float(self._path_min_clearance_cm or math.nan),
        )

    def _tracker_config(self) -> PathTrackerConfig:
        cfg = self.config
        return PathTrackerConfig(
            lookahead_distance_cm=cfg.path_lookahead_cm,
            bypass_nominal_speed_cm_s=cfg.bypass_nominal_speed_cm_s,
            bypass_max_speed_cm_s=cfg.bypass_max_speed_cm_s,
            bypass_min_speed_cm_s=cfg.bypass_min_speed_cm_s,
            max_vy_cm_s=cfg.max_vy_cm_s,
            max_planar_accel_cm_s2=cfg.max_planar_accel_cm_s2,
            max_planar_decel_cm_s2=cfg.max_planar_decel_cm_s2,
            path_complete_progress=cfg.path_complete_progress,
            path_complete_distance_cm=cfg.path_complete_distance_cm,
        )

    def _geometry_config(self) -> GeometryConfig:
        cfg = self.config
        return GeometryConfig(
            grid_resolution_cm=cfg.grid_resolution_cm,
            grid_x_min_cm=cfg.grid_x_min_cm,
            grid_x_max_cm=cfg.grid_x_max_cm,
            grid_y_min_cm=cfg.grid_y_min_cm,
            grid_y_max_cm=cfg.grid_y_max_cm,
            inflation_radius_cm=cfg.inflation_radius_cm,
            trajectory_extra_margin_cm=cfg.trajectory_extra_margin_cm,
            envelope_bin_cm=cfg.envelope_bin_cm,
            bezier_sample_count=cfg.bezier_sample_count,
        )

    def _update_activation(
        self,
        *,
        visual_desired: Command,
        radar_field: RadarObstacleField,
        radar_fresh: bool,
        now_s: float,
        dt_s: float,
    ) -> Command:
        cluster = self._activation_cluster(radar_field, radar_fresh)
        if cluster is None:
            self._acquire_count = 0
            if self.state == ContourBypassState.ACQUIRE:
                self._transition(ContourBypassState.NORMAL, "activation_lost")
            return visual_desired
        self._acquire_count += 1
        if self.state == ContourBypassState.NORMAL:
            self._transition(ContourBypassState.ACQUIRE, "activation_candidate")
        if self._acquire_count < max(1, self.config.activation_frames):
            return visual_desired
        self._transition(ContourBypassState.PLAN, "activation_confirmed")
        if not self._plan_once(cluster):
            self._transition(ContourBypassState.PLAN_FAILED, "collision_free_plan_unavailable")
            self._last_bypass_command = Command.zero("contour_plan_failed")
            return self._last_bypass_command
        self._transition(ContourBypassState.FOLLOW_BYPASS, "frozen_path_generated")
        tracking = self._tracker.update(
            local_x_cm=0.0,
            local_y_cm=0.0,
            local_yaw_deg=0.0,
            dt_s=dt_s,
        )
        self._last_tracking = tracking
        command = Command(
            tracking.command.vx_cm_s,
            tracking.command.vy_cm_s,
            0.0,
            self._slew_yaw(0.0, dt_s),
            "frozen_contour_bypass",
        )
        self._last_bypass_command = command
        return command

    def _activation_cluster(
        self, radar_field: RadarObstacleField, radar_fresh: bool
    ) -> np.ndarray | None:
        if not radar_fresh:
            return None
        clusters = self._stable_clusters(radar_field)
        candidates: list[np.ndarray] = []
        for cluster in clusters:
            ranges = np.linalg.norm(cluster, axis=1)
            if (
                float(np.min(ranges)) <= self.config.activation_radius_cm
                and float(np.min(np.abs(cluster[:, 1]))) <= self.config.activation_corridor_half_width_cm
            ):
                candidates.append(cluster)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda cluster: (
                float(np.min(np.abs(cluster[:, 1]))),
                float(np.min(np.linalg.norm(cluster, axis=1))),
                -len(cluster),
            ),
        )

    def _radar_points(self, radar_field: RadarObstacleField) -> np.ndarray:
        points = np.asarray(
            getattr(radar_field, "points_body_cm", np.empty((0, 2))), dtype=float
        ).reshape(-1, 2)
        if not len(points):
            return points
        finite = np.all(np.isfinite(points), axis=1)
        ranges = np.linalg.norm(points, axis=1)
        bearings = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        half_fov = 0.5 * self.config.radar_fov_deg
        valid = (
            finite
            & (points[:, 0] > self.config.radar_min_x_cm)
            & (ranges < self.config.radar_max_range_cm)
            & (np.abs(bearings) <= half_fov)
        )
        return points[valid]

    def _stable_clusters(self, radar_field: RadarObstacleField) -> list[np.ndarray]:
        points = self._radar_points(radar_field)
        if not len(points):
            return []
        remaining = set(range(len(points)))
        clusters: list[np.ndarray] = []
        while remaining:
            seed = remaining.pop()
            component = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                linked = {
                    index
                    for index in remaining
                    if abs(points[index, 0] - points[current, 0]) <= self.config.cluster_radius_x_cm
                    and abs(points[index, 1] - points[current, 1]) <= self.config.cluster_radius_y_cm
                }
                remaining.difference_update(linked)
                component.update(linked)
                frontier.extend(linked)
            if len(component) >= self.config.min_cluster_points:
                clusters.append(points[sorted(component)].copy())
        return clusters

    def _plan_once(self, cluster: np.ndarray) -> bool:
        # This is the sole method allowed to create a path.  It is invoked only
        # on the ACQUIRE -> PLAN transition, never from active-path updates.
        self.encounter_id += 1
        self.encounter_frame = (0.0, 0.0, 0.0)
        self._local_x_cm = self._local_y_cm = self._local_yaw_deg = 0.0
        frozen_cluster = np.asarray(cluster, dtype=float).reshape(-1, 2).copy()
        frozen_cluster.setflags(write=False)
        self.frozen_cluster_points = frozen_cluster
        self._cluster_point_count = len(frozen_cluster)
        self._cluster_centroid = np.mean(frozen_cluster, axis=0)
        geometry_cfg = self._geometry_config()
        occupancy = build_inflated_occupancy(frozen_cluster, geometry_cfg)
        contours = extract_external_contours(occupancy)
        contour = select_cluster_contour(contours, frozen_cluster)
        if len(contour) < 2:
            return False
        side = choose_bypass_side(
            frozen_cluster,
            side_deadband_cm=self.config.side_deadband_cm,
            occupancy=occupancy,
        )
        envelope = extract_side_envelope(
            contour, side, bin_cm=self.config.envelope_bin_cm
        )
        for retry in range(self.config.max_plan_retries + 1):
            controls = build_quintic_control_points(
                envelope,
                frozen_cluster,
                side,
                extra_margin_cm=self.config.trajectory_extra_margin_cm,
                outward_retry_cm=retry * self.config.plan_retry_step_cm,
            )
            samples = sample_quintic_bezier(controls, self.config.bezier_sample_count)
            clearance = path_minimum_clearance_cm(samples, frozen_cluster)
            if path_is_collision_free(
                samples,
                occupancy,
                minimum_clearance_cm=self.config.inflation_radius_cm,
            ):
                frozen_samples = samples.copy()
                frozen_samples.setflags(write=False)
                self.path_samples = frozen_samples
                self.path_control_points = tuple(
                    (float(point[0]), float(point[1])) for point in controls
                )
                self.locked_bypass_side = side
                self._frozen_occupancy = occupancy
                self._frozen_contour = contour.copy()
                self._plan_retry_count = retry
                self._path_min_clearance_cm = clearance
                self._plan_generation_count += 1
                self._tracker.set_path(frozen_samples)
                self._last_obstacle_bearing = math.degrees(
                    math.atan2(self._cluster_centroid[1], self._cluster_centroid[0])
                )
                self._fov_exit_armed = False
                self._fov_missing_count = 0
                self._fov_exit_confirmed = False
                if self._debug_plan_dir is not None:
                    self.save_debug_plan(
                        self._debug_plan_dir / f"bypass_plan_{self.encounter_id:03d}.png"
                    )
                logger.info(
                    "[CONTOUR-BYPASS][EVENT] path frozen encounter={} side={} points={} retries={} min_clearance={:.1f}",
                    self.encounter_id,
                    "left" if side > 0 else "right",
                    len(frozen_cluster),
                    retry,
                    clearance,
                )
                return True
        self._plan_retry_count = self.config.max_plan_retries
        self._path_min_clearance_cm = clearance
        logger.error(
            "[CONTOUR-BYPASS][EVENT] planning failed encounter={} retries={} min_clearance={:.1f}",
            self.encounter_id,
            self.config.max_plan_retries,
            clearance,
        )
        return False

    def _observe_fov_exit(
        self, radar_field: RadarObstacleField, radar_fresh: bool
    ) -> None:
        if self.state not in (
            ContourBypassState.FOLLOW_BYPASS,
            ContourBypassState.FOV_EXIT_CONFIRM,
        ):
            return
        if not radar_fresh:
            # A short stale interval must not cause go/stop oscillation or be
            # mistaken for the expected geometric FOV exit.
            return
        clusters = self._stable_clusters(radar_field)
        bearing: float | None = None
        if clusters:
            expected = self._predicted_obstacle_bearing()
            references = [expected]
            if self._last_obstacle_bearing is not None:
                references.append(self._last_obstacle_bearing)
            candidates = [
                (
                    min(
                        abs(
                            _wrap_deg(
                                math.degrees(
                                    math.atan2(np.mean(c[:, 1]), np.mean(c[:, 0]))
                                )
                                - reference
                            )
                        )
                        for reference in references
                    ),
                    math.degrees(math.atan2(np.mean(c[:, 1]), np.mean(c[:, 0]))),
                )
                for c in clusters
            ]
            delta, candidate_bearing = min(candidates, key=lambda item: item[0])
            if delta <= self.config.obstacle_association_bearing_deg:
                bearing = candidate_bearing
        if bearing is not None:
            self._last_obstacle_bearing = bearing
            self._fov_missing_count = 0
            if abs(bearing) >= self.config.fov_exit_arm_deg:
                self._fov_exit_armed = True
                if self.state == ContourBypassState.FOLLOW_BYPASS:
                    self._transition(ContourBypassState.FOV_EXIT_CONFIRM, "obstacle_reached_fov_edge")
            return
        if self._fov_exit_armed:
            self._fov_missing_count += 1
            if self._fov_missing_count >= self.config.fov_exit_missing_frames:
                self._fov_exit_confirmed = True
                self._fov_exit_progress = self._last_tracking.progress
                self._transition(ContourBypassState.RETURN_TO_ROAD, "expected_fov_exit")
                logger.info(
                    "[CONTOUR-BYPASS][EVENT] FOV_EXIT confirmed bearing={} missing_frames={}",
                    self._last_obstacle_bearing,
                    self._fov_missing_count,
                )

    def _predicted_obstacle_bearing(self) -> float:
        if self.frozen_cluster_points is None:
            return self._last_obstacle_bearing or 0.0
        centroid = np.mean(self.frozen_cluster_points, axis=0)
        delta_world = centroid - np.array((self._local_x_cm, self._local_y_cm))
        yaw = math.radians(self._local_yaw_deg)
        body_x = math.cos(yaw) * delta_world[0] + math.sin(yaw) * delta_world[1]
        body_y = -math.sin(yaw) * delta_world[0] + math.cos(yaw) * delta_world[1]
        return math.degrees(math.atan2(float(body_y), float(body_x)))

    def _yaw_command(self, tracking: PathTrackingResult, dt_s: float) -> float:
        if self.state in (
            ContourBypassState.FOLLOW_BYPASS,
            ContourBypassState.FOV_EXIT_CONFIRM,
        ):
            error = _wrap_deg(-self._local_yaw_deg)
            desired = float(
                np.clip(
                    self.config.yaw_hold_kp * error,
                    -self.config.yaw_hold_max_rate_deg_s,
                    self.config.yaw_hold_max_rate_deg_s,
                )
            )
            return self._slew_yaw(desired, dt_s)
        denominator = max(1e-6, 1.0 - self._fov_exit_progress)
        alpha = float(np.clip((tracking.progress - self._fov_exit_progress) / denominator, 0.0, 1.0))
        target_heading = _wrap_deg(alpha * _wrap_deg(tracking.tangent_heading_deg))
        error = _wrap_deg(target_heading - self._local_yaw_deg)
        desired = float(
            np.clip(
                self.config.return_yaw_kp * error,
                -self.config.return_max_yaw_rate_deg_s,
                self.config.return_max_yaw_rate_deg_s,
            )
        )
        return self._slew_yaw(desired, dt_s)

    def _slew_yaw(self, desired: float, dt_s: float) -> float:
        maximum_delta = self.config.max_yaw_accel_deg_s2 * max(0.0, float(dt_s))
        delta = float(np.clip(desired - self._last_yaw_rate_deg_s, -maximum_delta, maximum_delta))
        self._last_yaw_rate_deg_s += delta
        return self._last_yaw_rate_deg_s

    def _read_road(self, perception: Any) -> None:
        self._road_found = bool(getattr(perception, "is_road_found", False))
        self._road_confidence = float(getattr(perception, "confidence", 0.0) or 0.0)
        error = getattr(perception, "corrected_pixel_error", math.inf)
        try:
            self._road_pixel_error = float(error)
        except (TypeError, ValueError):
            self._road_pixel_error = math.inf

    def _road_rejoin_confirmed(self) -> bool:
        usable = bool(
            self._road_found
            and self._road_confidence >= self.config.rejoin_min_confidence
            and abs(self._road_pixel_error) <= self.config.rejoin_pixel_deadband_px
        )
        self._rejoin_confirm_count = self._rejoin_confirm_count + 1 if usable else 0
        return self._rejoin_confirm_count >= self.config.rejoin_confirm_frames

    def _begin_blend(self, now_s: float, command: Command, reason: str) -> None:
        self._blend_start_s = float(now_s)
        self._blend_from = command
        self._transition(ContourBypassState.REJOIN_BLEND, reason)

    def _update_blend(self, visual_desired: Command, now_s: float) -> Command:
        duration = max(1e-6, self.config.rejoin_blend_s)
        ratio = float(np.clip((now_s - self._blend_start_s) / duration, 0.0, 1.0))
        alpha = ratio * ratio * (3.0 - 2.0 * ratio)
        if ratio >= 1.0:
            self._reset_encounter("rejoin_blend_complete")
            return visual_desired
        command = Command(
            vx_cm_s=_lerp(self._blend_from.vx_cm_s, visual_desired.vx_cm_s, alpha),
            vy_cm_s=_lerp(self._blend_from.vy_cm_s, visual_desired.vy_cm_s, alpha),
            vz_cm_s=_lerp(self._blend_from.vz_cm_s, visual_desired.vz_cm_s, alpha),
            yaw_rate_deg_s=_lerp(
                self._blend_from.yaw_rate_deg_s,
                visual_desired.yaw_rate_deg_s,
                alpha,
            ),
            reason="contour_rejoin_blend",
        )
        self._last_bypass_command = command
        return command

    def _transition(self, state: ContourBypassState, reason: str) -> None:
        if state == self.state:
            return
        old = self.state
        self.previous_state = old
        self.state = state
        self.transition_reason = reason
        logger.info(
            "[CONTOUR-BYPASS][EVENT] {} -> {} reason={}",
            old.name,
            state.name,
            reason,
        )

    def _reset_encounter(self, reason: str) -> None:
        self._transition(ContourBypassState.NORMAL, reason)
        self.encounter_frame = None
        self.locked_bypass_side = None
        self.path_samples = None
        self.path_control_points = None
        self.frozen_cluster_points = None
        self._frozen_occupancy = None
        self._frozen_contour = None
        self._tracker.reset()
        self._acquire_count = 0
        self._rejoin_confirm_count = 0
        self._last_yaw_rate_deg_s = 0.0
        self._fov_exit_armed = False
        self._fov_missing_count = 0
        self._fov_exit_confirmed = False


def _wrap_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _lerp(start: float, end: float, alpha: float) -> float:
    return float(start) + (float(end) - float(start)) * float(alpha)


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


__all__ = [
    "ContourBypassConfig",
    "ContourBypassState",
    "ContourTrajectoryBypassPlanner",
]
