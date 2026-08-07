"""Static tube bypass using the physical front-half-plane radar returns.

Body coordinates are +X forward and +Y left.  During an encounter vision owns
only the path-tangent yaw command; forward speed is fixed at 60 percent of the
isolated visual limit and lateral motion is derived solely from radar clearance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math

import numpy as np

from FlightController.Solutions.Safety import Command, RadarObstacleField


class StaticRouteBypassState(str, Enum):
    NORMAL = "normal"
    DIVERGE_LEFT = "diverge_left"
    DIVERGE_RIGHT = "diverge_right"
    PASS_FORWARD_LEFT = "pass_forward_left"
    PASS_FORWARD_RIGHT = "pass_forward_right"
    SIDE_PASS_CONFIRM = "side_pass_confirm"
    CLEARANCE_RUN = "clearance_run"
    WAIT_VISUAL = "wait_visual"
    BLEND_BACK = "blend_back"
    PATH_LOST_HOLD = "path_lost_hold"
    TRACK_LOST_HOLD = "track_lost_hold"
    FAILSAFE_STOP = "failsafe_stop"
    TIMEOUT_STOP = "timeout_stop"


@dataclass(frozen=True)
class StaticRouteBypassConfig:
    # User-fixed geometry and speed policy.
    front_fov_deg: float = 180.0
    visual_max_vx_cm_s: float = 14.0
    avoidance_forward_ratio: float = 0.60
    tube_radius_cm: float = 15.0

    # Existing experiment geometry.
    road_half_width_cm: float = 25.0
    intrusion_half_width_cm: float = 75.0
    min_x_cm: float = 10.0
    lookahead_cm: float = 180.0
    min_confidence: float = 0.4
    min_cluster_points: int = 3
    cluster_grid_cm: float = 10.0
    cluster_radius_x_cm: float = 22.0
    cluster_radius_y_cm: float = 18.0
    association_radius_cm: float = 50.0
    side_deadband_cm: float = 5.0
    center_obstacle_default_bypass_side: str = "right"

    # UNVERIFIED_TUNING.
    target_surface_clearance_cm: float = 85.0
    reshift_surface_clearance_cm: float = 75.0
    max_outward_vy_cm_s: float = 8.0
    lateral_kp_s: float = 0.20
    ramp_in_s: float = 0.8
    lateral_decay_s: float = 1.0
    blend_back_s: float = 2.0
    edge_arm_deg: float = 80.0
    edge_missing_frames: int = 3
    activation_frames: int = 2
    clearance_frames: int = 3
    pass_complete_frames: int = 3
    rear_margin_cm: float = 20.0
    translation_credit_ratio: float = 0.70
    track_lost_hold_s: float = 1.0
    max_encounter_s: float = 40.0
    static_model_tolerance_cm: float = 25.0
    static_model_bad_frames: int = 3
    nominal_dt_s: float = 0.1

    @property
    def avoidance_vx_cm_s(self) -> float:
        return self.visual_max_vx_cm_s * self.avoidance_forward_ratio

    @property
    def half_fov_deg(self) -> float:
        return self.front_fov_deg / 2.0


@dataclass(frozen=True)
class StaticTubeObservation:
    points: np.ndarray
    surface_x_cm: float
    surface_y_cm: float
    surface_max_x_cm: float
    center_x_cm: float
    center_y_cm: float
    bearing_deg: float
    lateral_clearance_cm: float
    obstacle_side: int
    point_count: int


class StaticRouteBypassPlanner:
    """Bypass one isolated static tube while keeping vision-owned path yaw."""

    ACTIVE_STATES = {
        StaticRouteBypassState.DIVERGE_LEFT,
        StaticRouteBypassState.DIVERGE_RIGHT,
        StaticRouteBypassState.PASS_FORWARD_LEFT,
        StaticRouteBypassState.PASS_FORWARD_RIGHT,
        StaticRouteBypassState.SIDE_PASS_CONFIRM,
        StaticRouteBypassState.CLEARANCE_RUN,
        StaticRouteBypassState.WAIT_VISUAL,
        StaticRouteBypassState.BLEND_BACK,
        StaticRouteBypassState.PATH_LOST_HOLD,
        StaticRouteBypassState.TRACK_LOST_HOLD,
    }

    def __init__(self, config: StaticRouteBypassConfig | None = None) -> None:
        self.config = config or StaticRouteBypassConfig()
        self.state = StaticRouteBypassState.NORMAL
        self.previous_state = StaticRouteBypassState.NORMAL
        self.transition_reason = "initialized"
        self.encounter_id = 0
        self._locked_side: int | None = None
        self._intrusion_count = 0
        self._clearance_count = 0
        self._missing_count = 0
        self._pass_complete_count = 0
        self._forward_decrease_count = 0
        self._static_model_bad_count = 0
        self._encounter_started_s: float | None = None
        self._phase_started_s: float | None = None
        self._hold_started_s: float | None = None
        self._last_update_s: float | None = None
        self._last_seen_s: float | None = None
        self._last_observed_x_cm: float | None = None
        self._resume_state: StaticRouteBypassState | None = None
        self._observation: StaticTubeObservation | None = None
        self._predicted_center = np.empty((0,), dtype=float)
        self._edge_armed = False
        self._blend_started_s: float | None = None
        self._credited_translation_cm = 0.0
        self._credited_yaw_deg = 0.0
        self._last_applied = False
        self._last_radar_forward_clear = False
        self._last_outward_vy_cm_s = 0.0

    @property
    def active_bypass_side(self) -> int | None:
        return self._locked_side

    @property
    def target_y_cm(self) -> float | None:
        if self._locked_side is None:
            return None
        return self._locked_side * self.config.target_surface_clearance_cm

    def update(
        self,
        *,
        desired: Command,
        perception,
        radar_field: RadarObstacleField,
        now_s: float,
    ) -> Command:
        now = float(now_s)
        dt = self._step_dt(now)
        road_usable = self._road_usable(perception, desired)
        observation = self._observe(radar_field, tracking=self.state != StaticRouteBypassState.NORMAL)
        self._last_radar_forward_clear = radar_field.nearest_forward_obstacle_cm() is None
        self._accept_observation(observation, now)

        if self.state in {StaticRouteBypassState.FAILSAFE_STOP, StaticRouteBypassState.TIMEOUT_STOP}:
            return self._stop_command(desired, self.state.value)

        if self.state == StaticRouteBypassState.NORMAL:
            if not road_usable:
                self._intrusion_count = 0
                return desired
            if observation is None:
                self._intrusion_count = 0
                return desired
            self._intrusion_count += 1
            if self._intrusion_count < max(1, int(self.config.activation_frames)):
                return desired
            self._start_encounter(observation, radar_field, now)
            return self._avoidance_command(desired, now)

        if self._encounter_timed_out(now):
            self._transition(StaticRouteBypassState.TIMEOUT_STOP, "encounter_timeout", now)
            return self._stop_command(desired, "timeout_stop")

        if not road_usable:
            if self.state != StaticRouteBypassState.PATH_LOST_HOLD:
                self._resume_state = self.state
                self._transition(StaticRouteBypassState.PATH_LOST_HOLD, "path_tangent_lost", now)
            return self._stop_command(desired, "path_lost_hold")
        if self.state == StaticRouteBypassState.PATH_LOST_HOLD:
            resume = self._resume_state or self._diverge_state()
            self._transition(resume, "path_tangent_recovered", now)

        if self._static_model_bad_count >= max(1, int(self.config.static_model_bad_frames)):
            self._transition(StaticRouteBypassState.FAILSAFE_STOP, "static_model_mismatch", now)
            return self._stop_command(desired, "static_model_mismatch")

        if self.state == StaticRouteBypassState.TRACK_LOST_HOLD:
            if observation is not None:
                resume = self._resume_state or self._diverge_state()
                self._transition(resume, "track_reacquired", now)
            elif self._hold_started_s is not None and now - self._hold_started_s > self.config.track_lost_hold_s:
                self._transition(StaticRouteBypassState.FAILSAFE_STOP, "track_reacquire_timeout", now)
                return self._stop_command(desired, "track_reacquire_timeout")
            else:
                return self._stop_command(desired, "track_lost_hold")

        if self.state in {StaticRouteBypassState.DIVERGE_LEFT, StaticRouteBypassState.DIVERGE_RIGHT}:
            if observation is None:
                return self._enter_track_hold(desired, now, "unexpected_loss_during_diverge")
            if observation.lateral_clearance_cm >= self.config.target_surface_clearance_cm:
                self._clearance_count += 1
            else:
                self._clearance_count = 0
            if self._clearance_count >= max(1, int(self.config.clearance_frames)):
                self._transition(self._pass_state(), "target_lateral_clearance", now)
            return self._avoidance_command(desired, now)

        if self.state in {StaticRouteBypassState.PASS_FORWARD_LEFT, StaticRouteBypassState.PASS_FORWARD_RIGHT}:
            if observation is None:
                return self._enter_track_hold(desired, now, "unexpected_loss_before_edge")
            if observation.lateral_clearance_cm < self.config.reshift_surface_clearance_cm:
                self._transition(self._diverge_state(), "lateral_clearance_below_hysteresis", now)
                return self._avoidance_command(desired, now)
            if self._edge_ready(observation):
                self._edge_armed = True
                self._transition(StaticRouteBypassState.SIDE_PASS_CONFIRM, "entered_80_to_90_degree_band", now)
            return self._avoidance_command(desired, now)

        if self.state == StaticRouteBypassState.SIDE_PASS_CONFIRM:
            if observation is not None:
                self._missing_count = 0
                if not self._on_locked_side(observation) or abs(observation.bearing_deg) < self.config.edge_arm_deg:
                    self._edge_armed = False
                    self._transition(self._pass_state(), "obstacle_returned_from_side_band", now)
                return self._avoidance_command(desired, now)
            self._missing_count += 1
            if self._edge_armed and self._missing_count >= max(1, int(self.config.edge_missing_frames)):
                self._transition(StaticRouteBypassState.CLEARANCE_RUN, "expected_90_degree_edge_exit", now)
                return self._avoidance_command(desired, now)
            return self._avoidance_command(desired, now)

        if self.state == StaticRouteBypassState.CLEARANCE_RUN:
            if observation is not None and observation.center_x_cm > 0.0:
                self._edge_armed = abs(observation.bearing_deg) >= self.config.edge_arm_deg
                self._transition(
                    StaticRouteBypassState.SIDE_PASS_CONFIRM if self._edge_armed else self._pass_state(),
                    "obstacle_reentered_front_half_plane",
                    now,
                )
                return self._avoidance_command(desired, now)
            if self._pass_complete():
                self._pass_complete_count += 1
            else:
                self._pass_complete_count = 0
            if self._pass_complete_count >= max(1, int(self.config.pass_complete_frames)):
                self._transition(StaticRouteBypassState.WAIT_VISUAL, "tube_fully_behind_margin", now)
                return self._stop_command(desired, "wait_visual")
            return self._avoidance_command(desired, now)

        if self.state == StaticRouteBypassState.WAIT_VISUAL:
            if observation is not None and observation.center_x_cm > 0.0:
                self._transition(self._diverge_state(), "obstacle_reappeared_before_handoff", now)
                return self._avoidance_command(desired, now)
            self._blend_started_s = now
            self._transition(StaticRouteBypassState.BLEND_BACK, "visual_ready", now)
            return self._blend_command(desired, now)

        if self.state == StaticRouteBypassState.BLEND_BACK:
            acquisition = self._observe(radar_field, tracking=False)
            if acquisition is not None:
                self._accept_observation(acquisition, now)
                self._transition(self._diverge_state(), "obstacle_reappeared_during_blend", now)
                return self._avoidance_command(desired, now)
            command = self._blend_command(desired, now)
            assert self._blend_started_s is not None
            if now - self._blend_started_s >= self.config.blend_back_s:
                self.reset("blend_complete")
                return desired
            return command

        self._transition(StaticRouteBypassState.FAILSAFE_STOP, "invalid_state", now)
        return self._stop_command(desired, "invalid_state")

    def report_applied_command(self, final_command: Command, dt_s: float, applied: bool) -> None:
        self._last_applied = bool(applied)
        if not applied or self._predicted_center.size != 2 or self.state not in self.ACTIVE_STATES:
            return
        dt = max(0.0, min(0.5, float(dt_s)))
        if dt <= 0.0:
            return
        cfg = self.config
        velocity = np.asarray([final_command.vx_cm_s, final_command.vy_cm_s], dtype=float)
        credited = cfg.translation_credit_ratio * velocity * dt
        theta = math.radians(float(final_command.yaw_rate_deg_s) * dt)
        cosine, sine = math.cos(-theta), math.sin(-theta)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
        self._predicted_center = rotation @ (self._predicted_center - credited)
        self._credited_translation_cm += float(np.linalg.norm(credited))
        self._credited_yaw_deg += abs(float(final_command.yaw_rate_deg_s) * dt)

    def reset(self, reason: str = "manual_reset") -> None:
        previous = self.state
        self.state = StaticRouteBypassState.NORMAL
        self.previous_state = previous
        self.transition_reason = reason
        self._locked_side = None
        self._intrusion_count = 0
        self._clearance_count = 0
        self._missing_count = 0
        self._pass_complete_count = 0
        self._forward_decrease_count = 0
        self._static_model_bad_count = 0
        self._encounter_started_s = None
        self._phase_started_s = None
        self._hold_started_s = None
        self._last_seen_s = None
        self._last_observed_x_cm = None
        self._resume_state = None
        self._observation = None
        self._predicted_center = np.empty((0,), dtype=float)
        self._edge_armed = False
        self._blend_started_s = None
        self._last_outward_vy_cm_s = 0.0

    def diagnostics(self) -> dict[str, object]:
        observation = self._observation
        predicted = self._predicted_center if self._predicted_center.size == 2 else None
        return {
            "planner": "static_route",
            "state": self.state.value,
            "previous_state": self.previous_state.value,
            "transition_reason": self.transition_reason,
            "encounter_id": self.encounter_id,
            "active_bypass_side": _side_name(self._locked_side),
            "side_locked": self._locked_side is not None,
            "front_fov_deg": self.config.front_fov_deg,
            "visual_lateral_suppressed": self.state != StaticRouteBypassState.NORMAL,
            "visual_vy_cm_s": None,
            "radar_vy_cm_s": self._radar_vy(observation, self._last_update_s or 0.0),
            "avoidance_vx_cm_s": self.config.avoidance_vx_cm_s,
            "observation_valid": observation is not None,
            "cluster_point_count": 0 if observation is None else observation.point_count,
            "obstacle_bearing_deg": None if observation is None else observation.bearing_deg,
            "obstacle_surface_clearance_cm": None if observation is None else observation.lateral_clearance_cm,
            "obstacle_surface_x_cm": None if observation is None else observation.surface_x_cm,
            "obstacle_center_x_cm": None if observation is None else observation.center_x_cm,
            "obstacle_center_y_cm": None if observation is None else observation.center_y_cm,
            "predicted_center_x_cm": None if predicted is None else float(predicted[0]),
            "predicted_center_y_cm": None if predicted is None else float(predicted[1]),
            "edge_armed": self._edge_armed,
            "edge_missing_count": self._missing_count,
            "pass_complete_count": self._pass_complete_count,
            "credited_translation_cm": self._credited_translation_cm,
            "credited_yaw_deg": self._credited_yaw_deg,
            "last_applied": self._last_applied,
            "static_model_bad_count": self._static_model_bad_count,
            "front_corridor_clear": self._last_radar_forward_clear,
            "config": asdict(self.config),
        }

    def _start_encounter(self, observation: StaticTubeObservation, radar_field: RadarObstacleField, now: float) -> None:
        self.encounter_id += 1
        self._locked_side = self._choose_bypass_side(observation, radar_field)
        self._encounter_started_s = now
        self._credited_translation_cm = 0.0
        self._credited_yaw_deg = 0.0
        self._edge_armed = False
        self._transition(self._diverge_state(), "confirmed_static_obstacle", now)

    def _accept_observation(self, observation: StaticTubeObservation | None, now: float) -> None:
        self._observation = observation
        if observation is None:
            return
        measured = np.asarray([observation.center_x_cm, observation.center_y_cm], dtype=float)
        if self._predicted_center.size == 2:
            error = float(np.linalg.norm(measured - self._predicted_center))
            if self._last_applied and error > self.config.static_model_tolerance_cm:
                self._static_model_bad_count += 1
            else:
                self._static_model_bad_count = 0
        self._predicted_center = measured
        self._last_seen_s = now
        if self._last_observed_x_cm is not None and observation.center_x_cm <= self._last_observed_x_cm + 2.0:
            self._forward_decrease_count += 1
        else:
            self._forward_decrease_count = 0
        self._last_observed_x_cm = observation.center_x_cm

    def _observe(self, radar_field: RadarObstacleField, *, tracking: bool) -> StaticTubeObservation | None:
        points = np.asarray(getattr(radar_field, "points_body_cm", np.empty((0, 2))), dtype=float)
        if points.size == 0:
            return None
        points = points.reshape(-1, 2)
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        mask = (points[:, 0] >= 0.0) & (np.abs(angles) <= self.config.half_fov_deg + 1e-6)
        if tracking:
            mask &= np.linalg.norm(points, axis=1) <= 300.0
        else:
            mask &= (
                (points[:, 0] >= self.config.min_x_cm)
                & (points[:, 0] <= self.config.lookahead_cm)
                & (np.abs(points[:, 1]) <= self.config.intrusion_half_width_cm)
            )
        candidates = points[mask]
        if len(candidates) < self.config.min_cluster_points:
            return None
        cluster = self._associated_cluster(candidates, tracking)
        if cluster is None or len(cluster) < self.config.min_cluster_points:
            return None
        surface = np.median(cluster, axis=0)
        surface_range = float(np.linalg.norm(surface))
        if surface_range <= 1e-6:
            return None
        center = surface + self.config.tube_radius_cm * surface / surface_range
        bearing = math.degrees(math.atan2(float(center[1]), float(center[0])))
        side = 1 if center[1] > self.config.side_deadband_cm else -1 if center[1] < -self.config.side_deadband_cm else 0
        clearance = float(np.percentile(np.abs(cluster[:, 1]), 10.0))
        return StaticTubeObservation(
            points=cluster,
            surface_x_cm=float(surface[0]),
            surface_y_cm=float(surface[1]),
            surface_max_x_cm=float(np.max(cluster[:, 0])),
            center_x_cm=float(center[0]),
            center_y_cm=float(center[1]),
            bearing_deg=bearing,
            lateral_clearance_cm=clearance,
            obstacle_side=side,
            point_count=int(len(cluster)),
        )

    def _associated_cluster(self, candidates: np.ndarray, tracking: bool) -> np.ndarray | None:
        cfg = self.config
        if tracking and self._predicted_center.size == 2:
            distances = np.linalg.norm(candidates - self._predicted_center, axis=1)
            associated = candidates[distances <= cfg.association_radius_cm]
            if len(associated) >= cfg.min_cluster_points:
                return associated
        grid = max(1.0, cfg.cluster_grid_cm)
        cells: dict[tuple[int, int], list[int]] = {}
        for index, point in enumerate(candidates):
            key = (math.floor(float(point[0]) / grid), math.floor(float(point[1]) / grid))
            cells.setdefault(key, []).append(index)
        if not cells:
            return None
        peak = min(
            cells.values(),
            key=lambda indices: (
                -len(indices),
                float(np.median(candidates[indices, 0])),
                abs(float(np.median(candidates[indices, 1]))),
            ),
        )
        seed = np.median(candidates[peak], axis=0)
        return candidates[
            (np.abs(candidates[:, 0] - seed[0]) <= cfg.cluster_radius_x_cm)
            & (np.abs(candidates[:, 1] - seed[1]) <= cfg.cluster_radius_y_cm)
        ]

    def _choose_bypass_side(self, observation: StaticTubeObservation, radar_field: RadarObstacleField) -> int:
        if observation.obstacle_side > 0:
            return -1
        if observation.obstacle_side < 0:
            return 1
        left = radar_field.side_clearance_cm("left")
        right = radar_field.side_clearance_cm("right")
        left_score = math.inf if left is None else left
        right_score = math.inf if right is None else right
        if left_score > right_score:
            return 1
        if right_score > left_score:
            return -1
        return 1 if self.config.center_obstacle_default_bypass_side == "left" else -1

    def _avoidance_command(self, desired: Command, now: float) -> Command:
        vy = 0.0
        if self.state in {StaticRouteBypassState.DIVERGE_LEFT, StaticRouteBypassState.DIVERGE_RIGHT}:
            vy = self._radar_vy(self._observation, now)
            if abs(vy) > 1e-9:
                self._last_outward_vy_cm_s = vy
        elif self.state in {
            StaticRouteBypassState.PASS_FORWARD_LEFT,
            StaticRouteBypassState.PASS_FORWARD_RIGHT,
            StaticRouteBypassState.SIDE_PASS_CONFIRM,
        }:
            if self._observation is not None and self._observation.lateral_clearance_cm < self.config.target_surface_clearance_cm:
                vy = self._radar_vy(self._observation, now)
                if abs(vy) > 1e-9:
                    self._last_outward_vy_cm_s = vy
            elif self.state in {
                StaticRouteBypassState.PASS_FORWARD_LEFT,
                StaticRouteBypassState.PASS_FORWARD_RIGHT,
            } and self._phase_started_s is not None:
                remaining = 1.0 - (
                    now - self._phase_started_s
                ) / max(1e-6, self.config.lateral_decay_s)
                vy = self._last_outward_vy_cm_s * max(0.0, min(1.0, remaining))
        return Command(
            self.config.avoidance_vx_cm_s,
            vy,
            desired.vz_cm_s,
            desired.yaw_rate_deg_s,
            _append_reason(desired.reason, f"static_route:{self.state.value}"),
        )

    def _radar_vy(self, observation: StaticTubeObservation | None, now: float) -> float:
        if observation is None or self._locked_side is None:
            return 0.0
        error = max(0.0, self.config.target_surface_clearance_cm - observation.lateral_clearance_cm)
        magnitude = min(self.config.max_outward_vy_cm_s, self.config.lateral_kp_s * error)
        if error > 0.0:
            magnitude = max(1.0, magnitude)
        if self._phase_started_s is not None and self.state in {StaticRouteBypassState.DIVERGE_LEFT, StaticRouteBypassState.DIVERGE_RIGHT}:
            magnitude *= _smoothstep((now - self._phase_started_s) / max(1e-6, self.config.ramp_in_s))
        return self._locked_side * magnitude

    def _blend_command(self, desired: Command, now: float) -> Command:
        assert self._blend_started_s is not None
        alpha = _smoothstep((now - self._blend_started_s) / max(1e-6, self.config.blend_back_s))
        return Command(
            _lerp(self.config.avoidance_vx_cm_s, desired.vx_cm_s, alpha),
            _lerp(0.0, desired.vy_cm_s, alpha),
            desired.vz_cm_s,
            desired.yaw_rate_deg_s,
            _append_reason(desired.reason, f"static_route:blend_back:alpha={alpha:.2f}"),
        )

    def _edge_ready(self, observation: StaticTubeObservation) -> bool:
        return bool(
            self._on_locked_side(observation)
            and abs(observation.bearing_deg) >= self.config.edge_arm_deg
            and self._forward_decrease_count >= 2
        )

    def _on_locked_side(self, observation: StaticTubeObservation) -> bool:
        if self._locked_side is None:
            return False
        # Bypass left means the obstacle remains on the right, and vice versa.
        return observation.center_y_cm * self._locked_side < 0.0

    def _pass_complete(self) -> bool:
        if self._predicted_center.size != 2:
            return False
        return bool(
            float(self._predicted_center[0])
            + self.config.tube_radius_cm
            + self.config.rear_margin_cm
            <= 0.0
            and self._last_radar_forward_clear
        )

    def _enter_track_hold(self, desired: Command, now: float, reason: str) -> Command:
        self._resume_state = self.state
        self._transition(StaticRouteBypassState.TRACK_LOST_HOLD, reason, now)
        return self._stop_command(desired, "track_lost_hold")

    def _stop_command(self, desired: Command, reason: str) -> Command:
        return Command(0.0, 0.0, desired.vz_cm_s, 0.0, _append_reason(desired.reason, f"static_route:{reason}"))

    def _transition(self, new_state: StaticRouteBypassState, reason: str, now: float) -> None:
        if new_state == self.state:
            return
        self.previous_state = self.state
        self.state = new_state
        self.transition_reason = str(reason)
        self._phase_started_s = float(now)
        if new_state in {StaticRouteBypassState.TRACK_LOST_HOLD, StaticRouteBypassState.PATH_LOST_HOLD}:
            self._hold_started_s = float(now)

    def _diverge_state(self) -> StaticRouteBypassState:
        return StaticRouteBypassState.DIVERGE_LEFT if (self._locked_side or 1) > 0 else StaticRouteBypassState.DIVERGE_RIGHT

    def _pass_state(self) -> StaticRouteBypassState:
        return StaticRouteBypassState.PASS_FORWARD_LEFT if (self._locked_side or 1) > 0 else StaticRouteBypassState.PASS_FORWARD_RIGHT

    def _encounter_timed_out(self, now: float) -> bool:
        return bool(self._encounter_started_s is not None and now - self._encounter_started_s >= self.config.max_encounter_s)

    def _road_usable(self, perception, desired: Command) -> bool:
        return bool(
            perception is not None
            and getattr(perception, "is_road_found", False)
            and float(getattr(perception, "confidence", 0.0)) >= self.config.min_confidence
            and "road_lost" not in desired.reason
            and "visual_unavailable" not in desired.reason
        )

    def _step_dt(self, now: float) -> float:
        if self._last_update_s is None:
            dt = self.config.nominal_dt_s
        else:
            dt = max(0.0, min(0.5, now - self._last_update_s))
        self._last_update_s = now
        return dt


def _side_name(side: int | None) -> str | None:
    if side is None:
        return None
    return "left" if side > 0 else "right"


def _smoothstep(value: float) -> float:
    x = max(0.0, min(1.0, float(value)))
    return x * x * (3.0 - 2.0 * x)


def _lerp(start: float, end: float, alpha: float) -> float:
    return float(start) + (float(end) - float(start)) * float(alpha)


def _append_reason(reason: str, suffix: str) -> str:
    return f"{reason}+{suffix}" if reason else suffix


__all__ = [
    "StaticRouteBypassConfig",
    "StaticRouteBypassPlanner",
    "StaticRouteBypassState",
    "StaticTubeObservation",
]
