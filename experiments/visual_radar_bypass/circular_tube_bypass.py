"""Low-complexity inflated-circle bypass for the isolated tube experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math

import numpy as np

from FlightController.Solutions.Safety import Command, RadarObstacleField


class CircularBypassState(str, Enum):
    NORMAL = "normal"
    ORBIT_LEFT = "orbit_left"
    ORBIT_RIGHT = "orbit_right"
    RETURN_VISUAL = "return_visual"


@dataclass(frozen=True)
class CircularTubeBypassConfig:
    road_half_width_cm: float = 25.0
    intrusion_half_width_cm: float = 75.0
    min_x_cm: float = 10.0
    lookahead_cm: float = 180.0
    min_points: int = 3
    min_circle_fit_points: int = 5
    activate_frames: int = 2
    min_confidence: float = 0.4
    side_deadband_cm: float = 5.0
    center_obstacle_default_bypass_side: str = "right"

    tube_radius_cm: float = 15.0
    safety_radius_cm: float = 75.0
    min_fitted_tube_radius_cm: float = 2.0
    max_fitted_tube_radius_cm: float = 30.0
    max_circle_fit_rms_cm: float = 4.0
    tangent_speed_cm_s: float = 12.0
    radial_kp: float = 0.25
    max_radial_speed_cm_s: float = 4.0
    max_vx_cm_s: float = 14.0
    max_vy_cm_s: float = 10.0
    target_arc_deg: float = 90.0
    max_orbit_s: float = 12.0
    min_orbit_before_visual_return_s: float = 1.0
    visual_return_error_px: float = 50.0
    obstacle_lost_hold_s: float = 0.5

    return_blend_s: float = 1.5
    return_yaw_rate_limit_deg_s: float = 7.0
    nominal_dt_s: float = 0.1

    @property
    def orbit_radius_cm(self) -> float:
        return max(1.0, self.tube_radius_cm + self.safety_radius_cm)

    @property
    def effective_intrusion_half_width_cm(self) -> float:
        return max(
            abs(float(self.road_half_width_cm)),
            abs(float(self.intrusion_half_width_cm)),
            self.orbit_radius_cm,
        )


@dataclass(frozen=True)
class TubeObservation:
    surface_x_cm: float
    surface_y_cm: float
    center_x_cm: float
    center_y_cm: float
    tube_radius_cm: float
    fit_rms_cm: float | None
    circle_fit_used: bool
    point_count: int
    obstacle_side: int


class CircularTubeBypassPlanner:
    """Follow the tangent of one inflated tube circle, then blend to vision.

    Per update this performs one rectangular point mask, medians and constant
    size vector arithmetic.  It does not search a grid or score path candidates.
    """

    def __init__(self, config: CircularTubeBypassConfig | None = None) -> None:
        self.config = config or CircularTubeBypassConfig()
        self.state = CircularBypassState.NORMAL
        self._intrusion_count = 0
        self._active_bypass_side: int | None = None
        self._orbit_started_s: float | None = None
        self._return_started_s: float | None = None
        self._last_seen_s: float | None = None
        self._last_update_s: float | None = None
        self._arc_progress_rad = 0.0
        self._orbit_radius_cm = self.config.orbit_radius_cm
        self._observation: TubeObservation | None = None
        self._last_orbit_command = Command.zero("circular_tube_initial")
        self._exit_reason: str | None = None
        self._visual_return_armed = False

    @property
    def target_y_cm(self) -> float | None:
        if self._active_bypass_side is None:
            return None
        return self._active_bypass_side * self._orbit_radius_cm

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
        now = float(now_s)
        dt = self._step_dt(now)
        observation = self._observe(self._points(radar_field))
        self._observation = observation
        if observation is not None:
            self._intrusion_count += 1
            self._last_seen_s = now
        else:
            self._intrusion_count = 0

        if self.state == CircularBypassState.NORMAL:
            if not self._road_usable(perception):
                return desired
            if self._intrusion_count < max(1, int(self.config.activate_frames)):
                return desired
            assert observation is not None
            self._start_orbit(observation, now)
            self._visual_return_ready(perception, 0.0)
            return self._orbit_command(desired, observation, dt)

        if self.state in {
            CircularBypassState.ORBIT_LEFT,
            CircularBypassState.ORBIT_RIGHT,
        }:
            assert self._orbit_started_s is not None
            orbit_elapsed_s = max(0.0, now - self._orbit_started_s)
            if self._visual_return_ready(perception, orbit_elapsed_s):
                self._start_return(now, "visual_error")
                return self._return_command(desired, now)
            if self._arc_progress_rad >= math.radians(self.config.target_arc_deg):
                self._start_return(now, "arc_complete")
                return self._return_command(desired, now)
            if orbit_elapsed_s >= max(0.0, self.config.max_orbit_s):
                self._start_return(now, "orbit_timeout")
                return self._return_command(desired, now)
            if observation is None:
                if not self._recently_seen(now):
                    self._start_return(now, "obstacle_passed")
                    return self._return_command(desired, now)
                return self._last_orbit_command
            return self._orbit_command(desired, observation, dt)

        if self.state == CircularBypassState.RETURN_VISUAL:
            assert self._return_started_s is not None
            if now - self._return_started_s >= max(0.0, self.config.return_blend_s):
                self.reset()
                return desired
            return self._return_command(desired, now)

        self.reset()
        return desired

    def reset(self) -> None:
        self.state = CircularBypassState.NORMAL
        self._intrusion_count = 0
        self._active_bypass_side = None
        self._orbit_started_s = None
        self._return_started_s = None
        self._last_seen_s = None
        self._arc_progress_rad = 0.0
        self._orbit_radius_cm = self.config.orbit_radius_cm
        self._observation = None
        self._exit_reason = None
        self._visual_return_armed = False

    def diagnostics(self) -> dict[str, object]:
        observation = self._observation
        return {
            "planner": "circular_tube",
            "state": self.state.value,
            "target_y_cm": self.target_y_cm,
            "active_bypass_side": _side_name(self._active_bypass_side),
            "intrusion_count": self._intrusion_count,
            "cluster_point_count": 0 if observation is None else observation.point_count,
            "obstacle_center_x_cm": None if observation is None else observation.center_x_cm,
            "obstacle_center_y_cm": None if observation is None else observation.center_y_cm,
            "fitted_tube_radius_cm": None if observation is None else observation.tube_radius_cm,
            "circle_fit_rms_cm": None if observation is None else observation.fit_rms_cm,
            "circle_fit_used": False if observation is None else observation.circle_fit_used,
            "orbit_radius_cm": self._orbit_radius_cm,
            "arc_progress_deg": math.degrees(self._arc_progress_rad),
            "exit_reason": self._exit_reason,
            "visual_return_armed": self._visual_return_armed,
            "config": asdict(self.config),
        }

    def _start_orbit(self, observation: TubeObservation, now_s: float) -> None:
        self._active_bypass_side = self._opposite_side(observation.obstacle_side)
        self._orbit_radius_cm = max(
            1.0,
            observation.tube_radius_cm + self.config.safety_radius_cm,
        )
        self.state = self._orbit_state(self._active_bypass_side)
        self._orbit_started_s = float(now_s)
        self._return_started_s = None
        self._arc_progress_rad = 0.0
        self._exit_reason = None

    def _start_return(self, now_s: float, reason: str) -> None:
        self.state = CircularBypassState.RETURN_VISUAL
        self._return_started_s = float(now_s)
        self._exit_reason = reason

    def _orbit_command(
        self,
        desired: Command,
        observation: TubeObservation,
        dt_s: float,
    ) -> Command:
        cfg = self.config
        center = np.asarray(
            [observation.center_x_cm, observation.center_y_cm],
            dtype=float,
        )
        distance = max(1e-6, float(np.linalg.norm(center)))
        radial_out = -center / distance
        bypass_side = self._active_bypass_side or 1
        if bypass_side > 0:
            tangent = np.asarray([-radial_out[1], radial_out[0]], dtype=float)
            if tangent[0] < 0.0:
                tangent = -tangent
        else:
            tangent = np.asarray([radial_out[1], -radial_out[0]], dtype=float)
            if tangent[0] < 0.0:
                tangent = -tangent

        radial_error = distance - self._orbit_radius_cm
        radial_speed = _clamp(
            -cfg.radial_kp * radial_error,
            -cfg.max_radial_speed_cm_s,
            cfg.max_radial_speed_cm_s,
        )
        velocity = cfg.tangent_speed_cm_s * tangent + radial_speed * radial_out
        command = Command(
            _clamp(velocity[0], -cfg.max_vx_cm_s, cfg.max_vx_cm_s),
            _clamp(velocity[1], -cfg.max_vy_cm_s, cfg.max_vy_cm_s),
            desired.vz_cm_s,
            _clamp(
                desired.yaw_rate_deg_s,
                -cfg.return_yaw_rate_limit_deg_s,
                cfg.return_yaw_rate_limit_deg_s,
            ),
            _append_reason(
                desired.reason,
                f"circular_tube:{self.state.value}:r={distance:.1f}",
            ),
        )
        angular_rate = abs(float(cfg.tangent_speed_cm_s)) / self._orbit_radius_cm
        self._arc_progress_rad += max(0.0, float(dt_s)) * angular_rate
        self._last_orbit_command = command
        return command

    def _return_command(self, desired: Command, now_s: float) -> Command:
        cfg = self.config
        assert self._return_started_s is not None
        duration = max(1e-6, float(cfg.return_blend_s))
        alpha = _smoothstep((float(now_s) - self._return_started_s) / duration)
        yaw_target = _clamp(
            desired.yaw_rate_deg_s,
            -cfg.return_yaw_rate_limit_deg_s,
            cfg.return_yaw_rate_limit_deg_s,
        )
        return Command(
            _lerp(self._last_orbit_command.vx_cm_s, desired.vx_cm_s, alpha),
            _lerp(self._last_orbit_command.vy_cm_s, desired.vy_cm_s, alpha),
            _lerp(self._last_orbit_command.vz_cm_s, desired.vz_cm_s, alpha),
            _lerp(self._last_orbit_command.yaw_rate_deg_s, yaw_target, alpha),
            _append_reason(
                desired.reason,
                f"circular_tube:return_visual:alpha={alpha:.2f}",
            ),
        )

    def _observe(self, points: np.ndarray) -> TubeObservation | None:
        if points.size == 0:
            return None
        cfg = self.config
        selected = points[
            (points[:, 0] >= cfg.min_x_cm)
            & (points[:, 0] <= cfg.lookahead_cm)
            & (np.abs(points[:, 1]) <= cfg.effective_intrusion_half_width_cm)
        ]
        if len(selected) < max(1, int(cfg.min_points)):
            return None
        fit = _fit_circle(selected, cfg)
        surface = np.median(selected, axis=0)
        if fit is None:
            surface_distance = float(np.linalg.norm(surface))
            if surface_distance <= 1e-6:
                return None
            tube_radius_cm = float(cfg.tube_radius_cm)
            center = surface + tube_radius_cm * surface / surface_distance
            fit_rms_cm = None
            circle_fit_used = False
        else:
            center, tube_radius_cm, fit_rms_cm = fit
            circle_fit_used = True
        side = 1 if center[1] > cfg.side_deadband_cm else -1 if center[1] < -cfg.side_deadband_cm else 0
        return TubeObservation(
            surface_x_cm=float(surface[0]),
            surface_y_cm=float(surface[1]),
            center_x_cm=float(center[0]),
            center_y_cm=float(center[1]),
            tube_radius_cm=float(tube_radius_cm),
            fit_rms_cm=fit_rms_cm,
            circle_fit_used=circle_fit_used,
            point_count=int(len(selected)),
            obstacle_side=side,
        )

    def _visual_return_ready(self, perception, orbit_elapsed_s: float) -> bool:
        if not self._road_usable(perception):
            return False
        error = float(getattr(perception, "corrected_pixel_error", math.inf))
        if not math.isfinite(error):
            return False
        if abs(error) >= self.config.visual_return_error_px:
            self._visual_return_armed = True
            return False
        return bool(
            self._visual_return_armed
            and orbit_elapsed_s
            >= max(0.0, self.config.min_orbit_before_visual_return_s)
        )

    def _road_usable(self, perception) -> bool:
        return bool(
            perception is not None
            and getattr(perception, "is_road_found", False)
            and float(getattr(perception, "confidence", 0.0))
            >= self.config.min_confidence
        )

    def _recently_seen(self, now_s: float) -> bool:
        return bool(
            self._last_seen_s is not None
            and now_s - self._last_seen_s <= self.config.obstacle_lost_hold_s
        )

    def _step_dt(self, now_s: float) -> float:
        if self._last_update_s is None:
            dt = self.config.nominal_dt_s
        else:
            dt = max(0.0, min(0.5, now_s - self._last_update_s))
        self._last_update_s = float(now_s)
        return float(dt)

    def _opposite_side(self, obstacle_side: int) -> int:
        if obstacle_side > 0:
            return -1
        if obstacle_side < 0:
            return 1
        return 1 if self.config.center_obstacle_default_bypass_side == "left" else -1

    @staticmethod
    def _orbit_state(side: int) -> CircularBypassState:
        return CircularBypassState.ORBIT_LEFT if side > 0 else CircularBypassState.ORBIT_RIGHT

    @staticmethod
    def _points(radar_field: RadarObstacleField) -> np.ndarray:
        points = np.asarray(
            getattr(radar_field, "points_body_cm", np.empty((0, 2))),
            dtype=float,
        )
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        return points.reshape(-1, 2)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _fit_circle(
    points: np.ndarray,
    config: CircularTubeBypassConfig,
) -> tuple[np.ndarray, float, float] | None:
    """Fit one circle with a centered linear least-squares solve."""
    if len(points) < max(3, int(config.min_circle_fit_points)):
        return None
    mean = np.mean(points, axis=0)
    centered = points - mean
    design = np.column_stack(
        (2.0 * centered[:, 0], 2.0 * centered[:, 1], np.ones(len(points)))
    )
    squared = np.sum(centered * centered, axis=1)
    try:
        solution, _, rank, _ = np.linalg.lstsq(design, squared, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    local_center = solution[:2]
    radius_sq = float(solution[2] + np.dot(local_center, local_center))
    if not math.isfinite(radius_sq) or radius_sq <= 0.0:
        return None
    radius = math.sqrt(radius_sq)
    if not (
        config.min_fitted_tube_radius_cm
        <= radius
        <= config.max_fitted_tube_radius_cm
    ):
        return None
    center = mean + local_center
    radial_residuals = np.linalg.norm(points - center, axis=1) - radius
    rms = float(math.sqrt(float(np.mean(radial_residuals * radial_residuals))))
    if not math.isfinite(rms) or rms > config.max_circle_fit_rms_cm:
        return None
    return center, radius, rms


def _smoothstep(value: float) -> float:
    x = _clamp(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _lerp(start: float, end: float, alpha: float) -> float:
    return float(start) + float(alpha) * (float(end) - float(start))


def _side_name(side: int | None) -> str | None:
    if side is None:
        return None
    return "left" if side > 0 else "right" if side < 0 else "center"


def _append_reason(reason: str, suffix: str) -> str:
    return f"{reason}+{suffix}" if reason else suffix


__all__ = [
    "CircularBypassState",
    "CircularTubeBypassConfig",
    "CircularTubeBypassPlanner",
    "TubeObservation",
]
