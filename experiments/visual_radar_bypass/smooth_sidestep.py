"""Low-complexity locked sidestep with smooth visual-command blending.

The controlled demo contains one movable tubular obstacle and no neighbouring
obstacles.  This planner therefore uses one vectorised rectangular point mask
and a median, rather than a grid/cluster search or candidate-path scoring.
Once an encounter chooses a side, that side remains locked until the obstacle
has stayed clear and the avoidance command has blended back into vision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from FlightController.Solutions.Safety import Command, RadarObstacleField


class SmoothSidestepState(str, Enum):
    NORMAL = "normal"
    SHIFT_LEFT = "shift_left"
    SHIFT_RIGHT = "shift_right"
    BLEND_BACK = "blend_back"
    TIMEOUT_STOP = "timeout_stop"


@dataclass(frozen=True)
class SmoothSidestepConfig:
    road_half_width_cm: float = 25.0
    intrusion_half_width_cm: float = 75.0
    clearance_cm: float = 75.0
    activity_half_width_cm: float = 90.0
    min_x_cm: float = 10.0
    lookahead_cm: float = 180.0
    min_points: int = 3
    side_deadband_cm: float = 5.0
    center_obstacle_default_bypass_side: str = "right"

    shift_forward_speed_cm_s: float = 8.0
    shift_lateral_speed_cm_s: float = 10.0
    ramp_in_s: float = 1.0
    clear_hold_s: float = 2.0
    blend_back_s: float = 2.5
    max_sidestep_s: float = 9.0
    activate_frames: int = 2
    min_confidence: float = 0.4
    nominal_dt_s: float = 0.1

    @property
    def effective_intrusion_half_width_cm(self) -> float:
        return max(
            abs(float(self.road_half_width_cm)),
            abs(float(self.clearance_cm)),
            abs(float(self.intrusion_half_width_cm)),
        )


@dataclass(frozen=True)
class ObstacleObservation:
    center_x_cm: float
    center_y_cm: float
    point_count: int
    obstacle_side: int  # +1 left, -1 right, 0 centre


class SmoothSidestepPlanner:
    """Generate one locked, smoothly blended lateral manoeuvre per encounter."""

    def __init__(self, config: SmoothSidestepConfig | None = None) -> None:
        self.config = config or SmoothSidestepConfig()
        self.state = SmoothSidestepState.NORMAL
        self._locked_side: int | None = None
        self._blend_linear = 0.0
        self._intrusion_count = 0
        self._last_update_s: float | None = None
        self._last_seen_s: float | None = None
        self._sidestep_started_s: float | None = None
        self._observation: ObstacleObservation | None = None

    @property
    def target_y_cm(self) -> float | None:
        if self._locked_side is None:
            return None
        return self._locked_side * self.config.activity_half_width_cm

    @property
    def active_bypass_side(self) -> int | None:
        return self._locked_side

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

        if self.state == SmoothSidestepState.NORMAL:
            if not self._road_usable(perception):
                return desired
            if self._intrusion_count < max(1, int(self.config.activate_frames)):
                return desired
            assert observation is not None
            self._start(
                self._opposite_bypass_side(observation.obstacle_side),
                now,
            )
            self._move_blend_toward(1.0, dt, self.config.ramp_in_s)
            return self._blended_command(desired)

        if self.state == SmoothSidestepState.TIMEOUT_STOP:
            if self._recently_seen(now):
                return Command.zero(_append_reason(desired.reason, "smooth_sidestep_timeout_stop"))
            self.state = SmoothSidestepState.BLEND_BACK

        if self.state in {
            SmoothSidestepState.SHIFT_LEFT,
            SmoothSidestepState.SHIFT_RIGHT,
        }:
            if self._sidestep_timed_out(now):
                self.state = SmoothSidestepState.TIMEOUT_STOP
                return Command.zero(_append_reason(desired.reason, "smooth_sidestep_timeout_stop"))
            if observation is not None or self._recently_seen(now):
                self._move_blend_toward(1.0, dt, self.config.ramp_in_s)
                return self._blended_command(desired)
            self.state = SmoothSidestepState.BLEND_BACK

        if self.state == SmoothSidestepState.BLEND_BACK:
            if observation is not None:
                # A noisy reappearance belongs to the same encounter.  Resume
                # the already selected side instead of choosing again.
                self.state = self._shift_state(self._locked_side or 1)
                self._move_blend_toward(1.0, dt, self.config.ramp_in_s)
                return self._blended_command(desired)
            self._move_blend_toward(0.0, dt, self.config.blend_back_s)
            if self._blend_linear <= 0.0:
                self.reset()
                return desired
            return self._blended_command(desired)

        self.reset()
        return desired

    def reset(self) -> None:
        self.state = SmoothSidestepState.NORMAL
        self._locked_side = None
        self._blend_linear = 0.0
        self._intrusion_count = 0
        self._last_seen_s = None
        self._sidestep_started_s = None
        self._observation = None

    def diagnostics(self) -> dict[str, object]:
        observation = self._observation
        return {
            "planner": "smooth_sidestep",
            "state": self.state.value,
            "target_y_cm": self.target_y_cm,
            "active_bypass_side": _side_name(self._locked_side),
            "blend_linear": self._blend_linear,
            "blend_alpha": _smoothstep(self._blend_linear),
            "intrusion_count": self._intrusion_count,
            "cluster_point_count": 0 if observation is None else observation.point_count,
            "obstacle_center_x_cm": None if observation is None else observation.center_x_cm,
            "obstacle_center_y_cm": None if observation is None else observation.center_y_cm,
            "obstacle_side": None if observation is None else _side_name(observation.obstacle_side),
            "config": asdict(self.config),
        }

    def _start(self, bypass_side: int, now_s: float) -> None:
        self._locked_side = 1 if bypass_side > 0 else -1
        self._blend_linear = 0.0
        self._sidestep_started_s = float(now_s)
        self.state = self._shift_state(self._locked_side)

    def _blended_command(self, desired: Command) -> Command:
        side = self._locked_side or 1
        cfg = self.config
        alpha = _smoothstep(self._blend_linear)
        avoid_vx = min(
            max(0.0, float(desired.vx_cm_s)),
            max(0.0, float(cfg.shift_forward_speed_cm_s)),
        )
        avoid_vy = side * abs(float(cfg.shift_lateral_speed_cm_s))
        state_name = self.state.value
        return Command(
            _lerp(desired.vx_cm_s, avoid_vx, alpha),
            _lerp(desired.vy_cm_s, avoid_vy, alpha),
            desired.vz_cm_s,
            desired.yaw_rate_deg_s,
            _append_reason(
                desired.reason,
                f"smooth_sidestep:{state_name}:alpha={alpha:.2f}",
            ),
        )

    def _move_blend_toward(
        self,
        target: float,
        dt_s: float,
        duration_s: float,
    ) -> None:
        duration = max(1e-3, float(duration_s))
        step = max(0.0, float(dt_s)) / duration
        if target >= self._blend_linear:
            self._blend_linear = min(float(target), self._blend_linear + step)
        else:
            self._blend_linear = max(float(target), self._blend_linear - step)

    def _recently_seen(self, now_s: float) -> bool:
        return bool(
            self._last_seen_s is not None
            and now_s - self._last_seen_s <= max(0.0, self.config.clear_hold_s)
        )

    def _sidestep_timed_out(self, now_s: float) -> bool:
        return bool(
            self._sidestep_started_s is not None
            and now_s - self._sidestep_started_s
            >= max(0.0, self.config.max_sidestep_s)
        )

    def _step_dt(self, now_s: float) -> float:
        if self._last_update_s is None:
            dt = self.config.nominal_dt_s
        else:
            dt = max(0.0, min(0.5, now_s - self._last_update_s))
        self._last_update_s = now_s
        return float(dt)

    def _road_usable(self, perception) -> bool:
        return bool(
            perception is not None
            and getattr(perception, "is_road_found", False)
            and float(getattr(perception, "confidence", 0.0))
            >= self.config.min_confidence
        )

    def _observe(self, points: np.ndarray) -> ObstacleObservation | None:
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
        center_x = float(np.median(selected[:, 0]))
        center_y = float(np.median(selected[:, 1]))
        if center_y > cfg.side_deadband_cm:
            obstacle_side = 1
        elif center_y < -cfg.side_deadband_cm:
            obstacle_side = -1
        else:
            obstacle_side = 0
        return ObstacleObservation(
            center_x_cm=center_x,
            center_y_cm=center_y,
            point_count=int(len(selected)),
            obstacle_side=obstacle_side,
        )

    def _opposite_bypass_side(self, obstacle_side: int) -> int:
        if obstacle_side > 0:
            return -1
        if obstacle_side < 0:
            return 1
        return 1 if self.config.center_obstacle_default_bypass_side == "left" else -1

    @staticmethod
    def _shift_state(side: int) -> SmoothSidestepState:
        return (
            SmoothSidestepState.SHIFT_LEFT
            if side > 0
            else SmoothSidestepState.SHIFT_RIGHT
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


def _smoothstep(value: float) -> float:
    x = max(0.0, min(1.0, float(value)))
    return x * x * (3.0 - 2.0 * x)


def _lerp(start: float, end: float, alpha: float) -> float:
    return float(start) + float(alpha) * (float(end) - float(start))


def _side_name(side: int | None) -> str | None:
    if side is None:
        return None
    if side > 0:
        return "left"
    if side < 0:
        return "right"
    return "center"


def _append_reason(reason: str, suffix: str) -> str:
    return f"{reason}+{suffix}" if reason else suffix


__all__ = [
    "ObstacleObservation",
    "SmoothSidestepConfig",
    "SmoothSidestepPlanner",
    "SmoothSidestepState",
]
