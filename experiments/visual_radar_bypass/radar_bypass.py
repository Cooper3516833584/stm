"""Side-agnostic physical-obstacle radar sidestep for the isolated experiment.

Coordinate convention: body +X is forward and body +Y is left.  No obstacle
position is configured or injected.  The planner finds the densest real radar
cluster inside the bilateral intrusion envelope, infers its side per encounter,
and commands a direct lateral sidestep toward the opposite side.  The map guarantee that the
tubular test obstacle has no neighboring obstacles permits local cluster-based
planning; the global SafetyArbiter still receives the complete physical point
cloud.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math

import numpy as np

from FlightController.Solutions.Safety import Command, RadarObstacleField


class ObstacleBypassState(str, Enum):
    NORMAL = "normal"
    BYPASS_LEFT = "bypass_left"
    BYPASS_RIGHT = "bypass_right"
    FORWARD_RECOVERY = "forward_recovery"
    RETURN_CENTER = "return_center"


@dataclass(frozen=True)
class ObstacleBypassConfig:
    road_half_width_cm: float = 25.0
    intrusion_half_width_cm: float = 75.0
    activity_half_width_cm: float = 90.0
    clearance_cm: float = 75.0
    min_x_cm: float = 10.0
    lookahead_cm: float = 180.0
    bypass_speed_cm_s: float = 10.0
    bypass_lateral_speed_cm_s: float = 8.0
    max_yaw_rate_deg_s: float = 10.0
    activate_frames: int = 2
    release_s: float = 0.75
    max_bypass_s: float = 11.25
    min_confidence: float = 0.4
    return_pixel_deadband_px: float = 35.0

    # Forward-priority handoff from radar sidestep back to visual tracking.
    forward_recovery_s: float = 2.0
    forward_recovery_vx_cm_s: float = 10.0
    forward_recovery_lateral_decay_s: float = 0.4
    forward_recovery_visual_blend_s: float = 0.5
    forward_recovery_middle_visual_weight: float = 0.15
    forward_recovery_radar_reentry_cm: float = 80.0

    # Real tubular obstacle extraction.  A peak grid cell is expanded to a
    # small neighborhood so an obstacle crossing cell boundaries stays whole.
    cluster_grid_cm: float = 10.0
    cluster_radius_x_cm: float = 20.0
    cluster_radius_y_cm: float = 15.0
    min_cluster_points: int = 3
    side_deadband_cm: float = 5.0
    center_obstacle_default_bypass_side: str = "right"

    @property
    def effective_intrusion_half_width_cm(self) -> float:
        return max(
            abs(float(self.road_half_width_cm)),
            abs(float(self.clearance_cm)),
            abs(float(self.intrusion_half_width_cm)),
        )


@dataclass(frozen=True)
class ObstacleCluster:
    points: np.ndarray
    center_x_cm: float
    center_y_cm: float
    obstacle_side: int  # +1 left, -1 right, 0 centre


class ObstacleBypassPlanner:
    """Infer obstacle side from each physical point-cloud encounter."""

    def __init__(self, config: ObstacleBypassConfig | None = None) -> None:
        self.config = config or ObstacleBypassConfig()
        self.state = ObstacleBypassState.NORMAL
        self._intrusion_count = 0
        self._last_intrusion_s: float | None = None
        self._target_y_cm: float | None = None
        self._active_bypass_side: int | None = None
        self._bypass_started_s: float | None = None
        self._forward_recovery_started_s: float | None = None
        self._forward_recovery_elapsed_s = 0.0
        self._cluster_point_count = 0
        self._obstacle_center_x_cm: float | None = None
        self._obstacle_center_y_cm: float | None = None
        self._obstacle_side: int | None = None

    @property
    def target_y_cm(self) -> float | None:
        return self._target_y_cm

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
        cluster = self._detect_obstacle_cluster(self._points(radar_field))
        self._update_cluster_diagnostics(cluster)
        has_obstacle = cluster is not None
        if has_obstacle:
            self._intrusion_count += 1
            self._last_intrusion_s = float(now_s)
        else:
            self._intrusion_count = 0

        if self.state == ObstacleBypassState.NORMAL:
            if not self._road_usable(perception):
                return desired
            if self._intrusion_count < max(1, int(self.config.activate_frames)):
                return desired
            assert cluster is not None
            bypass_side = self._opposite_bypass_side(cluster.obstacle_side)
            self._set_bypass(bypass_side, now_s)
            return self._bypass_command(desired, bypass_side)

        if self.state in {
            ObstacleBypassState.BYPASS_LEFT,
            ObstacleBypassState.BYPASS_RIGHT,
        }:
            if (
                self._bypass_started_s is not None
                and now_s - self._bypass_started_s >= self.config.max_bypass_s
            ):
                return self._no_gap_command(desired)
            recently_blocked = (
                self._last_intrusion_s is not None
                and now_s - self._last_intrusion_s <= self.config.release_s
            )
            if has_obstacle or recently_blocked:
                bypass_side = self._active_bypass_side or 1
                return self._bypass_command(desired, bypass_side)
            if not self._visual_command_usable(perception, desired):
                self.reset()
                return desired
            if self.config.forward_recovery_s <= 0.0:
                self.reset()
                return desired
            self._start_forward_recovery(now_s)
            return self._forward_recovery_command(desired, now_s)

        if self.state == ObstacleBypassState.FORWARD_RECOVERY:
            if self._radar_requires_bypass(radar_field, has_obstacle):
                bypass_side = self._active_bypass_side or (
                    self._opposite_bypass_side(cluster.obstacle_side)
                    if cluster is not None
                    else 1
                )
                self._set_bypass(bypass_side, now_s)
                return self._bypass_command(desired, bypass_side)
            if not self._visual_command_usable(perception, desired):
                return Command.zero(
                    _append_reason(
                        desired.reason,
                        "tube_obstacle_forward_recovery_visual_hold",
                    )
                )
            assert self._forward_recovery_started_s is not None
            elapsed_s = max(0.0, now_s - self._forward_recovery_started_s)
            if elapsed_s >= max(0.0, self.config.forward_recovery_s):
                self.reset()
                return desired
            return self._forward_recovery_command(desired, now_s)

        self.reset()
        return desired

    def reset(self) -> None:
        self.state = ObstacleBypassState.NORMAL
        self._intrusion_count = 0
        self._last_intrusion_s = None
        self._target_y_cm = None
        self._active_bypass_side = None
        self._bypass_started_s = None
        self._forward_recovery_started_s = None
        self._forward_recovery_elapsed_s = 0.0
        self._cluster_point_count = 0
        self._obstacle_center_x_cm = None
        self._obstacle_center_y_cm = None
        self._obstacle_side = None

    def diagnostics(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "target_y_cm": self._target_y_cm,
            "active_bypass_side": _side_name(self._active_bypass_side),
            "intrusion_count": self._intrusion_count,
            "cluster_point_count": self._cluster_point_count,
            "obstacle_center_x_cm": self._obstacle_center_x_cm,
            "obstacle_center_y_cm": self._obstacle_center_y_cm,
            "obstacle_side": _side_name(self._obstacle_side),
            "forward_recovery_elapsed_s": self._forward_recovery_elapsed_s,
            "config": asdict(self.config),
        }

    def _road_usable(self, perception) -> bool:
        return bool(
            perception is not None
            and getattr(perception, "is_road_found", False)
            and float(getattr(perception, "confidence", 0.0))
            >= self.config.min_confidence
        )

    def _visual_command_usable(self, perception, desired: Command) -> bool:
        return bool(
            self._road_usable(perception)
            and "road_lost" not in str(desired.reason)
            and "visual_unavailable" not in str(desired.reason)
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

    def _bilateral_intrusion(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        cfg = self.config
        return points[
            (points[:, 0] >= cfg.min_x_cm)
            & (points[:, 0] <= cfg.lookahead_cm)
            & (np.abs(points[:, 1]) <= cfg.effective_intrusion_half_width_cm)
        ]

    def _detect_obstacle_cluster(
        self,
        points: np.ndarray,
    ) -> ObstacleCluster | None:
        intrusion = self._bilateral_intrusion(points)
        cfg = self.config
        if len(intrusion) < max(1, int(cfg.min_cluster_points)):
            return None

        grid = max(1.0, float(cfg.cluster_grid_cm))
        cells: dict[tuple[int, int], list[int]] = {}
        for index, (x_cm, y_cm) in enumerate(intrusion):
            key = (math.floor(float(x_cm) / grid), math.floor(float(y_cm) / grid))
            cells.setdefault(key, []).append(index)

        def peak_rank(indices: list[int]) -> tuple[float, float, float]:
            cell_points = intrusion[indices]
            return (
                -float(len(indices)),
                float(np.median(cell_points[:, 0])),
                abs(float(np.median(cell_points[:, 1]))),
            )

        peak_indices = min(cells.values(), key=peak_rank)
        peak_points = intrusion[peak_indices]
        seed_x = float(np.median(peak_points[:, 0]))
        seed_y = float(np.median(peak_points[:, 1]))
        cluster_points = intrusion[
            (np.abs(intrusion[:, 0] - seed_x) <= cfg.cluster_radius_x_cm)
            & (np.abs(intrusion[:, 1] - seed_y) <= cfg.cluster_radius_y_cm)
        ]
        if len(cluster_points) < max(1, int(cfg.min_cluster_points)):
            return None

        center_x = float(np.median(cluster_points[:, 0]))
        center_y = float(np.median(cluster_points[:, 1]))
        if center_y > cfg.side_deadband_cm:
            obstacle_side = 1
        elif center_y < -cfg.side_deadband_cm:
            obstacle_side = -1
        else:
            obstacle_side = 0
        return ObstacleCluster(
            points=cluster_points,
            center_x_cm=center_x,
            center_y_cm=center_y,
            obstacle_side=obstacle_side,
        )

    def _opposite_bypass_side(self, obstacle_side: int) -> int:
        if obstacle_side > 0:
            return -1
        if obstacle_side < 0:
            return 1
        return (
            1
            if self.config.center_obstacle_default_bypass_side == "left"
            else -1
        )

    def _set_bypass(self, bypass_side: int, now_s: float) -> None:
        self._active_bypass_side = 1 if bypass_side > 0 else -1
        self._target_y_cm = (
            self._active_bypass_side * self.config.activity_half_width_cm
        )
        self._bypass_started_s = float(now_s)
        self._forward_recovery_started_s = None
        self._forward_recovery_elapsed_s = 0.0
        self.state = (
            ObstacleBypassState.BYPASS_LEFT
            if bypass_side > 0
            else ObstacleBypassState.BYPASS_RIGHT
        )

    def _start_forward_recovery(self, now_s: float) -> None:
        self.state = ObstacleBypassState.FORWARD_RECOVERY
        self._bypass_started_s = None
        self._forward_recovery_started_s = float(now_s)
        self._forward_recovery_elapsed_s = 0.0

    def _radar_requires_bypass(
        self,
        radar_field: RadarObstacleField,
        has_obstacle: bool,
    ) -> bool:
        if has_obstacle:
            return True
        nearest = radar_field.nearest_forward_obstacle_cm()
        return bool(
            nearest is not None
            and nearest <= self.config.forward_recovery_radar_reentry_cm
        )

    def _update_cluster_diagnostics(
        self,
        cluster: ObstacleCluster | None,
    ) -> None:
        if cluster is None:
            self._cluster_point_count = 0
            self._obstacle_center_x_cm = None
            self._obstacle_center_y_cm = None
            self._obstacle_side = None
            return
        self._cluster_point_count = int(len(cluster.points))
        self._obstacle_center_x_cm = cluster.center_x_cm
        self._obstacle_center_y_cm = cluster.center_y_cm
        self._obstacle_side = cluster.obstacle_side

    def _bypass_command(self, desired: Command, bypass_side: int) -> Command:
        cfg = self.config
        yaw = _clamp(
            desired.yaw_rate_deg_s,
            -cfg.max_yaw_rate_deg_s,
            cfg.max_yaw_rate_deg_s,
        )
        side_sign = 1 if bypass_side > 0 else -1
        side = "left" if side_sign > 0 else "right"
        return Command(
            min(desired.vx_cm_s, cfg.bypass_speed_cm_s),
            side_sign * cfg.bypass_lateral_speed_cm_s,
            desired.vz_cm_s,
            yaw,
            _append_reason(
                desired.reason,
                f"tube_obstacle_sidestep:{side}",
            ),
        )

    def _forward_recovery_command(
        self,
        desired: Command,
        now_s: float,
    ) -> Command:
        cfg = self.config
        assert self._forward_recovery_started_s is not None
        elapsed_s = max(0.0, float(now_s) - self._forward_recovery_started_s)
        self._forward_recovery_elapsed_s = elapsed_s
        total_s = max(1e-6, float(cfg.forward_recovery_s))
        decay_s = min(total_s, max(0.0, cfg.forward_recovery_lateral_decay_s))
        blend_s = min(total_s, max(0.0, cfg.forward_recovery_visual_blend_s))
        middle_weight = _clamp(
            cfg.forward_recovery_middle_visual_weight,
            0.0,
            1.0,
        )
        side = self._active_bypass_side or 1

        initial = Command(
            min(desired.vx_cm_s, cfg.bypass_speed_cm_s),
            side * cfg.bypass_lateral_speed_cm_s,
            desired.vz_cm_s,
            _clamp(
                desired.yaw_rate_deg_s,
                -cfg.max_yaw_rate_deg_s,
                cfg.max_yaw_rate_deg_s,
            ),
            desired.reason,
        )
        forward = Command(
            max(0.0, cfg.forward_recovery_vx_cm_s),
            0.0,
            desired.vz_cm_s,
            0.0,
            desired.reason,
        )
        middle = _blend_command(forward, desired, middle_weight)

        if decay_s > 0.0 and elapsed_s < decay_s:
            phase = _smoothstep(elapsed_s / decay_s)
            command = _blend_command(initial, middle, phase)
            visual_weight = phase * middle_weight
        elif blend_s > 0.0 and elapsed_s > total_s - blend_s:
            phase = _smoothstep((elapsed_s - (total_s - blend_s)) / blend_s)
            command = _blend_command(middle, desired, phase)
            visual_weight = middle_weight + (1.0 - middle_weight) * phase
        else:
            command = middle
            visual_weight = middle_weight

        return Command(
            command.vx_cm_s,
            command.vy_cm_s,
            command.vz_cm_s,
            command.yaw_rate_deg_s,
            _append_reason(
                desired.reason,
                "tube_obstacle_forward_recovery:"
                f"t={elapsed_s:.2f}:visual={visual_weight:.2f}",
            ),
        )

    def _no_gap_command(self, desired: Command) -> Command:
        return Command(
            0.0,
            0.0,
            desired.vz_cm_s,
            0.0,
            _append_reason(desired.reason, "tube_obstacle_no_gap_stop"),
        )


def _side_name(side: int | None) -> str | None:
    if side is None:
        return None
    if side > 0:
        return "left"
    if side < 0:
        return "right"
    return "center"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _smoothstep(value: float) -> float:
    x = _clamp(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _blend_command(start: Command, end: Command, alpha: float) -> Command:
    weight = _clamp(alpha, 0.0, 1.0)

    def blend(a: float, b: float) -> float:
        return float(a) + weight * (float(b) - float(a))

    return Command(
        blend(start.vx_cm_s, end.vx_cm_s),
        blend(start.vy_cm_s, end.vy_cm_s),
        blend(start.vz_cm_s, end.vz_cm_s),
        blend(start.yaw_rate_deg_s, end.yaw_rate_deg_s),
        end.reason,
    )


def _append_reason(reason: str, suffix: str) -> str:
    return f"{reason}+{suffix}" if reason else suffix
