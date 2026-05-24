"""Safety arbiter and command gate for autonomy code.

This module does not modify the FC binary protocol. It only evaluates and
clips mission-level commands before callers may send them through
FC_Controller.send_realtime_control_data().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ObstacleUtils import mask_body_reflection, select_forward_corridor


@dataclass(frozen=True)
class Command:
    vx_cm_s: float = 0.0
    vy_cm_s: float = 0.0
    vz_cm_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    reason: str = ""

    @classmethod
    def zero(cls, reason: str = "zero") -> "Command":
        return cls(0.0, 0.0, 0.0, 0.0, reason)

    def as_fc_tuple(self) -> tuple[int, int, int, int]:
        return (
            round(self.vx_cm_s),
            round(self.vy_cm_s),
            round(self.vz_cm_s),
            round(self.yaw_rate_deg_s),
        )

    def clamp(self, config: "SafetyConfig") -> "Command":
        vx = _clip(self.vx_cm_s, -config.max_vx_cm_s, config.max_vx_cm_s)
        vy = _clip(self.vy_cm_s, -config.max_vy_cm_s, config.max_vy_cm_s)
        vz = _clip(self.vz_cm_s, -config.max_vz_cm_s, config.max_vz_cm_s)
        yaw = _clip(
            self.yaw_rate_deg_s,
            -config.max_yaw_rate_deg_s,
            config.max_yaw_rate_deg_s,
        )
        reason = self.reason
        if (vx, vy, vz, yaw) != (
            self.vx_cm_s,
            self.vy_cm_s,
            self.vz_cm_s,
            self.yaw_rate_deg_s,
        ):
            reason = _append_reason(reason, "clamped")
        return Command(vx, vy, vz, yaw, reason)


VelocityCommand = Command


@dataclass
class FlightHealth:
    fc_connected: bool = False
    fc_mode: int | None = None
    unlock: bool | None = None
    battery_v: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    radar_fresh: bool = False
    radar_max_age_s: float | None = None
    camera_ok: bool = True


@dataclass
class FlightStatus:
    connected: bool = False
    mode: int | None = None
    unlocked: bool | None = None
    battery_v: float | None = None
    alt_cm: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None


@dataclass
class SafetyConfig:
    require_fc: bool = True
    require_hold_pos_mode: bool = True
    hold_pos_mode: int = 2
    require_radar: bool = True
    radar_timeout_s: float = 0.5
    max_abs_roll_deg: float = 25.0
    max_abs_pitch_deg: float = 25.0
    min_battery_v: float | None = None
    max_vx_cm_s: float = 35.0
    max_vy_cm_s: float = 25.0
    max_vz_cm_s: float = 20.0
    max_yaw_rate_deg_s: float = 30.0

    # Compatibility and obstacle-gate fields used by earlier entry points.
    obstacle_stop_distance_cm: float = 80.0
    obstacle_slow_distance_cm: float = 150.0
    slow_speed_limit_cm_s: float = 12.0
    side_stop_distance_cm: float = 45.0

    @property
    def max_xy_speed_cm_s(self) -> float:
        return max(self.max_vx_cm_s, self.max_vy_cm_s)

    @property
    def max_z_speed_cm_s(self) -> float:
        return self.max_vz_cm_s

    @property
    def max_abs_roll_pitch_deg(self) -> float:
        return max(self.max_abs_roll_deg, self.max_abs_pitch_deg)


@dataclass
class SafetyDecision:
    command: Command
    allowed: bool
    hard_stop: bool
    reason: str


@dataclass
class SafetyResult:
    command: Command
    state: str
    reasons: list[str] = field(default_factory=list)
    nearest_forward_obstacle_cm: float | None = None


@dataclass
class RadarFieldConfig:
    max_distance_cm: float = 300.0
    body_x_half_cm: float = 25.0
    body_y_half_cm: float = 25.0
    forward_corridor_half_width_cm: float = 50.0
    min_obstacle_distance_cm: float = 10.0


class RadarObstacleField:
    """Filtered body-frame radar point cloud view."""

    def __init__(self, config: RadarFieldConfig | None = None):
        self.config = config or RadarFieldConfig()
        self.raw_points_body_cm = np.empty((0, 2), dtype=float)
        self.points_body_cm = np.empty((0, 2), dtype=float)
        self.updated_s = 0.0

    def update(self, points_body_cm: np.ndarray, now_s: float) -> "RadarObstacleField":
        points = _normalize_points(points_body_cm)
        if points.size:
            distances = np.linalg.norm(points, axis=1)
            points = points[distances <= self.config.max_distance_cm]
        self.raw_points_body_cm = points
        self.points_body_cm = self._remove_body_reflections(points)
        self.updated_s = now_s
        return self

    def nearest_forward_obstacle_cm(self) -> float | None:
        points = self.points_body_cm
        if points.size == 0:
            return None
        cfg = self.config
        forward = select_forward_corridor(
            points,
            min_x_cm=cfg.min_obstacle_distance_cm,
            half_width_cm=cfg.forward_corridor_half_width_cm,
        )
        if forward.size == 0:
            return None
        return float(np.min(forward[:, 0]))

    def side_clearance_cm(self, side: str) -> float | None:
        points = self.points_body_cm
        if points.size == 0:
            return None
        if side == "left":
            selected = points[points[:, 1] > 0]
        elif side == "right":
            selected = points[points[:, 1] < 0]
        else:
            raise ValueError("side must be 'left' or 'right'")
        if selected.size == 0:
            return None
        return float(np.min(np.abs(selected[:, 1])))

    def sector_clearance_cm(self, angle_deg: float, sector_half_width_deg: float = 12.0) -> float | None:
        points = self.points_body_cm
        if points.size == 0:
            return None
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        delta = np.abs(_wrap_deg_array(angles - angle_deg))
        selected = points[delta <= sector_half_width_deg]
        if selected.size == 0:
            return None
        return float(np.min(np.linalg.norm(selected, axis=1)))

    def _remove_body_reflections(self, points: np.ndarray) -> np.ndarray:
        cfg = self.config
        return mask_body_reflection(
            points,
            x_half_cm=cfg.body_x_half_cm,
            y_half_cm=cfg.body_y_half_cm,
        )


class SafetyArbiter:
    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()

    def evaluate(self, desired: Command, health: FlightHealth) -> SafetyDecision:
        hard_stop_reason = self._evaluate_hard_stop(health)
        if hard_stop_reason is not None:
            return SafetyDecision(
                command=Command.zero(f"safety_stop:{hard_stop_reason}"),
                allowed=False,
                hard_stop=True,
                reason=hard_stop_reason,
            )

        command = desired.clamp(self.config)
        reason = "ok"
        if command != desired:
            reason = "ok+clamped"
        return SafetyDecision(command=command, allowed=True, hard_stop=False, reason=reason)

    def filter(
        self,
        desired: Command,
        *,
        flight: FlightStatus,
        radar_connected: bool,
        radar_age_s: float | None,
        radar_field: RadarObstacleField,
        enable_flight: bool = False,
    ) -> SafetyResult:
        _ = enable_flight
        health = FlightHealth(
            fc_connected=flight.connected,
            fc_mode=flight.mode,
            unlock=flight.unlocked,
            battery_v=flight.battery_v,
            roll_deg=flight.roll_deg,
            pitch_deg=flight.pitch_deg,
            radar_fresh=bool(radar_connected and radar_age_s is not None and radar_age_s <= self.config.radar_timeout_s),
            radar_max_age_s=radar_age_s,
        )
        decision = self.evaluate(desired, health)
        nearest = radar_field.nearest_forward_obstacle_cm()
        if decision.hard_stop:
            return SafetyResult(decision.command, "HARD_STOP", [decision.reason], nearest)

        cmd = decision.command
        reasons: list[str] = []

        if nearest is not None and nearest <= self.config.obstacle_stop_distance_cm and cmd.vx_cm_s > 0:
            reasons.append("front_obstacle_stop")
            cmd = Command(
                0.0,
                cmd.vy_cm_s,
                cmd.vz_cm_s,
                cmd.yaw_rate_deg_s,
                _append_reason(cmd.reason, "front_obstacle_stop"),
            )
            return SafetyResult(cmd, "OBSTACLE_STOP", reasons, nearest)

        if nearest is not None and nearest <= self.config.obstacle_slow_distance_cm:
            if cmd.vx_cm_s > self.config.slow_speed_limit_cm_s:
                reasons.append("front_obstacle_slow")
                cmd = Command(
                    self.config.slow_speed_limit_cm_s,
                    cmd.vy_cm_s,
                    cmd.vz_cm_s,
                    cmd.yaw_rate_deg_s,
                    _append_reason(cmd.reason, "front_obstacle_slow"),
                )

        left_clearance = radar_field.side_clearance_cm("left")
        if left_clearance is not None and left_clearance <= self.config.side_stop_distance_cm and cmd.vy_cm_s > 0:
            reasons.append("left_side_blocked")
            cmd = Command(
                cmd.vx_cm_s,
                0.0,
                cmd.vz_cm_s,
                cmd.yaw_rate_deg_s,
                _append_reason(cmd.reason, "left_side_blocked"),
            )

        right_clearance = radar_field.side_clearance_cm("right")
        if right_clearance is not None and right_clearance <= self.config.side_stop_distance_cm and cmd.vy_cm_s < 0:
            reasons.append("right_side_blocked")
            cmd = Command(
                cmd.vx_cm_s,
                0.0,
                cmd.vz_cm_s,
                cmd.yaw_rate_deg_s,
                _append_reason(cmd.reason, "right_side_blocked"),
            )

        state = "OK" if not reasons else "LIMITED"
        return SafetyResult(cmd, state, reasons, nearest)

    def _evaluate_hard_stop(self, health: FlightHealth) -> str | None:
        cfg = self.config
        if cfg.require_fc and not health.fc_connected:
            return "fc_not_connected"
        if cfg.require_hold_pos_mode and health.fc_mode != cfg.hold_pos_mode:
            return "not_hold_pos_mode"
        if cfg.require_radar and not health.radar_fresh:
            return "radar_not_fresh"
        if health.roll_deg is not None and abs(health.roll_deg) > cfg.max_abs_roll_deg:
            return "roll_too_large"
        if health.pitch_deg is not None and abs(health.pitch_deg) > cfg.max_abs_pitch_deg:
            return "pitch_too_large"
        if cfg.min_battery_v is not None:
            if health.battery_v is not None and health.battery_v < cfg.min_battery_v:
                return "low_battery"
        return None


def flight_health_from_sources(
    fc=None,
    radar=None,
    multi_radar=None,
    radar_timeout_s: float = 0.5,
    camera_ok: bool = True,
) -> FlightHealth:
    fc_connected = False
    fc_mode = None
    unlock = None
    battery_v = None
    roll_deg = None
    pitch_deg = None

    if fc is not None:
        try:
            fc_connected = bool(getattr(fc, "connected", False))
            state = getattr(fc, "state", None)
            if state is not None:
                fc_mode = _field_value(getattr(state, "mode", None))
                unlock = _field_value(getattr(state, "unlock", None))
                battery_v = _field_value(getattr(state, "bat", None))
                roll_deg = _field_value(getattr(state, "roll", getattr(state, "rol", None)))
                pitch_deg = _field_value(getattr(state, "pit", None))
        except Exception:
            fc_connected = False

    radar_fresh = False
    radar_max_age_s = None
    try:
        if multi_radar is not None:
            radar_fresh = bool(multi_radar.is_fresh(max_age_s=radar_timeout_s))
            radar_max_age_s = multi_radar_age_s(multi_radar)
        elif radar is not None:
            radar_fresh = bool(radar.is_fresh(max_age_s=radar_timeout_s))
            radar_max_age_s = radar.get_last_frame_age_s()
    except Exception:
        radar_fresh = False
        radar_max_age_s = None

    return FlightHealth(
        fc_connected=fc_connected,
        fc_mode=fc_mode,
        unlock=unlock,
        battery_v=battery_v,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        radar_fresh=radar_fresh,
        radar_max_age_s=radar_max_age_s,
        camera_ok=camera_ok,
    )


def send_command_safely(
    fc,
    desired: Command,
    arbiter: SafetyArbiter,
    health: FlightHealth,
    *,
    dry_run: bool = True,
) -> SafetyDecision:
    decision = arbiter.evaluate(desired, health)
    cmd = decision.command
    if not dry_run and fc is not None:
        fc.send_realtime_control_data(
            round(cmd.vx_cm_s),
            round(cmd.vy_cm_s),
            round(cmd.vz_cm_s),
            round(cmd.yaw_rate_deg_s),
        )
    return decision


def flight_status_from_fc(fc: Any | None) -> FlightStatus:
    health = flight_health_from_sources(fc=fc, radar_timeout_s=0.0)
    return FlightStatus(
        connected=health.fc_connected,
        mode=health.fc_mode,
        unlocked=health.unlock,
        battery_v=health.battery_v,
        roll_deg=health.roll_deg,
        pitch_deg=health.pitch_deg,
    )


def multi_radar_age_s(multi_radar: Any) -> float | None:
    ages: list[float] = []
    for radar in getattr(multi_radar, "radars", []):
        age = radar.get_last_frame_age_s()
        if age is None:
            return None
        ages.append(float(age))
    if not ages:
        return None
    return max(ages)


def _field_value(field: Any) -> Any:
    return getattr(field, "value", field)


def _normalize_points(points_body_cm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_body_cm, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    return points.reshape(-1, 2)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _append_reason(reason: str, suffix: str) -> str:
    return f"{reason}+{suffix}" if reason else suffix


def _wrap_deg_array(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


__all__ = [
    "Command",
    "FlightHealth",
    "FlightStatus",
    "RadarFieldConfig",
    "RadarObstacleField",
    "SafetyArbiter",
    "SafetyConfig",
    "SafetyDecision",
    "SafetyResult",
    "VelocityCommand",
    "flight_health_from_sources",
    "flight_status_from_fc",
    "multi_radar_age_s",
    "send_command_safely",
]
