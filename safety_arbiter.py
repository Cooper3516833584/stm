"""Top-level safety arbiter for all autonomy modes."""

from __future__ import annotations

from dataclasses import dataclass, field

from autonomy_command import VelocityCommand
from autonomy_context import AutonomyContext
from local_world_model import LocalWorldModel


@dataclass
class SafetyConfig:
    require_radar: bool = True
    require_hold_pos_mode: bool = True
    hold_pos_mode: int = 2
    radar_timeout_s: float = 0.5
    min_battery_v: float = 10.0
    max_abs_roll_pitch_deg: float = 25.0
    obstacle_stop_distance_cm: float = 80.0
    obstacle_slow_distance_cm: float = 150.0
    slow_speed_limit_cm_s: float = 12.0
    max_xy_speed_cm_s: float = 35.0
    max_z_speed_cm_s: float = 25.0
    max_yaw_rate_deg_s: float = 30.0


@dataclass
class SafetyResult:
    command: VelocityCommand
    state: str
    reasons: list[str] = field(default_factory=list)
    nearest_forward_obstacle_cm: float | None = None


class SafetyArbiter:
    """Converts mission intent into a command that is allowed to reach the FC."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()

    def filter(
        self,
        desired: VelocityCommand,
        *,
        context: AutonomyContext,
        world: LocalWorldModel,
    ) -> SafetyResult:
        cfg = self.config
        reasons: list[str] = []

        hard_fault = self._hard_fault_reason(context)
        if hard_fault is not None:
            return SafetyResult(
                command=VelocityCommand.zero(f"safety_{hard_fault}"),
                state="HARD_STOP",
                reasons=[hard_fault],
                nearest_forward_obstacle_cm=world.nearest_forward_obstacle_cm(),
            )

        cmd = desired.clamp(
            max_xy_speed_cm_s=cfg.max_xy_speed_cm_s,
            max_z_speed_cm_s=cfg.max_z_speed_cm_s,
            max_yaw_rate_deg_s=cfg.max_yaw_rate_deg_s,
        )

        nearest = world.nearest_forward_obstacle_cm()
        if nearest is not None and nearest <= cfg.obstacle_stop_distance_cm and cmd.vx_cm_s > 0:
            reasons.append("front_obstacle_stop")
            cmd = VelocityCommand(
                0.0,
                cmd.vy_cm_s,
                cmd.vz_cm_s,
                cmd.yaw_rate_deg_s,
                f"{cmd.reason}+front_obstacle_stop",
            )
            return SafetyResult(cmd, "OBSTACLE_STOP", reasons, nearest)

        if nearest is not None and nearest <= cfg.obstacle_slow_distance_cm and cmd.vx_cm_s > cfg.slow_speed_limit_cm_s:
            reasons.append("front_obstacle_slow")
            cmd = VelocityCommand(
                cfg.slow_speed_limit_cm_s,
                cmd.vy_cm_s,
                cmd.vz_cm_s,
                cmd.yaw_rate_deg_s,
                f"{cmd.reason}+front_obstacle_slow",
            )
            return SafetyResult(cmd, "OBSTACLE_SLOW", reasons, nearest)

        return SafetyResult(cmd, "OK", reasons, nearest)

    def _hard_fault_reason(self, context: AutonomyContext) -> str | None:
        cfg = self.config
        health = context.health
        flight = context.flight

        if cfg.require_radar:
            if not health.radar_ok:
                return "radar_unavailable"
            if health.radar_age_s is None or health.radar_age_s > cfg.radar_timeout_s:
                return "radar_timeout"

        if cfg.require_hold_pos_mode and flight.mode is not None and flight.mode != cfg.hold_pos_mode:
            return "not_hold_pos_mode"

        if flight.battery_v is not None and 0.1 < flight.battery_v < cfg.min_battery_v:
            return "low_battery"

        if flight.roll_deg is not None and abs(flight.roll_deg) > cfg.max_abs_roll_pitch_deg:
            return "roll_too_large"
        if flight.pitch_deg is not None and abs(flight.pitch_deg) > cfg.max_abs_roll_pitch_deg:
            return "pitch_too_large"

        return None

