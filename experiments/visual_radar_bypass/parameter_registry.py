"""Central parameter and provenance registry for the isolated experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .smooth_sidestep import SmoothSidestepConfig


class ParameterSource(str, Enum):
    FIXED_REQUIREMENT = "FIXED_REQUIREMENT"
    EXISTING_PROJECT = "EXISTING_PROJECT"
    MATHEMATICAL = "MATHEMATICAL"
    AGENT_ASSUMPTION = "AGENT_ASSUMPTION"
    UNVERIFIED_TUNING = "UNVERIFIED_TUNING"


@dataclass(frozen=True)
class ExperimentSafetyConfig:
    """Experiment-local values passed to the unchanged production arbiter."""

    radar_timeout_s: float = 0.5
    max_vx_cm_s: float = 14.0
    max_vy_cm_s: float = 10.0
    max_yaw_rate_deg_s: float = 10.0
    obstacle_stop_distance_cm: float = 80.0
    obstacle_slow_distance_cm: float = 150.0
    slow_speed_limit_cm_s: float = 10.0
    side_stop_distance_cm: float = 45.0
    radar_max_distance_cm: float = 300.0
    radar_body_x_half_cm: float = 25.0
    radar_body_y_half_cm: float = 25.0
    radar_forward_corridor_half_width_cm: float = 75.0


@dataclass(frozen=True)
class ExperimentLoggingConfig:
    # UNVERIFIED_TUNING: trade diagnostic density against synchronous flushes.
    tuning_log_every_n: int = 2
    radar_snapshot_every_n: int = 5


def build_parameter_registry(
    smooth: SmoothSidestepConfig,
    safety: ExperimentSafetyConfig,
    logging: ExperimentLoggingConfig,
) -> list[dict[str, object]]:
    """Return the complete important-parameter registry for logs/reports."""

    e = ParameterSource.EXISTING_PROJECT
    u = ParameterSource.UNVERIFIED_TUNING
    rows = [
        _row("road_half_width_cm", smooth.road_half_width_cm, e, "road geometry", "wider road gate", "narrower road gate"),
        _row("intrusion_half_width_cm", smooth.intrusion_half_width_cm, e, "radar activation half-width", "earlier lateral activation", "later activation"),
        _row("clearance_cm", smooth.clearance_cm, e, "minimum configured lateral clearance envelope", "larger envelope", "smaller envelope", safety=True),
        _row("activity_half_width_cm", smooth.activity_half_width_cm, e, "reported sidestep activity extent", "larger allowed offset", "smaller allowed offset", safety=True),
        _row("min_x_cm", smooth.min_x_cm, e, "near radar gate", "ignore more close returns", "include closer returns", safety=True),
        _row("lookahead_cm", smooth.lookahead_cm, e, "far radar gate", "activate earlier", "activate later", safety=True),
        _row("min_points", smooth.min_points, e, "minimum obstacle evidence", "fewer false positives/more misses", "more sensitivity/noise"),
        _row("side_deadband_cm", smooth.side_deadband_cm, e, "centre-side classification deadband", "more centre defaults", "more side sensitivity"),
        _row("center_obstacle_default_bypass_side", smooth.center_obstacle_default_bypass_side, e, "deterministic centre bypass", "not numeric", "not numeric"),
        _row("shift_forward_speed_cm_s", smooth.shift_forward_speed_cm_s, u, "active forward target", "more forward progress and threshold interaction", "more conservative lateral-only motion", safety=True),
        _row("shift_lateral_speed_cm_s", smooth.shift_lateral_speed_cm_s, u, "active lateral target", "faster clearance/larger command step", "slower clearance", safety=True),
        _row("ramp_in_s", smooth.ramp_in_s, u, "smoothstep ramp-in duration", "smoother/slower response", "faster/larger delta"),
        _row("clear_hold_s", smooth.clear_hold_s, u, "radar-loss hysteresis", "stronger dropout immunity/longer sidestep", "earlier visual return"),
        _row("blend_back_s", smooth.blend_back_s, u, "visual handback duration", "smoother/slower handback", "faster/larger delta"),
        _row("max_sidestep_s", smooth.max_sidestep_s, u, "timeout-stop limit", "larger travel before stop", "earlier stop", safety=True),
        _row("activate_frames", smooth.activate_frames, u, "consecutive activation frames", "fewer false triggers/slower response", "faster/noisier response"),
        _row("min_confidence", smooth.min_confidence, e, "minimum visual confidence for encounter start", "stricter visual precondition", "weaker visual precondition"),
        _row("nominal_dt_s", smooth.nominal_dt_s, u, "first-frame integration interval", "larger first blend step", "smaller first blend step"),
        _row("radar_timeout_s", safety.radar_timeout_s, e, "radar freshness hard-stop threshold", "tolerate older radar", "stop sooner on stale radar", safety=True),
        _row("max_vx_cm_s", safety.max_vx_cm_s, e, "experiment forward velocity cap", "faster forward motion", "slower forward motion", safety=True),
        _row("max_vy_cm_s", safety.max_vy_cm_s, e, "experiment lateral velocity cap", "faster lateral motion", "slower lateral motion", safety=True),
        _row("max_yaw_rate_deg_s", safety.max_yaw_rate_deg_s, e, "experiment yaw-rate cap", "faster heading change", "slower heading change", safety=True),
        _row("obstacle_stop_distance_cm", safety.obstacle_stop_distance_cm, e, "minimum forward surface-x before vx stop", "earlier forward stop", "later forward stop", safety=True),
        _row("obstacle_slow_distance_cm", safety.obstacle_slow_distance_cm, e, "forward slowdown surface-x", "earlier slowdown", "later slowdown", safety=True),
        _row("slow_speed_limit_cm_s", safety.slow_speed_limit_cm_s, e, "SafetyArbiter slowed vx cap", "faster slowed motion", "slower motion", safety=True),
        _row("side_stop_distance_cm", safety.side_stop_distance_cm, e, "sideways obstacle stop clearance", "earlier lateral stop", "later lateral stop", safety=True),
        _row("radar_max_distance_cm", safety.radar_max_distance_cm, e, "radar point range", "more distant points/work", "fewer distant points"),
        _row("radar_forward_corridor_half_width_cm", safety.radar_forward_corridor_half_width_cm, e, "SafetyArbiter forward corridor", "more forward blockers", "narrower blocker set", safety=True),
        _row("tuning_log_every_n", logging.tuning_log_every_n, u, "structured command sampling interval", "less I/O/fewer samples", "more detail/more I/O"),
        _row("radar_snapshot_every_n", logging.radar_snapshot_every_n, u, "radar point snapshot interval", "less I/O/fewer snapshots", "more detail/more I/O"),
    ]
    return rows


def safety_parameter_registry(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if bool(row["safety_sensitive"])]


def _row(
    name: str,
    value: object,
    source: ParameterSource,
    purpose: str,
    increase_effect: str,
    decrease_effect: str,
    *,
    safety: bool = False,
) -> dict[str, object]:
    return {
        "parameter": name,
        "value": value,
        "source": source.value,
        "purpose": purpose,
        "increase_effect": increase_effect,
        "decrease_effect": decrease_effect,
        "requires_flight_tuning": source is ParameterSource.UNVERIFIED_TUNING,
        "safety_sensitive": safety,
    }


__all__ = [
    "ExperimentLoggingConfig",
    "ExperimentSafetyConfig",
    "ParameterSource",
    "build_parameter_registry",
    "safety_parameter_registry",
]
