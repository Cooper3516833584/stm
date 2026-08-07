"""Parameter provenance for the isolated static-route experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .static_route_bypass import StaticRouteBypassConfig


class ParameterSource(str, Enum):
    FIXED_REQUIREMENT = "FIXED_REQUIREMENT"
    EXISTING_PROJECT = "EXISTING_PROJECT"
    FLIGHT_VALIDATED = "FLIGHT_VALIDATED"
    UNVERIFIED_TUNING = "UNVERIFIED_TUNING"


@dataclass(frozen=True)
class ParameterRecord:
    parameter: str
    value: object
    source: ParameterSource
    purpose: str
    safety_sensitive: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "value": self.value,
            "source": self.source.value,
            "purpose": self.purpose,
            "requires_flight_tuning": self.source is ParameterSource.UNVERIFIED_TUNING,
            "safety_sensitive": self.safety_sensitive,
        }


def build_parameter_registry(
    config: StaticRouteBypassConfig,
    *,
    radar_timeout_s: float,
    tuning_log_every_n: int,
    radar_snapshot_every_n: int,
) -> list[dict[str, object]]:
    fixed = ParameterSource.FIXED_REQUIREMENT
    existing = ParameterSource.EXISTING_PROJECT
    validated = ParameterSource.FLIGHT_VALIDATED
    rows = [
        ParameterRecord("front_fov_deg", config.front_fov_deg, fixed, "front-half-plane planner search"),
        ParameterRecord("visual_max_vx_cm_s", config.visual_max_vx_cm_s, existing, "isolated visual speed limit", True),
        ParameterRecord("avoidance_forward_ratio", config.avoidance_forward_ratio, fixed, "60 percent avoidance speed", True),
        ParameterRecord("avoidance_vx_cm_s", config.avoidance_vx_cm_s, fixed, "fixed active forward target", True),
        ParameterRecord("tube_radius_cm", config.tube_radius_cm, existing, "known physical tube radius", True),
        ParameterRecord("lookahead_cm", config.lookahead_cm, existing, "initial acquisition distance", True),
        ParameterRecord("intrusion_half_width_cm", config.intrusion_half_width_cm, existing, "route intrusion gate", True),
        ParameterRecord("target_surface_clearance_cm", config.target_surface_clearance_cm, validated, "outward clearance target", True),
        ParameterRecord("reshift_surface_clearance_cm", config.reshift_surface_clearance_cm, validated, "clearance hysteresis", True),
        ParameterRecord("max_outward_vy_cm_s", config.max_outward_vy_cm_s, validated, "radar-only lateral correction cap", True),
        ParameterRecord("association_radius_cm", config.association_radius_cm, validated, "same-tube prediction gate", True),
        ParameterRecord("static_model_tolerance_cm", config.static_model_tolerance_cm, validated, "stationary-model residual gate", True),
        ParameterRecord("static_model_bad_frames", config.static_model_bad_frames, validated, "consecutive residual failures before stop", True),
        ParameterRecord("edge_arm_deg", config.edge_arm_deg, validated, "80-to-90 degree side-pass arming"),
        ParameterRecord("rear_margin_cm", config.rear_margin_cm, validated, "whole-tube rear margin", True),
        ParameterRecord("translation_credit_ratio", config.translation_credit_ratio, validated, "conservative command odometry", True),
        ParameterRecord("track_lost_hold_s", config.track_lost_hold_s, validated, "unexpected dropout hold", True),
        ParameterRecord("max_encounter_s", config.max_encounter_s, validated, "latched timeout", True),
        ParameterRecord("radar_timeout_s", radar_timeout_s, existing, "Safety radar freshness gate", True),
        ParameterRecord("obstacle_stop_distance_cm", 80.0, existing, "Safety forward stop", True),
        ParameterRecord("side_stop_distance_cm", 45.0, existing, "Safety lateral stop", True),
        ParameterRecord("body_x_half_cm", 25.0, fixed, "50 cm rectangular body longitudinal half-size", True),
        ParameterRecord("body_y_half_cm", 25.0, fixed, "50 cm rectangular body lateral half-size", True),
        ParameterRecord("side_corridor_x_half_cm", 25.0, fixed, "finite lateral swept-body corridor", True),
        ParameterRecord("tuning_log_every_n", tuning_log_every_n, existing, "command diagnostic sampling"),
        ParameterRecord("radar_snapshot_every_n", radar_snapshot_every_n, existing, "physical point snapshot sampling"),
    ]
    return [row.as_dict() for row in rows]


__all__ = ["ParameterRecord", "ParameterSource", "build_parameter_registry"]
