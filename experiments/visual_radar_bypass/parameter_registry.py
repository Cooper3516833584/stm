"""Parameter provenance for the isolated static-route experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .static_route_bypass import StaticRouteBypassConfig
from .purple_target_mission import PurpleTargetMissionConfig
from .visual_guidance import FrozenVisualConfig


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
    target_config: PurpleTargetMissionConfig | None = None,
    visual_config: FrozenVisualConfig | None = None,
) -> list[dict[str, object]]:
    fixed = ParameterSource.FIXED_REQUIREMENT
    existing = ParameterSource.EXISTING_PROJECT
    validated = ParameterSource.FLIGHT_VALIDATED
    unverified = ParameterSource.UNVERIFIED_TUNING
    v1 = StaticRouteBypassConfig()

    def v1_source(name: str) -> ParameterSource:
        return validated if getattr(config, name) == getattr(v1, name) else unverified

    rows = [
        ParameterRecord("front_fov_deg", config.front_fov_deg, fixed, "front-half-plane planner search"),
        ParameterRecord("visual_max_vx_cm_s", config.visual_max_vx_cm_s, v1_source("visual_max_vx_cm_s"), "isolated visual speed limit", True),
        ParameterRecord("avoidance_forward_ratio", config.avoidance_forward_ratio, fixed, "60 percent avoidance speed", True),
        ParameterRecord("avoidance_vx_cm_s", config.avoidance_vx_cm_s, v1_source("visual_max_vx_cm_s"), "fixed active forward target", True),
        ParameterRecord("diverge_vx_cm_s", config.active_diverge_vx_cm_s, v1_source("diverge_vx_cm_s"), "forward speed only for NORMAL-to-DIVERGE clearance acquisition", True),
        ParameterRecord("require_target_clearance_before_forward", config.require_target_clearance_before_forward, v1_source("require_target_clearance_before_forward"), "disable the v1 75 cm edge fallback before forward motion", True),
        ParameterRecord("tube_radius_cm", config.tube_radius_cm, existing, "known physical tube radius", True),
        ParameterRecord("lookahead_cm", config.lookahead_cm, existing, "initial acquisition distance", True),
        ParameterRecord("intrusion_half_width_cm", config.intrusion_half_width_cm, existing, "route intrusion gate", True),
        ParameterRecord("target_surface_clearance_cm", config.target_surface_clearance_cm, v1_source("target_surface_clearance_cm"), "outward clearance target", True),
        ParameterRecord("diverge_target_surface_clearance_cm", config.active_diverge_target_surface_clearance_cm, v1_source("diverge_target_surface_clearance_cm"), "DIVERGE-only lateral velocity target", True),
        ParameterRecord("reshift_surface_clearance_cm", config.reshift_surface_clearance_cm, v1_source("reshift_surface_clearance_cm"), "clearance hysteresis", True),
        ParameterRecord("max_outward_vy_cm_s", config.max_outward_vy_cm_s, v1_source("max_outward_vy_cm_s"), "radar-only lateral correction cap", True),
        ParameterRecord("lateral_kp_s", config.lateral_kp_s, v1_source("lateral_kp_s"), "radar-only lateral correction gain", True),
        ParameterRecord("ramp_in_s", config.ramp_in_s, v1_source("ramp_in_s"), "avoidance lateral ramp duration", True),
        ParameterRecord("association_radius_cm", config.association_radius_cm, validated, "same-tube prediction gate", True),
        ParameterRecord("static_model_tolerance_cm", config.static_model_tolerance_cm, validated, "stationary-model residual gate", True),
        ParameterRecord("static_model_bad_frames", config.static_model_bad_frames, validated, "consecutive residual failures before stop", True),
        ParameterRecord("edge_arm_deg", config.edge_arm_deg, validated, "80-to-90 degree side-pass arming"),
        ParameterRecord("clearance_run_s", config.clearance_run_s, v1_source("clearance_run_s"), "applied forward time without a newly acquired route obstacle", True),
        ParameterRecord("normal_activation_radius_cm", config.normal_activation_radius_cm, fixed, "maximum body-frame radius for NORMAL obstacle activation", True),
        ParameterRecord("clearance_reacquire_radius_cm", config.clearance_reacquire_radius_cm, fixed, "maximum body-frame radius for a new obstacle during timed clearance", True),
        ParameterRecord("trusted_point_cloud_max_x_cm", config.trusted_point_cloud_max_x_cm, existing, "trusted radar acquisition X limit", True),
        ParameterRecord("rear_margin_cm", config.rear_margin_cm, validated, "whole-tube rear margin", True),
        ParameterRecord("translation_credit_ratio", config.translation_credit_ratio, validated, "conservative command odometry", True),
        ParameterRecord("track_lost_hold_s", config.track_lost_hold_s, validated, "unexpected dropout hold", True),
        ParameterRecord("track_lost_forward_vx_cm_s", config.track_lost_forward_vx_cm_s, v1_source("track_lost_forward_vx_cm_s"), "forward speed while reacquiring a lost tracked obstacle", True),
        ParameterRecord("track_lost_use_guidance_yaw", config.track_lost_use_guidance_yaw, v1_source("track_lost_use_guidance_yaw"), "follow image-path heading during tracking dropout while suppressing visual vy", True),
        ParameterRecord("tracked_obstacle_disappear_distance_cm", config.tracked_obstacle_disappear_distance_cm, v1_source("tracked_obstacle_disappear_distance_cm"), "tracked obstacle center range treated as encounter exit", True),
        ParameterRecord("tracked_obstacle_disappear_frames", config.tracked_obstacle_disappear_frames, fixed, "consecutive 10 Hz confirmations beyond the disappearance range", True),
        ParameterRecord("max_encounter_s", config.max_encounter_s, v1_source("max_encounter_s"), "latched timeout; disabled when None", True),
        ParameterRecord("radar_timeout_s", radar_timeout_s, existing, "radar snapshot freshness and target-mission gate", True),
        ParameterRecord("safety_layer", "BYPASSED", fixed, "static-route planner output is sent directly", True),
        ParameterRecord("body_x_half_cm", 25.0, fixed, "50 cm rectangular body longitudinal half-size", True),
        ParameterRecord("body_y_half_cm", 25.0, fixed, "50 cm rectangular body lateral half-size", True),
        ParameterRecord("side_corridor_x_half_cm", 25.0, fixed, "finite lateral swept-body corridor", True),
        ParameterRecord("tuning_log_every_n", tuning_log_every_n, existing, "command diagnostic sampling"),
        ParameterRecord("radar_snapshot_every_n", radar_snapshot_every_n, existing, "physical point snapshot sampling"),
        ParameterRecord("control_rate_hz", 10.0, existing, "fused control-loop rate", True),
    ]
    if visual_config is not None:
        rows.extend(
            [
                ParameterRecord("camera_width", visual_config.camera_width, existing, "shared camera width"),
                ParameterRecord("camera_height", visual_config.camera_height, existing, "shared camera height"),
                ParameterRecord("camera_fps", visual_config.camera_fps, existing, "shared camera capture rate"),
                ParameterRecord("road_max_vx_cm_s", visual_config.max_vx_cm_s, unverified, "optimized road forward-speed cap", True),
                ParameterRecord("road_max_vy_cm_s", visual_config.max_vy_cm_s, unverified, "optimized road lateral-speed cap", True),
                ParameterRecord("road_yaw_kp", visual_config.tangent_kp_yaw, unverified, "optimized road tangent yaw gain", True),
                ParameterRecord("road_tangent_window_points", visual_config.tangent_window_points, unverified, "tangent window matched to the shorter 22 cm/s horizon", True),
                ParameterRecord("road_yaw_deadband_deg", visual_config.angle_deadband_deg, unverified, "optimized road yaw deadband"),
                ParameterRecord("road_max_yaw_rate_deg_s", visual_config.max_yaw_rate_deg_s, unverified, "optimized road yaw-rate cap", True),
                ParameterRecord("road_max_yaw_accel_deg_s2", visual_config.max_yaw_accel_deg_s2, unverified, "optimized road yaw acceleration cap", True),
                ParameterRecord("road_max_planar_accel_cm_s2", visual_config.max_planar_accel_cm_s2, unverified, "optimized road planar acceleration cap", True),
                ParameterRecord("road_max_planar_decel_cm_s2", visual_config.max_planar_decel_cm_s2, unverified, "speed-scaled independent road deceleration cap", True),
                ParameterRecord("road_curvature_yaw_ff_kp", visual_config.curvature_yaw_ff_kp, unverified, "speed-scaled signed-turn yaw feed-forward gain", True),
                ParameterRecord("road_curvature_yaw_ff_max_deg_s", visual_config.curvature_yaw_ff_max_deg_s, unverified, "speed-scaled signed-turn yaw feed-forward cap", True),
                ParameterRecord("road_curvature_yaw_ff_deadband_deg", visual_config.curvature_yaw_ff_deadband_deg, existing, "production signed-turn noise gate"),
                ParameterRecord("road_signed_turn_filter_tau_s", visual_config.signed_turn_filter_tau_s, existing, "production signed-turn direction filter"),
                ParameterRecord("road_corner_lookahead_start_deg", visual_config.corner_lookahead_start_deg, existing, "production corner lookahead activation"),
                ParameterRecord("road_corner_lookahead_full_deg", visual_config.corner_lookahead_full_deg, existing, "production full corner lookahead activation"),
                ParameterRecord("road_corner_min_lookahead_px", visual_config.corner_min_lookahead_px, existing, "shared-camera corner horizon preserving signed-turn sampling", True),
                ParameterRecord("road_corner_severity_release_tau_s", visual_config.corner_severity_release_tau_s, existing, "production corner-release hysteresis"),
                ParameterRecord("road_edge_recovery_start_ratio", visual_config.edge_recovery_start_ratio, unverified, "fixed-width normalized edge recovery start", True),
                ParameterRecord("road_edge_recovery_full_ratio", visual_config.edge_recovery_full_ratio, unverified, "fixed-width normalized full edge recovery", True),
                ParameterRecord("road_edge_recovery_lateral_kp", visual_config.edge_recovery_lateral_kp, unverified, "speed-scaled edge lateral gain", True),
                ParameterRecord("road_edge_recovery_max_vy_cm_s", visual_config.edge_recovery_max_vy_cm_s, unverified, "edge recovery lateral cap within fused profile", True),
                ParameterRecord("road_edge_yaw_start_ratio", visual_config.edge_yaw_start_ratio, unverified, "fixed-width inward yaw activation", True),
                ParameterRecord("road_edge_yaw_full_ratio", visual_config.edge_yaw_full_ratio, unverified, "fixed-width full inward yaw activation", True),
                ParameterRecord("road_edge_yaw_max_deg_s", visual_config.edge_yaw_max_deg_s, unverified, "speed-scaled inward edge yaw bias", True),
                ParameterRecord("road_edge_speed_slow_start_ratio", visual_config.edge_speed_slow_start_ratio, unverified, "fixed-width edge slowdown activation", True),
                ParameterRecord("road_edge_emergency_ratio", visual_config.edge_emergency_ratio, unverified, "fixed-width full edge slowdown activation", True),
                ParameterRecord("road_edge_emergency_vx_cap_cm_s", visual_config.edge_emergency_vx_cap_cm_s, unverified, "speed-scaled near-edge forward cap", True),
                ParameterRecord("target_max_dimension", visual_config.target_max_dimension, existing, "low-cost target downsample"),
                ParameterRecord("target_hue_min", visual_config.target_hue_min, existing, "purple HSV lower hue"),
                ParameterRecord("target_hue_max", visual_config.target_hue_max, existing, "purple HSV upper hue"),
                ParameterRecord("target_saturation_min", visual_config.target_saturation_min, existing, "purple HSV saturation gate"),
                ParameterRecord("target_value_min", visual_config.target_value_min, existing, "purple HSV value gate"),
                ParameterRecord("target_min_area_ratio", visual_config.target_min_area_ratio, existing, "purple connected-area gate"),
                ParameterRecord("target_max_rate_hz", visual_config.target_max_rate_hz, fixed, "match target work to the control loop"),
                ParameterRecord("target_stale_timeout_s", visual_config.target_stale_timeout_s, fixed, "target freshness gate", True),
            ]
        )
    if target_config is not None:
        target_purposes = {
            "high_planar_speed_cm_s": "100 cm target-pursuit vector-speed cap",
            "low_planar_speed_cm_s": "low-altitude target-pursuit vector-speed cap",
            "target_position_kp_s": "target planar position-error gain",
            "camera_ground_width_cm_at_reference": "measured ground width across the image",
            "camera_reference_altitude_cm": "altitude for the measured ground width",
            "camera_width_px": "image width used for pixel-to-centimetre scaling",
            "offset_filter_tau_s": "target offset low-pass time constant",
            "offset_filter_max_rate_px_s": "target offset slew cap",
            "max_planar_accel_cm_s2": "target planar vector acceleration cap",
            "acquire_confirm_frames": "initial target debounce",
            "clearance_confirm_frames": "radar-clear handoff debounce",
            "reach_confirm_frames": "target-centering debounce",
            "high_reach_x_px": "high-altitude vertical image threshold",
            "high_reach_y_px": "high-altitude lateral image threshold",
            "low_reach_x_px": "low-altitude vertical image threshold",
            "low_reach_y_px": "low-altitude lateral image threshold",
            "high_hover_s": "hover before descent",
            "low_hover_s": "hover before release",
            "target_altitude_cm": "payload calibration altitude",
            "return_altitude_cm": "road-follow return altitude",
            "max_vz_cm_s": "height-loop speed cap",
            "altitude_kp_s": "ALT_ADD proportional height gain",
            "altitude_tolerance_cm": "height completion band",
            "altitude_confirm_frames": "height completion debounce",
            "altitude_phase_timeout_s": "descent/climb timeout",
            "target_loss_timeout_s": "target reacquisition hold",
            "approach_timeout_s": "maximum time to high centering",
            "post_release_wait_s": "payload release settling wait",
        }
        for name, value in target_config.__dict__.items():
            rows.append(
                ParameterRecord(
                    f"target_mission.{name}",
                    value,
                    fixed if name not in {"target_position_kp_s", "offset_filter_tau_s", "offset_filter_max_rate_px_s", "max_planar_accel_cm_s2"} else unverified,
                    target_purposes[name],
                    name in {"high_planar_speed_cm_s", "low_planar_speed_cm_s", "target_position_kp_s", "camera_ground_width_cm_at_reference", "camera_reference_altitude_cm", "camera_width_px", "max_planar_accel_cm_s2", "target_altitude_cm", "return_altitude_cm", "max_vz_cm_s", "altitude_kp_s", "altitude_tolerance_cm", "altitude_phase_timeout_s", "approach_timeout_s"},
                )
            )
    return [row.as_dict() for row in rows]


__all__ = ["ParameterRecord", "ParameterSource", "build_parameter_registry"]
