"""Isolated copy of the final trajectory-vision orchestration.

This module deliberately does not add radar behavior to the production visual
files.  It calls their stable public APIs while freezing the flight-validated
v1 controller defaults.  Higher-speed profiles selectively enable speed-scaled
corner feed-forward and edge recovery from ``road_trajectory_main.py`` without
inheriting its 45 cm/s limits wholesale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from perception_pipeline import PerceptionPipeline
from road_perception import CameraOffsetCompensationConfig
from FlightController.Solutions.Safety import Command
from FlightController.Solutions.TrajectoryPointFollower import (
    TrajectoryPointFollower,
    TrajectoryPointFollowerConfig,
)


@dataclass(frozen=True)
class FrozenVisualConfig:
    camera_index: int = 7
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    npu_model_path: str = (
        "FlightController/Solutions/model/new_road_seg_v5_final_fp32.nb"
    )
    postprocess_mode: str = "fast-main"
    instance_selection: str = "highest-confidence"
    flight_height_m: float = 1.0
    target_enable: bool = True
    target_max_dimension: int = 256
    target_hue_min: float = 135.0
    target_hue_max: float = 179.0
    target_saturation_min: float = 90.0
    target_value_min: float = 60.0
    target_min_area_ratio: float = 0.005
    target_max_rate_hz: float = 10.0
    target_stale_timeout_s: float = 0.5

    max_vx_cm_s: float = 14.0
    max_vy_cm_s: float = 10.0
    max_yaw_rate_deg_s: float = 10.0
    reach_radius_px: float = 30.0
    min_forward_lookahead_px: float = 24.0
    max_forward_lookahead_px: float = 64.0
    lookahead_speed_gain_px_per_cm_s: float = 1.2
    latency_compensation_s: float = 0.134
    physical_road_width_cm: float = 50.0
    max_latency_prediction_px: float = 16.0
    tangent_window_points: int = 5
    tangent_kp_yaw: float = 0.25
    angle_deadband_deg: float = 3.0
    lateral_deadband_px: float = 8.0
    lateral_kp_cm_s_per_px: float = 0.10
    normal_max_vy_cm_s: float = 12.0
    curvature_yaw_ff_kp: float = 0.0
    curvature_yaw_ff_max_deg_s: float = 0.0
    curvature_yaw_ff_deadband_deg: float = 6.0
    signed_turn_filter_tau_s: float = 0.08
    corner_lookahead_start_deg: float = 30.0
    corner_lookahead_full_deg: float = 75.0
    corner_min_lookahead_px: float = 75.0
    corner_severity_release_tau_s: float = 0.25
    edge_recovery_start_ratio: float = 1.0
    edge_recovery_full_ratio: float = 1.0
    edge_recovery_lateral_kp: float = 0.22
    edge_recovery_max_vy_cm_s: float = 0.0
    edge_yaw_start_ratio: float = 1.0
    edge_yaw_full_ratio: float = 1.0
    edge_yaw_max_deg_s: float = 0.0
    edge_speed_slow_start_ratio: float = 1.0
    edge_emergency_ratio: float = 1.0
    edge_emergency_vx_cap_cm_s: float = 0.0
    yaw_sign: float = 1.0
    lateral_sign: float = -1.0
    target_filter_tau_s: float = 0.15
    tangent_filter_tau_s: float = 0.20
    target_filter_max_rate_px_s: float = 300.0
    tangent_filter_max_rate_deg_s: float = 45.0
    max_planar_accel_cm_s2: float = 24.0
    max_planar_decel_cm_s2: float = 24.0
    max_yaw_accel_deg_s2: float = 20.0
    road_loss_grace_s: float = 0.30
    road_loss_grace_vx_scale: float = 0.80
    road_loss_grace_vy_scale: float = 0.50
    road_loss_grace_yaw_scale: float = 0.70
    sharp_left_recovery_enabled: bool = False
    sharp_left_recovery_confirm_frames: int = 2
    sharp_left_recovery_reacquire_frames: int = 3
    sharp_left_recovery_history_frames: int = 5
    sharp_left_recovery_min_confirm_yaw_deg_s: float = 3.0
    sharp_left_recovery_min_hold_yaw_deg_s: float = 6.0
    sharp_left_recovery_max_hold_yaw_deg_s: float = 10.0
    sharp_left_recovery_timeout_s: float = 8.0
    degraded_speed_scale: float = 0.85
    curvature_slowdown_start_deg: float = 12.0
    curvature_full_slowdown_deg: float = 42.0
    min_curve_speed_cm_s: float = 12.0


@dataclass(frozen=True)
class VisualSample:
    perception: Any | None
    desired: Command
    perception_age_s: float
    perception_stale: bool
    camera_ok: bool
    frame: Any | None
    frame_time_s: float
    diagnostics: dict[str, object]
    road_guidance_usable: bool
    target: Any | None = None
    target_age_s: float = float("inf")
    target_stale: bool = True


class FrozenVisualGuidance:
    """Own the unchanged NPU perception and trajectory-point controller."""

    def __init__(self, config: FrozenVisualConfig | None = None, *, process_runtime=None) -> None:
        self.config = config or FrozenVisualConfig()
        cfg = self.config
        if process_runtime is None:
            self.pipeline = PerceptionPipeline(
                camera_index=cfg.camera_index,
                camera_width=cfg.camera_width,
                camera_height=cfg.camera_height,
                camera_fps=cfg.camera_fps,
                model_path="FlightController/Solutions/model/road_yolo11n_seg_128.onnx",
                npu_model_path=cfg.npu_model_path,
                inference_backend="npu",
                postprocess_mode=cfg.postprocess_mode,
                instance_selection=cfg.instance_selection,
                flight_height_m=cfg.flight_height_m,
                wb_enable=False,
                wb_r=1.0,
                wb_g=1.0,
                wb_b=1.0,
                offset_comp_config=CameraOffsetCompensationConfig(enabled=False),
                target_enable=cfg.target_enable,
                target_max_dimension=cfg.target_max_dimension,
                target_hue_min=cfg.target_hue_min,
                target_hue_max=cfg.target_hue_max,
                target_saturation_min=cfg.target_saturation_min,
                target_value_min=cfg.target_value_min,
                target_min_area_ratio=cfg.target_min_area_ratio,
                target_max_rate_hz=cfg.target_max_rate_hz,
                target_stale_timeout_s=cfg.target_stale_timeout_s,
            )
        else:
            from FlightController.Runtime import ProcessVisionPipeline

            self.pipeline = ProcessVisionPipeline(process_runtime)
        self.follower = TrajectoryPointFollower(
            TrajectoryPointFollowerConfig(
                image_width=cfg.camera_width,
                image_height=cfg.camera_height,
                max_vx_cm_s=cfg.max_vx_cm_s,
                max_vy_cm_s=cfg.max_vy_cm_s,
                max_yaw_rate_deg_s=cfg.max_yaw_rate_deg_s,
                reach_radius_px=cfg.reach_radius_px,
                min_forward_lookahead_px=cfg.min_forward_lookahead_px,
                max_forward_lookahead_px=cfg.max_forward_lookahead_px,
                lookahead_speed_gain_px_per_cm_s=(
                    cfg.lookahead_speed_gain_px_per_cm_s
                ),
                latency_compensation_s=cfg.latency_compensation_s,
                physical_road_width_cm=cfg.physical_road_width_cm,
                max_latency_prediction_px=cfg.max_latency_prediction_px,
                tangent_window_points=cfg.tangent_window_points,
                tangent_kp_yaw=cfg.tangent_kp_yaw,
                tangent_deadband_deg=cfg.angle_deadband_deg,
                lateral_deadband_px=cfg.lateral_deadband_px,
                lateral_kp_cm_s_per_px=cfg.lateral_kp_cm_s_per_px,
                normal_max_vy_cm_s=cfg.normal_max_vy_cm_s,
                curvature_yaw_ff_kp=cfg.curvature_yaw_ff_kp,
                curvature_yaw_ff_max_deg_s=cfg.curvature_yaw_ff_max_deg_s,
                curvature_yaw_ff_deadband_deg=cfg.curvature_yaw_ff_deadband_deg,
                signed_turn_filter_tau_s=cfg.signed_turn_filter_tau_s,
                corner_lookahead_start_deg=cfg.corner_lookahead_start_deg,
                corner_lookahead_full_deg=cfg.corner_lookahead_full_deg,
                corner_min_lookahead_px=cfg.corner_min_lookahead_px,
                corner_severity_release_tau_s=cfg.corner_severity_release_tau_s,
                edge_recovery_start_ratio=cfg.edge_recovery_start_ratio,
                edge_recovery_full_ratio=cfg.edge_recovery_full_ratio,
                edge_recovery_lateral_kp=cfg.edge_recovery_lateral_kp,
                edge_recovery_max_vy_cm_s=cfg.edge_recovery_max_vy_cm_s,
                edge_yaw_start_ratio=cfg.edge_yaw_start_ratio,
                edge_yaw_full_ratio=cfg.edge_yaw_full_ratio,
                edge_yaw_max_deg_s=cfg.edge_yaw_max_deg_s,
                edge_speed_slow_start_ratio=cfg.edge_speed_slow_start_ratio,
                edge_emergency_ratio=cfg.edge_emergency_ratio,
                edge_emergency_vx_cap_cm_s=cfg.edge_emergency_vx_cap_cm_s,
                yaw_sign=cfg.yaw_sign,
                lateral_sign=cfg.lateral_sign,
                target_filter_tau_s=cfg.target_filter_tau_s,
                tangent_filter_tau_s=cfg.tangent_filter_tau_s,
                target_filter_max_rate_px_s=cfg.target_filter_max_rate_px_s,
                tangent_filter_max_rate_deg_s=cfg.tangent_filter_max_rate_deg_s,
                max_planar_accel_cm_s2=cfg.max_planar_accel_cm_s2,
                max_planar_decel_cm_s2=cfg.max_planar_decel_cm_s2,
                max_yaw_accel_deg_s2=cfg.max_yaw_accel_deg_s2,
                lost_grace_s=cfg.road_loss_grace_s,
                lost_grace_vx_scale=cfg.road_loss_grace_vx_scale,
                lost_grace_vy_scale=cfg.road_loss_grace_vy_scale,
                lost_grace_yaw_scale=cfg.road_loss_grace_yaw_scale,
                sharp_left_recovery_enabled=cfg.sharp_left_recovery_enabled,
                sharp_left_recovery_confirm_frames=(
                    cfg.sharp_left_recovery_confirm_frames
                ),
                sharp_left_recovery_reacquire_frames=(
                    cfg.sharp_left_recovery_reacquire_frames
                ),
                sharp_left_recovery_history_frames=(
                    cfg.sharp_left_recovery_history_frames
                ),
                sharp_left_recovery_min_confirm_yaw_deg_s=(
                    cfg.sharp_left_recovery_min_confirm_yaw_deg_s
                ),
                sharp_left_recovery_min_hold_yaw_deg_s=(
                    cfg.sharp_left_recovery_min_hold_yaw_deg_s
                ),
                sharp_left_recovery_max_hold_yaw_deg_s=(
                    cfg.sharp_left_recovery_max_hold_yaw_deg_s
                ),
                sharp_left_recovery_timeout_s=cfg.sharp_left_recovery_timeout_s,
                degraded_speed_scale=cfg.degraded_speed_scale,
                curvature_slowdown_start_deg=cfg.curvature_slowdown_start_deg,
                curvature_full_slowdown_deg=cfg.curvature_full_slowdown_deg,
                min_curve_speed_cm_s=cfg.min_curve_speed_cm_s,
            )
        )

    def start(self) -> None:
        self.pipeline.start()

    def stop(self) -> None:
        self.pipeline.stop()

    def latest_perception(self):
        return self.pipeline.latest_perception()

    def disable_target(self) -> None:
        disable = getattr(self.pipeline, "disable_target", None)
        if callable(disable):
            disable()

    def set_sharp_left_recovery_armed(self, armed: bool) -> None:
        self.follower.set_sharp_left_recovery_armed(armed)

    def sample(self, now_s: float) -> VisualSample:
        perception, age_s, stale = self.pipeline.latest_perception()
        target, target_age_s, target_stale = self.pipeline.latest_target(
            max_age_s=self.config.target_stale_timeout_s
        )
        frame, frame_time_s = self.pipeline.latest_frame()
        camera_ok = bool(self.pipeline.camera_ok)
        perception_fresh = perception is not None and not stale
        usable = perception_fresh and camera_ok
        previous_state = self.follower.last_diagnostics.state
        # Debounce only a fresh single-frame road miss after tracking has begun.
        # Sensor failure and stale perception must still stop immediately.
        allow_lost_grace = bool(
            usable
            and previous_state
            in {
                "tracking",
                "road_lost_grace",
                "sharp_left_recovery",
                "sharp_left_recovery_timeout",
            }
        )
        desired = self.follower.update(
            perception if usable else None,
            now_s=now_s,
            allow_lost_grace=allow_lost_grace,
        )
        diagnostics = self.follower.last_diagnostics.as_dict()
        road_guidance_usable = bool(
            usable
            and diagnostics.get("state")
            in {
                "tracking",
                "road_lost_grace",
                "sharp_left_recovery",
                "sharp_left_recovery_timeout",
            }
        )
        return VisualSample(
            perception=perception,
            desired=desired,
            perception_age_s=float(age_s),
            perception_stale=bool(stale),
            camera_ok=camera_ok,
            frame=frame,
            frame_time_s=float(frame_time_s or 0.0),
            diagnostics=diagnostics,
            road_guidance_usable=road_guidance_usable,
            target=target,
            target_age_s=float(target_age_s),
            target_stale=bool(target_stale),
        )
