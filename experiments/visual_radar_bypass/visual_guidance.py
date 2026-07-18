"""Isolated copy of the final trajectory-vision orchestration.

This module deliberately does not add radar behavior to the production visual
files.  It calls their stable public APIs, while keeping the NPU model,
postprocess and compatible trajectory-controller configuration used by the
current ``road_trajectory_main.py``.  The obstacle experiment intentionally
retains its lower 10/8 cm/s planar limits; geometry, filtering, latency
compensation and curve handling match the final visual entry.
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
        "FlightController/Solutions/model/new_road_seg_v4_final_fp32.nb"
    )
    postprocess_mode: str = "fast-main"
    flight_height_m: float = 1.0

    max_vx_cm_s: float = 10.0
    max_vy_cm_s: float = 8.0
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
    yaw_sign: float = 1.0
    lateral_sign: float = -1.0
    target_filter_tau_s: float = 0.15
    tangent_filter_tau_s: float = 0.20
    target_filter_max_rate_px_s: float = 300.0
    tangent_filter_max_rate_deg_s: float = 45.0
    max_planar_accel_cm_s2: float = 24.0
    max_yaw_accel_deg_s2: float = 20.0
    degraded_speed_scale: float = 0.85
    curvature_slowdown_start_deg: float = 8.0
    curvature_full_slowdown_deg: float = 35.0
    min_curve_speed_cm_s: float = 10.0


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


class FrozenVisualGuidance:
    """Own the unchanged NPU perception and trajectory-point controller."""

    def __init__(self, config: FrozenVisualConfig | None = None) -> None:
        self.config = config or FrozenVisualConfig()
        cfg = self.config
        self.pipeline = PerceptionPipeline(
            camera_index=cfg.camera_index,
            camera_width=cfg.camera_width,
            camera_height=cfg.camera_height,
            camera_fps=cfg.camera_fps,
            model_path="FlightController/Solutions/model/road_yolo11n_seg_128.onnx",
            npu_model_path=cfg.npu_model_path,
            inference_backend="npu",
            postprocess_mode=cfg.postprocess_mode,
            flight_height_m=cfg.flight_height_m,
            wb_enable=False,
            wb_r=1.0,
            wb_g=1.0,
            wb_b=1.0,
            offset_comp_config=CameraOffsetCompensationConfig(enabled=False),
        )
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
                yaw_sign=cfg.yaw_sign,
                lateral_sign=cfg.lateral_sign,
                target_filter_tau_s=cfg.target_filter_tau_s,
                tangent_filter_tau_s=cfg.tangent_filter_tau_s,
                target_filter_max_rate_px_s=cfg.target_filter_max_rate_px_s,
                tangent_filter_max_rate_deg_s=cfg.tangent_filter_max_rate_deg_s,
                max_planar_accel_cm_s2=cfg.max_planar_accel_cm_s2,
                max_yaw_accel_deg_s2=cfg.max_yaw_accel_deg_s2,
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

    def sample(self, now_s: float) -> VisualSample:
        perception, age_s, stale = self.pipeline.latest_perception()
        frame, frame_time_s = self.pipeline.latest_frame()
        usable = perception is not None and not stale
        desired = self.follower.update(
            perception if usable else None,
            now_s=now_s,
        )
        return VisualSample(
            perception=perception,
            desired=desired,
            perception_age_s=float(age_s),
            perception_stale=bool(stale),
            camera_ok=bool(self.pipeline.camera_ok),
            frame=frame,
            frame_time_s=float(frame_time_s or 0.0),
            diagnostics=self.follower.last_diagnostics.as_dict(),
        )
