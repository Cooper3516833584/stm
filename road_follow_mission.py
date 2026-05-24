"""Mission logic for YOLO road-line following."""

from __future__ import annotations

from dataclasses import dataclass

from autonomy_command import VelocityCommand


@dataclass
class RoadFollowConfig:
    cruise_speed_cm_s: float = 20.0
    search_yaw_rate_deg_s: float = 12.0
    lost_timeout_s: float = 5.0
    min_confidence: float = 0.35
    image_width_px: float = 640.0
    pixel_yaw_gain: float = 18.0
    angle_yaw_gain: float = 0.35
    max_yaw_rate_deg_s: float = 25.0
    turn_slowdown_threshold_deg_s: float = 15.0


class RoadFollowMission:
    def __init__(self, config: RoadFollowConfig | None = None):
        self.config = config or RoadFollowConfig()
        self._lost_since_s: float | None = None

    def update(self, road_result, now_s: float) -> VelocityCommand:
        cfg = self.config
        if not self._road_is_usable(road_result):
            if self._lost_since_s is None:
                self._lost_since_s = now_s
            lost_for = now_s - self._lost_since_s
            if lost_for >= cfg.lost_timeout_s:
                return VelocityCommand.zero("road_lost_timeout")
            return VelocityCommand(0.0, 0.0, 0.0, cfg.search_yaw_rate_deg_s, "road_search")

        self._lost_since_s = None
        pixel_error = float(getattr(road_result, "corrected_pixel_error", getattr(road_result, "pixel_error", 0.0)))
        angle = float(getattr(road_result, "centerline_angle", 90.0))
        yaw_rate = self._compute_yaw_rate(pixel_error, angle)
        vx = cfg.cruise_speed_cm_s
        if abs(yaw_rate) >= cfg.turn_slowdown_threshold_deg_s:
            vx *= 0.6

        state = getattr(road_result, "road_state", "unknown")
        return VelocityCommand(vx, 0.0, 0.0, yaw_rate, f"road_follow_{state}")

    def _road_is_usable(self, road_result) -> bool:
        if road_result is None:
            return False
        if not bool(getattr(road_result, "is_road_found", False)):
            return False
        confidence = float(getattr(road_result, "confidence", 0.0))
        return confidence >= self.config.min_confidence

    def _compute_yaw_rate(self, pixel_error: float, centerline_angle_deg: float) -> float:
        cfg = self.config
        half_width = max(1.0, cfg.image_width_px / 2.0)
        normalized_error = pixel_error / half_width
        angle_error = 90.0 - centerline_angle_deg
        yaw_rate = normalized_error * cfg.pixel_yaw_gain + angle_error * cfg.angle_yaw_gain
        return max(-cfg.max_yaw_rate_deg_s, min(cfg.max_yaw_rate_deg_s, yaw_rate))

