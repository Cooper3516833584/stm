from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class PlannerConfig:
    max_speed_cm_s: float = 35.0
    approach_speed_cm_s: float = 20.0
    yaw_rate_limit_deg_s: float = 30.0
    obstacle_stop_distance_cm: float = 45.0
    obstacle_slow_distance_cm: float = 90.0
    target_center_deadband_px: float = 30.0
    camera_fov_deg: float = 70.0
    enable_free_flight: bool = False
    free_flight_speed_cm_s: float = 20.0
    forward_corridor_half_width_cm: float = 35.0
    min_obstacle_distance_cm: float = 10.0
    debounce_frames: int = 3


@dataclass
class TargetObservation:
    center_px: tuple[float, float]
    image_size: tuple[int, int]
    confidence: float
    class_name: str = "target"


@dataclass
class VelocityCommand:
    vx_cm_s: float
    vy_cm_s: float
    vz_cm_s: float
    yaw_rate_deg_s: float
    reason: str


class LocalPlanner:
    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        self._debounce_history: deque = deque(maxlen=max(1, self.config.debounce_frames))
        self._last_stable_state: str = "none"  # "none" | "stop" | "slow" | "cruise"

    def plan(
        self,
        *,
        obstacles_body_cm: np.ndarray,
        target: TargetObservation | None,
    ) -> VelocityCommand:
        if target is None:
            if not self.config.enable_free_flight:
                return VelocityCommand(0.0, 0.0, 0.0, 0.0, "no_target")
            return self._plan_free_flight(obstacles_body_cm)

        yaw_rate = self._target_yaw_rate(target)
        vx = self.config.approach_speed_cm_s
        reasons = ["target"]

        forward_distance = self._nearest_forward_obstacle_cm(obstacles_body_cm)
        if forward_distance is not None:
            if forward_distance < self.config.obstacle_stop_distance_cm:
                vx = 0.0
                reasons.append("obstacle_stop")
            elif forward_distance < self.config.obstacle_slow_distance_cm:
                vx *= 0.5
                reasons.append("obstacle_slow")

        vx = self._clip(vx, -self.config.max_speed_cm_s, self.config.max_speed_cm_s)
        yaw_rate = self._clip(
            yaw_rate,
            -self.config.yaw_rate_limit_deg_s,
            self.config.yaw_rate_limit_deg_s,
        )
        return VelocityCommand(vx, 0.0, 0.0, yaw_rate, "+".join(reasons))

    def _classify_obstacle_state(self, obstacles_body_cm: np.ndarray) -> str:
        d = self._nearest_forward_obstacle_cm(obstacles_body_cm)
        if d is None:
            return "none"
        if d < self.config.obstacle_stop_distance_cm:
            return "stop"
        if d < self.config.obstacle_slow_distance_cm:
            return "slow"
        return "cruise"

    def _debounce(self, raw_state: str) -> str:
        self._debounce_history.append(raw_state)
        n = self.config.debounce_frames
        # 新状态需要连续 N 帧一致才生效，否则保持旧状态
        if len(self._debounce_history) >= n:
            recent = list(self._debounce_history)[-n:]
            if all(s == raw_state for s in recent):
                self._last_stable_state = raw_state
        return self._last_stable_state

    def _plan_free_flight(self, obstacles_body_cm: np.ndarray) -> VelocityCommand:
        raw_state = self._classify_obstacle_state(obstacles_body_cm)
        stable_state = self._debounce(raw_state)
        reasons = ["free_flight"]

        if stable_state == "stop":
            vx = 0.0
            reasons.append("obstacle_stop")
        elif stable_state == "slow":
            vx = self.config.free_flight_speed_cm_s * 0.5
            reasons.append("obstacle_slow")
        elif stable_state == "none":
            vx = self.config.free_flight_speed_cm_s
        else:  # cruise
            vx = self.config.free_flight_speed_cm_s

        if raw_state != stable_state:
            reasons.append(f"debounce({raw_state})")

        vx = self._clip(vx, -self.config.max_speed_cm_s, self.config.max_speed_cm_s)
        return VelocityCommand(vx, 0.0, 0.0, 0.0, "+".join(reasons))

    def _target_yaw_rate(self, target: TargetObservation) -> float:
        width, _ = target.image_size
        if width <= 0:
            return 0.0
        offset_px = target.center_px[0] - width / 2.0
        if abs(offset_px) <= self.config.target_center_deadband_px:
            return 0.0
        normalized_offset = offset_px / (width / 2.0)
        angle_error_deg = normalized_offset * (self.config.camera_fov_deg / 2.0)
        return self._clip(
            angle_error_deg,
            -self.config.yaw_rate_limit_deg_s,
            self.config.yaw_rate_limit_deg_s,
        )

    def _nearest_forward_obstacle_cm(self, obstacles_body_cm: np.ndarray) -> float | None:
        points = np.asarray(obstacles_body_cm, dtype=float)
        if points.size == 0:
            return None
        points = points.reshape(-1, 2)
        half_width = self.config.forward_corridor_half_width_cm
        min_dist = self.config.min_obstacle_distance_cm
        forward = points[
            (points[:, 0] > min_dist) & (np.abs(points[:, 1]) < half_width)
        ]
        if forward.size == 0:
            return None
        return float(np.min(forward[:, 0]))

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))


__all__ = ["LocalPlanner", "PlannerConfig", "TargetObservation", "VelocityCommand"]
