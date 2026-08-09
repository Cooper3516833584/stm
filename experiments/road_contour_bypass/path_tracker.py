"""Monotonic lookahead tracker for a frozen encounter-frame path."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from FlightController.Solutions.Safety import Command


@dataclass(frozen=True)
class PathTrackerConfig:
    lookahead_distance_cm: float = 25.0
    bypass_nominal_speed_cm_s: float = 20.0
    bypass_max_speed_cm_s: float = 22.0
    bypass_min_speed_cm_s: float = 14.0
    max_vy_cm_s: float = 12.0
    max_planar_accel_cm_s2: float = 36.0
    max_planar_decel_cm_s2: float = 60.0
    path_complete_progress: float = 0.95
    path_complete_distance_cm: float = 22.0


@dataclass(frozen=True)
class PathTrackingResult:
    command: Command
    nearest_index: int
    target_index: int
    progress: float
    target_x_cm: float
    target_y_cm: float
    speed_cm_s: float
    tangent_heading_deg: float
    complete: bool


class FrozenPathTracker:
    """Track one immutable path; no radar data or curve fitting enters here."""

    def __init__(self, config: PathTrackerConfig | None = None) -> None:
        self.config = config or PathTrackerConfig()
        self.path_samples = np.empty((0, 2), dtype=float)
        self._arc_lengths = np.empty(0, dtype=float)
        self.nearest_index = 0
        self._last_velocity_body = np.zeros(2, dtype=float)
        self.last_result = PathTrackingResult(
            Command.zero("path_not_set"), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        )

    def set_path(self, path_samples: np.ndarray) -> None:
        path = np.asarray(path_samples, dtype=float).reshape(-1, 2).copy()
        if len(path) < 2 or not np.all(np.isfinite(path)):
            raise ValueError("path must contain at least two finite samples")
        path.setflags(write=False)
        self.path_samples = path
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self._arc_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        self.nearest_index = 0
        self._last_velocity_body[:] = 0.0

    def reset(self) -> None:
        self.path_samples = np.empty((0, 2), dtype=float)
        self._arc_lengths = np.empty(0, dtype=float)
        self.nearest_index = 0
        self._last_velocity_body[:] = 0.0

    def update(
        self,
        *,
        local_x_cm: float,
        local_y_cm: float,
        local_yaw_deg: float,
        dt_s: float,
    ) -> PathTrackingResult:
        if len(self.path_samples) < 2:
            return self.last_result
        position = np.array((local_x_cm, local_y_cm), dtype=float)
        remaining = self.path_samples[self.nearest_index :]
        forward_offset = int(np.argmin(np.linalg.norm(remaining - position, axis=1)))
        self.nearest_index += forward_offset
        total_length = max(1e-6, float(self._arc_lengths[-1]))
        progress = float(self._arc_lengths[self.nearest_index] / total_length)

        target_distance = self._arc_lengths[self.nearest_index] + self.config.lookahead_distance_cm
        target_index = int(np.searchsorted(self._arc_lengths, target_distance, side="left"))
        target_index = min(target_index, len(self.path_samples) - 1)
        target = self.path_samples[target_index]
        delta = target - position
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6 and target_index < len(self.path_samples) - 1:
            target_index += 1
            target = self.path_samples[target_index]
            delta = target - position
            distance = float(np.linalg.norm(delta))

        tangent_heading_deg = self.tangent_heading_deg(self.nearest_index)
        speed = self._speed_for_curvature(self.nearest_index)
        direction_encounter = delta / distance if distance > 1e-6 else np.zeros(2, dtype=float)
        desired_encounter = direction_encounter * speed
        yaw_rad = math.radians(local_yaw_deg)
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        desired_body = np.array(
            (
                c * desired_encounter[0] + s * desired_encounter[1],
                -s * desired_encounter[0] + c * desired_encounter[1],
            ),
            dtype=float,
        )
        desired_body[1] = float(
            np.clip(desired_body[1], -self.config.max_vy_cm_s, self.config.max_vy_cm_s)
        )
        velocity_body = self._slew_velocity(desired_body, max(0.0, float(dt_s)))
        self._last_velocity_body = velocity_body
        complete = bool(
            progress >= self.config.path_complete_progress
            or np.linalg.norm(position - self.path_samples[-1]) <= self.config.path_complete_distance_cm
        )
        command = (
            Command.zero("frozen_path_complete")
            if complete
            else Command(
                vx_cm_s=float(velocity_body[0]),
                vy_cm_s=float(velocity_body[1]),
                reason="frozen_contour_path",
            )
        )
        self.last_result = PathTrackingResult(
            command=command,
            nearest_index=self.nearest_index,
            target_index=target_index,
            progress=progress,
            target_x_cm=float(target[0]),
            target_y_cm=float(target[1]),
            speed_cm_s=float(np.linalg.norm(velocity_body)),
            tangent_heading_deg=tangent_heading_deg,
            complete=complete,
        )
        return self.last_result

    def tangent_heading_deg(self, index: int | None = None, ahead_cm: float = 15.0) -> float:
        if len(self.path_samples) < 2:
            return 0.0
        index = self.nearest_index if index is None else int(np.clip(index, 0, len(self.path_samples) - 1))
        target_distance = self._arc_lengths[index] + max(1.0, ahead_cm)
        target_index = min(
            int(np.searchsorted(self._arc_lengths, target_distance, side="left")),
            len(self.path_samples) - 1,
        )
        if target_index == index:
            target_index = min(index + 1, len(self.path_samples) - 1)
        tangent = self.path_samples[target_index] - self.path_samples[index]
        return math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))

    def _speed_for_curvature(self, index: int) -> float:
        heading_15 = self.tangent_heading_deg(index, ahead_cm=15.0)
        target_distance = self._arc_lengths[index] + 15.0
        index_15 = min(
            int(np.searchsorted(self._arc_lengths, target_distance, side="left")),
            len(self.path_samples) - 1,
        )
        heading_30 = self.tangent_heading_deg(index_15, ahead_cm=15.0)
        turn = abs(_wrap_deg(heading_30 - heading_15))
        cfg = self.config
        if turn <= 10.0:
            return cfg.bypass_max_speed_cm_s
        if turn >= 50.0:
            return cfg.bypass_min_speed_cm_s
        ratio = (turn - 10.0) / 40.0
        return cfg.bypass_nominal_speed_cm_s + ratio * (
            cfg.bypass_min_speed_cm_s - cfg.bypass_nominal_speed_cm_s
        )

    def _slew_velocity(self, desired: np.ndarray, dt_s: float) -> np.ndarray:
        if dt_s <= 0.0:
            return desired.copy()
        delta = desired - self._last_velocity_body
        delta_norm = float(np.linalg.norm(delta))
        accelerating = float(np.linalg.norm(desired)) > float(np.linalg.norm(self._last_velocity_body))
        limit = (
            self.config.max_planar_accel_cm_s2
            if accelerating
            else self.config.max_planar_decel_cm_s2
        ) * dt_s
        if delta_norm > limit > 0.0:
            delta *= limit / delta_norm
        return self._last_velocity_body + delta


PathTracker = FrozenPathTracker


def _wrap_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


__all__ = [
    "PathTrackerConfig",
    "PathTrackingResult",
    "FrozenPathTracker",
    "PathTracker",
]
