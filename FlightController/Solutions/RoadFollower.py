"""Road-following mission controller."""

from __future__ import annotations

from dataclasses import dataclass

from .Safety import Command


@dataclass
class RoadFollowerConfig:
    image_width: int = 640
    max_vx_cm_s: float = 25.0
    search_yaw_rate_deg_s: float = 12.0
    max_yaw_rate_deg_s: float = 25.0
    pixel_kp_yaw: float = 0.08
    angle_kp_yaw: float = 0.4
    pixel_deadband_px: float = 20.0
    lost_timeout_s: float = 5.0
    branch_preference: str = "auto"
    min_confidence: float = 0.35
    yaw_sign: float = 1.0
    ambiguous_speed_scale: float = 0.5


class RoadFollower:
    """Converts road_perception output into a desired velocity command."""

    def __init__(self, config: RoadFollowerConfig | None = None):
        self.config = config or RoadFollowerConfig()
        self._lost_since_s: float | None = None
        self._previous_branch_label: str | None = None

    def update(self, perception, now_s: float) -> Command:
        if not self._road_is_usable(perception):
            return self._lost_command(now_s)

        self._lost_since_s = None
        branch = getattr(perception, "selected_branch", None)
        if branch is None:
            branch = self._choose_fallback_branch(perception)

        pixel_error = self._control_pixel_error(perception, branch)
        angle = float(getattr(branch, "centerline_angle", getattr(perception, "centerline_angle", 90.0)))
        yaw_rate = self._compute_yaw_rate(pixel_error, angle)

        road_state = str(getattr(perception, "road_state", "unknown"))
        vx = self.config.max_vx_cm_s
        if road_state == "ambiguous":
            vx *= self.config.ambiguous_speed_scale
        elif road_state in {"fork", "intersection"} and str(getattr(perception, "branch_decision", "")) == "auto_best":
            vx *= 0.75

        label = str(getattr(branch, "label", "unknown"))
        if label != "unknown":
            self._previous_branch_label = label
        reason = f"road_follow:{road_state}:{label}"
        return Command(vx, 0.0, 0.0, yaw_rate, reason)

    def previous_branch_label(self) -> str | None:
        return self._previous_branch_label

    def _lost_command(self, now_s: float) -> Command:
        if self._lost_since_s is None:
            self._lost_since_s = now_s
        if now_s - self._lost_since_s >= self.config.lost_timeout_s:
            return Command.zero("road_lost_timeout")
        return Command(0.0, 0.0, 0.0, self.config.search_yaw_rate_deg_s, "road_search")

    def _road_is_usable(self, perception) -> bool:
        if perception is None:
            return False
        if not bool(getattr(perception, "is_road_found", False)):
            return False
        return float(getattr(perception, "confidence", 0.0)) >= self.config.min_confidence

    def _choose_fallback_branch(self, perception):
        branches = list(getattr(perception, "branches", []) or [])
        if not branches:
            return perception
        preferred = self.config.branch_preference
        labeled = [branch for branch in branches if getattr(branch, "label", None) == preferred]
        if labeled:
            return max(labeled, key=lambda branch: float(getattr(branch, "confidence", 0.0)))
        straight = [branch for branch in branches if getattr(branch, "label", None) == "straight"]
        if straight:
            return min(straight, key=lambda branch: abs(float(getattr(branch, "pixel_error", 0.0))))
        return min(branches, key=lambda branch: abs(float(getattr(branch, "pixel_error", 0.0))))

    def _control_pixel_error(self, perception, branch) -> float:
        if branch is not None and branch is getattr(perception, "selected_branch", None):
            return float(getattr(perception, "corrected_pixel_error", getattr(branch, "pixel_error", 0.0)))
        if branch is not None:
            return float(getattr(branch, "pixel_error", getattr(perception, "corrected_pixel_error", 0.0)))
        return float(getattr(perception, "corrected_pixel_error", 0.0))

    def _compute_yaw_rate(self, pixel_error: float, centerline_angle_deg: float) -> float:
        if abs(pixel_error) < self.config.pixel_deadband_px:
            pixel_error = 0.0
        angle_error = float(centerline_angle_deg) - 90.0
        yaw_rate = (
            self.config.pixel_kp_yaw * pixel_error
            + self.config.angle_kp_yaw * angle_error
        )
        yaw_rate *= self.config.yaw_sign
        return _clamp(
            yaw_rate,
            -self.config.max_yaw_rate_deg_s,
            self.config.max_yaw_rate_deg_s,
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


__all__ = ["RoadFollower", "RoadFollowerConfig"]
