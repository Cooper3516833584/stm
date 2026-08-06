"""Optional right-half-plane radar session followed by visual-only tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .radar_bypass import ObstacleBypassState


@dataclass(frozen=True)
class RightHalfHandoffConfig:
    normal_disable_s: float = 5.0


class RightHalfRadarHandoff:
    """Limit radar to clockwise 0..180 degrees, then retire it for the flight.

    Body +X is forward and +Y is left, so the requested clockwise half-plane
    is exactly ``y <= 0``.  Radar retirement is armed only by a completed
    FORWARD_RECOVERY -> NORMAL transition.  Any return to bypass before the
    stability timer expires cancels that timer.
    """

    def __init__(self, config: RightHalfHandoffConfig | None = None) -> None:
        self.config = config or RightHalfHandoffConfig()
        self.radar_disabled = False
        self._normal_started_s: float | None = None
        self.normal_elapsed_s = 0.0

    @staticmethod
    def filter_right_half_plane(points_body_cm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_body_cm, dtype=float)
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        points = points.reshape(-1, 2)
        finite = np.all(np.isfinite(points), axis=1)
        return points[finite & (points[:, 1] <= 0.0)]

    def observe(
        self,
        previous_state: ObstacleBypassState,
        current_state: ObstacleBypassState,
        now_s: float,
        bypass_pending: bool = False,
    ) -> bool:
        """Return True exactly once when radar processing should be stopped."""
        if self.radar_disabled:
            return False

        if bypass_pending or current_state in {
            ObstacleBypassState.BYPASS_LEFT,
            ObstacleBypassState.BYPASS_RIGHT,
        }:
            self._normal_started_s = None
            self.normal_elapsed_s = 0.0
            return False

        if (
            previous_state == ObstacleBypassState.FORWARD_RECOVERY
            and current_state == ObstacleBypassState.NORMAL
        ):
            self._normal_started_s = float(now_s)
            self.normal_elapsed_s = 0.0
            return False

        if self._normal_started_s is None:
            return False
        if current_state != ObstacleBypassState.NORMAL:
            self._normal_started_s = None
            self.normal_elapsed_s = 0.0
            return False

        self.normal_elapsed_s = max(0.0, float(now_s) - self._normal_started_s)
        if self.normal_elapsed_s < max(0.0, self.config.normal_disable_s):
            return False

        self.radar_disabled = True
        return True

    def diagnostics(self) -> dict[str, object]:
        return {
            "search_sector": "clockwise_0_to_180_deg",
            "body_half_plane": "y<=0",
            "normal_disable_s": self.config.normal_disable_s,
            "normal_elapsed_s": self.normal_elapsed_s,
            "radar_disabled": self.radar_disabled,
        }


__all__ = ["RightHalfHandoffConfig", "RightHalfRadarHandoff"]
