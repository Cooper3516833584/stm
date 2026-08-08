"""Flight indicator policy for the fused visual/radar entry point."""

from __future__ import annotations

import time
from typing import Callable


GREEN = (0, 255, 0)
RED = (255, 0, 0)
OFF = (0, 0, 0)

UNEXPECTED_PLANNER_STATES = frozenset(
    {
        "path_lost_hold",
        "track_lost_hold",
        "failsafe_stop",
        "timeout_stop",
    }
)


class FusionFlightIndicator:
    """Drive the FC indicator without repeating unchanged LED commands."""

    def __init__(self, fc, *, flash_period_s: float = 0.2):
        self.fc = fc
        self.flash_period_s = max(0.01, float(flash_period_s))
        self._color: tuple[int, int, int] | None = None

    def set_green(self) -> None:
        self._set_color(GREEN)

    def set_red(self) -> None:
        self._set_color(RED)

    def pre_takeoff_countdown(
        self,
        *,
        ready_wait_s: float = 15.0,
        warning_wait_s: float = 5.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.set_green()
        self.fc.set_digital_output(0, True)
        sleep_fn(float(ready_wait_s))
        self.set_red()
        sleep_fn(float(warning_wait_s))

    def update(
        self,
        *,
        now_s: float,
        avoiding: bool,
        unexpected: bool,
    ) -> None:
        if unexpected:
            phase = int(max(0.0, float(now_s)) / self.flash_period_s)
            self._set_color(RED if phase % 2 == 0 else OFF)
        elif avoiding:
            self.set_red()
        else:
            self.set_green()

    def _set_color(self, color: tuple[int, int, int]) -> None:
        if color == self._color:
            return
        self.fc.set_indicator_led(*color)
        self._color = color


def planner_state_name(planner) -> str:
    state = getattr(planner, "state", "normal")
    return str(getattr(state, "value", state))


def is_avoiding(*, planner_state: str, safety_state: str) -> bool:
    return bool(
        planner_state != "normal"
        or safety_state in {"LIMITED", "OBSTACLE_STOP"}
    )


def is_unexpected(
    *,
    planner_state: str,
    safety_state: str,
    decision_allowed: bool,
    radar_required: bool,
    radar_fresh: bool,
    camera_ok: bool,
    perception_stale: bool,
    road_found: bool,
) -> bool:
    return bool(
        planner_state in UNEXPECTED_PLANNER_STATES
        or safety_state == "HARD_STOP"
        or not decision_allowed
        or (radar_required and not radar_fresh)
        or not camera_ok
        or perception_stale
        or not road_found
    )


__all__ = [
    "FusionFlightIndicator",
    "GREEN",
    "OFF",
    "RED",
    "UNEXPECTED_PLANNER_STATES",
    "is_avoiding",
    "is_unexpected",
    "planner_state_name",
]
