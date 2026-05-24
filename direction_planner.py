"""Compatibility wrapper for relative body-frame navigation."""

from FlightController.Solutions.RelativeGoalNavigator import (
    RelativeGoalConfig as DirectionPlannerConfig,
    RelativeGoalNavigator as DirectionPlanner,
)

__all__ = ["DirectionPlanner", "DirectionPlannerConfig"]

