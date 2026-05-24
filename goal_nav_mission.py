"""Compatibility wrapper for relative goal navigation."""

from FlightController.Solutions.RelativeGoalNavigator import (
    RelativeGoalConfig as GoalNavConfig,
    RelativeGoalNavigator as GoalNavMission,
)

__all__ = ["GoalNavConfig", "GoalNavMission"]

