"""Compatibility wrapper for road following mission logic."""

from FlightController.Solutions.RoadFollower import (
    RoadFollower as RoadFollowMission,
    RoadFollowerConfig as RoadFollowConfig,
)

__all__ = ["RoadFollowConfig", "RoadFollowMission"]

