"""Compatibility wrapper for radar obstacle field helpers."""

from FlightController.Solutions.Safety import (
    RadarFieldConfig as LocalWorldModelConfig,
    RadarObstacleField as LocalWorldModel,
)

__all__ = ["LocalWorldModel", "LocalWorldModelConfig"]

