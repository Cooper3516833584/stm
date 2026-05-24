"""Compatibility wrapper for the official safety arbiter."""

from FlightController.Solutions.Safety import (
    FlightStatus,
    Command,
    FlightHealth,
    RadarFieldConfig,
    RadarObstacleField,
    SafetyArbiter,
    SafetyConfig,
    SafetyDecision,
    SafetyResult,
    VelocityCommand,
    flight_health_from_sources,
    flight_status_from_fc,
    multi_radar_age_s,
    send_command_safely,
)

__all__ = [
    "FlightStatus",
    "Command",
    "FlightHealth",
    "RadarFieldConfig",
    "RadarObstacleField",
    "SafetyArbiter",
    "SafetyConfig",
    "SafetyDecision",
    "SafetyResult",
    "VelocityCommand",
    "flight_health_from_sources",
    "flight_status_from_fc",
    "multi_radar_age_s",
    "send_command_safely",
]
