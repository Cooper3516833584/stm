"""Compatibility context helpers for older root-level imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from FlightController.Solutions.Safety import FlightStatus, flight_status_from_fc


@dataclass
class SensorHealth:
    radar_ok: bool = False
    camera_ok: bool = False
    fc_ok: bool = False
    radar_age_s: float | None = None
    camera_age_s: float | None = None
    loop_age_s: float = 0.0


@dataclass
class AutonomyContext:
    now_s: float
    health: SensorHealth = field(default_factory=SensorHealth)
    flight: FlightStatus = field(default_factory=FlightStatus)
    road: Any | None = None
    notes: list[str] = field(default_factory=list)


__all__ = ["AutonomyContext", "FlightStatus", "SensorHealth", "flight_status_from_fc"]

