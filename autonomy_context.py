"""Shared context dataclasses for top-level autonomy programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SensorHealth:
    radar_ok: bool = False
    camera_ok: bool = False
    fc_ok: bool = False
    radar_age_s: float | None = None
    camera_age_s: float | None = None
    loop_age_s: float = 0.0


@dataclass
class FlightStatus:
    connected: bool = False
    mode: int | None = None
    unlocked: bool | None = None
    battery_v: float | None = None
    alt_cm: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None


@dataclass
class Obstacle:
    x_cm: float
    y_cm: float
    distance_cm: float
    width_cm: float = 0.0
    depth_cm: float = 0.0
    kind: str = "unknown"
    confidence: float = 0.0
    last_seen_s: float = 0.0


@dataclass
class AutonomyContext:
    now_s: float
    health: SensorHealth = field(default_factory=SensorHealth)
    flight: FlightStatus = field(default_factory=FlightStatus)
    road: Any | None = None
    obstacles: list[Obstacle] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def flight_status_from_fc(fc: Any | None) -> FlightStatus:
    """Best-effort conversion from the existing FC state object."""
    if fc is None:
        return FlightStatus()

    state = getattr(fc, "state", None)
    if state is None:
        return FlightStatus(connected=bool(getattr(fc, "connected", False)))

    return FlightStatus(
        connected=bool(getattr(fc, "connected", False)),
        mode=_value(getattr(state, "mode", None)),
        unlocked=_value(getattr(state, "unlock", None)),
        battery_v=_value(getattr(state, "bat", None)),
        alt_cm=_value(getattr(state, "alt_add", None)),
        roll_deg=_value(getattr(state, "rol", None)),
        pitch_deg=_value(getattr(state, "pit", None)),
    )


def _value(field: Any) -> Any:
    return getattr(field, "value", field)

