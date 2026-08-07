"""Low-overhead structured diagnostics for the isolated experiment."""

from __future__ import annotations

from dataclasses import dataclass
from FlightController.Solutions.Safety import Command


@dataclass(frozen=True)
class DiagnosticEvent:
    event: str
    reason: str
    payload: dict[str, object]


class ExperimentDiagnosticsTracker:
    def __init__(self) -> None:
        self._last_final: Command | None = None
        self._last_state: str | None = None
        self._last_side: str | None = None
        self._last_encounter_id = 0
        self._last_override = False

    def observe(
        self,
        *,
        frame_id: int,
        now_s: float,
        dt_s: float,
        planner_elapsed_us: float,
        desired: Command,
        planned: Command,
        safe: Command,
        final: Command,
        planner_diagnostics: dict[str, object],
        safety_state: str,
        safety_reasons: list[str],
        decision_reason: str,
        nearest_forward_cm: float | None,
        raw_radar_point_count: int,
        radar_point_count: int,
    ) -> tuple[dict[str, object], list[DiagnosticEvent]]:
        delta = _command_delta(self._last_final, final)
        safety_override = safe != planned or final != safe
        state = str(planner_diagnostics.get("state", "unknown"))
        side_value = planner_diagnostics.get(
            "selected_side",
            planner_diagnostics.get("active_bypass_side"),
        )
        side = None if side_value is None else str(side_value)
        encounter_id = int(planner_diagnostics.get("encounter_id", 0) or 0)
        fallback_id = planner_diagnostics.get("fallback_id")

        payload = {
            "timestamp_s": float(now_s),
            "frame_id": int(frame_id),
            "dt_s": float(dt_s),
            "current_state": state,
            "previous_state": planner_diagnostics.get("previous_state"),
            "transition_reason": planner_diagnostics.get("transition_reason"),
            "encounter_id": encounter_id,
            "obstacle_distance_cm": planner_diagnostics.get("obstacle_surface_x_cm"),
            "nearest_forward_cm": nearest_forward_cm,
            "raw_radar_point_count": int(raw_radar_point_count),
            "radar_point_count": int(radar_point_count),
            "observation_valid": planner_diagnostics.get("observation_valid"),
            "fitted_center_cm": [
                planner_diagnostics.get("obstacle_center_x_cm"),
                planner_diagnostics.get("obstacle_center_y_cm"),
            ],
            "fitted_radius_cm": planner_diagnostics.get("fitted_tube_radius_cm"),
            "fit_rms_cm": planner_diagnostics.get("circle_fit_rms_cm"),
            "fit_valid": planner_diagnostics.get("circle_fit_used"),
            "desired": command_dict(desired),
            "planned": command_dict(planned),
            "safe": command_dict(safe),
            "final": command_dict(final),
            "selected_side": side,
            "selected_side_reason": planner_diagnostics.get("selected_side_reason"),
            "side_locked": bool(planner_diagnostics.get("side_locked", side is not None)),
            "blend_alpha": planner_diagnostics.get("blend_alpha"),
            "fallback_id": fallback_id,
            "fallback_reason": planner_diagnostics.get("fallback_reason"),
            "safety_arbiter_override": safety_override,
            "safety_state": safety_state,
            "safety_reasons": list(safety_reasons),
            "decision_reason": decision_reason,
            "delta_vx_cm_s": delta[0],
            "delta_vy_cm_s": delta[1],
            "delta_yaw_rate_deg_s": delta[2],
            "planner_elapsed_us": float(planner_elapsed_us),
        }
        events: list[DiagnosticEvent] = []
        if self._last_state is not None and state != self._last_state:
            events.append(
                DiagnosticEvent(
                    "state_transition",
                    str(planner_diagnostics.get("transition_reason", "state_changed")),
                    {"from": self._last_state, "to": state, "encounter_id": encounter_id},
                )
            )
        if side != self._last_side:
            events.append(
                DiagnosticEvent(
                    "side_lock" if side is not None else "side_unlock",
                    str(planner_diagnostics.get("selected_side_reason") or "encounter_complete"),
                    {"from": self._last_side, "to": side, "encounter_id": encounter_id},
                )
            )
            if self._last_side is not None and side is None:
                events.append(
                    DiagnosticEvent(
                        "encounter_end",
                        str(
                            planner_diagnostics.get("transition_reason")
                            or "encounter_complete"
                        ),
                        {"encounter_id": encounter_id},
                    )
                )
        if encounter_id != self._last_encounter_id:
            events.append(
                DiagnosticEvent(
                    "encounter_start",
                    str(planner_diagnostics.get("selected_side_reason") or "confirmed_obstacle"),
                    {"encounter_id": encounter_id, "selected_side": side},
                )
            )
        if fallback_id:
            events.append(
                DiagnosticEvent(
                    "fallback",
                    str(planner_diagnostics.get("fallback_reason") or fallback_id),
                    {"fallback_id": fallback_id, "encounter_id": encounter_id},
                )
            )
        if safety_override != self._last_override:
            events.append(
                DiagnosticEvent(
                    "safety_override_start" if safety_override else "safety_override_end",
                    ",".join(safety_reasons) or decision_reason,
                    {"active": safety_override, "state": safety_state},
                )
            )

        self._last_final = final
        self._last_state = state
        self._last_side = side
        self._last_encounter_id = encounter_id
        self._last_override = safety_override
        return payload, events


def command_dict(command: Command) -> dict[str, object]:
    return {
        "vx_cm_s": float(command.vx_cm_s),
        "vy_cm_s": float(command.vy_cm_s),
        "vz_cm_s": float(command.vz_cm_s),
        "yaw_rate_deg_s": float(command.yaw_rate_deg_s),
        "reason": command.reason,
    }


def _command_delta(
    previous: Command | None,
    current: Command,
) -> tuple[float, float, float]:
    if previous is None:
        return 0.0, 0.0, 0.0
    return (
        float(current.vx_cm_s - previous.vx_cm_s),
        float(current.vy_cm_s - previous.vy_cm_s),
        float(current.yaw_rate_deg_s - previous.yaw_rate_deg_s),
    )


__all__ = [
    "DiagnosticEvent",
    "ExperimentDiagnosticsTracker",
    "command_dict",
]
