"""FleetBus state projection for the road-patrol flight entry."""

from __future__ import annotations

from enum import IntEnum
import threading
import time

from fleet_bus.models import AckReason, AirFleetState, CommandId, NodeFlags


class RoadPatrolOperationState(IntEnum):
    """Task-specific values carried by ``ReportPayload.operation_state``."""

    STANDBY = 40
    TAKEOFF = 41
    LINE_FOLLOWING = 42
    OBSTACLE_AVOIDANCE = 43
    PAYLOAD_DELIVERY = 44
    PAYLOAD_RELEASED = 45
    LANDING = 46
    FAULT = 47


class RoadPatrolFleetStateProvider:
    """Publish task state without claiming an unavailable aircraft XY pose."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_s = time.monotonic()
        self._operation_state = RoadPatrolOperationState.STANDBY
        self._fc = None

    @property
    def operation_state(self) -> RoadPatrolOperationState:
        with self._lock:
            return self._operation_state

    def bind_fc(self, fc: object) -> None:
        with self._lock:
            self._fc = fc

    def set_operation_state(self, state: RoadPatrolOperationState) -> bool:
        state = RoadPatrolOperationState(state)
        with self._lock:
            if state == self._operation_state:
                return False
            self._operation_state = state
            return True

    def __call__(self) -> AirFleetState:
        with self._lock:
            operation_state = self._operation_state
            fc = self._fc

        fc_state = getattr(fc, "state", None)
        armed = bool(_number(getattr(fc_state, "unlock", 0.0)))
        altitude_cm = round(_number(getattr(fc_state, "alt_add", 0.0)))
        battery_cV = round(_number(getattr(fc_state, "bat", 0.0)) * 100.0)

        flags = int(NodeFlags.READY)
        if operation_state not in (
            RoadPatrolOperationState.STANDBY,
            RoadPatrolOperationState.LANDING,
            RoadPatrolOperationState.FAULT,
        ):
            flags |= int(NodeFlags.BUSY)
        if armed:
            flags |= int(NodeFlags.ARMED_OR_MOTOR_ACTIVE)

        return AirFleetState(
            node_flags=flags,
            node_uptime_ms=round((time.monotonic() - self._started_s) * 1000.0)
            & 0xFFFFFFFF,
            # There is no reliable aircraft XY source for this task.
            x_cm=0,
            y_cm=0,
            z_cm=max(-(2**31), min(2**31 - 1, altitude_cm)),
            battery_cV=max(0, min(0xFFFF, battery_cV)),
            operation_state=int(operation_state),
            pose_quality=0,
        )


def cruise_operation_state(
    *,
    target_mission: object | None,
    avoiding: bool,
) -> RoadPatrolOperationState:
    """Collapse planner and target-controller internals into operator stages."""

    if target_mission is not None and bool(
        getattr(target_mission, "mission_active", False)
    ):
        target_state = getattr(getattr(target_mission, "state", None), "value", "")
        if avoiding or target_state == "target_clearance":
            return RoadPatrolOperationState.OBSTACLE_AVOIDANCE
        if bool(getattr(target_mission, "payload_released", False)):
            return RoadPatrolOperationState.PAYLOAD_RELEASED
        return RoadPatrolOperationState.PAYLOAD_DELIVERY
    if avoiding:
        return RoadPatrolOperationState.OBSTACLE_AVOIDANCE
    return RoadPatrolOperationState.LINE_FOLLOWING


def wait_for_ground_takeoff_authorization(
    *,
    fleet_node: object,
    indicator: object,
    stop_event: threading.Event,
    receive_timeout_s: float = 0.1,
) -> bool:
    """Apply ground-issued prepare/start commands in the flight task thread."""

    commands = fleet_node.command_queue
    prepared = False
    while not stop_event.is_set():
        command = commands.receive(timeout=receive_timeout_s)
        if command is None:
            continue
        if command.command_id == int(CommandId.DRONE_PREPARE_MISSION):
            indicator.prepare_for_ground_countdown()
            prepared = True
            commands.complete(command)
            continue
        if command.command_id == int(CommandId.DRONE_START_MISSION):
            if not prepared:
                commands.fail(command, int(AckReason.NOT_READY))
                continue
            indicator.set_green()
            commands.complete(command)
            return True
        commands.fail(command, int(AckReason.UNSUPPORTED))
    return False


def _number(value: object, default: float = 0.0) -> float:
    candidate = getattr(value, "value", value)
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return default


__all__ = [
    "RoadPatrolFleetStateProvider",
    "RoadPatrolOperationState",
    "cruise_operation_state",
    "wait_for_ground_takeoff_authorization",
]
