from types import SimpleNamespace
import threading

from fleet_bus.models import NodeFlags
from experiments.visual_radar_bypass.road_patrol_fleet import (
    RoadPatrolFleetStateProvider,
    RoadPatrolOperationState,
    cruise_operation_state,
    wait_for_ground_takeoff_authorization,
)


def mission(state, *, active=True, released=False):
    return SimpleNamespace(
        state=SimpleNamespace(value=state),
        mission_active=active,
        payload_released=released,
    )


def test_state_provider_never_claims_an_xy_pose():
    provider = RoadPatrolFleetStateProvider()
    provider.bind_fc(
        SimpleNamespace(
            state=SimpleNamespace(
                unlock=SimpleNamespace(value=1),
                alt_add=SimpleNamespace(value=93.6),
                bat=SimpleNamespace(value=11.7),
            )
        )
    )
    provider.set_operation_state(RoadPatrolOperationState.LINE_FOLLOWING)

    state = provider()

    assert not state.node_flags & int(NodeFlags.POSE_VALID)
    assert state.node_flags & int(NodeFlags.READY)
    assert state.node_flags & int(NodeFlags.ARMED_OR_MOTOR_ACTIVE)
    assert (state.x_cm, state.y_cm, state.z_cm) == (0, 0, 94)
    assert state.operation_state == int(RoadPatrolOperationState.LINE_FOLLOWING)


def test_cruise_state_prioritizes_avoidance_during_payload_delivery():
    assert cruise_operation_state(
        target_mission=mission("target_approach"), avoiding=False
    ) == RoadPatrolOperationState.PAYLOAD_DELIVERY
    assert cruise_operation_state(
        target_mission=mission("target_clearance"), avoiding=False
    ) == RoadPatrolOperationState.OBSTACLE_AVOIDANCE


def test_actual_release_has_a_distinct_durable_state():
    assert cruise_operation_state(
        target_mission=mission("post_release_wait", released=True),
        avoiding=False,
    ) == RoadPatrolOperationState.PAYLOAD_RELEASED
    assert cruise_operation_state(
        target_mission=mission("complete", active=False, released=True),
        avoiding=False,
    ) == RoadPatrolOperationState.LINE_FOLLOWING


def test_ground_prepare_turns_red_and_start_turns_green_before_authorization():
    events = []
    prepare = SimpleNamespace(command_id=0x24)
    start = SimpleNamespace(command_id=0x23)

    class Commands:
        def __init__(self):
            self.pending = [prepare, start]

        def receive(self, timeout=None):
            events.append(("receive", timeout))
            return self.pending.pop(0)

        def complete(self, command):
            events.append(("complete", command.command_id))

        def fail(self, command, error_code):
            events.append(("fail", command.command_id, error_code))

    class Indicator:
        def prepare_for_ground_countdown(self):
            events.append(("indicator", "red-and-payload-on"))

        def set_green(self):
            events.append(("indicator", "green"))

    authorized = wait_for_ground_takeoff_authorization(
        fleet_node=SimpleNamespace(command_queue=Commands()),
        indicator=Indicator(),
        stop_event=threading.Event(),
    )

    assert authorized
    assert events == [
        ("receive", 0.1),
        ("indicator", "red-and-payload-on"),
        ("complete", 0x24),
        ("receive", 0.1),
        ("indicator", "green"),
        ("complete", 0x23),
    ]


def test_start_before_prepare_is_failed_and_does_not_authorize_takeoff():
    start = SimpleNamespace(command_id=0x23)
    failed = []

    class Commands:
        def receive(self, timeout=None):
            stop.set()
            return start

        def complete(self, command):
            raise AssertionError("start must not complete before prepare")

        def fail(self, command, error_code):
            failed.append((command.command_id, error_code))

    stop = threading.Event()
    authorized = wait_for_ground_takeoff_authorization(
        fleet_node=SimpleNamespace(command_queue=Commands()),
        indicator=SimpleNamespace(),
        stop_event=stop,
    )

    assert not authorized
    assert failed == [(0x23, 2)]
