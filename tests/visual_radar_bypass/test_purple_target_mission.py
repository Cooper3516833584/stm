from types import SimpleNamespace

import pytest

from FlightController.Solutions.Safety import Command
from experiments.visual_radar_bypass.purple_target_mission import (
    PurpleTargetMissionController,
    PurpleTargetMissionState,
)


ROAD = Command(22.0, 3.0, 0.0, 4.0, "road")


def _target(sequence, x=100.0, y=0.0, found=True):
    return SimpleNamespace(
        found=found,
        offset_x_px=x if found else None,
        offset_y_px=y if found else None,
        capture_time_s=float(sequence),
        error=None,
    )


def _step(
    controller,
    now,
    target,
    *,
    stale=False,
    planner="normal",
    radar=True,
    altitude=100.0,
):
    return controller.update(
        now_s=now,
        road_desired=ROAD,
        target=target,
        target_stale=stale,
        planner_state=planner,
        radar_fresh=radar,
        altitude_cm=altitude,
    )


def _acquire_and_clear(controller, x=100.0, y=0.0):
    for index in range(3):
        decision = _step(controller, index * 0.1, _target(index + 1, x, y))
    assert decision.state == PurpleTargetMissionState.TARGET_CLEARANCE
    for index in range(3, 6):
        decision = _step(controller, index * 0.1, _target(index + 1, x, y))
        if decision.state == PurpleTargetMissionState.TARGET_APPROACH:
            break
    assert decision.state == PurpleTargetMissionState.TARGET_APPROACH
    return decision


def test_search_uses_road_without_target_control_until_three_unique_detections():
    controller = PurpleTargetMissionController()

    first = _step(controller, 0.0, _target(1))
    repeated = _step(controller, 0.1, _target(1))
    second = _step(controller, 0.2, _target(2))

    assert first.desired == repeated.desired == second.desired == ROAD
    assert not first.mission_owns_command
    assert controller.state == PurpleTargetMissionState.ROAD_SEARCH

    confirmed = _step(controller, 0.3, _target(3))
    assert confirmed.mission_owns_command
    assert confirmed.desired == Command.zero("purple_target:clearance_wait")
    assert controller.state == PurpleTargetMissionState.TARGET_CLEARANCE


@pytest.mark.parametrize(
    ("x", "y", "yaw_sign"),
    [
        (100.0, 100.0, -1),
        (100.0, -100.0, 1),
        (-100.0, 100.0, -1),
        (-100.0, -100.0, 1),
    ],
)
def test_target_bearing_covers_all_quadrants_and_turns_in_place(x, y, yaw_sign):
    controller = PurpleTargetMissionController()
    _acquire_and_clear(controller, x, y)

    decision = _step(controller, 0.6, _target(7, x, y))

    assert decision.desired.vx_cm_s == 0.0
    assert decision.desired.vy_cm_s == 0.0
    assert decision.desired.yaw_rate_deg_s * yaw_sign > 0.0
    assert abs(decision.desired.yaw_rate_deg_s) <= 18.0


def test_aligned_target_reaches_fixed_speed_through_existing_acceleration_limit():
    controller = PurpleTargetMissionController()
    _acquire_and_clear(controller)

    commands = [
        _step(controller, 0.6 + index * 0.1, _target(7 + index, 100.0, 0.0)).desired
        for index in range(4)
    ]

    assert commands[0].vx_cm_s == pytest.approx(3.6)
    assert commands[-1].vx_cm_s == pytest.approx(13.2)
    assert all(command.vy_cm_s == 0.0 for command in commands)
    assert all(command.yaw_rate_deg_s == 0.0 for command in commands)


def test_obstacle_pauses_target_yaw_and_requires_three_clear_frames_to_resume():
    controller = PurpleTargetMissionController()
    _acquire_and_clear(controller, 100.0, 20.0)
    _step(controller, 0.6, _target(7, 100.0, 20.0))

    paused = _step(
        controller,
        0.7,
        _target(8, 100.0, 20.0),
        planner="diverge_left",
    )
    assert paused.state == PurpleTargetMissionState.TARGET_CLEARANCE
    assert paused.desired == Command.zero("purple_target:avoidance_pause")

    for index in range(2):
        waiting = _step(controller, 0.8 + index * 0.1, _target(9 + index, 100.0, 20.0))
        assert waiting.state == PurpleTargetMissionState.TARGET_CLEARANCE
    resumed = _step(controller, 1.0, _target(11, 100.0, 20.0))
    assert resumed.state == PurpleTargetMissionState.TARGET_APPROACH
    moving_again = _step(controller, 1.1, _target(12, 100.0, 0.0))
    assert moving_again.desired.vx_cm_s == pytest.approx(3.6)


def test_new_radar_conflict_pauses_before_planner_state_changes():
    controller = PurpleTargetMissionController()
    _acquire_and_clear(controller, 100.0, 20.0)

    decision = controller.update(
        now_s=0.6,
        road_desired=ROAD,
        target=_target(7, 100.0, 20.0),
        target_stale=False,
        planner_state="normal",
        radar_fresh=True,
        altitude_cm=100.0,
        obstacle_conflict=True,
    )

    assert decision.state == PurpleTargetMissionState.TARGET_CLEARANCE
    assert decision.desired.yaw_rate_deg_s == 0.0


def test_full_terminal_sequence_releases_once_and_returns_to_road_height():
    controller = PurpleTargetMissionController()
    _acquire_and_clear(controller, 0.0, 0.0)
    now = 0.6
    sequence = 7

    for _ in range(3):
        decision = _step(controller, now, _target(sequence, 0.0, 0.0))
        now += 0.1
        sequence += 1
    assert decision.state == PurpleTargetMissionState.HIGH_HOVER

    decision = _step(controller, now + 2.0, _target(sequence, 0.0, 0.0))
    assert decision.state == PurpleTargetMissionState.DESCEND_60
    now += 2.1
    sequence += 1
    for _ in range(3):
        decision = _step(
            controller,
            now,
            _target(sequence, 0.0, 0.0),
            altitude=60.0,
        )
        now += 0.1
        sequence += 1
    assert decision.state == PurpleTargetMissionState.LOW_CALIBRATE

    for _ in range(3):
        decision = _step(
            controller,
            now,
            _target(sequence, 0.0, 0.0),
            altitude=60.0,
        )
        now += 0.1
        sequence += 1
    assert decision.state == PurpleTargetMissionState.LOW_HOVER

    decision = _step(
        controller,
        now + 1.0,
        _target(sequence, 0.0, 0.0),
        altitude=60.0,
    )
    assert decision.state == PurpleTargetMissionState.RELEASE_PENDING
    assert controller.release_is_authorized(
        planner_state="normal",
        radar_fresh=True,
        safety_state="OK",
        command_allowed=True,
        final_command=Command.zero(),
    )
    controller.mark_payload_released(now + 1.0)
    assert controller.payload_released
    assert controller.consume_disable_target_request()
    assert not controller.consume_disable_target_request()

    _step(controller, now + 2.1, None, stale=True, altitude=60.0)
    assert controller.state == PurpleTargetMissionState.ASCEND_100
    for index in range(3):
        decision = _step(
            controller,
            now + 2.2 + index * 0.1,
            None,
            stale=True,
            altitude=100.0,
        )
    assert controller.state == PurpleTargetMissionState.COMPLETE
    final = _step(controller, now + 2.6, None, stale=True, altitude=100.0)
    assert final.desired == ROAD
    assert not final.mission_owns_command


def test_target_loss_holds_two_seconds_then_aborts_and_never_requests_release():
    controller = PurpleTargetMissionController()
    _acquire_and_clear(controller)

    hold = _step(controller, 1.0, _target(20, found=False))
    assert hold.desired.vx_cm_s == 0.0
    assert controller.state == PurpleTargetMissionState.TARGET_APPROACH

    aborted = _step(controller, 3.01, _target(21, found=False), altitude=100.0)
    assert aborted.state == PurpleTargetMissionState.ABORT_RECOVERY
    assert controller.consume_disable_target_request()
    assert not controller.payload_released
    returned = _step(controller, 3.11, None, stale=True, altitude=100.0)
    assert returned.desired == ROAD
    assert controller.state == PurpleTargetMissionState.ABORTED


def test_approach_timeout_aborts_after_thirty_seconds():
    controller = PurpleTargetMissionController()
    _acquire_and_clear(controller, 100.0, 0.0)

    decision = _step(controller, 30.21, _target(99, 100.0, 0.0), altitude=80.0)

    assert decision.state == PurpleTargetMissionState.ABORT_RECOVERY
    assert decision.desired.vz_cm_s > 0.0
    assert decision.desired.vx_cm_s == 0.0


def test_release_requires_normal_planner_fresh_radar_and_clean_safety():
    controller = PurpleTargetMissionController()
    controller.state = PurpleTargetMissionState.RELEASE_PENDING
    controller._release_requested = True

    assert not controller.release_is_authorized(
        planner_state="diverge_left",
        radar_fresh=True,
        safety_state="OK",
        command_allowed=True,
        final_command=Command.zero(),
    )
    assert not controller.release_is_authorized(
        planner_state="normal",
        radar_fresh=False,
        safety_state="OK",
        command_allowed=True,
        final_command=Command.zero(),
    )
    assert not controller.release_is_authorized(
        planner_state="normal",
        radar_fresh=True,
        safety_state="OBSTACLE_STOP",
        command_allowed=True,
        final_command=Command.zero(),
    )
    assert not controller.release_is_authorized(
        planner_state="normal",
        radar_fresh=True,
        safety_state="OK",
        command_allowed=True,
        final_command=Command(1.0, 0.0, 0.0, 0.0, "moving"),
    )
    assert not controller.release_is_authorized(
        planner_state="normal",
        radar_fresh=True,
        safety_state="OK",
        command_allowed=True,
        final_command=Command.zero(),
        obstacle_clear=False,
    )
