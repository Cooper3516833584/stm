from argparse import Namespace
import sys

from FlightController.Solutions.Safety import Command
import goal_nav_main


class _FakeNavigator:
    def __init__(self, commands):
        self.commands = list(commands)
        self.calls = 0

    def update(self, radar_field, now_s=None):
        self.calls += 1
        if len(self.commands) == 1:
            return self.commands[0]
        return self.commands.pop(0)


def _args(**overrides):
    values = {
        "forward_step_cm": 40.0,
        "min_step_s": 0.35,
        "max_step_s": 2.0,
        "turn_step_s": 0.45,
        "hold_after_step_s": 0.35,
        "dry_run": False,
        "no_fc": False,
        "no_radar": False,
        "enable_flight": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_goal_nav_defaults_for_forward_avoidance(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["goal_nav_main.py"])
    args = goal_nav_main.parse_args()
    assert args.takeoff_height_cm == 150.0
    assert args.scan_fov_deg == 150.0
    assert args.candidate_step_deg == 2.0
    assert args.candidate_edge_margin_deg == 10.0
    assert args.forward_step_cm == 40.0
    assert args.cruise_speed_cm_s == 10.0
    assert args.min_forward_speed_cm_s == 5.0
    assert not goal_nav_main._is_actual_dry_run(args)


def test_safety_switches_force_dry_run():
    assert goal_nav_main._is_actual_dry_run(_args(no_radar=True))
    assert goal_nav_main._is_actual_dry_run(_args(no_fc=True))
    assert goal_nav_main._is_actual_dry_run(_args(dry_run=True))
    assert not goal_nav_main._is_actual_dry_run(_args())


def test_step_duration_uses_forward_distance_and_bounds():
    args = _args()
    assert goal_nav_main._step_duration_for_command(Command(20, 0, 0, 0, "forward"), args) == 2.0
    assert goal_nav_main._step_duration_for_command(Command(200, 0, 0, 0, "forward"), args) == 0.35
    assert goal_nav_main._step_duration_for_command(Command(0, 0, 0, 10, "turn"), args) == 0.45


def test_no_path_finishes_mission_with_zero_command():
    runtime = goal_nav_main.StepRuntime()
    nav = _FakeNavigator([Command.zero("blocked_no_path_dir_0_clear_60")])

    cmd = goal_nav_main._next_step_command(
        runtime=runtime,
        navigator=nav,
        radar_field=None,
        now_s=0.0,
        args=_args(),
    )

    assert cmd.as_fc_tuple() == (0, 0, 0, 0)
    assert runtime.mission_done
    assert runtime.done_reason.startswith("blocked_no_path")


def test_step_execution_stops_before_replanning():
    runtime = goal_nav_main.StepRuntime()
    nav = _FakeNavigator([Command(20, 0, 0, 0, "forward_clear")])
    args = _args()

    first = goal_nav_main._next_step_command(
        runtime=runtime,
        navigator=nav,
        radar_field=None,
        now_s=0.0,
        args=args,
    )
    assert first.vx_cm_s == 20
    assert runtime.phase == "execute"

    running = goal_nav_main._next_step_command(
        runtime=runtime,
        navigator=nav,
        radar_field=None,
        now_s=1.0,
        args=args,
    )
    assert running.vx_cm_s == 20
    assert runtime.phase == "execute"

    stopped = goal_nav_main._next_step_command(
        runtime=runtime,
        navigator=nav,
        radar_field=None,
        now_s=2.1,
        args=args,
    )
    assert stopped.as_fc_tuple() == (0, 0, 0, 0)
    assert stopped.reason.startswith("step_complete")
    assert runtime.phase == "hold"

    held = goal_nav_main._next_step_command(
        runtime=runtime,
        navigator=nav,
        radar_field=None,
        now_s=2.2,
        args=args,
    )
    assert held.as_fc_tuple() == (0, 0, 0, 0)
    assert held.reason == "step_hold"


def test_execute_phase_uses_live_forward_speed_update():
    runtime = goal_nav_main.StepRuntime()
    nav = _FakeNavigator([
        Command(20, 0, 0, 0, "forward_clear_220"),
        Command(8, 0, 0, 0, "forward_clear_120"),
    ])
    args = _args()

    first = goal_nav_main._next_step_command(
        runtime=runtime,
        navigator=nav,
        radar_field=None,
        now_s=0.0,
        args=args,
    )
    assert first.vx_cm_s == 20

    running = goal_nav_main._next_step_command(
        runtime=runtime,
        navigator=nav,
        radar_field=None,
        now_s=0.5,
        args=args,
    )
    assert running.vx_cm_s == 8
    assert runtime.active_command.vx_cm_s == 8
    assert runtime.phase == "execute"
