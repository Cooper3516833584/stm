import circular_tube_bypass_main
from experiments.visual_radar_bypass import main as experiment_main


def test_no_argument_entry_expands_to_requested_real_flight_command():
    args = experiment_main.parse_args(circular_tube_bypass_main.build_argv())

    assert args.bypass_planner == "legacy"
    assert args.circular_tube_bypass
    assert args.enable_flight
    assert args.auto_takeoff
    assert args.confirm_visual_radar_flight_test
    assert args.takeoff_height_cm == 100
    assert args.duration_s == 60.0
    assert not args.right_half_radar_then_visual


def test_entry_allows_safe_duration_and_height_overrides():
    args = experiment_main.parse_args(
        circular_tube_bypass_main.build_argv(
            ["--duration-s", "30", "--takeoff-height-cm", "80"]
        )
    )

    assert args.duration_s == 30.0
    assert args.takeoff_height_cm == 80


def test_entry_delegates_expanded_arguments(monkeypatch):
    captured = []
    monkeypatch.setattr(circular_tube_bypass_main, "run_experiment", captured.append)

    circular_tube_bypass_main.main(["--duration-s", "20"])

    assert captured == [
        [*circular_tube_bypass_main.DEFAULT_FLIGHT_ARGV, "--duration-s", "20"]
    ]
