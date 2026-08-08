from experiments.visual_radar_bypass.flight_indicator import (
    FusionFlightIndicator,
    GREEN,
    OFF,
    RED,
    YELLOW,
    is_avoiding,
    is_unexpected,
)


class _FakeFC:
    def __init__(self):
        self.calls = []

    def set_indicator_led(self, r, g, b):
        self.calls.append(("led", r, g, b))

    def set_digital_output(self, channel, on):
        self.calls.append(("digital", channel, on))


def test_pre_takeoff_countdown_orders_green_output_wait_and_red_warning():
    fc = _FakeFC()
    waits = []
    indicator = FusionFlightIndicator(fc)

    indicator.pre_takeoff_countdown(sleep_fn=waits.append)

    assert fc.calls == [
        ("led", *GREEN),
        ("digital", 0, True),
        ("led", *RED),
    ]
    assert waits == [15.0, 5.0]


def test_indicator_uses_green_for_road_yellow_for_target_red_for_avoidance_and_flashes_errors():
    fc = _FakeFC()
    indicator = FusionFlightIndicator(fc)

    indicator.update(now_s=0.0, avoiding=False, unexpected=False)
    indicator.update(now_s=0.1, avoiding=False, unexpected=False)
    indicator.update(
        now_s=0.2,
        avoiding=False,
        unexpected=False,
        target_active=True,
    )
    indicator.update(
        now_s=0.3,
        avoiding=False,
        unexpected=False,
        target_active=True,
    )
    indicator.update(
        now_s=0.4,
        avoiding=True,
        unexpected=False,
        target_active=True,
    )
    indicator.update(
        now_s=0.5,
        avoiding=False,
        unexpected=False,
        target_active=True,
    )
    indicator.update(
        now_s=0.61,
        avoiding=False,
        unexpected=True,
        target_active=True,
    )
    indicator.update(
        now_s=0.81,
        avoiding=False,
        unexpected=True,
        target_active=True,
    )
    indicator.update(now_s=1.01, avoiding=False, unexpected=False)

    assert fc.calls == [
        ("led", *GREEN),
        ("led", *YELLOW),
        ("led", *RED),
        ("led", *YELLOW),
        ("led", *OFF),
        ("led", *RED),
        ("led", *GREEN),
    ]


def test_indicator_state_classification():
    assert not is_avoiding(planner_state="normal", safety_state="OK")
    assert not is_avoiding(planner_state="normal", safety_state="LIMITED")
    assert is_avoiding(planner_state="diverge_left", safety_state="OK")
    assert is_avoiding(planner_state="diverge_left", safety_state="LIMITED")
    assert is_avoiding(planner_state="normal", safety_state="OBSTACLE_STOP")

    common = {
        "planner_state": "normal",
        "safety_state": "OK",
        "decision_allowed": True,
        "radar_required": True,
        "radar_fresh": True,
        "camera_ok": True,
        "perception_stale": False,
        "road_found": True,
    }
    assert not is_unexpected(**common)
    assert is_unexpected(**{**common, "radar_fresh": False})
    assert is_unexpected(**{**common, "road_found": False})
    assert is_unexpected(**{**common, "planner_state": "failsafe_stop"})
