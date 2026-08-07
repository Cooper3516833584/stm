from types import SimpleNamespace

import numpy as np

from FlightController.Solutions.RoadObstacleBypassPlanner import (
    RoadBypassConfig,
    RoadBypassState,
    RoadObstacleBypassPlanner,
)
from FlightController.Solutions.Safety import Command, RadarObstacleField
from experiments.visual_radar_bypass.static_route_bypass import (
    StaticRouteBypassPlanner,
    StaticRouteBypassState,
)


PERCEPTION = SimpleNamespace(
    is_road_found=True,
    confidence=1.0,
    corrected_pixel_error=0.0,
)
DESIRED = Command(10.0, 0.0, 0.0, 0.0, "road")


def _field(points):
    field = RadarObstacleField()
    field.update(np.asarray(points, dtype=float), 0.0)
    return field


def test_generic_bypass_confirmation_time_is_invariant_at_sweep_rates():
    field = _field([[100.0, 0.0], [102.0, 2.0], [104.0, -2.0]])
    for rate_hz in (10.0, 20.0, 30.0, 50.0):
        planner = RoadObstacleBypassPlanner(
            RoadBypassConfig(enabled=True, activate_frames=2, confirmation_period_s=0.1)
        )
        for now_s in np.arange(0.0, 0.1, 1.0 / rate_hz):
            planner.update(desired=DESIRED, perception=PERCEPTION, radar_field=field, now_s=now_s)
            assert planner.state == RoadBypassState.NORMAL
        planner.update(desired=DESIRED, perception=PERCEPTION, radar_field=field, now_s=0.10)
        assert planner.state != RoadBypassState.NORMAL


def test_static_route_activation_time_is_invariant_at_sweep_rates():
    field = _field(
        [[100.0, value] for value in (-4.0, -2.0, 0.0, 2.0, 4.0)]
    )
    for rate_hz in (10.0, 20.0, 30.0, 50.0):
        planner = StaticRouteBypassPlanner()
        for now_s in np.arange(0.0, 0.1, 1.0 / rate_hz):
            planner.update(desired=DESIRED, perception=PERCEPTION, radar_field=field, now_s=now_s)
            assert planner.state == StaticRouteBypassState.NORMAL
        planner.update(desired=DESIRED, perception=PERCEPTION, radar_field=field, now_s=0.10)
        assert planner.state in {
            StaticRouteBypassState.DIVERGE_LEFT,
            StaticRouteBypassState.DIVERGE_RIGHT,
        }
