"""Synthetic, no-camera, no-radar-device and no-flight-controller validation.

This module only constructs in-memory RadarObstacleField objects.  It never
imports the flight entry point, opens serial devices or calls connect_fc().
"""

from __future__ import annotations

import argparse
import json
import math
import time
from types import SimpleNamespace
from typing import Callable

import numpy as np

from FlightController.Solutions.Safety import (
    Command,
    FlightStatus,
    RadarFieldConfig,
    RadarObstacleField,
    SafetyArbiter,
    SafetyConfig,
)

from .circular_tube_bypass import CircularTubeBypassPlanner
from .parameter_registry import ExperimentSafetyConfig
from .radar_bypass import ObstacleBypassConfig, ObstacleBypassPlanner
from .smooth_sidestep import SmoothSidestepPlanner


PlannerFactory = Callable[[], object]


def benchmark_planners(
    *,
    iterations: int = 3000,
    warmup: int = 200,
    seed: int = 3516833584,
) -> dict[str, dict[str, float]]:
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    sequence = _synthetic_sequence(seed)
    factories: dict[str, PlannerFactory] = {
        "legacy_basic": lambda: ObstacleBypassPlanner(
            ObstacleBypassConfig(forward_recovery_s=0.0)
        ),
        "legacy_forward_recovery": ObstacleBypassPlanner,
        "smooth_sidestep": SmoothSidestepPlanner,
        "circular_tube_bypass": CircularTubeBypassPlanner,
    }
    results: dict[str, dict[str, float]] = {}
    for name, factory in factories.items():
        planner = factory()
        now_s = 1.0
        for index in range(warmup):
            field, desired = sequence[index % len(sequence)]
            planner.update(
                desired=desired,
                perception=_perception(),
                radar_field=field,
                now_s=now_s,
            )
            now_s += 0.1
        samples_ns = np.empty(iterations, dtype=np.int64)
        for index in range(iterations):
            field, desired = sequence[index % len(sequence)]
            started_ns = time.perf_counter_ns()
            planner.update(
                desired=desired,
                perception=_perception(),
                radar_field=field,
                now_s=now_s,
            )
            samples_ns[index] = time.perf_counter_ns() - started_ns
            now_s += 0.1
        samples_us = samples_ns.astype(float) / 1000.0
        results[name] = {
            "mean_us": float(np.mean(samples_us)),
            "p50_us": float(np.percentile(samples_us, 50)),
            "p95_us": float(np.percentile(samples_us, 95)),
            "samples": int(iterations),
        }
    return results


def run_offline_validation() -> dict[str, object]:
    """Assert selected-planner stability across radar/Safety thresholds."""

    planner = SmoothSidestepPlanner()
    desired = Command(14.0, -6.0, 0.0, 2.0, "synthetic_visual")
    right_79 = _cluster_field(79.0, -40.0)
    right_81 = _cluster_field(81.0, -40.0)
    empty = _field(np.empty((0, 2), dtype=float))
    safety_config = ExperimentSafetyConfig()
    arbiter = SafetyArbiter(
        SafetyConfig(
            require_fc=False,
            require_hold_pos_mode=False,
            require_unlocked=False,
            require_radar=False,
            max_vx_cm_s=safety_config.max_vx_cm_s,
            max_vy_cm_s=safety_config.max_vy_cm_s,
            max_yaw_rate_deg_s=safety_config.max_yaw_rate_deg_s,
            obstacle_stop_distance_cm=(
                safety_config.obstacle_stop_distance_cm
            ),
            obstacle_slow_distance_cm=(
                safety_config.obstacle_slow_distance_cm
            ),
            slow_speed_limit_cm_s=safety_config.slow_speed_limit_cm_s,
            side_stop_distance_cm=safety_config.side_stop_distance_cm,
        )
    )
    flight = FlightStatus()

    normal = planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=empty,
        now_s=1.0,
    )
    assert normal == desired
    planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=right_81,
        now_s=1.1,
    )
    planner.update(
        desired=desired,
        perception=_perception(),
        radar_field=right_79,
        now_s=1.2,
    )
    assert planner.active_bypass_side == 1

    # Complete ramp-in, then alternate across the 80 cm Safety threshold.  A
    # separate unit scenario injects opposite-side noise to verify side lock;
    # using a real opposite-side cluster here would correctly trigger the
    # production side-clearance gate and obscure the forward-threshold check.
    for step in range(1, 13):
        planner.update(
            desired=desired,
            perception=_perception(),
            radar_field=right_81,
            now_s=1.2 + step * 0.1,
        )
    outputs: list[Command] = []
    states: list[str] = []
    sides: list[int | None] = []
    overrides = 0
    for index in range(40):
        field = (right_79, right_81)[index % 2]
        planned = planner.update(
            desired=desired,
            perception=_perception(),
            radar_field=field,
            now_s=2.5 + index * 0.1,
        )
        safe = arbiter.filter(
            planned,
            flight=flight,
            radar_connected=True,
            radar_age_s=0.0,
            radar_field=field,
        )
        outputs.append(safe.command)
        states.append(planner.state.value)
        sides.append(planner.active_bypass_side)
        overrides += int(safe.command != planned)

    assert set(states) == {"shift_left"}
    assert set(sides) == {1}
    assert all(command.vx_cm_s == 0.0 for command in outputs)
    assert all(command.vy_cm_s > 0.0 for command in outputs)
    assert all(command.yaw_rate_deg_s == desired.yaw_rate_deg_s for command in outputs)
    assert overrides == 0

    deltas = [
        (
            abs(later.vx_cm_s - earlier.vx_cm_s),
            abs(later.vy_cm_s - earlier.vy_cm_s),
            abs(later.yaw_rate_deg_s - earlier.yaw_rate_deg_s),
        )
        for earlier, later in zip(outputs, outputs[1:])
    ]
    return {
        "frames": len(outputs),
        "state_switches": sum(a != b for a, b in zip(states, states[1:])),
        "side_switches": sum(a != b for a, b in zip(sides, sides[1:])),
        "safety_arbiter_overrides": overrides,
        "max_abs_delta_vx_cm_s": max(delta[0] for delta in deltas),
        "max_abs_delta_vy_cm_s": max(delta[1] for delta in deltas),
        "max_abs_delta_yaw_rate_deg_s": max(delta[2] for delta in deltas),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="No-FC bypass benchmark")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3516833584)
    args = parser.parse_args(argv)
    result = {
        "no_flight_controller": True,
        "benchmark": benchmark_planners(
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
        ),
        "stability": run_offline_validation(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _synthetic_sequence(seed: int) -> list[tuple[RadarObstacleField, Command]]:
    rng = np.random.default_rng(seed)
    desired = Command(14.0, -6.0, 0.0, 2.0, "synthetic_visual")
    sequence: list[tuple[RadarObstacleField, Command]] = []
    for index in range(120):
        phase = index % 40
        if phase < 24:
            x_cm = 80.0 + float(rng.normal(0.0, 1.2))
            y_cm = -40.0 + float(rng.normal(0.0, 1.0))
            field = _tube_arc_field(x_cm, y_cm)
        elif phase < 32:
            field = _field(np.empty((0, 2), dtype=float))
        else:
            x_cm = 80.0 + float(rng.normal(0.0, 1.2))
            y_cm = 40.0 + float(rng.normal(0.0, 1.0))
            field = _tube_arc_field(x_cm, y_cm)
        sequence.append((field, desired))
    return sequence


def _tube_arc_field(center_x_cm: float, center_y_cm: float) -> RadarObstacleField:
    angles = np.linspace(math.radians(125.0), math.radians(235.0), 16)
    points = np.column_stack(
        (
            center_x_cm + 15.0 * np.cos(angles),
            center_y_cm + 15.0 * np.sin(angles),
        )
    )
    return _field(points)


def _cluster_field(center_x_cm: float, center_y_cm: float) -> RadarObstacleField:
    offsets = np.asarray(
        [(-1.5, -1.0), (-0.5, 0.5), (0.5, -0.5), (1.5, 1.0)],
        dtype=float,
    )
    return _field(offsets + np.asarray([center_x_cm, center_y_cm]))


def _field(points: np.ndarray) -> RadarObstacleField:
    safety = ExperimentSafetyConfig()
    return RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=safety.radar_max_distance_cm,
            body_x_half_cm=safety.radar_body_x_half_cm,
            body_y_half_cm=safety.radar_body_y_half_cm,
            forward_corridor_half_width_cm=(
                safety.radar_forward_corridor_half_width_cm
            ),
        )
    ).update(np.asarray(points, dtype=float), now_s=1.0)


def _perception() -> SimpleNamespace:
    return SimpleNamespace(
        is_road_found=True,
        confidence=0.9,
        corrected_pixel_error=100.0,
    )


if __name__ == "__main__":
    main()
