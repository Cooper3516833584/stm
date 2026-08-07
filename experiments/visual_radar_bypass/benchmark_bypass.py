"""No-hardware regression assertions and deterministic planner benchmark."""

from __future__ import annotations

import argparse
import json
import math
import time
from types import SimpleNamespace

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
from .radar_bypass import ObstacleBypassConfig, ObstacleBypassPlanner
from .smooth_sidestep import SmoothSidestepPlanner
from .static_route_bypass import StaticRouteBypassPlanner, StaticRouteBypassState


DT_S = 0.1


def _perception():
    return SimpleNamespace(is_road_found=True, confidence=0.95, corrected_pixel_error=0.0)


def _desired(vy: float = -6.0, yaw: float = 0.0) -> Command:
    return Command(14.0, vy, 0.0, yaw, "trajectory_point_follow:single")


def _field(points: np.ndarray, now_s: float) -> RadarObstacleField:
    return RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=300.0,
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=75.0,
        )
    ).update(points, now_s)


def _tube_surface(center: np.ndarray, radius_cm: float = 15.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, 36, endpoint=False)
    points = center + radius_cm * np.column_stack((np.cos(angles), np.sin(angles)))
    return points[np.linalg.norm(points, axis=1) <= 300.0]


def _propagate_static_center(center: np.ndarray, command: Command, dt_s: float) -> np.ndarray:
    translated = center - np.asarray([command.vx_cm_s, command.vy_cm_s]) * dt_s
    theta = math.radians(command.yaw_rate_deg_s * dt_s)
    cosine, sine = math.cos(-theta), math.sin(-theta)
    return np.asarray([[cosine, -sine], [sine, cosine]]) @ translated


def run_static_closed_loop(*, yaw_rate_deg_s: float = 0.0) -> dict[str, object]:
    planner = StaticRouteBypassPlanner()
    arbiter = SafetyArbiter(
        SafetyConfig(
            require_fc=False,
            require_hold_pos_mode=False,
            require_radar=False,
            max_vx_cm_s=14.0,
            max_vy_cm_s=10.0,
            max_yaw_rate_deg_s=10.0,
            obstacle_stop_distance_cm=80.0,
            obstacle_slow_distance_cm=150.0,
            slow_speed_limit_cm_s=10.0,
            side_stop_distance_cm=45.0,
        )
    )
    center = np.asarray([150.0, -30.0])
    states: list[str] = []
    sides: list[int | None] = []
    commands: list[Command] = []
    minimum_surface_clearance = math.inf
    visual_return_before_pass = False
    completion_center: np.ndarray | None = None

    for frame in range(450):
        now = 1.0 + frame * DT_S
        points = _tube_surface(center)
        field = _field(points, now)
        desired = _desired(vy=-8.0, yaw=yaw_rate_deg_s)
        planned = planner.update(
            desired=desired,
            perception=_perception(),
            radar_field=field,
            now_s=now,
        )
        safe = arbiter.filter(
            planned,
            flight=FlightStatus(),
            radar_connected=True,
            radar_age_s=0.0,
            radar_field=field,
        ).command
        planner.report_applied_command(safe, DT_S, True)
        center = _propagate_static_center(center, safe, DT_S)
        states.append(planner.state.value)
        sides.append(planner.active_bypass_side)
        commands.append(safe)
        minimum_surface_clearance = min(
            minimum_surface_clearance,
            max(0.0, float(np.linalg.norm(center)) - planner.config.tube_radius_cm),
        )
        if planner.state not in {StaticRouteBypassState.BLEND_BACK, StaticRouteBypassState.NORMAL}:
            visual_return_before_pass |= safe.vy_cm_s < -1e-6
        if planner.state == StaticRouteBypassState.BLEND_BACK and completion_center is None:
            completion_center = center.copy()
        if planner.state == StaticRouteBypassState.NORMAL and planner.encounter_id > 0:
            break

    side_switches = sum(
        1
        for first, second in zip(sides, sides[1:])
        if first is not None and second is not None and first != second
    )
    assert completion_center is not None, f"static-route did not complete: {planner.diagnostics()}"
    assert completion_center[0] + planner.config.tube_radius_cm + planner.config.rear_margin_cm <= 1.0
    assert not visual_return_before_pass
    assert side_switches == 0
    assert all(abs(command.vx_cm_s) <= 14.0 for command in commands)
    assert all(abs(command.vy_cm_s) <= 10.0 for command in commands)
    assert all(abs(command.yaw_rate_deg_s) <= 10.0 for command in commands)
    return {
        "frames": len(commands),
        "completed": True,
        "completion_center_cm": completion_center.tolist(),
        "minimum_radial_surface_clearance_cm": minimum_surface_clearance,
        "state_switches": sum(a != b for a, b in zip(states, states[1:])),
        "side_switches": side_switches,
        "max_delta_vx_cm_s": _max_delta([command.vx_cm_s for command in commands]),
        "max_delta_vy_cm_s": _max_delta([command.vy_cm_s for command in commands]),
        "max_delta_yaw_deg_s": _max_delta([command.yaw_rate_deg_s for command in commands]),
    }


def run_compatibility_assertions() -> dict[str, object]:
    field = _field(np.asarray([[99.0, -39.0], [100.0, -40.0], [101.0, -41.0], [102.0, -42.0]]), 1.0)
    planners = {
        "legacy_basic": ObstacleBypassPlanner(ObstacleBypassConfig(forward_recovery_s=0.0)),
        "legacy_recovery": ObstacleBypassPlanner(),
        "smooth_sidestep": SmoothSidestepPlanner(),
        "circular_bypass": CircularTubeBypassPlanner(),
        "static_route": StaticRouteBypassPlanner(),
    }
    result: dict[str, object] = {}
    for name, planner in planners.items():
        planner.update(desired=_desired(), perception=_perception(), radar_field=field, now_s=1.0)
        command = planner.update(desired=_desired(), perception=_perception(), radar_field=field, now_s=1.1)
        values = command.as_fc_tuple()
        assert all(math.isfinite(float(value)) for value in values)
        assert abs(command.vx_cm_s) <= 14.0
        assert abs(command.vy_cm_s) <= 10.0
        assert abs(command.yaw_rate_deg_s) <= 10.0
        result[name] = {"state": planner.state.value, "command": values}
    return result


def run_microbenchmark(iterations: int = 1000) -> dict[str, object]:
    field = _field(np.asarray([[99.0, -39.0], [100.0, -40.0], [101.0, -41.0], [102.0, -42.0]]), 1.0)
    factories = {
        "legacy_basic": lambda: ObstacleBypassPlanner(ObstacleBypassConfig(forward_recovery_s=0.0)),
        "legacy_recovery": ObstacleBypassPlanner,
        "smooth_sidestep": SmoothSidestepPlanner,
        "circular_bypass": CircularTubeBypassPlanner,
        "static_route": StaticRouteBypassPlanner,
    }
    output: dict[str, object] = {}
    for name, factory in factories.items():
        planner = factory()
        samples: list[float] = []
        for index in range(iterations):
            started = time.perf_counter_ns()
            command = planner.update(
                desired=_desired(),
                perception=_perception(),
                radar_field=field,
                now_s=1.0 + index * DT_S,
            )
            samples.append((time.perf_counter_ns() - started) / 1000.0)
            report = getattr(planner, "report_applied_command", None)
            if callable(report):
                report(command, DT_S, False)
        values = np.asarray(samples)
        output[name] = {
            "mean_us": float(np.mean(values)),
            "p50_us": float(np.percentile(values, 50.0)),
            "p95_us": float(np.percentile(values, 95.0)),
        }
    return output


def _max_delta(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(abs(second - first) for first, second in zip(values, values[1:]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="No-FC static-route regression and benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--assert-only", action="store_true")
    args = parser.parse_args(argv)
    result = {
        "static_straight": run_static_closed_loop(),
        "static_curved": run_static_closed_loop(yaw_rate_deg_s=2.0),
        "compatibility": run_compatibility_assertions(),
    }
    if not args.assert_only:
        result["benchmark"] = run_microbenchmark(max(1, args.iterations))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
