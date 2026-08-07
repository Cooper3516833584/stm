import ast
from pathlib import Path

import road_follow_main
import road_trajectory_main
from experiments.visual_radar_bypass import main as static_route_main


def test_frozen_static_route_uses_process_runtime_by_default_without_flight():
    args = static_route_main.parse_args([])
    assert args.bypass_planner == "static-route"
    assert args.runtime_mode == "process"
    assert not args.enable_flight
    assert not args.auto_takeoff


def test_road_trajectory_dry_run_keeps_process_runtime_and_no_fc():
    args = road_follow_main.parse_args(
        road_trajectory_main.build_argv(["--dry-run", "--no-fc"])
    )
    assert args.road_controller == "trajectory-point"
    assert args.runtime_mode == "process"
    assert args.dry_run
    assert args.no_fc
    assert not args.enable_flight
    assert not args.auto_takeoff


def test_board_benchmark_source_cannot_import_flight_controller_connector():
    path = Path("FlightController/tools/bench_process_runtime.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("FCConnector" in module for module in imported)
