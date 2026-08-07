"""Replay recorded physical radar frames through static-route without hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from FlightController.Solutions.Safety import Command, RadarFieldConfig, RadarObstacleField
from .static_route_bypass import StaticRouteBypassPlanner


def replay(session_dir: str | Path) -> dict[str, object]:
    root = Path(session_dir)
    log_path = root / "radar.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError(f"radar log missing: {log_path}")
    planner = StaticRouteBypassPlanner()
    field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=300.0,
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=75.0,
        )
    )
    perception = SimpleNamespace(is_road_found=True, confidence=0.95)
    desired = Command(14.0, -6.0, 0.0, 0.0, "recorded_replay")
    states: list[str] = []
    valid_observations = 0
    front_180_points = 0
    frames = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        points_name = record.get("points_file")
        if not points_name:
            continue
        points_path = Path(points_name)
        if not points_path.is_file():
            points_path = root / "radar_points" / points_path.name
        with np.load(points_path) as payload:
            points = np.asarray(payload["points_body_cm"], dtype=float)
        now = 1.0 + frames * 0.1
        field.update(points, now)
        planner.update(
            desired=desired,
            perception=perception,
            radar_field=field,
            now_s=now,
        )
        # Recorded replay is deliberately not treated as executed movement.
        planner.report_applied_command(Command.zero("replay_not_applied"), 0.1, False)
        diagnostics = planner.diagnostics()
        valid_observations += int(bool(diagnostics["observation_valid"]))
        body_points = np.asarray(field.points_body_cm)
        if body_points.size:
            front_180_points += int(np.count_nonzero(body_points[:, 0] >= 0.0))
        states.append(planner.state.value)
        frames += 1
    if frames == 0:
        raise RuntimeError(f"no readable radar frames in {log_path}")
    return {
        "session": str(root),
        "frames": frames,
        "valid_observation_frames": valid_observations,
        "front_180_point_total": front_180_points,
        "states": sorted(set(states)),
        "final_state": planner.state.value,
        "command_progress_applied": planner.diagnostics()["credited_translation_cm"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a SessionRecorder radar directory")
    parser.add_argument("session_dir")
    args = parser.parse_args(argv)
    print(json.dumps(replay(args.session_dir), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
