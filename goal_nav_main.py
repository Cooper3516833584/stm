"""Main placeholder program for radar obstacle scan + relative goal navigation.

This program does not use pose estimation yet. The goal is interpreted in the
current body frame, or --forward-test can be used for early straight-ahead
avoidance tests. Automatic landing is intentionally left manual for this phase.
"""

from __future__ import annotations

import argparse
import time

from loguru import logger

from autonomy_context import AutonomyContext, SensorHealth, flight_status_from_fc
from autonomy_hardware import build_dual_radar, connect_fc, send_fc_command, stop_fc
from direction_planner import DirectionPlanner, DirectionPlannerConfig
from goal_nav_mission import GoalNavConfig, GoalNavMission
from local_world_model import LocalWorldModel, LocalWorldModelConfig
from safety_arbiter import SafetyArbiter, SafetyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Radar goal navigation placeholder")
    parser.add_argument("--enable-flight", action="store_true", help="Actually send commands to FC")
    parser.add_argument("--connect-fc", action="store_true", help="Connect FC even in dry-run")
    parser.add_argument("--fc-port", default=None)
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--goal-x-cm", type=float, default=300.0)
    parser.add_argument("--goal-y-cm", type=float, default=0.0)
    parser.add_argument("--forward-test", action="store_true")
    parser.add_argument("--loop-hz", type=float, default=20.0)
    parser.add_argument("--max-distance-cm", type=float, default=300.0)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_file:
        logger.add(args.log_file, enqueue=True, level="DEBUG")

    enable_flight = bool(args.enable_flight)
    fc = None
    radar = None
    world = LocalWorldModel(LocalWorldModelConfig(max_distance_cm=args.max_distance_cm))
    planner = DirectionPlanner(DirectionPlannerConfig())
    mission = GoalNavMission(
        GoalNavConfig(
            goal_x_cm=args.goal_x_cm,
            goal_y_cm=args.goal_y_cm,
            forward_test=args.forward_test,
        ),
        planner=planner,
    )
    safety = SafetyArbiter(SafetyConfig())
    period_s = 1.0 / max(args.loop_hz, 0.1)

    try:
        if enable_flight or args.connect_fc:
            fc = connect_fc(args.fc_port)
        radar = build_dual_radar(args.upper_port, args.lower_port)
        radar.start()

        logger.info(
            f"[GOAL] started enable_flight={enable_flight} "
            f"goal=({args.goal_x_cm},{args.goal_y_cm}) forward_test={args.forward_test}"
        )
        while True:
            loop_start = time.perf_counter()
            points = radar.get_obstacle_points_body_cm(max_distance_cm=args.max_distance_cm)
            snapshot = world.update_from_radar_points(points, loop_start)
            radar_age = world.radar_age_s(loop_start)
            context = AutonomyContext(
                now_s=loop_start,
                health=SensorHealth(
                    radar_ok=bool(radar.connected),
                    camera_ok=False,
                    fc_ok=bool(getattr(fc, "connected", False)) if fc is not None else False,
                    radar_age_s=radar_age,
                ),
                flight=flight_status_from_fc(fc),
                obstacles=snapshot.obstacles,
            )
            desired = mission.update(world)
            safe = safety.filter(desired, context=context, world=world)
            send_fc_command(fc, safe.command, enable_flight)

            obstacle_summary = ",".join(
                f"{obs.kind}@({obs.x_cm:.0f},{obs.y_cm:.0f})" for obs in snapshot.obstacles[:4]
            )
            logger.info(
                "[GOAL] desired={} safe={} safety={} nearest={} obstacles={}".format(
                    desired.as_fc_tuple(),
                    safe.command.as_fc_tuple(),
                    safe.state,
                    safe.nearest_forward_obstacle_cm,
                    obstacle_summary or "none",
                )
            )

            elapsed = time.perf_counter() - loop_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
    except KeyboardInterrupt:
        logger.info("[GOAL] interrupted by user")
    finally:
        stop_fc(fc, enable_flight)
        if radar is not None:
            radar.stop()
        logger.info("[GOAL] stopped")


if __name__ == "__main__":
    main()

