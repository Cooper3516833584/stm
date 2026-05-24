"""Relative goal navigation demo entry point.

This is a relative-direction demo, not a global autonomous navigation
solution. It does not unlock, take off, or land the aircraft.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relative goal navigation demo")
    parser.add_argument("--goal-x-cm", type=float, default=200.0,
                        help="目标在机体前方距离 (cm)。默认 200 仅供 dry-run 测试，实飞须显式指定")
    parser.add_argument("--goal-y-cm", type=float, default=0.0)
    parser.add_argument("--fc-port", default=None)
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--no-fc", action="store_true")
    parser.add_argument("--connect-fc", action="store_true", help="Compatibility flag; FC connects whenever --no-fc is absent")
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-flight", action="store_true")
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--radar-timeout-s", type=float, default=0.5)
    parser.add_argument("--max-distance-cm", type=float, default=300.0)
    parser.add_argument("--body-x-half-cm", type=float, default=25.0)
    parser.add_argument("--body-y-half-cm", type=float, default=25.0)
    parser.add_argument("--corridor-half-width-cm", type=float, default=50.0)
    parser.add_argument("--cruise-speed-cm-s", type=float, default=20.0)
    parser.add_argument("--yaw-rate-limit-deg-s", type=float, default=25.0)
    parser.add_argument("--yaw-kp", type=float, default=0.5)
    parser.add_argument("--arrive-distance-cm", type=float, default=30.0)
    parser.add_argument("--forward-test", action="store_true")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_file)

    from FlightController.Components import MultiRadar, RadarConfig
    from FlightController.Components.FCConnector import FCConnectConfig, connect_fc
    from FlightController.Solutions.RelativeGoalNavigator import (
        RelativeGoalConfig,
        RelativeGoalNavigator,
    )
    from FlightController.Solutions.Safety import (
        Command,
        RadarFieldConfig,
        RadarObstacleField,
        SafetyArbiter,
        SafetyConfig,
        flight_health_from_sources,
        flight_status_from_fc,
        multi_radar_age_s,
        send_command_safely,
    )

    actual_dry_run = bool(args.dry_run or args.no_fc or not args.enable_flight)
    if actual_dry_run:
        logger.warning("[SAFETY] dry-run mode: no non-zero velocity will be sent. Add --enable-flight to allow real output")
    if args.no_radar:
        logger.warning("[SAFETY] no-radar relative goal demo only; not for flight")

    fc = None
    multi_radar = None
    radar_field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=args.max_distance_cm,
            body_x_half_cm=args.body_x_half_cm,
            body_y_half_cm=args.body_y_half_cm,
            forward_corridor_half_width_cm=args.corridor_half_width_cm,
        )
    )
    navigator = RelativeGoalNavigator(
        RelativeGoalConfig(
            goal_x_cm=args.goal_x_cm,
            goal_y_cm=args.goal_y_cm,
            cruise_speed_cm_s=args.cruise_speed_cm_s,
            yaw_rate_limit_deg_s=args.yaw_rate_limit_deg_s,
            yaw_kp=args.yaw_kp,
            arrive_distance_cm=args.arrive_distance_cm,
            forward_test=args.forward_test,
        )
    )
    arbiter = SafetyArbiter(
        SafetyConfig(
            require_fc=not args.no_fc,
            require_hold_pos_mode=not args.no_fc,
            require_radar=not args.no_radar,
            radar_timeout_s=args.radar_timeout_s,
            max_vx_cm_s=args.cruise_speed_cm_s,
            max_yaw_rate_deg_s=args.yaw_rate_limit_deg_s,
        )
    )
    period_s = 1.0 / max(args.loop_hz, 0.1)

    try:
        if not args.no_fc:
            fc = connect_fc(FCConnectConfig(port=args.fc_port, mode=2, timeout_s=10.0))
            logger.info("[GOAL-DEMO] FC connected and switched to HOLD_POS mode; no unlock/takeoff is performed")

        if not args.no_radar:
            multi_radar = MultiRadar(_radar_configs(args.upper_port, args.lower_port))
            multi_radar.start()

        logger.info(
            "[GOAL-DEMO] started dry_run={} relative_goal=({:.0f},{:.0f}) no_radar={}".format(
                actual_dry_run,
                args.goal_x_cm,
                args.goal_y_cm,
                args.no_radar,
            )
        )
        last_log_s = 0.0
        while True:
            loop_start = time.perf_counter()

            if multi_radar is not None:
                points = multi_radar.get_obstacle_points_body_cm(max_distance_cm=args.max_distance_cm)
                radar_field.update(points, loop_start)
                radar_age_s = multi_radar_age_s(multi_radar)
                radar_connected = bool(multi_radar.connected and multi_radar.is_fresh(max_age_s=args.radar_timeout_s))
            else:
                radar_field.update(np.empty((0, 2), dtype=float), loop_start)
                radar_age_s = 0.0
                radar_connected = True

            desired = navigator.update(radar_field, now_s=loop_start)
            health = flight_health_from_sources(
                fc=fc,
                multi_radar=multi_radar,
                radar_timeout_s=args.radar_timeout_s,
            )
            safe = arbiter.filter(
                desired,
                flight=flight_status_from_fc(fc),
                radar_connected=radar_connected,
                radar_age_s=radar_age_s,
                radar_field=radar_field,
                enable_flight=not actual_dry_run,
            )
            decision = send_command_safely(
                fc,
                safe.command,
                arbiter,
                health,
                dry_run=actual_dry_run,
            )

            if loop_start - last_log_s >= 1.0:
                last_log_s = loop_start
                logger.info(
                    "[GOAL-DEMO] relative_goal=({:.0f},{:.0f}) desired=(vx={} yaw={}) "
                    "safe=(vx={} yaw={}) safety={} sent={} radar_fresh={}".format(
                        args.goal_x_cm,
                        args.goal_y_cm,
                        round(desired.vx_cm_s),
                        round(desired.yaw_rate_deg_s),
                        round(safe.command.vx_cm_s),
                        round(safe.command.yaw_rate_deg_s),
                        decision.reason,
                        bool(not actual_dry_run and fc is not None),
                        radar_connected if multi_radar is not None else "disabled",
                    )
                )

            _sleep_to_rate(loop_start, period_s)
    except KeyboardInterrupt:
        logger.info("[GOAL-DEMO] interrupted")
    finally:
        if fc is not None:
            try:
                health = flight_health_from_sources(
                    fc=fc,
                    multi_radar=multi_radar,
                    radar_timeout_s=args.radar_timeout_s,
                )
                send_command_safely(
                    fc,
                    Command.zero("shutdown"),
                    arbiter,
                    health,
                    dry_run=False,
                )
                time.sleep(0.05)
            finally:
                fc.close()
        if multi_radar is not None:
            multi_radar.stop()
        logger.info("[GOAL-DEMO] stopped")


def _radar_configs(upper_port: str, lower_port: str):
    from FlightController.Components import RadarConfig

    return [
        RadarConfig("upper", 0, (0.0, 0.0), 0.0, port=upper_port),
        RadarConfig("lower", 1, (0.96, 0.15), 0.0, port=lower_port, mount_mirror_y=True),
    ]


def _setup_logging(log_file: str | None) -> None:
    if not log_file:
        return
    path = Path(log_file)
    if str(path).replace("\\", "/").startswith("/tmp/"):
        logger.warning("Avoid writing logs to /tmp on the target board")
    logger.add(str(path), enqueue=True, level="DEBUG")


def _sleep_to_rate(loop_start: float, period_s: float) -> None:
    elapsed = time.perf_counter() - loop_start
    if elapsed < period_s:
        time.sleep(period_s - elapsed)


if __name__ == "__main__":
    main()
