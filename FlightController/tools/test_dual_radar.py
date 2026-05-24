"""Dual D500 radar avoidance ground test.

Default behavior is dry-run. Non-zero FC commands are sent only when
--enable-flight is explicitly provided. The low-level FC protocol is not
modified here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (root, root.parent):
        value = str(p)
        if value not in sys.path:
            sys.path.insert(0, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual D500 radar avoidance test")
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--fc-port", default=None)
    parser.add_argument("--no-fc", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run; dry-run is also default unless --enable-flight is set")
    parser.add_argument("--enable-flight", action="store_true", help="Actually send non-zero velocity commands")
    parser.add_argument("--lower-mirror-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lower-mount-x-cm", type=float, default=0.96)
    parser.add_argument("--lower-mount-y-cm", type=float, default=0.15)
    parser.add_argument("--max-distance-cm", type=float, default=300.0)
    parser.add_argument("--stop-distance-cm", type=float, default=80.0)
    parser.add_argument("--slow-distance-cm", type=float, default=150.0)
    parser.add_argument("--corridor-half-width-cm", type=float, default=50.0)
    parser.add_argument("--body-x-half-cm", type=float, default=25.0)
    parser.add_argument("--body-y-half-cm", type=float, default=25.0)
    parser.add_argument("--radar-timeout-s", type=float, default=0.5)
    parser.add_argument("--min-battery-v", type=float, default=None)
    parser.add_argument("--loop-hz", type=float, default=30.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debug-dump", action="store_true")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def _format_age_ms(age_s: object) -> str:
    if age_s is None:
        return "None"
    try:
        return f"{float(age_s) * 1000.0:.0f}ms"
    except (TypeError, ValueError):
        return "None"


def _log_stale_radars(logger, health: dict[str, object]) -> None:
    for state in health.get("radars", []):
        if not isinstance(state, dict):
            continue
        age = state.get("last_frame_age_s")
        connected = bool(state.get("connected"))
        if connected and age is not None:
            try:
                if float(age) <= float(health.get("max_age_s", 0.5)):
                    continue
            except (TypeError, ValueError):
                pass
        name = state.get("name", "radar")
        age_str = "None" if age is None else f"{float(age):.2f}s"
        logger.warning(f"[SAFETY_STOP] radar_stale: {name} last_frame_age={age_str}")


def main() -> None:
    _setup_path()

    import numpy as np
    from loguru import logger

    from FlightController.Components.FCConnector import FCConnectConfig, connect_fc
    from FlightController.Components import MultiRadar, RadarConfig
    from FlightController.Solutions.LocalPlanner import LocalPlanner, PlannerConfig
    from FlightController.Solutions.Safety import (
        Command as SafeCommand,
        RadarFieldConfig,
        RadarObstacleField,
        SafetyArbiter,
        SafetyConfig,
        flight_status_from_fc,
        flight_health_from_sources,
        multi_radar_age_s,
        send_command_safely,
    )

    args = parse_args()
    if args.log_file:
        logger.add(args.log_file, enqueue=True, level="DEBUG")

    actual_dry_run = bool(args.dry_run or args.no_fc or not args.enable_flight)
    fc = None
    if not args.no_fc:
        fc = connect_fc(FCConnectConfig(port=args.fc_port, mode=2, timeout_s=10.0))
        logger.info("[FC] connected and switched to HOLD_POS mode; no unlock/takeoff is performed")

    if actual_dry_run:
        logger.warning("[SAFETY] dry-run mode: no non-zero velocity will be sent. Add --enable-flight to allow real output")

    configs = [
        RadarConfig(
            name="upper",
            index=0,
            mount_xy_cm=(0.0, 0.0),
            mount_yaw_deg=0.0,
            port=args.upper_port,
        ),
        RadarConfig(
            name="lower",
            index=1,
            mount_xy_cm=(args.lower_mount_x_cm, args.lower_mount_y_cm),
            mount_yaw_deg=0.0,
            mount_mirror_y=args.lower_mirror_y,
            port=args.lower_port,
        ),
    ]
    logger.info(
        "[DUAL-RADAR] upper port={} mount_xy=(0.00cm, 0.00cm) yaw=0.0deg mirror_y=False",
        args.upper_port,
    )
    logger.info(
        "[DUAL-RADAR] lower port={} mount_xy=({:.2f}cm, {:.2f}cm) yaw=0.0deg mirror_y={}",
        args.lower_port,
        args.lower_mount_x_cm,
        args.lower_mount_y_cm,
        args.lower_mirror_y,
    )
    multi_radar = MultiRadar(configs)
    radar_field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=args.max_distance_cm,
            body_x_half_cm=args.body_x_half_cm,
            body_y_half_cm=args.body_y_half_cm,
            forward_corridor_half_width_cm=args.corridor_half_width_cm,
        )
    )
    planner = LocalPlanner(
        PlannerConfig(
            enable_free_flight=True,
            free_flight_speed_cm_s=20.0,
            max_speed_cm_s=50.0,
            obstacle_stop_distance_cm=args.stop_distance_cm,
            obstacle_slow_distance_cm=args.slow_distance_cm,
            forward_corridor_half_width_cm=args.corridor_half_width_cm,
        )
    )
    safety = SafetyArbiter(
        SafetyConfig(
            require_fc=not args.no_fc,
            require_hold_pos_mode=not args.no_fc,
            radar_timeout_s=args.radar_timeout_s,
            min_battery_v=args.min_battery_v,
            obstacle_stop_distance_cm=args.stop_distance_cm,
            obstacle_slow_distance_cm=args.slow_distance_cm,
        )
    )
    send_enabled = bool(fc is not None and not actual_dry_run)

    try:
        multi_radar.start()
        logger.info("[DUAL-RADAR] waiting for data...")

        wait_start = time.perf_counter()
        while not multi_radar.connected and time.perf_counter() - wait_start <= 10.0:
            time.sleep(1.0)
            for radar in multi_radar.radars:
                stats = radar.get_radar_latency_stats()
                logger.info(
                    "[WAIT] {} connected={} frames_ok={} crc_errors={} age_ms={:.0f}".format(
                        radar.name,
                        radar.connected,
                        stats["serial_frames_ok"],
                        stats["crc_errors"],
                        stats["last_sample_age_ms"],
                    )
                )
        if not multi_radar.connected:
            logger.warning("[DUAL-RADAR] at least one radar is not connected; safety will hold zero velocity")

        period = 1.0 / max(args.loop_hz, 0.1)
        loop_count = 0
        wall_start = time.perf_counter()
        work_total_s = 0.0
        loop_times: list[float] = []
        upper_counts: list[int] = []
        lower_counts: list[int] = []
        body_filtered_counts: list[int] = []
        obs_distances: list[float] = []

        while True:
            t0 = time.perf_counter()
            all_points = multi_radar.get_obstacle_points_body_cm(max_distance_cm=args.max_distance_cm)
            radar_field.update(all_points, t0)

            raw_upper = int(np.count_nonzero(multi_radar.radars[0].map.data != -1))
            raw_lower = int(np.count_nonzero(multi_radar.radars[1].map.data != -1))
            body_filtered_count = len(radar_field.raw_points_body_cm) - len(radar_field.points_body_cm)

            local_command = planner.plan(obstacles_body_cm=radar_field.points_body_cm, target=None)
            desired = SafeCommand(
                local_command.vx_cm_s,
                local_command.vy_cm_s,
                local_command.vz_cm_s,
                local_command.yaw_rate_deg_s,
                local_command.reason,
            )
            radar_age_s = multi_radar_age_s(multi_radar)
            health = flight_health_from_sources(
                fc=fc,
                multi_radar=multi_radar,
                radar_timeout_s=args.radar_timeout_s,
            )
            safety_result = safety.filter(
                desired,
                flight=flight_status_from_fc(fc),
                radar_connected=bool(multi_radar.connected and multi_radar.is_fresh(max_age_s=args.radar_timeout_s)),
                radar_age_s=radar_age_s,
                radar_field=radar_field,
                enable_flight=send_enabled,
            )
            safe_command = safety_result.command
            send_decision = send_command_safely(
                fc,
                safe_command,
                safety,
                health,
                dry_run=actual_dry_run,
            )

            work_s = time.perf_counter() - t0
            loop_count += 1
            work_total_s += work_s
            loop_times.append(work_s * 1000.0)
            upper_counts.append(raw_upper)
            lower_counts.append(raw_lower)
            body_filtered_counts.append(body_filtered_count)
            if safety_result.nearest_forward_obstacle_cm is not None:
                obs_distances.append(safety_result.nearest_forward_obstacle_cm)

            if args.debug_dump and len(radar_field.points_body_cm) > 0:
                fwd = radar_field.points_body_cm[radar_field.points_body_cm[:, 0] > 10]
                corridor = fwd[abs(fwd[:, 1]) < args.corridor_half_width_cm]
                if len(corridor) > 0:
                    dists = np.linalg.norm(corridor, axis=1)
                    logger.info(
                        "[DEBUG] corridor_points={} nearest={:.0f} farthest={:.0f}".format(
                            len(corridor),
                            dists.min(),
                            dists.max(),
                        )
                    )

            if loop_count >= 50:
                wall_elapsed_s = max(0.001, time.perf_counter() - wall_start)
                health = multi_radar.get_health_snapshot(max_age_s=args.radar_timeout_s)
                t_arr = np.array(loop_times)
                u_arr = np.array(upper_counts)
                l_arr = np.array(lower_counts)
                b_arr = np.array(body_filtered_counts)
                effective_hz = loop_count / wall_elapsed_s
                cpu_pct = work_total_s / wall_elapsed_s * 100.0
                obs_str = f"{np.mean(obs_distances):.0f}cm" if obs_distances else "none"
                radar_states = health["radars"]
                upper_state = radar_states[0] if len(radar_states) > 0 else {}
                lower_state = radar_states[1] if len(radar_states) > 1 else {}
                upper_age = _format_age_ms(upper_state.get("last_frame_age_s"))
                lower_age = _format_age_ms(lower_state.get("last_frame_age_s"))
                logger.info(
                    "[DUAL-RADAR] fresh={} upper_age={} lower_age={} upper_pts={:.0f} "
                    "lower_pts={:.0f} body_masked={:.0f} front={} vx={} reason={}".format(
                        health["fresh"],
                        upper_age,
                        lower_age,
                        u_arr.mean(),
                        l_arr.mean(),
                        b_arr.mean(),
                        obs_str,
                        round(safe_command.vx_cm_s),
                        safe_command.reason,
                    )
                )
                if send_decision.hard_stop and send_decision.reason == "radar_not_fresh":
                    _log_stale_radars(logger, health)
                logger.info(
                    "FC_mode={} | points={:.0f}+{:.0f} body_filtered={:.0f} | "
                    "front={} desired={} safe={} safety={} radar_age={} "
                    "loop={:.1f}/{:.1f}/{:.1f}ms eff={:.0f}Hz cpu~{:.0f}%".format(
                        getattr(getattr(fc, "state", None), "mode", None).value if fc is not None else "no-fc",
                        u_arr.mean(),
                        l_arr.mean(),
                        b_arr.mean(),
                        obs_str,
                        desired.as_fc_tuple(),
                        safe_command.as_fc_tuple(),
                        safety_result.state,
                        f"{radar_age_s:.3f}s" if radar_age_s is not None else "none",
                        t_arr.mean(),
                        t_arr.max(),
                        t_arr.min(),
                        effective_hz,
                        cpu_pct,
                    )
                )
                loop_count = 0
                wall_start = time.perf_counter()
                work_total_s = 0.0
                loop_times.clear()
                upper_counts.clear()
                lower_counts.clear()
                body_filtered_counts.clear()
                obs_distances.clear()

            elapsed = time.perf_counter() - t0
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        logger.info("[DUAL-RADAR] interrupted")
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
                    SafeCommand.zero("shutdown"),
                    safety,
                    health,
                    dry_run=False,
                )
                time.sleep(0.05)
            finally:
                fc.close()
        multi_radar.stop()
        logger.info("[DUAL-RADAR] stopped")


if __name__ == "__main__":
    main()
