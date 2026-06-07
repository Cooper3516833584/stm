"""Relative goal navigation demo entry point.

This is a relative-direction demo, not a global autonomous navigation
solution. It does not unlock, take off, or land the aircraft.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
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
    parser.add_argument("--obstacle-clearance-cm", type=float, default=80.0,
                        help="Hard obstacle clearance for goal avoidance, default 80cm")
    parser.add_argument("--clearance-release-cm", type=float, default=90.0,
                        help="Clearance required to resume forward motion after blocking, default 90cm")
    parser.add_argument("--scan-fov-deg", type=float, default=150.0,
                        help="Goal-avoidance scan FOV in degrees, default front 150deg")
    parser.add_argument("--candidate-edge-margin-deg", type=float, default=10.0,
                        help="Margin removed from scan FOV edges for candidate directions, default 10deg")
    parser.add_argument("--lookahead-cm", type=float, default=220.0,
                        help="Candidate direction lookahead distance, default 220cm")
    parser.add_argument("--avoid-begin-distance-cm", type=float, default=150.0,
                        help="Distance where early avoidance and speed shaping begin, default 150cm")
    parser.add_argument("--candidate-step-deg", type=float, default=5.0,
                        help="Candidate direction step in degrees, default 5deg")
    parser.add_argument("--align-start-deg", type=float, default=10.0,
                        help="Start turn-in-place above this selected direction error")
    parser.add_argument("--align-stop-deg", type=float, default=3.0,
                        help="Continue turn-in-place until error is below this threshold")
    parser.add_argument("--min-turn-yaw-rate-deg-s", type=float, default=6.0,
                        help="Minimum yaw rate while turning in place")
    parser.add_argument("--min-forward-speed-cm-s", type=float, default=8.0,
                        help="Minimum forward speed between clearance and avoidance distances")
    parser.add_argument("--record-dir", default="/media/sdcard/stm_records",
                        help="Directory on SD card for session recording")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable camera/radar session recording")
    parser.add_argument("--record-frame-every-n", type=int, default=10,
                        help="Save one camera frame every N control loops")
    parser.add_argument("--record-radar-every-n", type=int, default=1,
                        help="Save one radar metadata/point snapshot every N control loops")
    parser.add_argument("--record-jpeg-quality", type=int, default=85)
    parser.add_argument("--record-camera-index", type=int, default=9,
                        help="Camera index used only for goal-nav recording")
    parser.add_argument("--record-camera-width", type=int, default=640)
    parser.add_argument("--record-camera-height", type=int, default=480)
    parser.add_argument("--record-camera-fps", type=int, default=30)
    parser.add_argument("--no-record-camera", action="store_true",
                        help="Record radar only; do not open a camera in goal_nav_main")
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
    from FlightController.Solutions.SessionRecorder import SessionRecorder, SessionRecorderConfig
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

    actual_dry_run = _is_actual_dry_run(args)
    if actual_dry_run:
        logger.warning("[SAFETY] dry-run mode: no non-zero velocity will be sent. Add --enable-flight to allow real output")
    if args.no_radar:
        logger.warning("[SAFETY] no-radar forces dry-run; relative goal demo is not allowed to fly without radar")
    if abs(args.goal_y_cm) > 1e-6:
        logger.warning(
            "[GOAL-DEMO] goal_y_cm is non-zero. Current goal_nav_main is a front-only local "
            "obstacle avoidance demo and does not integrate global pose/yaw. Use --goal-y-cm 0 "
            "for reliable avoidance tests."
        )

    fc = None
    multi_radar = None
    record_cap = None
    recorder = SessionRecorder(
        SessionRecorderConfig(
            root_dir=args.record_dir,
            enabled=not args.no_record,
            mode="goal_nav",
            frame_every_n=args.record_frame_every_n,
            radar_every_n=args.record_radar_every_n,
            jpeg_quality=args.record_jpeg_quality,
        )
    )
    radar_field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=args.max_distance_cm,
            body_x_half_cm=args.body_x_half_cm,
            body_y_half_cm=args.body_y_half_cm,
            forward_corridor_half_width_cm=max(
                args.corridor_half_width_cm,
                args.obstacle_clearance_cm,
            ),
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
            scan_fov_deg=args.scan_fov_deg,
            candidate_edge_margin_deg=args.candidate_edge_margin_deg,
            candidate_step_deg=args.candidate_step_deg,
            obstacle_clearance_cm=args.obstacle_clearance_cm,
            clearance_release_cm=args.clearance_release_cm,
            lookahead_cm=args.lookahead_cm,
            avoid_begin_distance_cm=args.avoid_begin_distance_cm,
            align_start_deg=args.align_start_deg,
            align_stop_deg=args.align_stop_deg,
            min_turn_yaw_rate_deg_s=args.min_turn_yaw_rate_deg_s,
            min_forward_speed_cm_s=args.min_forward_speed_cm_s,
            allow_sideways_velocity=False,
        )
    )
    arbiter = SafetyArbiter(
        SafetyConfig(
            require_fc=not args.no_fc,
            require_hold_pos_mode=not args.no_fc,
            require_radar=not args.no_radar,
            radar_timeout_s=args.radar_timeout_s,
            max_vx_cm_s=args.cruise_speed_cm_s,
            max_vy_cm_s=0.0,
            max_yaw_rate_deg_s=args.yaw_rate_limit_deg_s,
            obstacle_stop_distance_cm=args.obstacle_clearance_cm,
            obstacle_slow_distance_cm=args.avoid_begin_distance_cm,
            slow_speed_limit_cm_s=min(args.cruise_speed_cm_s, 12.0),
            side_stop_distance_cm=args.obstacle_clearance_cm,
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

        if not args.no_record and not args.no_record_camera:
            record_cap = _open_record_camera(args)
            if record_cap is None:
                logger.warning("[GOAL-DEMO] record camera open failed; radar metadata will still be recorded")

        logger.info(
            "[GOAL-DEMO] started dry_run={} relative_goal=({:.0f},{:.0f}) "
            "no_radar={} clearance={:.0f}cm fov={:.0f}deg lookahead={:.0f}cm".format(
                actual_dry_run,
                args.goal_x_cm,
                args.goal_y_cm,
                args.no_radar,
                args.obstacle_clearance_cm,
                args.scan_fov_deg,
                args.lookahead_cm,
            )
        )
        last_log_s = 0.0
        loop_count = 0
        while True:
            loop_start = time.perf_counter()
            camera_ok, frame = _read_record_camera(record_cap)
            recorder.record_frame(loop_count=loop_count, now_s=loop_start, frame=frame, label="goal")

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
            recorder.record_radar(
                loop_count=loop_count,
                now_s=loop_start,
                radar_field=radar_field,
                multi_radar=multi_radar,
                radar_age_s=radar_age_s,
                radar_connected=radar_connected,
                desired=desired,
                safe_command=safe.command,
                decision_reason=decision.reason,
                extra={
                    "camera_ok": bool(camera_ok),
                    "goal_x_cm": float(args.goal_x_cm),
                    "goal_y_cm": float(args.goal_y_cm),
                },
            )
            recorder.record_command(
                loop_count=loop_count,
                now_s=loop_start,
                desired=desired,
                safe_command=safe.command,
                decision_reason=decision.reason,
                extra={"camera_ok": bool(camera_ok)},
            )

            if loop_start - last_log_s >= 1.0:
                last_log_s = loop_start
                logger.info(
                    "[GOAL-DEMO] relative_goal=({:.0f},{:.0f}) desired=(vx={} vy={} yaw={}) "
                    "safe=(vx={} vy={} yaw={}) reason={} safety={} sent={} radar_fresh={}".format(
                        args.goal_x_cm,
                        args.goal_y_cm,
                        round(desired.vx_cm_s),
                        round(desired.vy_cm_s),
                        round(desired.yaw_rate_deg_s),
                        round(safe.command.vx_cm_s),
                        round(safe.command.vy_cm_s),
                        round(safe.command.yaw_rate_deg_s),
                        desired.reason,
                        decision.reason,
                        bool(not actual_dry_run and fc is not None),
                        radar_connected if multi_radar is not None else "disabled",
                    )
                )

            loop_count += 1
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
        if record_cap is not None:
            record_cap.release()
        recorder.close()
        logger.info("[GOAL-DEMO] stopped")


def _radar_configs(upper_port: str, lower_port: str):
    from FlightController.Components import RadarConfig

    return [
        RadarConfig("upper", 0, (0.0, 0.0), 0.0, port=upper_port),
        RadarConfig("lower", 1, (0.96, 0.15), 0.0, port=lower_port, mount_mirror_y=True),
    ]


def _is_actual_dry_run(args: argparse.Namespace) -> bool:
    return bool(
        args.dry_run
        or args.no_fc
        or args.no_radar
        or not args.enable_flight
    )


def _open_record_camera(args: argparse.Namespace):
    cap = cv2.VideoCapture(args.record_camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.record_camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.record_camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.record_camera_fps)
    return cap


def _read_record_camera(cap) -> tuple[bool, np.ndarray | None]:
    if cap is None or not cap.isOpened():
        return False, None
    ok, frame = cap.read()
    if not ok:
        return False, None
    return True, frame


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
