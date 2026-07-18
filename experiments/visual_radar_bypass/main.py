"""Independent real-vision + physical-radar tubular-obstacle experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
from loguru import logger

from FlightController.Components import MultiRadar, RadarConfig
from FlightController.Components.FCConnector import FCConnectConfig, connect_fc
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
from FlightController.Solutions.SessionRecorder import (
    SessionRecorder,
    SessionRecorderConfig,
)

from .flight_runtime import (
    FlightRuntimeConfig,
    auto_takeoff,
    land_and_wait_for_lock,
    wait_for_radars,
    wait_for_visual_road,
)
from .radar_bypass import ObstacleBypassPlanner
from .smooth_sidestep import SmoothSidestepPlanner
from .visual_guidance import FrozenVisualConfig, FrozenVisualGuidance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated real-vision/physical-radar tubular-obstacle test"
    )
    parser.add_argument("--camera-index", type=int, default=7)
    parser.add_argument("--model-npu", default=FrozenVisualConfig().npu_model_path)
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--fc-port", default=None)
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--radar-timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--bypass-planner",
        choices=("legacy", "smooth-sidestep"),
        default="legacy",
        help="Select the unchanged legacy planner or the isolated smooth sidestep",
    )
    parser.add_argument("--record-dir", default="/media/sdcard/stm_records")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--enable-flight", action="store_true")
    parser.add_argument("--auto-takeoff", action="store_true")
    parser.add_argument(
        "--confirm-visual-radar-flight-test",
        action="store_true",
        help="Acknowledge real unlock/takeoff using live camera and physical radars",
    )
    parser.add_argument("--takeoff-height-cm", type=int, default=100)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.model_npu):
        raise FileNotFoundError(f"required NPU model missing: {args.model_npu}")
    if args.loop_hz <= 0.0:
        raise ValueError("--loop-hz must be greater than zero")
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be greater than zero")
    if args.enable_flight:
        missing = []
        if not args.auto_takeoff:
            missing.append("--auto-takeoff")
        if not args.confirm_visual_radar_flight_test:
            missing.append("--confirm-visual-radar-flight-test")
        if missing:
            raise ValueError("--enable-flight requires " + ", ".join(missing))
        if args.no_record:
            raise ValueError("real flight test requires recording")
        if not 40 <= args.takeoff_height_cm <= 100:
            raise ValueError("flight-test takeoff height must be within 40..100cm")
        if args.duration_s > 120.0:
            raise ValueError("real flight test duration cannot exceed 120s")
    elif args.auto_takeoff or args.confirm_visual_radar_flight_test:
        raise ValueError("takeoff/confirmation options require --enable-flight")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    actual_flight = bool(args.enable_flight)
    visual_config = FrozenVisualConfig(
        camera_index=args.camera_index,
        npu_model_path=args.model_npu,
    )
    flight_config = FlightRuntimeConfig(
        takeoff_height_cm=args.takeoff_height_cm,
    )
    session_mode = (
        "isolated_visual_radar_smooth_sidestep"
        if args.bypass_planner == "smooth-sidestep"
        else "isolated_visual_radar_tube_obstacle"
    )
    recorder = SessionRecorder(
        SessionRecorderConfig(
            root_dir=args.record_dir,
            enabled=not args.no_record,
            mode=session_mode,
            frame_every_n=10,
            radar_every_n=1,
            video_enabled=True,
            video_every_n=2,
            video_fps=5.0,
            metadata={
                "argv": list(sys.argv),
                "visual_config": vars(visual_config),
                "physical_obstacle": (
                    "real movable tube; position is inferred from physical radar points"
                ),
                "radar_points": "physical only; no synthetic injection",
                "bypass_planner": args.bypass_planner,
            },
        )
    )
    if actual_flight and not recorder.enabled:
        raise RuntimeError("flight test refused because session recording is unavailable")
    sink_id = _setup_logging(recorder.runtime_log_path)

    guidance = FrozenVisualGuidance(visual_config)
    radars = MultiRadar(_radar_configs(args.upper_port, args.lower_port))
    radar_field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=300.0,
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=75.0,
        )
    )
    planner = (
        SmoothSidestepPlanner()
        if args.bypass_planner == "smooth-sidestep"
        else ObstacleBypassPlanner()
    )
    arbiter = SafetyArbiter(
        SafetyConfig(
            require_fc=actual_flight,
            require_hold_pos_mode=actual_flight,
            require_unlocked=actual_flight,
            require_radar=True,
            radar_timeout_s=args.radar_timeout_s,
            max_vx_cm_s=visual_config.max_vx_cm_s,
            max_vy_cm_s=visual_config.max_vy_cm_s,
            max_yaw_rate_deg_s=visual_config.max_yaw_rate_deg_s,
            obstacle_stop_distance_cm=80.0,
            obstacle_slow_distance_cm=150.0,
            slow_speed_limit_cm_s=10.0,
        )
    )

    fc = None
    flight_owned = False
    interrupted = False
    guidance_started = False
    radars_started = False
    period_s = 1.0 / args.loop_hz
    try:
        guidance.start()
        guidance_started = True
        radars.start()
        radars_started = True
        wait_for_radars(radars, timeout_s=5.0, max_age_s=args.radar_timeout_s)
        wait_for_visual_road(guidance, timeout_s=10.0, consecutive_frames=3)

        if actual_flight:
            fc = connect_fc(FCConnectConfig(port=args.fc_port, mode=2, timeout_s=10.0))
            flight_owned = True
            auto_takeoff(fc, flight_config)
        else:
            logger.warning(
                "[VIS-RADAR] dry run: real camera/radars active, no FC connection"
            )

        start_s = time.perf_counter()
        last_log_s = 0.0
        loop_count = 0
        while time.perf_counter() - start_s < args.duration_s:
            loop_start = time.perf_counter()
            sample = guidance.sample(loop_start)
            points = radars.get_obstacle_points_body_cm(max_distance_cm=300.0)
            radar_field.update(points, loop_start)
            radar_age_s = multi_radar_age_s(radars)
            radar_fresh = bool(
                radars.connected
                and radars.is_fresh(max_age_s=args.radar_timeout_s)
            )
            planned = planner.update(
                desired=sample.desired,
                perception=sample.perception,
                radar_field=radar_field,
                now_s=loop_start,
            )
            health = flight_health_from_sources(
                fc=fc,
                multi_radar=radars,
                radar_timeout_s=args.radar_timeout_s,
                camera_ok=sample.camera_ok,
            )
            safe = arbiter.filter(
                planned,
                flight=flight_status_from_fc(fc),
                radar_connected=radar_fresh,
                radar_age_s=radar_age_s,
                radar_field=radar_field,
                enable_flight=actual_flight,
            )
            decision = send_command_safely(
                fc,
                safe.command,
                arbiter,
                health,
                dry_run=not actual_flight,
            )
            extra = {
                "visual": {
                    "road_found": bool(
                        getattr(sample.perception, "is_road_found", False)
                    ),
                    "confidence": _float_or_none(
                        getattr(sample.perception, "confidence", None)
                    ),
                    "pixel_error": _float_or_none(
                        getattr(sample.perception, "corrected_pixel_error", None)
                    ),
                    "angle_deg": _float_or_none(
                        getattr(sample.perception, "centerline_angle", None)
                    ),
                    "age_s": sample.perception_age_s,
                    "stale": sample.perception_stale,
                    "camera_ok": sample.camera_ok,
                    "controller": sample.diagnostics,
                },
                "tube_obstacle_bypass": planner.diagnostics(),
                "sent": bool(actual_flight and decision.allowed),
            }
            if recorder.frame_due(loop_count):
                recorder.record_frame(
                    loop_count=loop_count,
                    now_s=loop_start,
                    frame=sample.frame,
                    label="road",
                    source_time_s=sample.frame_time_s,
                    extra=extra,
                )
            recorder.record_radar(
                loop_count=loop_count,
                now_s=loop_start,
                radar_field=radar_field,
                multi_radar=radars,
                radar_age_s=radar_age_s,
                radar_connected=radar_fresh,
                desired=sample.desired,
                safe_command=safe.command,
                decision_reason=decision.reason,
                extra=extra,
            )
            recorder.record_command(
                loop_count=loop_count,
                now_s=loop_start,
                desired=sample.desired,
                safe_command=safe.command,
                decision_reason=decision.reason,
                extra=extra,
            )
            if loop_start - last_log_s >= 1.0:
                last_log_s = loop_start
                logger.info(
                    "[VIS-RADAR] road={} err={} angle={} radar={} bypass={} "
                    "target_y={} desired={} planned={} safe={} sent={}",
                    getattr(sample.perception, "is_road_found", False),
                    _float_or_none(
                        getattr(sample.perception, "corrected_pixel_error", None)
                    ),
                    _float_or_none(
                        getattr(sample.perception, "centerline_angle", None)
                    ),
                    radar_fresh,
                    planner.state.value,
                    planner.target_y_cm,
                    sample.desired.as_fc_tuple(),
                    planned.as_fc_tuple(),
                    safe.command.as_fc_tuple(),
                    bool(actual_flight and decision.allowed),
                )
            loop_count += 1
            _sleep_to_rate(loop_start, period_s)
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("[VIS-RADAR] interrupted")
    finally:
        if fc is not None:
            try:
                if flight_owned:
                    land_and_wait_for_lock(fc, flight_config)
                elif fc.connected:
                    fc.stablize()
            finally:
                fc.close()
        if guidance_started:
            guidance.stop()
        if radars_started:
            radars.stop()
        recorder.close()
        if sink_id is not None:
            logger.remove(sink_id)
        logger.info(
            "[VIS-RADAR] stopped interrupted={} actual_flight={}",
            interrupted,
            actual_flight,
        )


def _radar_configs(upper_port: str, lower_port: str) -> list[RadarConfig]:
    return [
        RadarConfig("upper", 0, (0.0, 0.0), 0.0, port=upper_port),
        RadarConfig(
            "lower",
            1,
            (0.96, 0.15),
            0.0,
            port=lower_port,
            mount_mirror_y=True,
        ),
    ]


def _setup_logging(log_path: Path | None) -> int | None:
    if log_path is None:
        return None
    return logger.add(str(log_path), enqueue=True, encoding="utf-8")


def _sleep_to_rate(loop_start: float, period_s: float) -> None:
    remaining = period_s - (time.perf_counter() - loop_start)
    if remaining > 0.0:
        time.sleep(remaining)


def _float_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


if __name__ == "__main__":
    main()
