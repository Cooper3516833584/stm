"""Independent real-vision + physical-radar tubular-obstacle experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import time

import numpy as np
from loguru import logger
from road_perception import CameraOffsetCompensationConfig

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
from .circular_tube_bypass import (
    CircularTubeBypassConfig,
    CircularTubeBypassPlanner,
)
from .radar_bypass import ObstacleBypassConfig, ObstacleBypassPlanner
from .right_half_handoff import RightHalfRadarHandoff
from .smooth_sidestep import SmoothSidestepPlanner
from .static_route_bypass import (
    StaticRouteBypassConfig,
    StaticRouteBypassPlanner,
)
from FlightController.Runtime import (
    LoopRateMonitor,
    ProcessRadarClient,
    ProcessRuntime,
    ProcessRuntimeConfig,
)
from .parameter_registry import build_parameter_registry
from .visual_guidance import FrozenVisualConfig, FrozenVisualGuidance


DEFAULT_BYPASS_PLANNER = "static-route"
EXPERIMENTAL_PROFILE_NAME = "static-route-22cm-experiment"
EXPERIMENTAL_PROFILE_STATUS = "EXPERIMENTAL_UNVALIDATED"


def build_experimental_visual_config(
    *, camera_index: int = 7, npu_model_path: str | None = None
) -> FrozenVisualConfig:
    """Build the current 22 cm/s trial without changing the frozen v1 defaults."""
    base = FrozenVisualConfig(camera_index=camera_index)
    if npu_model_path is not None:
        base = replace(base, npu_model_path=npu_model_path)
    return replace(
        base,
        max_vx_cm_s=22.0,
        max_vy_cm_s=12.0,
        max_yaw_rate_deg_s=18.0,
        min_forward_lookahead_px=28.0,
        max_forward_lookahead_px=88.0,
        lookahead_speed_gain_px_per_cm_s=1.4,
        max_latency_prediction_px=24.0,
        tangent_kp_yaw=0.45,
        angle_deadband_deg=4.0,
        lateral_deadband_px=16.0,
        target_filter_tau_s=0.12,
        tangent_filter_tau_s=0.13,
        target_filter_max_rate_px_s=500.0,
        tangent_filter_max_rate_deg_s=100.0,
        max_planar_accel_cm_s2=36.0,
        max_yaw_accel_deg_s2=50.0,
        degraded_speed_scale=0.90,
        curvature_slowdown_start_deg=18.0,
        curvature_full_slowdown_deg=52.0,
        min_curve_speed_cm_s=15.0,
    )


def build_experimental_static_route_config(
    *, tube_radius_cm: float = 15.0, visual_max_vx_cm_s: float = 22.0
) -> StaticRouteBypassConfig:
    """Apply only the unfrozen avoidance changes needed by the 22 cm/s trial."""
    return StaticRouteBypassConfig(
        tube_radius_cm=tube_radius_cm,
        visual_max_vx_cm_s=visual_max_vx_cm_s,
        max_outward_vy_cm_s=12.0,
        lateral_kp_s=0.25,
        ramp_in_s=0.7,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Isolated real-vision/physical-radar tubular-obstacle test"
    )
    parser.add_argument("--camera-index", type=int, default=7)
    parser.add_argument("--model-npu", default=FrozenVisualConfig().npu_model_path)
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--fc-port", default=None)
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument(
        "--runtime-mode",
        choices=("process", "threaded"),
        default="process",
        help="Isolate vision/radar in spawned processes (default) or use the legacy threads",
    )
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--radar-timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--bypass-planner",
        choices=("legacy", "smooth-sidestep", "static-route"),
        default=DEFAULT_BYPASS_PLANNER,
        help="Select static-route (default) or an unchanged earlier planner",
    )
    parser.add_argument(
        "--bypass-forward-transition-s",
        type=float,
        default=2.0,
        help="Legacy planner forward-priority radar-to-vision handoff duration",
    )
    parser.add_argument(
        "--right-half-radar-then-visual",
        action="store_true",
        help=(
            "Use only clockwise 0..180 degree radar points, then stop radar "
            "after forward recovery has remained normal for 5 seconds"
        ),
    )
    parser.add_argument(
        "--circular-tube-bypass",
        action="store_true",
        help="Follow a low-complexity inflated circle around the detected tube",
    )
    parser.add_argument("--tube-radius-cm", type=float, default=15.0)
    parser.add_argument("--tube-safety-radius-cm", type=float, default=75.0)
    parser.add_argument("--record-dir", default="/data/stm_records")
    parser.add_argument("--tuning-log-every-n", type=int, default=2)
    parser.add_argument("--radar-snapshot-every-n", type=int, default=5)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--enable-flight", action="store_true")
    parser.add_argument("--auto-takeoff", action="store_true")
    parser.add_argument(
        "--confirm-visual-radar-flight-test",
        action="store_true",
        help="Acknowledge real unlock/takeoff using live camera and physical radars",
    )
    parser.add_argument("--takeoff-height-cm", type=int, default=100)
    args = parser.parse_args(raw_argv)
    if (
        args.right_half_radar_then_visual or args.circular_tube_bypass
    ) and "--bypass-planner" not in raw_argv:
        # Preserve the historical standalone flags after static-route becomes
        # the default planner.
        args.bypass_planner = "legacy"
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.model_npu):
        raise FileNotFoundError(f"required NPU model missing: {args.model_npu}")
    if args.loop_hz <= 0.0:
        raise ValueError("--loop-hz must be greater than zero")
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be greater than zero")
    if args.bypass_forward_transition_s < 0.0:
        raise ValueError("--bypass-forward-transition-s cannot be negative")
    if args.tuning_log_every_n <= 0:
        raise ValueError("--tuning-log-every-n must be a positive integer")
    if args.radar_snapshot_every_n <= 0:
        raise ValueError("--radar-snapshot-every-n must be a positive integer")
    if args.right_half_radar_then_visual:
        if args.bypass_planner != "legacy":
            raise ValueError("--right-half-radar-then-visual requires legacy planner")
        if args.bypass_forward_transition_s <= 0.0:
            raise ValueError(
                "--right-half-radar-then-visual requires a positive "
                "--bypass-forward-transition-s"
            )
    if args.circular_tube_bypass:
        if args.bypass_planner != "legacy":
            raise ValueError("--circular-tube-bypass requires legacy planner selection")
        if args.right_half_radar_then_visual:
            raise ValueError(
                "--circular-tube-bypass and --right-half-radar-then-visual "
                "are independent experiments and cannot be combined"
            )
    if args.tube_radius_cm <= 0.0:
        raise ValueError("--tube-radius-cm must be greater than zero")
    if args.tube_safety_radius_cm <= 0.0:
        raise ValueError("--tube-safety-radius-cm must be greater than zero")
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
    visual_config = build_experimental_visual_config(
        camera_index=args.camera_index,
        npu_model_path=args.model_npu,
    )
    flight_config = FlightRuntimeConfig(
        takeoff_height_cm=args.takeoff_height_cm,
    )
    static_route_config = build_experimental_static_route_config(
        tube_radius_cm=args.tube_radius_cm,
        visual_max_vx_cm_s=visual_config.max_vx_cm_s,
    )
    parameter_registry = build_parameter_registry(
        static_route_config,
        radar_timeout_s=args.radar_timeout_s,
        tuning_log_every_n=args.tuning_log_every_n,
        radar_snapshot_every_n=args.radar_snapshot_every_n,
    )
    process_runtime = None
    if args.runtime_mode == "process":
        process_runtime = ProcessRuntime(
            ProcessRuntimeConfig(
                camera_index=visual_config.camera_index,
                camera_width=visual_config.camera_width,
                camera_height=visual_config.camera_height,
                camera_fps=visual_config.camera_fps,
                npu_model_path=visual_config.npu_model_path,
                inference_backend="npu",
                postprocess_mode=visual_config.postprocess_mode,
                instance_selection=visual_config.instance_selection,
                flight_height_m=visual_config.flight_height_m,
                offset_comp_config=CameraOffsetCompensationConfig(enabled=False),
                upper_port=args.upper_port,
                lower_port=args.lower_port,
                radar_timeout_s=args.radar_timeout_s,
            )
        )
    if args.circular_tube_bypass:
        session_mode = "isolated_visual_radar_circular_tube"
    elif args.bypass_planner == "smooth-sidestep":
        session_mode = "isolated_visual_radar_smooth_sidestep"
    elif args.bypass_planner == "static-route":
        session_mode = "isolated_visual_radar_static_route"
    else:
        session_mode = "isolated_visual_radar_tube_obstacle"
    recorder = SessionRecorder(
        SessionRecorderConfig(
            root_dir=args.record_dir,
            enabled=not args.no_record,
            mode=session_mode,
            frame_every_n=10,
            radar_every_n=args.radar_snapshot_every_n,
            frame_rate_hz=1.0,
            radar_rate_hz=10.0 / max(1, args.radar_snapshot_every_n),
            command_rate_hz=10.0 / max(1, args.tuning_log_every_n),
            video_enabled=True,
            video_every_n=2,
            video_fps=5.0,
            frame_ring_descriptor=(
                process_runtime.frame_ring_descriptor if process_runtime is not None else None
            ),
            metadata={
                "argv": list(sys.argv),
                "visual_config": vars(visual_config),
                "physical_obstacle": (
                    "one isolated static tube on the requested route"
                ),
                "radar_points": "physical only; no synthetic injection",
                "bypass_planner": args.bypass_planner,
                "avoidance_profile": (
                    EXPERIMENTAL_PROFILE_NAME
                    if args.bypass_planner == DEFAULT_BYPASS_PLANNER
                    else None
                ),
                "avoidance_profile_status": (
                    EXPERIMENTAL_PROFILE_STATUS
                    if args.bypass_planner == DEFAULT_BYPASS_PLANNER
                    else None
                ),
                "right_half_radar_then_visual": args.right_half_radar_then_visual,
                "circular_tube_bypass": args.circular_tube_bypass,
                "tube_radius_cm": args.tube_radius_cm,
                "tube_safety_radius_cm": args.tube_safety_radius_cm,
                "parameter_registry": parameter_registry,
            },
        )
    )
    if actual_flight and not recorder.healthy:
        if process_runtime is not None:
            process_runtime.stop()
        raise RuntimeError("flight test refused because session recording is unavailable")
    sink_id = _setup_logging(recorder.log_sink if recorder.enabled else None)

    guidance = FrozenVisualGuidance(visual_config, process_runtime=process_runtime)
    radars = (
        ProcessRadarClient(process_runtime, max_age_s=args.radar_timeout_s)
        if process_runtime is not None
        else MultiRadar(_radar_configs(args.upper_port, args.lower_port))
    )
    radar_field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=300.0,
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=75.0,
            side_corridor_x_half_cm=25.0,
        )
    )
    if args.circular_tube_bypass:
        planner = CircularTubeBypassPlanner(
            CircularTubeBypassConfig(
                tube_radius_cm=args.tube_radius_cm,
                safety_radius_cm=args.tube_safety_radius_cm,
            )
        )
    elif args.bypass_planner == "smooth-sidestep":
        planner = SmoothSidestepPlanner()
    elif args.bypass_planner == "static-route":
        planner = StaticRouteBypassPlanner(static_route_config)
    else:
        planner = ObstacleBypassPlanner(
            ObstacleBypassConfig(
                forward_recovery_s=args.bypass_forward_transition_s,
            )
        )
    right_half_handoff = (
        RightHalfRadarHandoff() if args.right_half_radar_then_visual else None
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
            side_stop_distance_cm=45.0,
        )
    )
    visual_only_arbiter = SafetyArbiter(
        SafetyConfig(
            require_fc=actual_flight,
            require_hold_pos_mode=actual_flight,
            require_unlocked=actual_flight,
            require_radar=False,
            radar_timeout_s=args.radar_timeout_s,
            max_vx_cm_s=visual_config.max_vx_cm_s,
            max_vy_cm_s=visual_config.max_vy_cm_s,
            max_yaw_rate_deg_s=visual_config.max_yaw_rate_deg_s,
        )
    )

    fc = None
    flight_owned = False
    interrupted = False
    guidance_started = False
    radars_started = False
    period_s = 1.0 / args.loop_hz
    loop_monitor = LoopRateMonitor(args.loop_hz)
    try:
        if process_runtime is not None:
            process_runtime.start()
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
        previous_loop_s = start_s
        previous_final_command = Command.zero("initial")
        last_log_s = 0.0
        loop_count = 0
        while time.perf_counter() - start_s < args.duration_s:
            loop_start = time.perf_counter()
            dt_s = max(0.0, min(0.5, loop_start - previous_loop_s))
            previous_loop_s = loop_start
            sample = guidance.sample(loop_start)
            planner_elapsed_us = 0.0
            previous_planner_state = planner.state
            radar_retired = bool(
                right_half_handoff is not None
                and right_half_handoff.radar_disabled
            )
            if radar_retired:
                radar_age_s = None
                radar_fresh = False
                planned = sample.desired
            else:
                points = radars.get_obstacle_points_body_cm(max_distance_cm=300.0)
                if right_half_handoff is not None:
                    points = right_half_handoff.filter_right_half_plane(points)
                radar_field.update(points, loop_start)
                radar_age_s = multi_radar_age_s(radars)
                radar_fresh = bool(
                    radars.connected
                    and radars.is_fresh(max_age_s=args.radar_timeout_s)
                )
                planner_started_ns = time.perf_counter_ns()
                planned = planner.update(
                    desired=sample.desired,
                    perception=sample.perception,
                    radar_field=radar_field,
                    now_s=loop_start,
                )
                planner_elapsed_us = (
                    time.perf_counter_ns() - planner_started_ns
                ) / 1000.0
                if (
                    right_half_handoff is not None
                    and right_half_handoff.observe(
                        previous_planner_state,
                        planner.state,
                        loop_start,
                        bypass_pending=bool(
                            planner.diagnostics().get("intrusion_count", 0)
                        ),
                    )
                ):
                    logger.warning(
                        "[VIS-RADAR] right-half radar phase complete; "
                        "stopping radars and continuing with visual trajectory only"
                    )
                    radars.stop()
                    radars_started = False
                    radar_field.update(np.empty((0, 2), dtype=float), loop_start)
                    radar_age_s = None
                    radar_fresh = False
                    radar_retired = True
                    planned = sample.desired
            active_arbiter = visual_only_arbiter if radar_retired else arbiter
            health = flight_health_from_sources(
                fc=fc,
                multi_radar=None if radar_retired else radars,
                radar_timeout_s=args.radar_timeout_s,
                camera_ok=sample.camera_ok,
            )
            safe = active_arbiter.filter(
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
                active_arbiter,
                health,
                dry_run=not actual_flight,
            )
            command_applied = bool(actual_flight and decision.allowed)
            report_applied = getattr(planner, "report_applied_command", None)
            if callable(report_applied):
                report_applied(decision.command, dt_s, command_applied)
            planner_diagnostics = planner.diagnostics()
            planner_diagnostics["visual_vy_cm_s"] = sample.desired.vy_cm_s
            planner_diagnostics["planner_elapsed_us"] = planner_elapsed_us
            if planner.state != previous_planner_state:
                logger.info(
                    "[VIS-RADAR][EVENT] state_transition={} -> {} reason={} encounter={} side={}",
                    previous_planner_state.value,
                    planner.state.value,
                    planner_diagnostics.get("transition_reason"),
                    planner_diagnostics.get("encounter_id"),
                    planner_diagnostics.get("active_bypass_side"),
                )
            final_delta = {
                "vx_cm_s": decision.command.vx_cm_s - previous_final_command.vx_cm_s,
                "vy_cm_s": decision.command.vy_cm_s - previous_final_command.vy_cm_s,
                "yaw_rate_deg_s": (
                    decision.command.yaw_rate_deg_s
                    - previous_final_command.yaw_rate_deg_s
                ),
            }
            previous_final_command = decision.command
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
                "tube_obstacle_bypass": planner_diagnostics,
                "commands": {
                    "desired": sample.desired.as_fc_tuple(),
                    "planned": planned.as_fc_tuple(),
                    "safe": safe.command.as_fc_tuple(),
                    "final": decision.command.as_fc_tuple(),
                    "safety_state": safe.state,
                    "safety_reasons": safe.reasons,
                    "nearest_forward_obstacle_cm": safe.nearest_forward_obstacle_cm,
                    "left_side_clearance_cm": safe.left_side_clearance_cm,
                    "right_side_clearance_cm": safe.right_side_clearance_cm,
                    "safety_override": bool(
                        safe.command != planned or decision.command != safe.command
                    ),
                    "final_delta": final_delta,
                },
                "right_half_handoff": (
                    right_half_handoff.diagnostics()
                    if right_half_handoff is not None
                    else None
                ),
                "sent": command_applied,
            }
            if recorder.frame_due(loop_count, loop_start):
                recorder.record_frame(
                    loop_count=loop_count,
                    now_s=loop_start,
                    frame=sample.frame,
                    label="road",
                    source_time_s=sample.frame_time_s,
                    extra=extra,
                )
            if not radar_retired:
                recorder.record_radar(
                    loop_count=loop_count,
                    now_s=loop_start,
                    radar_field=radar_field,
                    multi_radar=radars,
                    radar_age_s=radar_age_s,
                    radar_connected=radar_fresh,
                    desired=sample.desired,
                    safe_command=decision.command,
                    decision_reason=decision.reason,
                    extra=extra,
                )
            if loop_count % args.tuning_log_every_n == 0:
                recorder.record_command(
                    loop_count=loop_count,
                    now_s=loop_start,
                    desired=sample.desired,
                    safe_command=decision.command,
                    decision_reason=decision.reason,
                    extra=extra,
                )
            if loop_start - last_log_s >= 1.0:
                last_log_s = loop_start
                logger.info(
                    "[VIS-RADAR] road={} err={} angle={} radar={} retired={} bypass={} "
                    "target_y={} desired={} planned={} safe={} sent={}",
                    getattr(sample.perception, "is_road_found", False),
                    _float_or_none(
                        getattr(sample.perception, "corrected_pixel_error", None)
                    ),
                    _float_or_none(
                        getattr(sample.perception, "centerline_angle", None)
                    ),
                    radar_fresh,
                    radar_retired,
                    planner.state.value,
                    planner.target_y_cm,
                    sample.desired.as_fc_tuple(),
                    planned.as_fc_tuple(),
                    safe.command.as_fc_tuple(),
                    bool(actual_flight and decision.allowed),
                )
                timing = loop_monitor.snapshot()
                runtime_health = process_runtime.health(loop_start) if process_runtime else None
                logger.info(
                    "[RUNTIME] target_hz={:.1f} actual_hz={:.2f} work_p95_ms={:.2f} "
                    "work_p99_ms={:.2f} jitter_p99_ms={:.2f} deadline_miss={} "
                    "vision_age_ms={} radar_age_ms={} vision_drop={} radar_drop={} vq={} rq={}",
                    timing.target_hz,
                    timing.achieved_hz,
                    timing.work_p95_ms,
                    timing.work_p99_ms,
                    timing.jitter_p99_ms,
                    timing.deadline_misses,
                    _float_or_none(
                        runtime_health.vision_age_s * 1000.0
                        if runtime_health else sample.perception_age_s * 1000.0
                    ),
                    _float_or_none(
                        runtime_health.radar_age_s * 1000.0
                        if runtime_health else (radar_age_s or 0.0) * 1000.0
                    ),
                    runtime_health.vision_publish_drops if runtime_health else None,
                    runtime_health.radar_publish_drops if runtime_health else None,
                    runtime_health.vision_queue_depth if runtime_health else None,
                    runtime_health.radar_queue_depth if runtime_health else None,
                )
            loop_monitor.record(loop_start, time.perf_counter())
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
        logger.info(
            "[VIS-RADAR] stopped interrupted={} actual_flight={}",
            interrupted,
            actual_flight,
        )
        if process_runtime is not None:
            process_runtime.stop_workers()
        if sink_id is not None:
            logger.remove(sink_id)
        recorder.close()
        if process_runtime is not None:
            process_runtime.stop()


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


def _setup_logging(log_target) -> int | None:
    if log_target is None:
        return None
    if callable(log_target):
        return logger.add(log_target, enqueue=False)
    return logger.add(str(log_target), enqueue=True, encoding="utf-8")


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
