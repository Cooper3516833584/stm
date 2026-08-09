"""Independent real-vision + physical-radar tubular-obstacle experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import threading
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
from .flight_indicator import (
    FusionFlightIndicator,
    is_avoiding,
    is_unexpected,
    planner_state_name,
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
from .purple_target_mission import (
    PurpleTargetMissionConfig,
    PurpleTargetMissionController,
)
from .road_patrol_fleet import (
    RoadPatrolFleetStateProvider,
    RoadPatrolOperationState,
    cruise_operation_state,
    wait_for_ground_takeoff_authorization,
)
from .visual_guidance import FrozenVisualConfig, FrozenVisualGuidance
from fleet_bus import attach_air_fleet_node


DEFAULT_BYPASS_PLANNER = "static-route"
EXPERIMENTAL_PROFILE_NAME = "static-route-22cm-experiment"
EXPERIMENTAL_PROFILE_STATUS = "EXPERIMENTAL_UNVALIDATED"
FLEET_LANDING_REPORT_GRACE_S = 1.2


def build_experimental_visual_config(
    *,
    camera_index: int = 7,
    npu_model_path: str | None = None,
    target_enable: bool = True,
) -> FrozenVisualConfig:
    """Build the current 22 cm/s trial without changing the frozen v1 defaults."""
    base = FrozenVisualConfig(camera_index=camera_index)
    if npu_model_path is not None:
        base = replace(base, npu_model_path=npu_model_path)
    return replace(
        base,
        target_enable=bool(target_enable),
        max_vx_cm_s=22.0,
        max_vy_cm_s=12.0,
        max_yaw_rate_deg_s=18.0,
        min_forward_lookahead_px=28.0,
        max_forward_lookahead_px=88.0,
        lookahead_speed_gain_px_per_cm_s=1.4,
        max_latency_prediction_px=24.0,
        # The 22 cm/s horizon is roughly 60--82 px rather than 75--130 px;
        # use a three-point tangent window so signed-turn samples do not all
        # collapse to the same bounded tangent.
        tangent_window_points=3,
        tangent_kp_yaw=0.45,
        angle_deadband_deg=4.0,
        lateral_deadband_px=16.0,
        lateral_kp_cm_s_per_px=0.10,
        normal_max_vy_cm_s=12.0,
        # The production 45 cm/s profile uses 0.30 / 18 deg/s.  Scale the
        # feed-forward by the available yaw authority (18 / 55) instead of
        # importing the aggressive values unchanged.
        curvature_yaw_ff_kp=0.10,
        curvature_yaw_ff_max_deg_s=6.0,
        curvature_yaw_ff_deadband_deg=6.0,
        signed_turn_filter_tau_s=0.08,
        corner_lookahead_start_deg=30.0,
        corner_lookahead_full_deg=75.0,
        # Pixel geometry and path sampling are shared with the production
        # camera.  Keep 75 px so the signed-turn horizon still spans enough of
        # a sharp corner after adapting the tangent window for 22 cm/s.
        corner_min_lookahead_px=75.0,
        corner_severity_release_tau_s=0.25,
        # path_width_px is fixed, so the production dimensionless edge gates
        # transfer directly.  Velocity/yaw magnitudes are capped for 22 cm/s.
        edge_recovery_start_ratio=0.55,
        edge_recovery_full_ratio=0.90,
        edge_recovery_lateral_kp=0.16,
        edge_recovery_max_vy_cm_s=12.0,
        edge_yaw_start_ratio=0.75,
        edge_yaw_full_ratio=0.95,
        edge_yaw_max_deg_s=3.0,
        edge_speed_slow_start_ratio=0.90,
        edge_emergency_ratio=0.95,
        edge_emergency_vx_cap_cm_s=18.5,
        target_filter_tau_s=0.12,
        tangent_filter_tau_s=0.13,
        target_filter_max_rate_px_s=500.0,
        tangent_filter_max_rate_deg_s=100.0,
        max_planar_accel_cm_s2=36.0,
        # 120 cm/s^2 * (22 / 45) ~= 58.7; round to a conservative 60.
        max_planar_decel_cm_s2=60.0,
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
        clearance_run_s=1.5,
        normal_activation_radius_cm=100.0,
        clearance_reacquire_radius_cm=80.0,
        max_encounter_s=None,
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
    parser.add_argument(
        "--hc14-port",
        default=None,
        help=(
            "HC-14 CH340 serial port; default is auto-detection by USB "
            "VID:PID 1a86:7523"
        ),
    )
    parser.add_argument("--hc14-baudrate", type=int, default=None)
    parser.add_argument("--hc14-connect-timeout-s", type=float, default=5.0)
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
    parser.add_argument(
        "--disable-target-mission",
        action="store_true",
        help="Disable the purple-target payload mission in static-route mode",
    )
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
    if args.hc14_baudrate is not None and args.hc14_baudrate <= 0:
        raise ValueError("--hc14-baudrate must be greater than zero")
    if args.hc14_connect_timeout_s <= 0.0:
        raise ValueError("--hc14-connect-timeout-s must be greater than zero")
    if args.bypass_forward_transition_s < 0.0:
        raise ValueError("--bypass-forward-transition-s cannot be negative")
    if args.tuning_log_every_n <= 0:
        raise ValueError("--tuning-log-every-n must be a positive integer")
    if args.radar_snapshot_every_n <= 0:
        raise ValueError("--radar-snapshot-every-n must be a positive integer")
    if (
        args.bypass_planner == "static-route"
        and not args.disable_target_mission
        and args.right_half_radar_then_visual
    ):
        raise ValueError(
            "purple target mission requires radars for the whole mission; "
            "disable the target mission before using right-half radar retirement"
        )
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
    target_mission_enabled = bool(
        args.bypass_planner == "static-route"
        and not args.disable_target_mission
        and not args.circular_tube_bypass
    )
    visual_config = build_experimental_visual_config(
        camera_index=args.camera_index,
        npu_model_path=args.model_npu,
        target_enable=target_mission_enabled,
    )
    target_mission_config = PurpleTargetMissionConfig(
        target_vx_cm_s=visual_config.max_vx_cm_s * 0.60,
        yaw_kp=visual_config.tangent_kp_yaw,
        yaw_deadband_deg=visual_config.angle_deadband_deg,
        max_yaw_rate_deg_s=visual_config.max_yaw_rate_deg_s,
        max_yaw_accel_deg_s2=visual_config.max_yaw_accel_deg_s2,
        forward_bearing_limit_deg=visual_config.curvature_slowdown_start_deg,
        offset_filter_tau_s=visual_config.target_filter_tau_s,
        offset_filter_max_rate_px_s=visual_config.target_filter_max_rate_px_s,
        max_planar_accel_cm_s2=visual_config.max_planar_accel_cm_s2,
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
        target_config=target_mission_config if target_mission_enabled else None,
        visual_config=visual_config,
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
                target_enable=visual_config.target_enable,
                target_max_dimension=visual_config.target_max_dimension,
                target_hue_min=visual_config.target_hue_min,
                target_hue_max=visual_config.target_hue_max,
                target_saturation_min=visual_config.target_saturation_min,
                target_value_min=visual_config.target_value_min,
                target_min_area_ratio=visual_config.target_min_area_ratio,
                target_max_rate_hz=visual_config.target_max_rate_hz,
                target_stale_timeout_s=visual_config.target_stale_timeout_s,
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
                "target_mission_enabled": target_mission_enabled,
                "target_mission_config": (
                    target_mission_config.__dict__ if target_mission_enabled else None
                ),
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
    target_mission = (
        PurpleTargetMissionController(target_mission_config)
        if target_mission_enabled
        else None
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
    indicator = None
    fleet_node = None
    fleet_stop_event = threading.Event()
    fleet_state = RoadPatrolFleetStateProvider()
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
            fleet_node = attach_air_fleet_node(
                None,
                None,
                fleet_stop_event,
                readonly=True,
                allow_start_mission=True,
                state_provider=fleet_state,
                hc14_port=args.hc14_port,
                hc14_baudrate=args.hc14_baudrate,
                connect_timeout_s=args.hc14_connect_timeout_s,
            )
            # The radio link is a precondition for ground authorization.  Do
            # not even open the flight-controller connection until HC-14 is
            # present, uniquely identified, and successfully opened.
            fc = connect_fc(FCConnectConfig(port=args.fc_port, mode=2, timeout_s=10.0))
            flight_owned = True
            fleet_state.bind_fc(fc)
            indicator = FusionFlightIndicator(fc)
            logger.warning(
                "[VIS-RADAR] initialization complete; waiting for ground "
                "prepare/countdown/start sequence"
            )
            fleet_state.set_operation_state(RoadPatrolOperationState.TAKEOFF)
            authorized = wait_for_ground_takeoff_authorization(
                fleet_node=fleet_node,
                indicator=indicator,
                stop_event=fleet_stop_event,
            )
            if authorized:
                auto_takeoff(fc, flight_config)
                fleet_state.set_operation_state(
                    RoadPatrolOperationState.LINE_FOLLOWING
                )
            else:
                interrupted = True
                fleet_state.set_operation_state(
                    RoadPatrolOperationState.LANDING
                )
                indicator.set_red()
                logger.warning(
                    "[VIS-RADAR] takeoff cancelled before ground authorization"
                )
        else:
            logger.warning(
                "[VIS-RADAR] dry run: real camera/radars active, no FC connection"
            )

        start_s = time.perf_counter()
        previous_loop_s = start_s
        previous_final_command = Command.zero("initial")
        last_log_s = 0.0
        loop_count = 0
        while (
            time.perf_counter() - start_s < args.duration_s
            and not fleet_stop_event.is_set()
        ):
            loop_start = time.perf_counter()
            dt_s = max(0.0, min(0.5, loop_start - previous_loop_s))
            previous_loop_s = loop_start
            sample = guidance.sample(loop_start)
            planner_elapsed_us = 0.0
            previous_planner_state = planner.state
            previous_target_state = (
                target_mission.state if target_mission is not None else None
            )
            flight_status = flight_status_from_fc(fc)
            radar_retired = bool(
                right_half_handoff is not None
                and right_half_handoff.radar_disabled
            )
            if radar_retired:
                radar_age_s = None
                radar_fresh = False
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

            obstacle_conflict = False
            conflict_probe = getattr(planner, "has_obstacle_conflict", None)
            if not radar_retired and radar_fresh and callable(conflict_probe):
                obstacle_conflict = bool(conflict_probe(radar_field))

            target_decision = None
            selected_desired = sample.desired
            if target_mission is not None:
                target_decision = target_mission.update(
                    now_s=loop_start,
                    road_desired=sample.desired,
                    target=sample.target,
                    target_stale=sample.target_stale,
                    planner_state=planner_state_name(planner),
                    radar_fresh=radar_fresh,
                    altitude_cm=flight_status.alt_cm,
                    obstacle_conflict=obstacle_conflict,
                )
                selected_desired = target_decision.desired

            if radar_retired:
                planned = selected_desired
            else:
                planner_started_ns = time.perf_counter_ns()
                planner_options = {}
                if target_decision is not None and target_decision.mission_owns_command:
                    planner_options = {
                        "guidance_usable": target_decision.guidance_usable,
                        "preserve_guidance_yaw": target_decision.preserve_guidance_yaw,
                    }
                planned = planner.update(
                    desired=selected_desired,
                    perception=sample.perception,
                    radar_field=radar_field,
                    now_s=loop_start,
                    **planner_options,
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
                    planned = selected_desired
            active_arbiter = visual_only_arbiter if radar_retired else arbiter
            health = flight_health_from_sources(
                fc=fc,
                multi_radar=None if radar_retired else radars,
                radar_timeout_s=args.radar_timeout_s,
                camera_ok=sample.camera_ok,
            )
            safe = active_arbiter.filter(
                planned,
                flight=flight_status,
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
            payload_release_event = False
            if target_mission is not None and target_mission.release_is_authorized(
                planner_state=planner_state_name(planner),
                radar_fresh=radar_fresh,
                safety_state=safe.state,
                command_allowed=decision.allowed,
                final_command=decision.command,
                obstacle_clear=not bool(
                    planner.diagnostics().get("observation_valid", False)
                ),
            ):
                if actual_flight:
                    if fc is None:
                        raise RuntimeError("payload release requested without FC")
                    fc.set_digital_output(0, False)
                    logger.warning("[PURPLE-TARGET] payload released: digital output 0=False")
                else:
                    logger.warning("[PURPLE-TARGET] dry-run payload release simulated")
                target_mission.mark_payload_released(loop_start)
                payload_release_event = True
            if (
                target_mission is not None
                and target_mission.consume_disable_target_request()
            ):
                guidance.disable_target()
                logger.info("[PURPLE-TARGET] target detector stopped; road NPU remains active")
            if indicator is not None:
                state_name = planner_state_name(planner)
                avoiding = is_avoiding(
                    planner_state=state_name,
                    safety_state=safe.state,
                )
                target_observation_ok = bool(
                    sample.target is not None
                    and getattr(sample.target, "found", False)
                    and not sample.target_stale
                )
                mission_guidance_required = bool(
                    target_mission is not None
                    and target_mission.mission_active
                    and target_mission.target_required
                )
                mission_nonvisual_phase = bool(
                    target_mission is not None
                    and target_mission.mission_active
                    and not target_mission.target_required
                )
                indicator.update(
                    now_s=loop_start,
                    avoiding=avoiding,
                    unexpected=is_unexpected(
                        planner_state=state_name,
                        safety_state=safe.state,
                        decision_allowed=decision.allowed,
                        radar_required=not radar_retired,
                        radar_fresh=radar_fresh,
                        camera_ok=sample.camera_ok,
                        perception_stale=(
                            not target_observation_ok
                            if mission_guidance_required
                            else (False if mission_nonvisual_phase else sample.perception_stale)
                        ),
                        road_found=(
                            target_observation_ok
                            if mission_guidance_required
                            else (
                                True
                                if mission_nonvisual_phase
                                else bool(getattr(sample.perception, "is_road_found", False))
                            )
                        ),
                    ),
                    # Mission-active begins only after the configured target
                    # confirmation frames and remains true through completion
                    # or abort recovery.  The indicator policy, rather than
                    # the mission controller, owns the FC LED command.
                    target_active=bool(
                        target_mission is not None
                        and target_mission.mission_active
                    ),
                )
                fleet_state.set_operation_state(
                    cruise_operation_state(
                        target_mission=target_mission,
                        avoiding=avoiding,
                    )
                )
            command_applied = bool(actual_flight and decision.allowed)
            report_applied = getattr(planner, "report_applied_command", None)
            if callable(report_applied):
                report_applied(decision.command, dt_s, command_applied)
            planner_diagnostics = planner.diagnostics()
            planner_diagnostics["visual_vy_cm_s"] = sample.desired.vy_cm_s
            planner_diagnostics["selected_guidance_vy_cm_s"] = selected_desired.vy_cm_s
            planner_diagnostics["planner_elapsed_us"] = planner_elapsed_us
            target_mission_diagnostics = (
                target_mission.diagnostics(loop_start)
                if target_mission is not None
                else None
            )
            if planner.state != previous_planner_state:
                logger.info(
                    "[VIS-RADAR][EVENT] state_transition={} -> {} reason={} encounter={} side={}",
                    previous_planner_state.value,
                    planner.state.value,
                    planner_diagnostics.get("transition_reason"),
                    planner_diagnostics.get("encounter_id"),
                    planner_diagnostics.get("active_bypass_side"),
                )
            if (
                target_mission is not None
                and target_mission.state != previous_target_state
            ):
                logger.info(
                    "[PURPLE-TARGET][EVENT] state_transition={} -> {} reason={}",
                    getattr(previous_target_state, "value", previous_target_state),
                    target_mission.state.value,
                    target_mission.transition_reason,
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
                "purple_target": {
                    "enabled_for_mission": target_mission is not None,
                    "found": bool(
                        sample.target is not None
                        and getattr(sample.target, "found", False)
                        and not sample.target_stale
                    ),
                    "offset_x_px": _float_or_none(
                        getattr(sample.target, "offset_x_px", None)
                    ),
                    "offset_y_px": _float_or_none(
                        getattr(sample.target, "offset_y_px", None)
                    ),
                    "capture_time_s": _float_or_none(
                        getattr(sample.target, "capture_time_s", None)
                    ),
                    "age_s": _float_or_none(sample.target_age_s),
                    "stale": sample.target_stale,
                    "error": getattr(sample.target, "error", None),
                    "mission": target_mission_diagnostics,
                    "payload_release_event": payload_release_event,
                },
                "tube_obstacle_bypass": planner_diagnostics,
                "commands": {
                    "road_desired": sample.desired.as_fc_tuple(),
                    "desired": selected_desired.as_fc_tuple(),
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
                    desired=selected_desired,
                    safe_command=decision.command,
                    decision_reason=decision.reason,
                    extra=extra,
                )
            if loop_count % args.tuning_log_every_n == 0:
                recorder.record_command(
                    loop_count=loop_count,
                    now_s=loop_start,
                    desired=selected_desired,
                    safe_command=decision.command,
                    decision_reason=decision.reason,
                    extra=extra,
                )
            if loop_start - last_log_s >= 1.0:
                last_log_s = loop_start
                logger.info(
                    "[VIS-RADAR] road={} err={} angle={} radar={} retired={} bypass={} "
                    "target_y={} mission={} desired={} planned={} safe={} sent={}",
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
                    (
                        target_mission.state.value
                        if target_mission is not None
                        else "disabled"
                    ),
                    selected_desired.as_fc_tuple(),
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
        if flight_owned:
            fleet_state.set_operation_state(RoadPatrolOperationState.LANDING)
        if indicator is not None:
            indicator.set_red()
        logger.warning("[VIS-RADAR] interrupted")
    except BaseException:
        if flight_owned:
            fleet_state.set_operation_state(RoadPatrolOperationState.LANDING)
        else:
            fleet_state.set_operation_state(RoadPatrolOperationState.FAULT)
        if indicator is not None:
            indicator.set_red()
        raise
    finally:
        if fc is not None:
            try:
                if flight_owned:
                    fleet_state.set_operation_state(
                        RoadPatrolOperationState.LANDING
                    )
                    land_and_wait_for_lock(fc, flight_config)
                elif fc.connected:
                    fc.stablize()
            finally:
                fc.close()
        if fleet_node is not None:
            if flight_owned:
                time.sleep(FLEET_LANDING_REPORT_GRACE_S)
            fleet_node.close()
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
