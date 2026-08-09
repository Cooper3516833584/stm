"""Independent visual-road + frozen radar-contour bypass entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import sys
import threading
import time

from loguru import logger
import numpy as np
from road_perception import CameraOffsetCompensationConfig

from FlightController.Components import MultiRadar, RadarConfig
from FlightController.Components.FCConnector import FCConnectConfig, connect_fc
from FlightController.Runtime import (
    LoopRateMonitor,
    ProcessRadarClient,
    ProcessRuntime,
    ProcessRuntimeConfig,
)
from FlightController.Solutions.Safety import (
    Command,
    RadarFieldConfig,
    RadarObstacleField,
    multi_radar_age_s,
)
from FlightController.Solutions.SessionRecorder import (
    SessionRecorder,
    SessionRecorderConfig,
)
from fleet_bus import attach_air_fleet_node

from experiments.visual_radar_bypass.flight_indicator import FusionFlightIndicator
from experiments.visual_radar_bypass.flight_runtime import (
    FlightRuntimeConfig,
    auto_takeoff,
    land_and_wait_for_lock,
    wait_for_radars,
    wait_for_visual_road,
)
from experiments.visual_radar_bypass.purple_target_mission import (
    PurpleTargetMissionConfig,
    PurpleTargetMissionController,
)
from experiments.visual_radar_bypass.road_patrol_fleet import (
    RoadPatrolFleetStateProvider,
    RoadPatrolOperationState,
    cruise_operation_state,
    wait_for_ground_takeoff_authorization,
)
from experiments.visual_radar_bypass.visual_guidance import FrozenVisualConfig, FrozenVisualGuidance

from .planner import (
    ContourBypassConfig,
    ContourBypassState,
    ContourTrajectoryBypassPlanner,
)


FLEET_LANDING_REPORT_GRACE_S = 1.2


def build_experimental_visual_config(
    *,
    camera_index: int = 7,
    npu_model_path: str | None = None,
    target_enable: bool = True,
) -> FrozenVisualConfig:
    """Use the existing frozen visual stack with the current 22 cm/s profile."""
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
        tangent_window_points=3,
        tangent_kp_yaw=0.45,
        angle_deadband_deg=4.0,
        lateral_deadband_px=16.0,
        lateral_kp_cm_s_per_px=0.10,
        normal_max_vy_cm_s=12.0,
        curvature_yaw_ff_kp=0.10,
        curvature_yaw_ff_max_deg_s=6.0,
        curvature_yaw_ff_deadband_deg=6.0,
        signed_turn_filter_tau_s=0.08,
        corner_lookahead_start_deg=30.0,
        corner_lookahead_full_deg=75.0,
        corner_min_lookahead_px=75.0,
        corner_severity_release_tau_s=0.25,
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
        max_planar_decel_cm_s2=60.0,
        max_yaw_accel_deg_s2=50.0,
        degraded_speed_scale=0.90,
        curvature_slowdown_start_deg=18.0,
        curvature_full_slowdown_deg=52.0,
        min_curve_speed_cm_s=15.0,
    )


@dataclass(frozen=True)
class DirectCommandLimits:
    max_abs_vx_cm_s: float = 22.0
    max_abs_vy_cm_s: float = 12.0
    max_abs_vz_cm_s: float = 20.0
    max_abs_yaw_rate_deg_s: float = 18.0


def clamp_task_command(
    command: Command,
    limits: DirectCommandLimits | None = None,
) -> Command:
    """Apply finite validation and this task's actuator-range limits only."""
    cfg = limits or DirectCommandLimits()
    values = (
        command.vx_cm_s,
        command.vy_cm_s,
        command.vz_cm_s,
        command.yaw_rate_deg_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("direct flight command contains a non-finite value")
    clipped = (
        float(np.clip(values[0], -cfg.max_abs_vx_cm_s, cfg.max_abs_vx_cm_s)),
        float(np.clip(values[1], -cfg.max_abs_vy_cm_s, cfg.max_abs_vy_cm_s)),
        float(np.clip(values[2], -cfg.max_abs_vz_cm_s, cfg.max_abs_vz_cm_s)),
        float(
            np.clip(
                values[3],
                -cfg.max_abs_yaw_rate_deg_s,
                cfg.max_abs_yaw_rate_deg_s,
            )
        ),
    )
    reason = command.reason
    if clipped != tuple(float(value) for value in values):
        reason = f"{reason};task_range_clamped" if reason else "task_range_clamped"
    return Command(*clipped, reason=reason)


def send_direct_command(
    fc,
    command: Command,
    dry_run: bool,
    limits: DirectCommandLimits | None = None,
) -> Command:
    """Round and directly send once; return the exact command integrated later."""
    bounded = clamp_task_command(command, limits)
    applied = Command(
        float(round(bounded.vx_cm_s)),
        float(round(bounded.vy_cm_s)),
        float(round(bounded.vz_cm_s)),
        float(round(bounded.yaw_rate_deg_s)),
        bounded.reason,
    )
    if not dry_run:
        if fc is None:
            raise RuntimeError("real direct send requested without an FC connection")
        fc.send_realtime_control_data(
            round(bounded.vx_cm_s),
            round(bounded.vy_cm_s),
            round(bounded.vz_cm_s),
            round(bounded.yaw_rate_deg_s),
        )
    return applied


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual road patrol with encounter-frozen inflated-contour bypass"
    )
    parser.add_argument("--camera-index", type=int, default=7)
    parser.add_argument("--model-npu", default=FrozenVisualConfig().npu_model_path)
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--fc-port", default=None)
    parser.add_argument(
        "--hc14-port",
        default=None,
        help="HC-14 CH340 port; default auto-detects USB VID:PID 1a86:7523",
    )
    parser.add_argument("--hc14-baudrate", type=int, default=None)
    parser.add_argument("--hc14-connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--runtime-mode", choices=("process", "threaded"), default="process")
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=100.0)
    parser.add_argument("--radar-timeout-s", type=float, default=0.5)
    parser.add_argument("--record-dir", default="/data/stm_records")
    parser.add_argument("--tuning-log-every-n", type=int, default=2)
    parser.add_argument("--radar-snapshot-every-n", type=int, default=5)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--disable-target-mission", action="store_true")
    parser.add_argument("--enable-flight", action="store_true")
    parser.add_argument("--auto-takeoff", action="store_true")
    parser.add_argument(
        "--confirm-road-contour-bypass-flight-test",
        action="store_true",
        help="Acknowledge this independent direct-command real-flight experiment",
    )
    parser.add_argument("--takeoff-height-cm", type=int, default=100)
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.model_npu):
        raise FileNotFoundError(f"required NPU model missing: {args.model_npu}")
    if args.loop_hz <= 0.0 or args.duration_s <= 0.0:
        raise ValueError("loop rate and duration must be positive")
    if args.radar_timeout_s <= 0.0:
        raise ValueError("--radar-timeout-s must be positive")
    if args.hc14_baudrate is not None and args.hc14_baudrate <= 0:
        raise ValueError("--hc14-baudrate must be greater than zero")
    if args.hc14_connect_timeout_s <= 0.0:
        raise ValueError("--hc14-connect-timeout-s must be greater than zero")
    if args.tuning_log_every_n <= 0 or args.radar_snapshot_every_n <= 0:
        raise ValueError("recording intervals must be positive integers")
    if args.enable_flight:
        missing: list[str] = []
        if not args.auto_takeoff:
            missing.append("--auto-takeoff")
        if not args.confirm_road_contour_bypass_flight_test:
            missing.append("--confirm-road-contour-bypass-flight-test")
        if missing:
            raise ValueError("--enable-flight requires " + ", ".join(missing))
        if args.no_record:
            raise ValueError("real flight requires session recording")
        if not 40 <= args.takeoff_height_cm <= 100:
            raise ValueError("real-flight takeoff height must be within 40..100cm")
        if args.duration_s > 120.0:
            raise ValueError("real-flight duration cannot exceed 120s")
    elif args.auto_takeoff or args.confirm_road_contour_bypass_flight_test:
        raise ValueError("takeoff/confirmation options require --enable-flight")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    actual_flight = bool(args.enable_flight)
    target_mission_enabled = not args.disable_target_mission
    visual_config = build_experimental_visual_config(
        camera_index=args.camera_index,
        npu_model_path=args.model_npu,
        target_enable=target_mission_enabled,
    )
    planner_config = ContourBypassConfig()
    direct_limits = DirectCommandLimits(
        max_abs_vx_cm_s=visual_config.max_vx_cm_s,
        max_abs_vy_cm_s=visual_config.max_vy_cm_s,
        max_abs_yaw_rate_deg_s=visual_config.max_yaw_rate_deg_s,
    )
    flight_config = FlightRuntimeConfig(takeoff_height_cm=args.takeoff_height_cm)
    target_config = PurpleTargetMissionConfig(
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
    process_runtime = _build_process_runtime(args, visual_config)
    recorder = SessionRecorder(
        SessionRecorderConfig(
            root_dir=args.record_dir,
            enabled=not args.no_record,
            mode="road_contour_bypass",
            frame_rate_hz=1.0,
            radar_rate_hz=args.loop_hz / max(1, args.radar_snapshot_every_n),
            command_rate_hz=args.loop_hz / max(1, args.tuning_log_every_n),
            video_enabled=True,
            video_every_n=2,
            video_fps=5.0,
            frame_ring_descriptor=(
                process_runtime.frame_ring_descriptor if process_runtime is not None else None
            ),
            metadata={
                "argv": list(sys.argv),
                "mode": "road_contour_bypass",
                "visual_config": vars(visual_config),
                "planner_config": vars(planner_config),
                "direct_command_limits": vars(direct_limits),
                "safety_arbiter_enabled": False,
                "path_policy": "plan_once_freeze_until_encounter_complete",
                "target_mission_enabled": target_mission_enabled,
                "hc14_port": args.hc14_port,
                "hc14_baudrate": args.hc14_baudrate,
                "hc14_connect_timeout_s": args.hc14_connect_timeout_s,
            },
        )
    )
    if actual_flight and not recorder.healthy:
        if process_runtime is not None:
            process_runtime.stop()
        raise RuntimeError("real flight refused because session recording is unavailable")
    sink_id = _setup_logging(recorder.log_sink if recorder.enabled else None)
    guidance = FrozenVisualGuidance(visual_config, process_runtime=process_runtime)
    radars = (
        ProcessRadarClient(process_runtime, max_age_s=args.radar_timeout_s)
        if process_runtime is not None
        else MultiRadar(_radar_configs(args.upper_port, args.lower_port))
    )
    radar_field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=planner_config.radar_max_range_cm,
            body_x_half_cm=25.0,
            body_y_half_cm=25.0,
            forward_corridor_half_width_cm=planner_config.activation_corridor_half_width_cm,
            side_corridor_x_half_cm=25.0,
        )
    )
    debug_dir = recorder.session_dir / "plans" if recorder.session_dir is not None else None
    planner = ContourTrajectoryBypassPlanner(planner_config, debug_plan_dir=debug_dir)
    target_mission = PurpleTargetMissionController(target_config) if target_mission_enabled else None

    fc = None
    flight_owned = False
    guidance_started = False
    radars_started = False
    indicator = None
    fleet_node = None
    fleet_stop_event = threading.Event()
    fleet_state = RoadPatrolFleetStateProvider()
    period_s = 1.0 / args.loop_hz
    loop_monitor = LoopRateMonitor(args.loop_hz)
    interrupted = False
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
            # Ground authorization depends on the direct HC-14 link. Refuse to
            # open the FC connection until that radio is uniquely resolved and
            # connected, matching the current road-patrol preflight ordering.
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
            fc = connect_fc(FCConnectConfig(port=args.fc_port, mode=2, timeout_s=10.0))
            flight_owned = True
            fleet_state.bind_fc(fc)
            indicator = FusionFlightIndicator(fc)
            fleet_state.set_operation_state(RoadPatrolOperationState.TAKEOFF)
            if wait_for_ground_takeoff_authorization(
                fleet_node=fleet_node,
                indicator=indicator,
                stop_event=fleet_stop_event,
            ):
                auto_takeoff(fc, flight_config)
                fleet_state.set_operation_state(RoadPatrolOperationState.LINE_FOLLOWING)
            else:
                interrupted = True
                fleet_state.set_operation_state(RoadPatrolOperationState.LANDING)
        else:
            logger.warning(
                "[CONTOUR-BYPASS] dry run: live camera/radars, no FC send; environmental command gate disabled"
            )

        started_s = time.perf_counter()
        previous_loop_s = started_s
        last_log_s = 0.0
        loop_count = 0
        while (
            time.perf_counter() - started_s < args.duration_s
            and not fleet_stop_event.is_set()
        ):
            loop_start = time.perf_counter()
            dt_s = max(0.0, min(0.5, loop_start - previous_loop_s))
            previous_loop_s = loop_start
            sample = guidance.sample(loop_start)
            points = radars.get_obstacle_points_body_cm(
                max_distance_cm=planner_config.radar_max_range_cm
            )
            radar_field.update(points, loop_start)
            radar_age_s = multi_radar_age_s(radars)
            radar_fresh = bool(
                radars.connected and radars.is_fresh(max_age_s=args.radar_timeout_s)
            )

            selected_visual = sample.desired
            target_decision = None
            if target_mission is not None:
                target_decision = target_mission.update(
                    now_s=loop_start,
                    road_desired=sample.desired,
                    target=sample.target,
                    target_stale=sample.target_stale,
                    planner_state=planner.state.value,
                    radar_fresh=radar_fresh,
                    altitude_cm=_fc_altitude(fc),
                    obstacle_conflict=_activation_conflict(radar_field, planner_config),
                )
                selected_visual = target_decision.desired

            # Exactly one module owns the final command in each state: visual
            # (including the existing target mission) or the frozen-path planner.
            planned = planner.update(
                visual_desired=selected_visual,
                perception=sample.perception,
                radar_field=radar_field,
                radar_fresh=radar_fresh,
                now_s=loop_start,
                dt_s=dt_s,
            )
            applied = send_direct_command(
                fc,
                planned,
                dry_run=not actual_flight,
                limits=direct_limits,
            )
            planner.report_applied_command(applied, dt_s)

            payload_release_event = False
            if target_mission is not None and target_mission.release_is_authorized(
                planner_state=planner.state.value,
                radar_fresh=radar_fresh,
                safety_state="OK",
                command_allowed=True,
                final_command=applied,
                obstacle_clear=not _activation_conflict(radar_field, planner_config),
            ):
                if actual_flight:
                    fc.set_digital_output(0, False)
                target_mission.mark_payload_released(loop_start)
                payload_release_event = True
            if target_mission is not None and target_mission.consume_disable_target_request():
                guidance.disable_target()

            diagnostics = planner.diagnostics()
            avoiding = planner.state not in (
                ContourBypassState.NORMAL,
                ContourBypassState.ACQUIRE,
            )
            if indicator is not None:
                indicator.update(
                    now_s=loop_start,
                    avoiding=avoiding,
                    unexpected=bool(
                        planner.state == ContourBypassState.PLAN_FAILED
                        or not sample.camera_ok
                    ),
                    target_active=bool(target_mission and target_mission.mission_active),
                )
                fleet_state.set_operation_state(
                    cruise_operation_state(target_mission=target_mission, avoiding=avoiding)
                )

            extra = {
                "visual": {
                    "road_found": bool(getattr(sample.perception, "is_road_found", False)),
                    "confidence": _float_or_none(getattr(sample.perception, "confidence", None)),
                    "pixel_error": _float_or_none(
                        getattr(sample.perception, "corrected_pixel_error", None)
                    ),
                    "age_s": sample.perception_age_s,
                    "stale": sample.perception_stale,
                    "camera_ok": sample.camera_ok,
                    "controller": sample.diagnostics,
                },
                "planner": diagnostics,
                "purple_target": {
                    "enabled": target_mission is not None,
                    "mission": (
                        target_mission.diagnostics(loop_start) if target_mission is not None else None
                    ),
                    "payload_release_event": payload_release_event,
                },
                "commands": {
                    "visual": selected_visual.as_fc_tuple(),
                    "planner": planned.as_fc_tuple(),
                    "final": applied.as_fc_tuple(),
                    "safety": "disabled",
                    "single_send_owner": "experiments.road_contour_bypass.main",
                },
                "safety_arbiter_enabled": False,
                "sent": actual_flight,
            }
            if recorder.frame_due(loop_count, loop_start):
                recorder.record_frame(
                    loop_count=loop_count,
                    now_s=loop_start,
                    frame=sample.frame,
                    label="road_contour",
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
                desired=selected_visual,
                safe_command=applied,
                decision_reason="direct_task_send",
                extra=extra,
            )
            recorder.record_command(
                loop_count=loop_count,
                now_s=loop_start,
                desired=selected_visual,
                safe_command=applied,
                decision_reason="direct_task_send_safety_disabled",
                extra=extra,
            )
            if loop_start - last_log_s >= 1.0:
                last_log_s = loop_start
                logger.info(
                    "[CONTOUR-BYPASS] state={} progress={:.3f} pose=({:.1f},{:.1f},{:.1f}) "
                    "radar_fresh={} bearing={} road={} err={} cmd={} frozen={} generations={}",
                    diagnostics["state"],
                    diagnostics["path_progress"],
                    diagnostics["local_x"],
                    diagnostics["local_y"],
                    diagnostics["local_yaw"],
                    radar_fresh,
                    diagnostics["last_obstacle_bearing"],
                    diagnostics["road_found"],
                    diagnostics["road_pixel_error"],
                    applied.as_fc_tuple(),
                    diagnostics["path_frozen"],
                    diagnostics["plan_generation_count"],
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
        logger.warning("[CONTOUR-BYPASS] interrupted")
    except BaseException:
        fleet_state.set_operation_state(
            RoadPatrolOperationState.LANDING if flight_owned else RoadPatrolOperationState.FAULT
        )
        if indicator is not None:
            indicator.set_red()
        raise
    finally:
        if fc is not None:
            try:
                if flight_owned:
                    fleet_state.set_operation_state(RoadPatrolOperationState.LANDING)
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
        if process_runtime is not None:
            process_runtime.stop_workers()
        if sink_id is not None:
            logger.remove(sink_id)
        recorder.close()
        if process_runtime is not None:
            process_runtime.stop()
        logger.info(
            "[CONTOUR-BYPASS] stopped interrupted={} actual_flight={}",
            interrupted,
            actual_flight,
        )


def _build_process_runtime(args, visual_config):
    if args.runtime_mode != "process":
        return None
    return ProcessRuntime(
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


def _activation_conflict(radar_field: RadarObstacleField, config: ContourBypassConfig) -> bool:
    points = np.asarray(radar_field.points_body_cm, dtype=float).reshape(-1, 2)
    if not len(points):
        return False
    ranges = np.linalg.norm(points, axis=1)
    bearings = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    valid = (
        (points[:, 0] > config.radar_min_x_cm)
        & (ranges <= config.activation_radius_cm)
        & (np.abs(points[:, 1]) <= config.activation_corridor_half_width_cm)
        & (np.abs(bearings) <= config.radar_fov_deg * 0.5)
    )
    return int(np.count_nonzero(valid)) >= config.min_cluster_points


def _fc_altitude(fc) -> float | None:
    if fc is None:
        return None
    try:
        return float(fc.state.alt_add.value)
    except (AttributeError, TypeError, ValueError):
        return None


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
    return logger.add(log_target, enqueue=False) if callable(log_target) else logger.add(
        str(log_target), enqueue=True, encoding="utf-8"
    )


def _sleep_to_rate(loop_start: float, period_s: float) -> None:
    remaining = period_s - (time.perf_counter() - loop_start)
    if remaining > 0.0:
        time.sleep(remaining)


def _float_or_none(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


if __name__ == "__main__":
    main()
