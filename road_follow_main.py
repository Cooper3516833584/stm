"""Segmentation-based road-following entry point.

Default behavior is dry-run and camera-only road following: radar acquisition
and obstacle avoidance are disabled unless ``--enable-radar`` is explicitly
provided. Non-zero FC commands are sent only when ``--enable-flight`` is
provided. Automatic takeoff additionally needs ``--auto-takeoff``. On Ctrl+C,
an enabled in-flight session requests the FC's native in-place landing and
waits for it to lock.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from loguru import logger

from perception_pipeline import PerceptionPipeline


ROAD_WIDTH_CM = 50.0
ROAD_HALF_WIDTH_CM = ROAD_WIDTH_CM / 2.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Road-following dry-run / flight entry")
    parser.add_argument("--camera-index", type=int, default=7,
                        help="Road-following camera index (default: 7 on OpenSTLinux)")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera", default=None, help="Deprecated alias; use --camera-index for numeric V4L2 devices")
    parser.add_argument("--width", type=int, default=None, help="Deprecated alias for --camera-width")
    parser.add_argument("--height", type=int, default=None, help="Deprecated alias for --camera-height")
    parser.add_argument("--fps", type=int, default=None, help="Deprecated alias for --camera-fps")
    parser.add_argument(
        "--road-model-backend",
        choices=["npu", "cpu"],
        default="npu",
        help="Road inference backend (default: npu; cpu keeps the legacy small YOLO model)",
    )
    parser.add_argument(
        "--road-postprocess-mode",
        choices=["fast-main", "full"],
        default="fast-main",
        help="Road postprocess: fast sparse main road (default) or full-resolution main road",
    )
    parser.add_argument(
        "--model",
        default="FlightController/Solutions/model/road_yolo11n_seg_128.onnx",
        help="Legacy lightweight YOLO ONNX path used by --road-model-backend cpu",
    )
    parser.add_argument(
        "--model-npu",
        default="FlightController/Solutions/model/new_road_seg_v4_final_fp32.nb",
        help="Semantic segmentation .nb path used by --road-model-backend npu",
    )
    parser.add_argument("--model-path", default=None, help="Deprecated alias for --model")
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--fc-port", default=None)
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--no-fc", action="store_true")
    parser.add_argument("--connect-fc", action="store_true", help="Connect FC for status only")
    radar_group = parser.add_mutually_exclusive_group()
    radar_group.add_argument(
        "--enable-radar",
        dest="no_radar",
        action="store_false",
        help="Opt in to radar acquisition and obstacle avoidance (disabled by default)",
    )
    radar_group.add_argument(
        "--no-radar",
        dest="no_radar",
        action="store_true",
        help="Disable radar acquisition and obstacle avoidance (default)",
    )
    parser.set_defaults(no_radar=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-flight", action="store_true")
    parser.add_argument(
        "--auto-takeoff",
        action="store_true",
        help="Unlock and use the FC one-key takeoff before road following; requires --enable-flight",
    )
    parser.add_argument(
        "--takeoff-height-cm",
        type=int,
        default=100,
        help="One-key takeoff target height in cm (40..500; default: 100)",
    )
    parser.add_argument("--post-unlock-delay-s", type=float, default=2.0)
    parser.add_argument("--takeoff-timeout-s", type=float, default=25.0)
    parser.add_argument("--takeoff-height-tolerance-cm", type=float, default=15.0)
    parser.add_argument(
        "--min-takeoff-battery-v",
        type=float,
        default=10.5,
        help="Minimum battery voltage before and during automatic takeoff (default: 10.5)",
    )
    parser.add_argument("--takeoff-low-battery-confirm-frames", type=int, default=3)
    parser.add_argument("--landing-timeout-s", type=float, default=30.0)
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--wb-enable", action="store_true",
                        help="Enable software white balance correction for camera color cast")
    parser.add_argument("--wb-r", type=float, default=1.00,
                        help="White balance R channel gain (default: 1.00; calibrate cam#7 before enabling)")
    parser.add_argument("--wb-g", type=float, default=1.00,
                        help="White balance G channel gain (default: 1.00)")
    parser.add_argument("--wb-b", type=float, default=1.00,
                        help="White balance B channel gain (default: 1.00; calibrate cam#7 before enabling)")
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--debug-every-n", type=int, default=30)
    parser.add_argument("--debug-image-dir", default=None, help="Deprecated alias for --debug-dir")
    parser.add_argument("--debug-image-every", type=int, default=None, help="Deprecated alias for --debug-every-n")
    parser.add_argument("--record-dir", default="/media/sdcard/stm_records",
                        help="Directory on SD card for session recording")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable camera/radar session recording")
    parser.add_argument("--record-frame-every-n", type=int, default=10,
                        help="Save one JPEG keyframe every N control loops")
    parser.add_argument("--no-record-video", action="store_true",
                        help="Disable continuous camera AVI recording")
    parser.add_argument("--record-video-every-n", type=int, default=1,
                        help="Append one video frame every N control loops (default: 1)")
    parser.add_argument("--record-video-fps", type=float, default=10.0,
                        help="Playback FPS stored in camera.avi (default: 10)")
    parser.add_argument("--record-frame-queue-size", type=int, default=8,
                        help="Bounded asynchronous camera writer queue size")
    parser.add_argument("--record-radar-every-n", type=int, default=1,
                        help="Save one radar metadata/point snapshot every N control loops")
    parser.add_argument("--record-jpeg-quality", type=int, default=85)
    parser.add_argument("--radar-timeout-s", type=float, default=0.5)
    parser.add_argument("--max-distance-cm", type=float, default=300.0)
    parser.add_argument("--body-x-half-cm", type=float, default=25.0)
    parser.add_argument("--body-y-half-cm", type=float, default=25.0)
    parser.add_argument(
        "--corridor-half-width-cm",
        type=float,
        default=ROAD_HALF_WIDTH_CM,
        help="Radar forward corridor half-width (default: 25cm for a 50cm road)",
    )
    parser.add_argument("--max-vx-cm-s", type=float, default=25.0)
    parser.add_argument("--max-vy-cm-s", type=float, default=5.0)
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=25.0)
    parser.add_argument("--road-pixel-kp-vy", type=float, default=0.03,
                        help="Cross-track pixel error to lateral velocity gain")
    parser.add_argument("--road-pixel-kp-yaw", type=float, default=0.0,
                        help="Legacy cross-track pixel error to yaw gain (default: disabled)")
    parser.add_argument("--road-angle-kp-yaw", type=float, default=0.25)
    parser.add_argument("--road-target-centerline-angle-deg", type=float, default=90.0,
                        help="Camera-image angle that corresponds to aircraft-forward road alignment")
    parser.add_argument("--road-angle-deadband-deg", type=float, default=3.0)
    parser.add_argument("--road-pixel-filter-tau-s", type=float, default=0.35)
    parser.add_argument("--road-angle-filter-tau-s", type=float, default=0.35)
    parser.add_argument("--road-pixel-filter-max-rate-px-s", type=float, default=300.0)
    parser.add_argument("--road-angle-filter-max-rate-deg-s", type=float, default=45.0)
    parser.add_argument("--road-yaw-sign", type=float, default=1.0)
    parser.add_argument("--road-lateral-sign", type=float, default=-1.0,
                        help="Image-right road error to FC body-Y mapping (default: -1)")
    parser.add_argument("--road-search-yaw-rate-deg-s", type=float, default=0.0,
                        help="Yaw while road is lost; zero safely holds heading")
    parser.add_argument("--road-heading-slowdown-start-deg", type=float, default=30.0)
    parser.add_argument("--road-heading-stop-deg", type=float, default=70.0)
    parser.add_argument("--road-bypass-enable", action="store_true",
                        help="Enable radar-assisted in-road bypass for branches/vines intruding into the road center")
    parser.add_argument(
        "--road-half-width-cm",
        type=float,
        default=ROAD_HALF_WIDTH_CM,
        help="Physical road half-width (default: 25cm, i.e. 50cm full width)",
    )
    parser.add_argument("--road-edge-margin-cm", type=float, default=25.0)
    parser.add_argument("--road-bypass-lookahead-cm", type=float, default=180.0)
    parser.add_argument("--road-bypass-min-x-cm", type=float, default=40.0)
    parser.add_argument(
        "--road-bypass-intrusion-half-width-cm",
        type=float,
        default=ROAD_HALF_WIDTH_CM,
        help="Obstacle intrusion half-width (default: the 25cm road half-width)",
    )
    parser.add_argument("--road-bypass-clearance-cm", type=float, default=75.0)
    parser.add_argument("--road-bypass-speed-cm-s", type=float, default=12.0)
    parser.add_argument("--road-bypass-lateral-step-cm", type=float, default=10.0)
    parser.add_argument("--road-bypass-guide-distance-cm", type=float, default=150.0)
    parser.add_argument("--road-bypass-yaw-kp", type=float, default=0.75)
    parser.add_argument("--road-bypass-max-yaw-bias-deg-s", type=float, default=15.0)
    parser.add_argument("--road-bypass-yaw-sign", type=float, default=1.0)
    parser.add_argument("--road-bypass-activate-frames", type=int, default=2)
    parser.add_argument("--road-bypass-release-s", type=float, default=0.5)
    parser.add_argument("--road-bypass-min-confidence", type=float, default=0.4)
    parser.add_argument("--road-bypass-return-pixel-deadband-px", type=float, default=35.0)
    parser.add_argument("--offset-comp-enable", action="store_true")
    parser.add_argument("--enable-offset-comp", action="store_true", help="Deprecated alias for --offset-comp-enable")
    parser.add_argument("--flight-height-m", type=float, default=1.0,
                        help="飞行高度 (m), 用于计算该高度的 meters-per-pixel")
    parser.add_argument(
        "--cam-forward-offset-m",
        type=float,
        default=-0.0787,
        help="Road camera body-frame X offset in metres (default: -0.0787; rear of body centre)",
    )
    parser.add_argument("--meters-per-pixel-x", type=float, default=None)
    parser.add_argument("--offset-correction-sign", type=float, default=1.0)
    parser.add_argument("--offset-max-correction-px", type=float, default=120.0)
    parser.add_argument("--pipeline-latency-s", type=float, default=0.0)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args = _normalize_args(args)
    _validate_flight_args(args)

    from road_perception import CameraOffsetCompensationConfig
    from FlightController.Components import MultiRadar, RadarConfig
    from FlightController.Components.FCConnector import FCConnectConfig, connect_fc
    from FlightController.Solutions.RoadObstacleBypassPlanner import (
        RoadBypassConfig,
        RoadObstacleBypassPlanner,
    )
    from FlightController.Solutions.RoadFollower import RoadFollower, RoadFollowerConfig
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

    selected_model = args.model_npu if args.road_model_backend == "npu" else args.model
    if not os.path.isfile(selected_model):
        msg = (
            f"[ROAD] {args.road_model_backend} model missing, "
            f"perception lost: {selected_model}"
        )
        if args.require_model:
            raise FileNotFoundError(msg)
        logger.warning(msg)

    actual_dry_run = bool(args.dry_run or args.no_fc or not args.enable_flight)
    if actual_dry_run:
        logger.warning("[SAFETY] dry-run mode: no non-zero velocity will be sent. Add --enable-flight to allow real output")
    if args.no_radar:
        logger.info("[ROAD] camera-only mode: radar acquisition and obstacle avoidance are disabled")
    if args.road_bypass_enable and args.road_half_width_cm <= args.road_edge_margin_cm:
        logger.warning(
            "[ROAD] 50cm road leaves no safe lateral bypass corridor after the "
            "25cm body/edge margin; bypass will use its no-gap slowdown behavior"
        )

    fc = None
    multi_radar = None
    pipeline = None
    flight_owned = False
    interrupted = False
    log_sink_id = None
    recorder = SessionRecorder(
        SessionRecorderConfig(
            root_dir=args.record_dir,
            enabled=not args.no_record,
            mode="road_follow",
            frame_every_n=args.record_frame_every_n,
            radar_every_n=args.record_radar_every_n,
            jpeg_quality=args.record_jpeg_quality,
            video_enabled=not args.no_record_video,
            video_every_n=args.record_video_every_n,
            video_fps=args.record_video_fps,
            frame_queue_size=args.record_frame_queue_size,
            metadata={
                "argv": list(sys.argv),
                "arguments": dict(vars(args)),
                "control_design": "heading-yaw+lateral-cross-track",
            },
        )
    )
    default_log_path = recorder.runtime_log_path
    log_sink_id = _setup_logging(args.log_file or default_log_path)
    radar_field = RadarObstacleField(
        RadarFieldConfig(
            max_distance_cm=args.max_distance_cm,
            body_x_half_cm=args.body_x_half_cm,
            body_y_half_cm=args.body_y_half_cm,
            forward_corridor_half_width_cm=args.corridor_half_width_cm,
        )
    )
    follower = RoadFollower(
        RoadFollowerConfig(
            image_width=args.camera_width,
            max_vx_cm_s=args.max_vx_cm_s,
            max_vy_cm_s=args.max_vy_cm_s,
            max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
            search_yaw_rate_deg_s=args.road_search_yaw_rate_deg_s,
            pixel_kp_vy=args.road_pixel_kp_vy,
            pixel_kp_yaw=args.road_pixel_kp_yaw,
            angle_kp_yaw=args.road_angle_kp_yaw,
            pixel_filter_tau_s=args.road_pixel_filter_tau_s,
            angle_filter_tau_s=args.road_angle_filter_tau_s,
            pixel_filter_max_rate_px_s=args.road_pixel_filter_max_rate_px_s,
            angle_filter_max_rate_deg_s=args.road_angle_filter_max_rate_deg_s,
            target_centerline_angle_deg=args.road_target_centerline_angle_deg,
            angle_deadband_deg=args.road_angle_deadband_deg,
            yaw_sign=args.road_yaw_sign,
            lateral_sign=args.road_lateral_sign,
            heading_slowdown_start_deg=args.road_heading_slowdown_start_deg,
            heading_stop_deg=args.road_heading_stop_deg,
        )
    )
    bypass_planner = RoadObstacleBypassPlanner(
        RoadBypassConfig(
            enabled=bool(args.road_bypass_enable and not args.no_radar),
            road_half_width_cm=args.road_half_width_cm,
            road_edge_margin_cm=args.road_edge_margin_cm,
            min_x_cm=args.road_bypass_min_x_cm,
            lookahead_cm=args.road_bypass_lookahead_cm,
            intrusion_half_width_cm=args.road_bypass_intrusion_half_width_cm,
            bypass_clearance_cm=args.road_bypass_clearance_cm,
            lateral_step_cm=args.road_bypass_lateral_step_cm,
            guide_distance_cm=args.road_bypass_guide_distance_cm,
            bypass_speed_cm_s=args.road_bypass_speed_cm_s,
            bypass_yaw_kp=args.road_bypass_yaw_kp,
            max_bypass_yaw_bias_deg_s=args.road_bypass_max_yaw_bias_deg_s,
            max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
            bypass_yaw_sign=args.road_bypass_yaw_sign,
            activate_frames=args.road_bypass_activate_frames,
            release_s=args.road_bypass_release_s,
            min_confidence=args.road_bypass_min_confidence,
            return_pixel_deadband_px=args.road_bypass_return_pixel_deadband_px,
        )
    )
    arbiter = SafetyArbiter(
        SafetyConfig(
            require_fc=not args.no_fc,
            require_hold_pos_mode=not args.no_fc,
            require_unlocked=bool(args.enable_flight and not args.no_fc),
            require_radar=not args.no_radar,
            radar_timeout_s=args.radar_timeout_s,
            max_vx_cm_s=args.max_vx_cm_s,
            max_vy_cm_s=args.max_vy_cm_s,
            max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
        )
    )
    offset_comp = CameraOffsetCompensationConfig(
        enabled=bool(args.offset_comp_enable or args.enable_offset_comp),
        cam_forward_offset_m=args.cam_forward_offset_m,
        meters_per_pixel_x=args.meters_per_pixel_x,
        correction_sign=args.offset_correction_sign,
        max_correction_px=args.offset_max_correction_px,
        pipeline_latency_s=args.pipeline_latency_s,
    )
    period_s = 1.0 / max(args.loop_hz, 0.1)
    telemetry_tracker = _FCTelemetryTracker()

    try:
        if not args.no_fc:
            fc = connect_fc(FCConnectConfig(port=args.fc_port, mode=2, timeout_s=10.0))
            logger.info("[ROAD] FC connected and switched to HOLD_POS mode")

        if not args.no_radar:
            multi_radar = MultiRadar(_radar_configs(args.upper_port, args.lower_port))
            multi_radar.start()

        pipeline = PerceptionPipeline(
            camera_index=args.camera_index,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            model_path=args.model,
            npu_model_path=args.model_npu,
            inference_backend=args.road_model_backend,
            postprocess_mode=args.road_postprocess_mode,
            flight_height_m=args.flight_height_m,
            wb_enable=bool(args.wb_enable),
            wb_r=args.wb_r,
            wb_g=args.wb_g,
            wb_b=args.wb_b,
            offset_comp_config=offset_comp,
        )
        pipeline.start()

        if args.auto_takeoff:
            flight_owned = True
            _auto_takeoff(fc, args)

        logger.info(
            "[ROAD] started dry_run={} no_radar={} mode=single-road camera={} "
            "backend={} model={} postprocess={}".format(
                actual_dry_run,
                args.no_radar,
                args.camera_index,
                args.road_model_backend,
                selected_model,
                args.road_postprocess_mode,
            )
        )
        loop_count = 0
        last_log_s = 0.0
        while True:
            loop_start = time.perf_counter()

            # ── 1. Non-blocking: read latest perception + frame from pipeline
            perception, percept_age_s, percept_stale = pipeline.latest_perception()
            camera_ok = pipeline.camera_ok

            frame, frame_ts = pipeline.latest_frame()
            frame_age_s = (
                max(0.0, loop_start - frame_ts)
                if frame_ts is not None and frame_ts > 0.0
                else None
            )
            # ── 2. Road following (uses latest perception, or lost if stale)
            if perception is None or percept_stale:
                desired = follower.update(None, now_s=loop_start)
            else:
                desired = follower.update(perception, now_s=loop_start)

            # ── 3. Radar (unchanged)
            if multi_radar is not None:
                points = multi_radar.get_obstacle_points_body_cm(max_distance_cm=args.max_distance_cm)
                radar_field.update(points, loop_start)
                radar_age_s = multi_radar_age_s(multi_radar)
                radar_connected = bool(multi_radar.connected and multi_radar.is_fresh(max_age_s=args.radar_timeout_s))
            else:
                radar_field.update(np.empty((0, 2), dtype=float), loop_start)
                radar_age_s = 0.0
                radar_connected = True

            # ── 4. Planning / safety / FC send (unchanged)
            planned = bypass_planner.update(
                desired=desired,
                perception=perception,
                radar_field=radar_field,
                now_s=loop_start,
            )
            health = flight_health_from_sources(
                fc=fc,
                multi_radar=multi_radar,
                radar_timeout_s=args.radar_timeout_s,
                camera_ok=bool(camera_ok),
            )
            safe = arbiter.filter(
                planned,
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

            fc_telemetry = telemetry_tracker.update(fc, loop_start)
            controller_diagnostics = follower.last_diagnostics.as_dict()
            diagnostic_extra = _road_record_extra(
                perception,
                camera_ok,
                planned,
                bypass_planner,
                controller_diagnostics=controller_diagnostics,
                perception_age_s=percept_age_s,
                perception_stale=percept_stale,
                frame_age_s=frame_age_s,
                fc_telemetry=fc_telemetry,
            )
            diagnostic_extra["send_gate"] = {
                "command": _command_extra(decision.command),
                "allowed": bool(decision.allowed),
                "hard_stop": bool(decision.hard_stop),
                "reason": decision.reason,
                "sent": bool(not actual_dry_run and fc is not None),
            }
            diagnostic_frame = _annotate_road_frame(
                frame,
                perception=perception,
                loop_count=loop_count,
                controller_diagnostics=controller_diagnostics,
                safe_command=safe.command,
                fc_telemetry=fc_telemetry,
                perception_age_s=percept_age_s,
                perception_stale=percept_stale,
            )
            frame_record_path = recorder.record_frame(
                loop_count=loop_count,
                now_s=loop_start,
                frame=diagnostic_frame,
                label="road",
                source_time_s=frame_ts,
                extra={
                    "perception_age_s": _float_or_none(percept_age_s),
                    "perception_stale": bool(percept_stale),
                    "frame_age_s": _float_or_none(frame_age_s),
                    "road_state": diagnostic_extra["road_state"],
                    "controller": controller_diagnostics,
                    "fc": fc_telemetry,
                },
            )
            diagnostic_extra["frame_record_path"] = frame_record_path

            # ── 5. Recording
            if multi_radar is not None:
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
                    extra=diagnostic_extra,
                )
            recorder.record_command(
                loop_count=loop_count,
                now_s=loop_start,
                desired=desired,
                safe_command=safe.command,
                decision_reason=decision.reason,
                extra=diagnostic_extra,
            )

            if loop_start - last_log_s >= 1.0:
                last_log_s = loop_start
                _log_road_summary(
                    args=args,
                    perception=perception,
                    desired=desired,
                    planned=planned,
                    safe_command=safe.command,
                    safety_reason=decision.reason,
                    sent=bool(not actual_dry_run and fc is not None),
                    radar_fresh=radar_connected if multi_radar is not None else "disabled",
                    fc=fc,
                    bypass_planner=bypass_planner,
                    controller_diagnostics=controller_diagnostics,
                    perception_age_s=percept_age_s,
                    perception_stale=percept_stale,
                    frame_age_s=frame_age_s,
                    fc_telemetry=fc_telemetry,
                )

            loop_count += 1
            _sleep_to_rate(loop_start, period_s)
    except KeyboardInterrupt:
        interrupted = True
        logger.info("[ROAD] interrupted")
    finally:
        if fc is not None:
            try:
                if flight_owned or (interrupted and not actual_dry_run):
                    _land_and_wait_for_lock(fc, args)
                else:
                    health = flight_health_from_sources(
                        fc=fc,
                        multi_radar=multi_radar,
                        radar_timeout_s=args.radar_timeout_s,
                        camera_ok=True,
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
        if pipeline is not None:
            pipeline.stop()
        if multi_radar is not None:
            multi_radar.stop()
        logger.info("[ROAD] stopped")
        recorder.close()
        if log_sink_id is not None:
            logger.remove(log_sink_id)


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.width is not None:
        args.camera_width = args.width
    if args.height is not None:
        args.camera_height = args.height
    if args.fps is not None:
        args.camera_fps = args.fps
    if args.model_path:
        args.model = args.model_path
        args.road_model_backend = "cpu"
    if args.debug_image_dir:
        args.debug_dir = args.debug_image_dir
    if args.debug_image_every is not None:
        args.debug_every_n = args.debug_image_every
    if args.camera is not None:
        try:
            args.camera_index = int(args.camera)
        except ValueError:
            args.camera_index = args.camera
    return args


def _validate_flight_args(args: argparse.Namespace) -> None:
    """Reject combinations that could make automatic flight ambiguous."""
    if args.auto_takeoff and (not args.enable_flight or args.dry_run or args.no_fc):
        raise ValueError(
            "--auto-takeoff requires --enable-flight and cannot be combined with --dry-run or --no-fc"
        )
    if not 40 <= args.takeoff_height_cm <= 500:
        raise ValueError("--takeoff-height-cm must be within the FC one-key takeoff range of 40..500")
    if args.no_radar and args.road_bypass_enable:
        raise ValueError("--road-bypass-enable requires --enable-radar")
    for option in (
        "post_unlock_delay_s",
        "takeoff_timeout_s",
        "takeoff_height_tolerance_cm",
        "min_takeoff_battery_v",
        "takeoff_low_battery_confirm_frames",
        "landing_timeout_s",
        "record_frame_every_n",
        "record_radar_every_n",
        "record_video_every_n",
        "record_video_fps",
        "record_frame_queue_size",
    ):
        if getattr(args, option) <= 0:
            raise ValueError(f"--{option.replace('_', '-')} must be greater than zero")
    for option in ("max_vx_cm_s", "max_vy_cm_s", "max_yaw_rate_deg_s"):
        if getattr(args, option) < 0:
            raise ValueError(f"--{option.replace('_', '-')} cannot be negative")
    if args.road_heading_stop_deg <= args.road_heading_slowdown_start_deg:
        raise ValueError(
            "--road-heading-stop-deg must be greater than --road-heading-slowdown-start-deg"
        )
    if not 0.0 <= args.road_target_centerline_angle_deg <= 180.0:
        raise ValueError("--road-target-centerline-angle-deg must be within 0..180")
    if args.road_angle_deadband_deg < 0.0:
        raise ValueError("--road-angle-deadband-deg cannot be negative")
    for option in (
        "road_pixel_filter_tau_s",
        "road_angle_filter_tau_s",
        "road_pixel_filter_max_rate_px_s",
        "road_angle_filter_max_rate_deg_s",
    ):
        if getattr(args, option) < 0.0:
            raise ValueError(f"--{option.replace('_', '-')} cannot be negative")


def _wait_for_fc_mode(fc, target_mode: int, timeout_s: float = 5.0) -> None:
    """Set an FC mode and confirm the reported state changed."""
    fc.set_flight_mode(target_mode)
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if not fc.connected:
            raise RuntimeError("FC disconnected while changing flight mode")
        if int(fc.state.mode.value) == target_mode:
            return
        time.sleep(0.05)
    raise RuntimeError(
        f"FC mode change timed out: expected mode={target_mode}, got mode={fc.state.mode.value}"
    )


def _auto_takeoff(fc, args: argparse.Namespace) -> None:
    """Use the guarded one-key takeoff sequence from the optical-flow test."""
    if fc is None or not fc.connected:
        raise RuntimeError("FC is not connected; automatic takeoff refused")
    if bool(fc.state.unlock.value):
        raise RuntimeError("FC is already unlocked; automatic takeoff refused")
    if float(fc.state.bat.value) <= 1.0:
        raise RuntimeError("FC has not reported a valid battery voltage; automatic takeoff refused")
    if float(fc.state.bat.value) < args.min_takeoff_battery_v:
        raise RuntimeError(
            f"battery voltage too low for automatic takeoff: "
            f"{float(fc.state.bat.value):.2f} V < {args.min_takeoff_battery_v:.2f} V"
        )

    logger.info("[ROAD] automatic takeoff: switching FC to PROGRAM mode")
    _wait_for_fc_mode(fc, fc.PROGRAM_MODE)
    logger.info("[ROAD] automatic takeoff: requesting unlock")
    fc.unlock()
    unlock_deadline = time.perf_counter() + 5.0
    while time.perf_counter() < unlock_deadline:
        if not fc.connected:
            raise RuntimeError("FC disconnected while waiting for unlock")
        if bool(fc.state.unlock.value):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("FC unlock confirmation timed out")

    logger.info("[ROAD] unlock confirmed; waiting {:.1f}s before takeoff", args.post_unlock_delay_s)
    time.sleep(args.post_unlock_delay_s)
    if not fc.connected or not bool(fc.state.unlock.value):
        raise RuntimeError("FC is no longer unlocked before takeoff")

    logger.info("[ROAD] requesting one-key takeoff to {} cm", args.takeoff_height_cm)
    fc.take_off(args.takeoff_height_cm)
    deadline = time.perf_counter() + args.takeoff_timeout_s
    minimum_height_cm = args.takeoff_height_cm - args.takeoff_height_tolerance_cm
    low_battery_frames = 0
    while time.perf_counter() < deadline:
        if not fc.connected:
            raise RuntimeError("FC disconnected during takeoff")
        if not bool(fc.state.unlock.value):
            raise RuntimeError("FC locked unexpectedly during takeoff")
        battery_v = float(fc.state.bat.value)
        if battery_v < args.min_takeoff_battery_v:
            low_battery_frames += 1
        else:
            low_battery_frames = 0
        if low_battery_frames >= args.takeoff_low_battery_confirm_frames:
            raise RuntimeError(
                f"battery voltage stayed too low during takeoff: "
                f"{battery_v:.2f} V < {args.min_takeoff_battery_v:.2f} V"
            )
        altitude_cm = float(fc.state.alt_add.value)
        if altitude_cm >= minimum_height_cm:
            _wait_for_fc_mode(fc, fc.HOLD_POS_MODE)
            fc.stablize()
            logger.info("[ROAD] takeoff complete at alt_add={:.1f} cm; HOLD_POS active", altitude_cm)
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"takeoff height confirmation timed out: alt_add={float(fc.state.alt_add.value):.1f} cm, "
        f"target={args.takeoff_height_cm} cm"
    )


def _land_and_wait_for_lock(fc, args: argparse.Namespace) -> bool:
    """Stop road-following commands, request native landing, and never air-lock."""
    try:
        if not fc.connected:
            logger.error("[ROAD] FC disconnected; unable to request landing. Take over with the RC immediately.")
            return False
        if not bool(fc.state.unlock.value):
            logger.info("[ROAD] FC is already locked; landing is not required")
            return True

        logger.warning("[ROAD] stopping road following and requesting native in-place landing")
        fc.stablize()
        time.sleep(0.1)
        fc.land()
        deadline = time.perf_counter() + args.landing_timeout_s
        next_land_request = time.perf_counter() + 2.0
        next_status = 0.0
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if not fc.connected:
                logger.error("[ROAD] FC disconnected during landing. Take over with the RC immediately.")
                return False
            if not bool(fc.state.unlock.value):
                logger.info("[ROAD] landing confirmed: FC locked")
                return True
            if now >= next_land_request:
                fc.land()
                next_land_request = now + 2.0
            if now >= next_status:
                logger.info(
                    "[ROAD] landing: alt_add={:.1f} cm unlock={}",
                    float(fc.state.alt_add.value),
                    bool(fc.state.unlock.value),
                )
                next_status = now + 1.0
            time.sleep(0.1)
        logger.error(
            "[ROAD] landing confirmation timed out; land was requested but motors were not force-locked. "
            "Take over with the RC immediately."
        )
        return False
    except Exception as exc:
        logger.exception("[ROAD] landing request failed: {}. Take over with the RC immediately.", exc)
        return False


def _open_camera(args: argparse.Namespace):
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    return cap


def _read_camera(cap) -> tuple[bool, np.ndarray | None]:
    if cap is None or not cap.isOpened():
        return False, None
    ok, frame = cap.read()
    if not ok:
        return False, None
    return True, frame


def _radar_configs(upper_port: str, lower_port: str):
    from FlightController.Components import RadarConfig

    return [
        RadarConfig("upper", 0, (0.0, 0.0), 0.0, port=upper_port),
        RadarConfig("lower", 1, (0.96, 0.15), 0.0, port=lower_port, mount_mirror_y=True),
    ]


class _FCTelemetryTracker:
    """Sample FC state and estimate measured yaw rate from fresh telemetry."""

    def __init__(self) -> None:
        self._last_update_count: int | None = None
        self._last_yaw_deg: float | None = None
        self._last_state_time_s: float | None = None
        self._yaw_rate_deg_s: float | None = None

    def update(self, fc, now_s: float) -> dict[str, object]:
        if fc is None:
            return {"connected": False}
        state = getattr(fc, "state", None)
        if state is None:
            return {"connected": bool(getattr(fc, "connected", False))}

        update_count = int(getattr(state, "update_count", 0))
        state_time_s = _float_or_none(getattr(state, "last_update_monotonic", None))
        yaw_deg = _state_float(state, "yaw")
        if update_count != self._last_update_count and yaw_deg is not None:
            if self._last_yaw_deg is not None and self._last_state_time_s is not None and state_time_s is not None:
                dt_s = state_time_s - self._last_state_time_s
                if 0.001 <= dt_s <= 1.0:
                    yaw_delta_deg = (yaw_deg - self._last_yaw_deg + 180.0) % 360.0 - 180.0
                    self._yaw_rate_deg_s = yaw_delta_deg / dt_s
            self._last_update_count = update_count
            self._last_yaw_deg = yaw_deg
            self._last_state_time_s = state_time_s

        telemetry_age_s = (
            max(0.0, now_s - state_time_s)
            if state_time_s is not None and state_time_s > 0.0
            else None
        )
        return {
            "connected": bool(getattr(fc, "connected", False)),
            "update_count": update_count,
            "telemetry_time_perf_s": state_time_s,
            "telemetry_age_s": _float_or_none(telemetry_age_s),
            "yaw_deg": yaw_deg,
            "yaw_rate_deg_s": _float_or_none(self._yaw_rate_deg_s),
            "roll_deg": _state_float(state, "rol"),
            "pitch_deg": _state_float(state, "pit"),
            "alt_add_cm": _state_float(state, "alt_add"),
            "vel_x_cm_s": _state_float(state, "vel_x"),
            "vel_y_cm_s": _state_float(state, "vel_y"),
            "vel_z_cm_s": _state_float(state, "vel_z"),
            "battery_v": _state_float(state, "bat"),
            "mode": _state_value(state, "mode"),
            "unlock": _state_value(state, "unlock"),
        }


def _log_road_summary(
    *,
    args: argparse.Namespace,
    perception,
    desired,
    planned,
    safe_command,
    safety_reason: str,
    sent: bool,
    radar_fresh,
    fc,
    bypass_planner=None,
    controller_diagnostics: dict[str, object] | None = None,
    perception_age_s: float | None = None,
    perception_stale: bool = False,
    frame_age_s: float | None = None,
    fc_telemetry: dict[str, object] | None = None,
) -> None:
    state = getattr(perception, "road_state", "lost")
    err = float(getattr(perception, "pixel_error", 0.0)) if perception is not None else 0.0
    corr = float(getattr(perception, "corrected_pixel_error", 0.0)) if perception is not None else 0.0
    angle = float(getattr(perception, "centerline_angle", 90.0)) if perception is not None else 90.0
    conf = float(getattr(perception, "confidence", 0.0)) if perception is not None else 0.0
    controller_diagnostics = controller_diagnostics or {}
    fc_telemetry = fc_telemetry or {}
    fc_mode = fc_telemetry.get("mode")
    bypass_state = (
        getattr(getattr(bypass_planner, "state", None), "value", "disabled")
        if bypass_planner is not None
        else "disabled"
    )
    bypass_y = getattr(bypass_planner, "last_target_y_cm", None) if bypass_planner is not None else None
    logger.info(
        "[ROAD] state={} mode=single-road err={:.0f} corr={:.0f} angle={:.0f} conf={:.2f} "
        "ctrl=(angle_err={} px_yaw={} angle_yaw={} speed_scale={}) "
        "desired=(vx={} vy={} yaw={}) planned=(vx={} vy={} yaw={}) safe=(vx={} vy={} yaw={}) "
        "actual=(yaw={} yaw_rate={} vx={} vy={}) ages=(frame={} perception={} stale={}) "
        "bypass={} bypass_y={} safety={} sent={} radar_fresh={} fc_mode={}".format(
            state,
            err,
            corr,
            angle,
            conf,
            _round_or_none(controller_diagnostics.get("angle_error_deg")),
            _round_or_none(controller_diagnostics.get("pixel_yaw_term_deg_s")),
            _round_or_none(controller_diagnostics.get("angle_yaw_term_deg_s")),
            _round_or_none(controller_diagnostics.get("heading_speed_scale"), 2),
            round(desired.vx_cm_s),
            round(desired.vy_cm_s),
            round(desired.yaw_rate_deg_s),
            round(planned.vx_cm_s),
            round(planned.vy_cm_s),
            round(planned.yaw_rate_deg_s),
            round(safe_command.vx_cm_s),
            round(safe_command.vy_cm_s),
            round(safe_command.yaw_rate_deg_s),
            _round_or_none(fc_telemetry.get("yaw_deg")),
            _round_or_none(fc_telemetry.get("yaw_rate_deg_s")),
            _round_or_none(fc_telemetry.get("vel_x_cm_s")),
            _round_or_none(fc_telemetry.get("vel_y_cm_s")),
            _round_or_none(frame_age_s, 3),
            _round_or_none(perception_age_s, 3),
            bool(perception_stale),
            bypass_state,
            _float_or_none(bypass_y),
            safety_reason,
            sent,
            radar_fresh,
            fc_mode if fc_mode is not None else "no-fc",
        )
    )


def _road_record_extra(
    perception,
    camera_ok: bool,
    planned=None,
    bypass_planner=None,
    *,
    controller_diagnostics: dict[str, object] | None = None,
    perception_age_s: float | None = None,
    perception_stale: bool = False,
    frame_age_s: float | None = None,
    fc_telemetry: dict[str, object] | None = None,
) -> dict[str, object]:
    raw_error = _float_or_none(getattr(perception, "pixel_error", None))
    corrected_error = _float_or_none(getattr(perception, "corrected_pixel_error", None))
    centerline_points = list(getattr(perception, "centerline_points", []) or [])
    extra = {
        "camera_ok": bool(camera_ok),
        "road_state": getattr(perception, "road_state", "lost") if perception is not None else "lost",
        "branch": "disabled",
        "pixel_error": raw_error,
        "corrected_pixel_error": corrected_error,
        "offset_correction_px": (
            corrected_error - raw_error
            if corrected_error is not None and raw_error is not None
            else None
        ),
        "centerline_angle": _float_or_none(getattr(perception, "centerline_angle", None)),
        "path_width_px": _float_or_none(getattr(perception, "path_width_px", None)),
        "confidence": _float_or_none(getattr(perception, "confidence", None)),
        "is_road_found": bool(getattr(perception, "is_road_found", False)) if perception is not None else False,
        "debug_msg": getattr(perception, "debug_msg", "") if perception is not None else "",
        "centerline_point_count": len(centerline_points),
        "centerline_first": _point_extra(centerline_points[0]) if centerline_points else None,
        "centerline_last": _point_extra(centerline_points[-1]) if centerline_points else None,
        "perception_age_s": _float_or_none(perception_age_s),
        "perception_stale": bool(perception_stale),
        "frame_age_s": _float_or_none(frame_age_s),
        "controller": dict(controller_diagnostics or {}),
        "fc": dict(fc_telemetry or {}),
    }
    if bypass_planner is not None:
        extra.update(_bypass_record_extra(bypass_planner))
    if planned is not None:
        extra["planned"] = _command_extra(planned)
    return extra


def _annotate_road_frame(
    frame,
    *,
    perception,
    loop_count: int,
    controller_diagnostics: dict[str, object],
    safe_command,
    fc_telemetry: dict[str, object],
    perception_age_s: float | None,
    perception_stale: bool,
):
    if frame is None:
        return None
    try:
        output = np.asarray(frame).copy()
        height, width = output.shape[:2]
        cv2.line(output, (width // 2, 0), (width // 2, height - 1), (0, 255, 0), 1)

        points = list(getattr(perception, "centerline_points", []) or [])
        if len(points) >= 2:
            polyline = np.asarray(
                [[round(float(point[0])), round(float(point[1]))] for point in points],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(output, [polyline], False, (0, 0, 255), 3, cv2.LINE_AA)

        road_state = getattr(perception, "road_state", "lost") if perception is not None else "lost"
        found = bool(getattr(perception, "is_road_found", False)) if perception is not None else False
        confidence = _float_or_none(getattr(perception, "confidence", None))
        pixel_error = _float_or_none(getattr(perception, "corrected_pixel_error", None))
        filtered_pixel_error = _float_or_none(controller_diagnostics.get("filtered_pixel_error_px"))
        angle = _float_or_none(getattr(perception, "centerline_angle", None))
        filtered_angle = _float_or_none(controller_diagnostics.get("centerline_angle_deg"))
        lines = [
            (
                f"loop={loop_count} road={road_state} found={found} conf={_display_float(confidence, 2)} "
                f"age={_display_float(perception_age_s, 3)}s stale={bool(perception_stale)}"
            ),
            (
                f"pixel={_display_float(pixel_error, 1)}->{_display_float(filtered_pixel_error, 1)}px "
                f"angle={_display_float(angle, 1)}->{_display_float(filtered_angle, 1)}deg "
                f"angle_err={_display_float(controller_diagnostics.get('angle_error_deg'), 1)}deg"
            ),
            (
                f"ctrl px_yaw={_display_float(controller_diagnostics.get('pixel_yaw_term_deg_s'), 1)} "
                f"angle_yaw={_display_float(controller_diagnostics.get('angle_yaw_term_deg_s'), 1)} "
                f"scale={_display_float(controller_diagnostics.get('heading_speed_scale'), 2)}"
            ),
            (
                f"safe vx={_display_float(getattr(safe_command, 'vx_cm_s', None), 1)} "
                f"vy={_display_float(getattr(safe_command, 'vy_cm_s', None), 1)} "
                f"yaw={_display_float(getattr(safe_command, 'yaw_rate_deg_s', None), 1)}"
            ),
            (
                f"fc yaw={_display_float(fc_telemetry.get('yaw_deg'), 1)} "
                f"yaw_rate={_display_float(fc_telemetry.get('yaw_rate_deg_s'), 1)} "
                f"vel=({_display_float(fc_telemetry.get('vel_x_cm_s'), 1)},"
                f"{_display_float(fc_telemetry.get('vel_y_cm_s'), 1)})"
            ),
        ]
        overlay_height = min(height, 12 + len(lines) * 20)
        shade = output[:overlay_height].copy()
        shade[:] = 0
        output[:overlay_height] = cv2.addWeighted(output[:overlay_height], 0.35, shade, 0.65, 0.0)
        text_color = (0, 0, 255) if perception_stale or not found else (255, 255, 255)
        for index, line in enumerate(lines):
            cv2.putText(
                output,
                line,
                (8, 18 + index * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
                cv2.LINE_AA,
            )
        return output
    except Exception as exc:
        logger.warning(f"[REC] diagnostic overlay failed: {type(exc).__name__}: {exc}")
        return frame


def _command_extra(command) -> dict[str, object] | None:
    if command is None:
        return None
    return {
        "vx_cm_s": _float_or_none(getattr(command, "vx_cm_s", None)),
        "vy_cm_s": _float_or_none(getattr(command, "vy_cm_s", None)),
        "vz_cm_s": _float_or_none(getattr(command, "vz_cm_s", None)),
        "yaw_rate_deg_s": _float_or_none(getattr(command, "yaw_rate_deg_s", None)),
        "reason": getattr(command, "reason", ""),
    }


def _bypass_record_extra(bypass_planner) -> dict[str, object]:
    state = getattr(getattr(bypass_planner, "state", None), "value", "disabled")
    return {
        "road_bypass_state": state,
        "road_bypass_target_y_cm": _float_or_none(getattr(bypass_planner, "last_target_y_cm", None)),
        "road_bypass_active_side": getattr(bypass_planner, "active_side", None),
    }


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _round_or_none(value, digits: int = 1):
    value = _float_or_none(value)
    return None if value is None else round(value, digits)


def _display_float(value, digits: int = 1) -> str:
    value = _float_or_none(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def _state_value(state, name: str):
    field = getattr(state, name, None)
    return getattr(field, "value", field)


def _state_float(state, name: str) -> float | None:
    return _float_or_none(_state_value(state, name))


def _point_extra(point) -> list[float] | None:
    try:
        values = list(point)
        return [float(values[0]), float(values[1])]
    except (TypeError, ValueError, IndexError):
        return None


def _setup_logging(log_file: str | Path | None) -> int | None:
    if not log_file:
        return None
    path = Path(log_file)
    if str(path).replace("\\", "/").startswith("/tmp/"):
        logger.warning("Avoid writing logs to /tmp on the target board")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sink_id = logger.add(
            str(path),
            enqueue=True,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        )
    except (OSError, ValueError) as exc:
        logger.warning(f"[ROAD] runtime file logging disabled for {path}: {exc}")
        return None
    logger.info(f"[ROAD] runtime log: {path}")
    return sink_id


def _debug_image_path(debug_dir: str | None, every: int, loop_count: int) -> str | None:
    if not debug_dir or every <= 0 or loop_count % every != 0:
        return None
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    return str(path / f"road_{loop_count:06d}.jpg")


def _sleep_to_rate(loop_start: float, period_s: float) -> None:
    elapsed = time.perf_counter() - loop_start
    if elapsed < period_s:
        time.sleep(period_s - elapsed)


if __name__ == "__main__":
    main()
