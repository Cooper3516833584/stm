"""YOLO road-following entry point.

Default behavior is dry-run. Non-zero FC commands are sent only when
--enable-flight is explicitly provided. This file does not unlock, take off,
or land the aircraft.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import cv2
import numpy as np
from loguru import logger

from perception_pipeline import PerceptionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Road-following dry-run / flight entry")
    parser.add_argument("--camera-index", type=int, default=9)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera", default=None, help="Deprecated alias; use --camera-index for numeric V4L2 devices")
    parser.add_argument("--width", type=int, default=None, help="Deprecated alias for --camera-width")
    parser.add_argument("--height", type=int, default=None, help="Deprecated alias for --camera-height")
    parser.add_argument("--fps", type=int, default=None, help="Deprecated alias for --camera-fps")
    parser.add_argument("--model", default="FlightController/Solutions/model/road_yolo11n_seg.onnx")
    parser.add_argument("--model-npu", default=None,
                        help=".nb NPU compiled model path (overrides --model)")
    parser.add_argument("--model-path", default=None, help="Deprecated alias for --model")
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--fc-port", default=None)
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--no-fc", action="store_true")
    parser.add_argument("--connect-fc", action="store_true", help="Connect FC for status only")
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-flight", action="store_true")
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--branch", choices=["auto", "straight", "left", "right"], default="auto")
    parser.add_argument("--branch-preference", choices=["auto", "straight", "left", "right"], default=None)
    parser.add_argument("--branch-policy", choices=["center", "left", "right"], default=None, help="Deprecated alias")
    parser.add_argument("--wb-enable", action="store_true",
                        help="Enable software white balance correction for camera color cast")
    parser.add_argument("--wb-r", type=float, default=2.78,
                        help="White balance R channel gain (default: 2.78 for cam#9 cyan cast)")
    parser.add_argument("--wb-g", type=float, default=1.00,
                        help="White balance G channel gain (default: 1.00)")
    parser.add_argument("--wb-b", type=float, default=1.26,
                        help="White balance B channel gain (default: 1.26 for cam#9 cyan cast)")
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--debug-every-n", type=int, default=30)
    parser.add_argument("--debug-image-dir", default=None, help="Deprecated alias for --debug-dir")
    parser.add_argument("--debug-image-every", type=int, default=None, help="Deprecated alias for --debug-every-n")
    parser.add_argument("--record-dir", default="/media/sdcard/stm_records",
                        help="Directory on SD card for session recording")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable camera/radar session recording")
    parser.add_argument("--record-frame-every-n", type=int, default=10,
                        help="Save one camera frame every N control loops")
    parser.add_argument("--record-radar-every-n", type=int, default=1,
                        help="Save one radar metadata/point snapshot every N control loops")
    parser.add_argument("--record-jpeg-quality", type=int, default=85)
    parser.add_argument("--radar-timeout-s", type=float, default=0.5)
    parser.add_argument("--max-distance-cm", type=float, default=300.0)
    parser.add_argument("--body-x-half-cm", type=float, default=25.0)
    parser.add_argument("--body-y-half-cm", type=float, default=25.0)
    parser.add_argument("--corridor-half-width-cm", type=float, default=50.0)
    parser.add_argument("--max-vx-cm-s", type=float, default=25.0)
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=25.0)
    parser.add_argument("--road-bypass-enable", action="store_true",
                        help="Enable radar-assisted in-road bypass for branches/vines intruding into the road center")
    parser.add_argument("--road-half-width-cm", type=float, default=120.0)
    parser.add_argument("--road-edge-margin-cm", type=float, default=25.0)
    parser.add_argument("--road-bypass-lookahead-cm", type=float, default=180.0)
    parser.add_argument("--road-bypass-min-x-cm", type=float, default=40.0)
    parser.add_argument("--road-bypass-intrusion-half-width-cm", type=float, default=80.0)
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
    parser.add_argument("--flight-height-m", type=float, default=2.0,
                        help="飞行高度 (m), 用于计算该高度的 meters-per-pixel")
    parser.add_argument("--cam-forward-offset-m", type=float, default=0.10)
    parser.add_argument("--meters-per-pixel-x", type=float, default=None)
    parser.add_argument("--offset-correction-sign", type=float, default=1.0)
    parser.add_argument("--offset-max-correction-px", type=float, default=120.0)
    parser.add_argument("--pipeline-latency-s", type=float, default=0.0)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args = _normalize_args(args)
    _setup_logging(args.log_file)

    import road_perception
    from road_perception import CameraOffsetCompensationConfig, CameraWhiteBalanceConfig
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

    road_perception.MODEL_PATH = args.model
    if args.model_npu:
        road_perception.MODEL_PATH_NPU = args.model_npu
        road_perception._AUTO_USE_NPU = True
        logger.info("NPU .nb model configured: {}", args.model_npu)
    if not os.path.isfile(args.model):
        msg = f"[ROAD] model missing, perception lost: {args.model}"
        if args.require_model:
            raise FileNotFoundError(msg)
        logger.warning(msg)

    actual_dry_run = bool(args.dry_run or args.no_fc or not args.enable_flight)
    if actual_dry_run:
        logger.warning("[SAFETY] dry-run mode: no non-zero velocity will be sent. Add --enable-flight to allow real output")
    if args.no_radar:
        logger.warning("[SAFETY] no-radar mode, ground vision debug only; not recommended for flight")

    fc = None
    multi_radar = None
    recorder = SessionRecorder(
        SessionRecorderConfig(
            root_dir=args.record_dir,
            enabled=not args.no_record,
            mode="road_follow",
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
            forward_corridor_half_width_cm=args.corridor_half_width_cm,
        )
    )
    follower = RoadFollower(
        RoadFollowerConfig(
            image_width=args.camera_width,
            max_vx_cm_s=args.max_vx_cm_s,
            max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
            branch_preference=args.branch,
        )
    )
    bypass_planner = RoadObstacleBypassPlanner(
        RoadBypassConfig(
            enabled=bool(args.road_bypass_enable),
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
            require_radar=not args.no_radar,
            radar_timeout_s=args.radar_timeout_s,
            max_vx_cm_s=args.max_vx_cm_s,
            max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
        )
    )
    wb_config = CameraWhiteBalanceConfig(
        enabled=bool(args.wb_enable),
        r_gain=args.wb_r,
        g_gain=args.wb_g,
        b_gain=args.wb_b,
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

    try:
        if not args.no_fc:
            fc = connect_fc(FCConnectConfig(port=args.fc_port, mode=2, timeout_s=10.0))
            logger.info("[ROAD] FC connected and switched to HOLD_POS mode; no unlock/takeoff is performed")

        if not args.no_radar:
            multi_radar = MultiRadar(_radar_configs(args.upper_port, args.lower_port))
            multi_radar.start()

        pipeline = PerceptionPipeline(
            camera_index=args.camera_index,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            model_path=args.model,
            flight_height_m=args.flight_height_m,
            branch_preference=args.branch,
            wb_enable=bool(args.wb_enable),
            wb_r=args.wb_r,
            wb_g=args.wb_g,
            wb_b=args.wb_b,
        )
        pipeline.start()

        logger.info(
            "[ROAD] started dry_run={} no_radar={} branch={} camera={} model={}".format(
                actual_dry_run,
                args.no_radar,
                args.branch,
                args.camera_index,
                args.model,
            )
        )
        loop_count = 0
        last_log_s = 0.0
        while True:
            loop_start = time.perf_counter()

            # ── 1. Non-blocking: read latest perception + frame from pipeline
            perception, _percept_age, percept_stale = pipeline.latest_perception()
            camera_ok = pipeline.camera_ok

            # Recording frame (decimated inside SessionRecorder)
            frame, _frame_ts = pipeline.latest_frame()
            recorder.record_frame(loop_count=loop_count, now_s=loop_start, frame=frame, label="road")
            debug_path = _debug_image_path(args.debug_dir, args.debug_every_n, loop_count)

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

            # ── 5. Recording (unchanged)
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
                extra=_road_record_extra(perception, camera_ok, planned, bypass_planner),
            )
            recorder.record_command(
                loop_count=loop_count,
                now_s=loop_start,
                desired=desired,
                safe_command=safe.command,
                decision_reason=decision.reason,
                extra={
                    "camera_ok": bool(camera_ok),
                    **_bypass_record_extra(bypass_planner),
                    "planned": _command_extra(planned),
                },
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
                )

            loop_count += 1
            _sleep_to_rate(loop_start, period_s)
    except KeyboardInterrupt:
        logger.info("[ROAD] interrupted")
    finally:
        pipeline.stop()
        if fc is not None:
            try:
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
        if multi_radar is not None:
            multi_radar.stop()
        recorder.close()
        logger.info("[ROAD] stopped")


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.width is not None:
        args.camera_width = args.width
    if args.height is not None:
        args.camera_height = args.height
    if args.fps is not None:
        args.camera_fps = args.fps
    if args.model_path:
        args.model = args.model_path
    if args.debug_image_dir:
        args.debug_dir = args.debug_image_dir
    if args.debug_image_every is not None:
        args.debug_every_n = args.debug_image_every
    if args.branch_policy == "center":
        args.branch = "straight"
    elif args.branch_policy in {"left", "right"}:
        args.branch = args.branch_policy
    elif args.branch_preference:
        args.branch = args.branch_preference
    if args.camera is not None:
        try:
            args.camera_index = int(args.camera)
        except ValueError:
            args.camera_index = args.camera
    return args


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
) -> None:
    selected = getattr(perception, "selected_branch", None)
    branch = getattr(selected, "label", "none")
    state = getattr(perception, "road_state", "lost")
    err = float(getattr(perception, "pixel_error", 0.0)) if perception is not None else 0.0
    corr = float(getattr(perception, "corrected_pixel_error", 0.0)) if perception is not None else 0.0
    angle = float(getattr(perception, "centerline_angle", 90.0)) if perception is not None else 90.0
    conf = float(getattr(perception, "confidence", 0.0)) if perception is not None else 0.0
    fc_mode = getattr(getattr(getattr(fc, "state", None), "mode", None), "value", None)
    bypass_state = (
        getattr(getattr(bypass_planner, "state", None), "value", "disabled")
        if bypass_planner is not None
        else "disabled"
    )
    bypass_y = getattr(bypass_planner, "last_target_y_cm", None) if bypass_planner is not None else None
    logger.info(
        "[ROAD] state={} branch={} err={:.0f} corr={:.0f} angle={:.0f} conf={:.2f} "
        "desired=(vx={} yaw={}) planned=(vx={} yaw={}) safe=(vx={} yaw={}) "
        "bypass={} bypass_y={} safety={} sent={} radar_fresh={} fc_mode={}".format(
            state,
            branch if branch != "none" else args.branch,
            err,
            corr,
            angle,
            conf,
            round(desired.vx_cm_s),
            round(desired.yaw_rate_deg_s),
            round(planned.vx_cm_s),
            round(planned.yaw_rate_deg_s),
            round(safe_command.vx_cm_s),
            round(safe_command.yaw_rate_deg_s),
            bypass_state,
            _float_or_none(bypass_y),
            safety_reason,
            sent,
            radar_fresh,
            fc_mode if fc_mode is not None else "no-fc",
        )
    )


def _road_record_extra(perception, camera_ok: bool, planned=None, bypass_planner=None) -> dict[str, object]:
    selected = getattr(perception, "selected_branch", None)
    branch = getattr(selected, "label", "none")
    extra = {
        "camera_ok": bool(camera_ok),
        "road_state": getattr(perception, "road_state", "lost") if perception is not None else "lost",
        "branch": branch,
        "pixel_error": _float_or_none(getattr(perception, "pixel_error", None)),
        "corrected_pixel_error": _float_or_none(getattr(perception, "corrected_pixel_error", None)),
        "centerline_angle": _float_or_none(getattr(perception, "centerline_angle", None)),
        "confidence": _float_or_none(getattr(perception, "confidence", None)),
        "is_road_found": bool(getattr(perception, "is_road_found", False)) if perception is not None else False,
    }
    if bypass_planner is not None:
        extra.update(_bypass_record_extra(bypass_planner))
    if planned is not None:
        extra["planned"] = _command_extra(planned)
    return extra


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
        return float(value)
    except (TypeError, ValueError):
        return None


def _setup_logging(log_file: str | None) -> None:
    if not log_file:
        return
    path = Path(log_file)
    if str(path).replace("\\", "/").startswith("/tmp/"):
        logger.warning("Avoid writing logs to /tmp on the target board")
    logger.add(str(path), enqueue=True, level="DEBUG")


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
