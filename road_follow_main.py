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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Road-following dry-run / flight entry")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera", default=None, help="Deprecated alias; use --camera-index for numeric V4L2 devices")
    parser.add_argument("--width", type=int, default=None, help="Deprecated alias for --camera-width")
    parser.add_argument("--height", type=int, default=None, help="Deprecated alias for --camera-height")
    parser.add_argument("--fps", type=int, default=None, help="Deprecated alias for --camera-fps")
    parser.add_argument("--model", default="FlightController/Solutions/model/road_yolo11n_seg.onnx")
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
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--debug-every-n", type=int, default=30)
    parser.add_argument("--debug-image-dir", default=None, help="Deprecated alias for --debug-dir")
    parser.add_argument("--debug-image-every", type=int, default=None, help="Deprecated alias for --debug-every-n")
    parser.add_argument("--radar-timeout-s", type=float, default=0.5)
    parser.add_argument("--max-distance-cm", type=float, default=300.0)
    parser.add_argument("--body-x-half-cm", type=float, default=25.0)
    parser.add_argument("--body-y-half-cm", type=float, default=25.0)
    parser.add_argument("--corridor-half-width-cm", type=float, default=50.0)
    parser.add_argument("--max-vx-cm-s", type=float, default=25.0)
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=25.0)
    parser.add_argument("--offset-comp-enable", action="store_true")
    parser.add_argument("--enable-offset-comp", action="store_true", help="Deprecated alias for --offset-comp-enable")
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
    from road_perception import CameraOffsetCompensationConfig
    from FlightController.Components import MultiRadar, RadarConfig
    from FlightController.Components.FCConnector import FCConnectConfig, connect_fc
    from FlightController.Solutions.RoadFollower import RoadFollower, RoadFollowerConfig
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
    model_missing_logged = False
    if not os.path.isfile(args.model):
        msg = f"[ROAD] model missing, perception lost: {args.model}"
        if args.require_model:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        model_missing_logged = True

    actual_dry_run = bool(args.dry_run or args.no_fc or not args.enable_flight)
    if actual_dry_run:
        logger.warning("[SAFETY] dry-run mode: no non-zero velocity will be sent. Add --enable-flight to allow real output")
    if args.no_radar:
        logger.warning("[SAFETY] no-radar mode, ground vision debug only; not recommended for flight")

    fc = None
    multi_radar = None
    cap = None
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

        cap = _open_camera(args)
        if cap is None:
            logger.warning("[ROAD] camera open failed; perception will stay lost")

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
            ok, frame = _read_camera(cap)
            debug_path = _debug_image_path(args.debug_dir, args.debug_every_n, loop_count)

            if not ok or frame is None:
                perception = None
                desired = Command.zero("camera_failed")
            else:
                try:
                    perception = road_perception.get_road_perception(
                        frame,
                        debug_save_path=debug_path,
                        offset_comp_config=offset_comp,
                        branch_preference=args.branch,
                        previous_branch_label=follower.previous_branch_label(),
                    )
                    if not model_missing_logged and "ONNX model not found" in getattr(perception, "debug_msg", ""):
                        logger.warning(f"[ROAD] model missing, perception lost: {args.model}")
                        model_missing_logged = True
                    desired = follower.update(perception, now_s=loop_start)
                except Exception as exc:
                    logger.warning(f"[ROAD] perception lost: {type(exc).__name__}: {exc}")
                    perception = None
                    desired = follower.update(None, now_s=loop_start)

            if multi_radar is not None:
                points = multi_radar.get_obstacle_points_body_cm(max_distance_cm=args.max_distance_cm)
                radar_field.update(points, loop_start)
                radar_age_s = multi_radar_age_s(multi_radar)
                radar_connected = bool(multi_radar.connected and multi_radar.is_fresh(max_age_s=args.radar_timeout_s))
            else:
                radar_field.update(np.empty((0, 2), dtype=float), loop_start)
                radar_age_s = 0.0
                radar_connected = True

            health = flight_health_from_sources(
                fc=fc,
                multi_radar=multi_radar,
                radar_timeout_s=args.radar_timeout_s,
                camera_ok=bool(ok),
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
                _log_road_summary(
                    args=args,
                    perception=perception,
                    desired=desired,
                    safe_command=safe.command,
                    safety_reason=decision.reason,
                    sent=bool(not actual_dry_run and fc is not None),
                    radar_fresh=radar_connected if multi_radar is not None else "disabled",
                    fc=fc,
                )

            loop_count += 1
            _sleep_to_rate(loop_start, period_s)
    except KeyboardInterrupt:
        logger.info("[ROAD] interrupted")
    finally:
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
        if cap is not None:
            cap.release()
        if multi_radar is not None:
            multi_radar.stop()
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
    safe_command,
    safety_reason: str,
    sent: bool,
    radar_fresh,
    fc,
) -> None:
    selected = getattr(perception, "selected_branch", None)
    branch = getattr(selected, "label", "none")
    state = getattr(perception, "road_state", "lost")
    err = float(getattr(perception, "pixel_error", 0.0)) if perception is not None else 0.0
    corr = float(getattr(perception, "corrected_pixel_error", 0.0)) if perception is not None else 0.0
    angle = float(getattr(perception, "centerline_angle", 90.0)) if perception is not None else 90.0
    conf = float(getattr(perception, "confidence", 0.0)) if perception is not None else 0.0
    fc_mode = getattr(getattr(getattr(fc, "state", None), "mode", None), "value", None)
    logger.info(
        "[ROAD] state={} branch={} err={:.0f} corr={:.0f} angle={:.0f} conf={:.2f} "
        "desired=(vx={} yaw={}) safe=(vx={} yaw={}) safety={} sent={} radar_fresh={} fc_mode={}".format(
            state,
            branch if branch != "none" else args.branch,
            err,
            corr,
            angle,
            conf,
            round(desired.vx_cm_s),
            round(desired.yaw_rate_deg_s),
            round(safe_command.vx_cm_s),
            round(safe_command.yaw_rate_deg_s),
            safety_reason,
            sent,
            radar_fresh,
            fc_mode if fc_mode is not None else "no-fc",
        )
    )


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
