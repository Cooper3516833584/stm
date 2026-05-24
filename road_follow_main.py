"""Main placeholder program for YOLO road-line following.

All safety decisions pass through SafetyArbiter before any FC command is sent.
Automatic landing is intentionally not implemented here; landing remains manual
for this phase.
"""

from __future__ import annotations

import argparse
import time

from loguru import logger

from autonomy_context import AutonomyContext, SensorHealth, flight_status_from_fc
from autonomy_hardware import build_dual_radar, connect_fc, open_camera, send_fc_command, stop_fc
from local_world_model import LocalWorldModel, LocalWorldModelConfig
from road_follow_mission import RoadFollowConfig, RoadFollowMission
from safety_arbiter import SafetyArbiter, SafetyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO road-line following placeholder")
    parser.add_argument("--enable-flight", action="store_true", help="Actually send commands to FC")
    parser.add_argument("--connect-fc", action="store_true", help="Connect FC even in dry-run")
    parser.add_argument("--fc-port", default=None)
    parser.add_argument("--upper-port", default="/dev/ttySTM4")
    parser.add_argument("--lower-port", default="/dev/ttySTM9")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loop-hz", type=float, default=20.0)
    parser.add_argument("--max-distance-cm", type=float, default=300.0)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--debug-image", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_file:
        logger.add(args.log_file, enqueue=True, level="DEBUG")

    from road_perception import get_road_perception

    enable_flight = bool(args.enable_flight)
    fc = None
    radar = None
    camera = None
    world = LocalWorldModel(LocalWorldModelConfig(max_distance_cm=args.max_distance_cm))
    mission = RoadFollowMission(RoadFollowConfig(image_width_px=float(args.width)))
    safety = SafetyArbiter(SafetyConfig())
    period_s = 1.0 / max(args.loop_hz, 0.1)

    try:
        if enable_flight or args.connect_fc:
            fc = connect_fc(args.fc_port)
        radar = build_dual_radar(args.upper_port, args.lower_port)
        radar.start()
        camera = open_camera(args.camera, args.width, args.height, args.fps)

        logger.info(f"[ROAD] started enable_flight={enable_flight}")
        while True:
            loop_start = time.perf_counter()
            ok, frame = camera.read()
            road_result = None
            if ok and frame is not None:
                road_result = get_road_perception(frame, debug_save_path=args.debug_image)

            points = radar.get_obstacle_points_body_cm(max_distance_cm=args.max_distance_cm)
            snapshot = world.update_from_radar_points(points, loop_start)
            radar_age = world.radar_age_s(loop_start)
            context = AutonomyContext(
                now_s=loop_start,
                health=SensorHealth(
                    radar_ok=bool(radar.connected),
                    camera_ok=bool(ok),
                    fc_ok=bool(getattr(fc, "connected", False)) if fc is not None else False,
                    radar_age_s=radar_age,
                    camera_age_s=0.0 if ok else None,
                ),
                flight=flight_status_from_fc(fc),
                road=road_result,
                obstacles=snapshot.obstacles,
            )
            desired = mission.update(road_result, loop_start)
            safe = safety.filter(desired, context=context, world=world)
            send_fc_command(fc, safe.command, enable_flight)

            logger.info(
                "[ROAD] road={} desired={} safe={} safety={} obs={} nearest={}".format(
                    getattr(road_result, "road_state", "none"),
                    desired.as_fc_tuple(),
                    safe.command.as_fc_tuple(),
                    safe.state,
                    len(snapshot.obstacles),
                    safe.nearest_forward_obstacle_cm,
                )
            )

            elapsed = time.perf_counter() - loop_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
    except KeyboardInterrupt:
        logger.info("[ROAD] interrupted by user")
    finally:
        stop_fc(fc, enable_flight)
        if camera is not None:
            camera.close()
        if radar is not None:
            radar.stop()
        logger.info("[ROAD] stopped")


if __name__ == "__main__":
    main()

