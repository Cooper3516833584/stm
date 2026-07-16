"""Capture one downward road-camera frame at a 1 m fixed-point hover.

This is a flight test utility.  It uses the existing Navigation.pointing_takeoff
routine with only the upper radar, and fixes the navigation waypoint at the
radar origin so it does not request horizontal movement.  By default it only
prints the planned operation; physical actions require the explicit
``--execute`` flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2


DEFAULT_OUTPUT = "/home/root/Desktop/ObstacleAvoidanceDrone/road_camera_video7_at_1m.jpg"


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root, root.parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-point 1 m flight: capture one /dev/video7 road-camera frame, then land."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually unlock, take off, capture, and land. Omit this flag for a no-hardware dry run.",
    )
    parser.add_argument("--port", default=None, help="Flight-controller serial path; default is auto-detect.")
    parser.add_argument("--upper-radar-port", default="/dev/ttySTM4", help="Upper-radar serial path.")
    parser.add_argument("--radar-ready-timeout-s", type=float, default=12.0)
    parser.add_argument("--camera-index", type=int, default=7, help="Road camera OpenCV index (default: 7).")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--target-height-cm", type=int, default=100, help="Target hover height from alt_add.")
    parser.add_argument("--height-tolerance-cm", type=int, default=8)
    parser.add_argument("--climb-speed-cm-s", type=int, default=20)
    parser.add_argument("--first-lift-cm", type=int, default=60)
    parser.add_argument("--hover-settle-s", type=float, default=2.0, help="Time to remain in HOLD_POS before capture.")
    parser.add_argument("--camera-warmup-frames", type=int, default=10)
    parser.add_argument("--landing-timeout-s", type=float, default=25.0)
    parser.add_argument("--landing-alt-threshold-cm", type=float, default=10.0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JPEG output path on the board.")
    return parser.parse_args()


def _open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open road camera /dev/video{args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    for _ in range(max(1, args.camera_warmup_frames)):
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Road camera preflight read failed")
    return cap


def _capture_frame(cap: cv2.VideoCapture, args: argparse.Namespace) -> object:
    for _ in range(max(1, args.camera_warmup_frames)):
        cap.grab()
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Road camera capture failed")
    return frame


def _wait_for_upper_radar(radar, timeout_s: float) -> None:
    deadline = time.perf_counter() + max(0.1, timeout_s)
    while time.perf_counter() < deadline:
        if radar.connected and radar.is_fresh(max_age_s=0.5):
            stats = radar.get_radar_latency_stats()
            print(
                "Upper radar ready: "
                f"frames={stats.get('serial_frames_ok', 0)}, "
                f"latest_age={float(stats.get('last_sample_age_ms', 0.0)):.0f}ms"
            )
            return
        time.sleep(0.1)
    raise RuntimeError(f"Upper radar is not fresh on {radar._serial.port if radar._serial else 'unknown port'}")


def _land_and_lock_when_safe(fc, args: argparse.Namespace) -> bool:
    """Request landing; only lock after altitude confirms ground proximity."""
    print("Requesting fixed-point landing...")
    try:
        fc.stablize()
        fc.set_flight_mode(fc.PROGRAM_MODE)
        fc.land()
        deadline = time.perf_counter() + max(1.0, args.landing_timeout_s)
        while time.perf_counter() < deadline:
            alt_cm = float(getattr(fc.state.alt_add, "value", 9999.0))
            unlocked = bool(getattr(fc.state.unlock, "value", False))
            if not unlocked:
                print("Landing confirmed: flight controller is locked.")
                return True
            if alt_cm <= args.landing_alt_threshold_cm:
                fc.lock()
                fc.wait_for_lock(timeout_s=5)
                print(f"Landing confirmed at alt_add={alt_cm:.1f} cm; motors locked.")
                return True
            time.sleep(0.1)
        print("ERROR: landing confirmation timed out; land command was sent, but motors were NOT force-locked.")
        return False
    except Exception as exc:
        print(f"ERROR: landing request failed: {exc}")
        return False


def _print_plan(args: argparse.Namespace) -> None:
    print("No-hardware dry run. Add --execute to arm the flight test.")
    print(f"camera=/dev/video{args.camera_index} ({args.camera_width}x{args.camera_height}@{args.camera_fps:g})")
    print(f"upper radar={args.upper_radar_port}; hover target={args.target_height_cm} cm (alt_add)")
    print(f"output={args.output}")
    print("sequence: camera/radar preflight -> Navigation.pointing_takeoff at radar origin -> JPEG capture -> land")


def main() -> int:
    args = parse_args()
    if not 40 <= args.target_height_cm <= 500:
        raise SystemExit("--target-height-cm must be in [40, 500]")
    if args.height_tolerance_cm < 0:
        raise SystemExit("--height-tolerance-cm must be non-negative")
    if not args.execute:
        _print_plan(args)
        return 0

    _setup_path()
    from FlightController import FC_Controller
    from FlightController.Components.LDRadar_Driver import LD_Radar
    from FlightController.Solutions.Navigation import Navigation

    output = Path(args.output)
    cap: cv2.VideoCapture | None = None
    fc = None
    radar = None
    navigation = None
    airborne = False
    try:
        # Fail before arming if the road camera cannot deliver frames.
        cap = _open_camera(args)
        print(
            "Camera preflight OK: "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"@{cap.get(cv2.CAP_PROP_FPS):.1f} fps"
        )

        fc = FC_Controller()
        fc.start_listen_serial(serial_dev=args.port, block_until_connected=True)
        fc.wait_for_connection()
        print(
            "FC connected: "
            f"alt_add={fc.state.alt_add.value}cm alt_fused={fc.state.alt_fused.value}cm"
        )

        # Only the upper radar is opened.  Its current solved pose becomes the
        # navigation origin, so the (0, 0) waypoint does not request translation.
        radar = LD_Radar(name="upper", index=0, mount_xy_cm=(0.0, 0.0), mount_yaw_deg=0.0)
        radar.start(com=args.upper_radar_port, radar_type="D500")
        _wait_for_upper_radar(radar, args.radar_ready_timeout_s)

        navigation = Navigation(fc=fc, radar=radar)
        navigation.start(mode="radar")
        navigation.calibrate_basepoint(wait=True)
        navigation.pointing_takeoff(
            point=(0.0, 0.0),
            target_height=args.target_height_cm,
            first_lift=args.first_lift_cm,
            lock_pos_thres=15,
            lock_pos_time=1.0,
            lock_timeout=12,
            hover_timeout=12,
            height_timeout=15,
        )
        airborne = True
        fc.stablize()

        alt_cm = float(fc.state.alt_add.value)
        if alt_cm < args.target_height_cm - args.height_tolerance_cm:
            raise RuntimeError(
                f"Takeoff did not reach target: alt_add={alt_cm:.1f}cm, "
                f"required >= {args.target_height_cm - args.height_tolerance_cm}cm"
            )
        print(f"Hovering in HOLD_POS: alt_add={alt_cm:.1f} cm; settling for {args.hover_settle_s:.1f} s")
        time.sleep(max(0.0, args.hover_settle_s))

        frame = _capture_frame(cap, args)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame):
            raise RuntimeError(f"Failed to write JPEG: {output}")
        print(f"Captured road-camera frame: {output} (alt_add={fc.state.alt_add.value}cm)")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; requesting landing if takeoff was started.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        if cap is not None:
            cap.release()
        if navigation is not None:
            navigation.stop(join=True)
        if radar is not None:
            radar.stop(joined=True)
        if fc is not None:
            if airborne or bool(getattr(fc.state.unlock, "value", False)):
                _land_and_lock_when_safe(fc, args)
            fc.close()


if __name__ == "__main__":
    raise SystemExit(main())
