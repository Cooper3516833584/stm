import argparse
from pathlib import Path
import sys
import time


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root, root.parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def main() -> None:
    _setup_path()

    from FlightController.Components import MultiRadar, RadarConfig

    parser = argparse.ArgumentParser(description="Smoke test two direct-connected D500 radars.")
    parser.add_argument("--front-port", default=None, help="Front radar serial device path.")
    parser.add_argument("--rear-port", default=None, help="Rear radar serial device path.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Test duration.")
    args = parser.parse_args()

    configs = [
        RadarConfig(
            name="front",
            index=0,
            mount_xy_cm=(0.0, 12.0),
            mount_yaw_deg=0.0,
            port=args.front_port,
        ),
        RadarConfig(
            name="rear",
            index=1,
            mount_xy_cm=(0.0, -12.0),
            mount_yaw_deg=180.0,
            port=args.rear_port,
        ),
    ]
    multi_radar = MultiRadar(configs)
    try:
        multi_radar.start()
        end_time = time.perf_counter() + args.seconds
        while time.perf_counter() < end_time:
            points = multi_radar.get_obstacle_points_body_cm()
            print(f"dual_radar connected={multi_radar.connected} points={len(points)}")
            time.sleep(0.5)
    finally:
        multi_radar.stop()


if __name__ == "__main__":
    main()
