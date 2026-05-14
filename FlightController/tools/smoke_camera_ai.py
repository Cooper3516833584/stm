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

    from FlightController.Components import CameraConfig, CameraSource
    from FlightController.Solutions.TargetDetector import TargetDetector

    parser = argparse.ArgumentParser(description="Smoke test headless camera and AI target detector.")
    parser.add_argument("--device", default=None, help="Camera device path or index.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Test duration.")
    parser.add_argument("--class-name", default=None, help="Optional target class filter.")
    args = parser.parse_args()

    device = _parse_device(args.device)
    camera = CameraSource(CameraConfig(device=device))
    try:
        camera.open()
        detector = TargetDetector()
        end_time = time.perf_counter() + args.seconds
        while time.perf_counter() < end_time:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("camera read failed")
                time.sleep(0.1)
                continue
            result = detector.detect_best(frame, class_name=args.class_name)
            print("detection:", result)
            time.sleep(0.1)
    finally:
        camera.close()


def _parse_device(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
