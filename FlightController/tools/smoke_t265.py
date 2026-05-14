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

    from FlightController.Components.RealSense import T265

    t265 = None
    try:
        t265 = T265(connection="raw")
        t265.start(print_update=False)
        end_time = time.perf_counter() + 5.0
        while time.perf_counter() < end_time:
            print("T265:", t265.get_xy_yaw_cm())
            time.sleep(0.5)
    finally:
        if t265 is not None and getattr(t265, "running", False):
            t265.stop()


if __name__ == "__main__":
    main()
