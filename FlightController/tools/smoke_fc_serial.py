import argparse
from pathlib import Path
import sys


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root, root.parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def main() -> None:
    _setup_path()

    from FlightController import FC_Controller

    parser = argparse.ArgumentParser(description="Smoke test Lingxiao flight controller serial connection.")
    parser.add_argument("--port", default=None, help="Serial device path. Defaults to resolver auto-detect.")
    args = parser.parse_args()

    fc = FC_Controller()
    try:
        fc.start_listen_serial(serial_dev=args.port, block_until_connected=True)
        fc.wait_for_connection()
        state = fc.state
        print(
            "FC STATE:",
            f"mode={state.mode.value}",
            f"unlock={state.unlock.value}",
            f"bat={state.bat.value:.2f}V",
            f"yaw={state.yaw.value:.2f}deg",
            f"alt_add={state.alt_add.value}cm",
        )
    finally:
        fc.close()


if __name__ == "__main__":
    main()
