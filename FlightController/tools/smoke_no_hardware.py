from pathlib import Path
import sys

import numpy as np


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root, root.parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def main() -> None:
    _setup_path()

    from FlightController import FC_Controller
    from FlightController.Components import MultiRadar, RadarConfig
    from FlightController.Solutions.LocalPlanner import LocalPlanner

    fc = FC_Controller()
    configs = [
        RadarConfig(name="front", index=0, mount_xy_cm=(0.0, 12.0), mount_yaw_deg=0.0),
        RadarConfig(name="rear", index=1, mount_xy_cm=(0.0, -12.0), mount_yaw_deg=180.0),
    ]
    multi_radar = MultiRadar(configs)
    planner = LocalPlanner()
    command = planner.plan(obstacles_body_cm=np.empty((0, 2)), target=None)
    assert command.reason == "no_target"
    _ = (fc, multi_radar)
    print("NO HARDWARE SMOKE OK")


if __name__ == "__main__":
    main()
