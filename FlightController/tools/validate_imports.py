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
    from FlightController.Components import CameraConfig, CameraSource
    from FlightController.Components import LD_Radar, MultiRadar, RadarConfig
    from FlightController.Components.RealSense import T265_Pose_Frame
    from FlightController.Solutions.AutonomousNavigator import AutonomousNavigatorConfig
    from FlightController.Solutions.LocalPlanner import LocalPlanner
    from FlightController.Solutions.TargetDetector import TargetDetector
    from FlightController.Solutions.Navigation import Navigation

    _ = (
        FC_Controller,
        LD_Radar,
        MultiRadar,
        RadarConfig,
        CameraSource,
        CameraConfig,
        T265_Pose_Frame,
        Navigation,
        TargetDetector,
        LocalPlanner,
        AutonomousNavigatorConfig,
    )
    print("IMPORT VALIDATION OK")


if __name__ == "__main__":
    main()
