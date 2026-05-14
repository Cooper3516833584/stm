import argparse
import compileall
import importlib
from pathlib import Path
import sys

import numpy as np


REQUIRED_MODULES = [
    ("loguru", "loguru"),
    ("pyserial", "serial"),
    ("attrs", "attr"),
    ("simple-pid", "simple_pid"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("opencv-python/debian-opencv", "cv2"),
    ("matplotlib", "matplotlib"),
    ("onnxruntime", "onnxruntime"),
    ("pyrealsense2", "pyrealsense2"),
]


def _setup_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    for path in (root, root.parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


def _check_python() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError(f"Python >= 3.11 is required, got {sys.version}")
    print(f"[OK] Python: {sys.version.split()[0]}")


def _check_modules(skip_pyrealsense2: bool) -> None:
    missing = []
    for label, import_name in REQUIRED_MODULES:
        if skip_pyrealsense2 and import_name == "pyrealsense2":
            print("[SKIP] pyrealsense2")
            continue
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:
            missing.append(f"{label} ({import_name}): {exc}")
            print(f"[FAIL] {label}: {exc}")
            continue
        version = getattr(module, "__version__", None)
        if version is None and import_name == "serial":
            version = getattr(module, "VERSION", None)
        if import_name == "onnxruntime":
            providers = module.get_available_providers()
            if "CPUExecutionProvider" not in providers:
                missing.append(f"{label}: CPUExecutionProvider not available, providers={providers}")
                print(f"[FAIL] {label}: CPUExecutionProvider not available, providers={providers}")
                continue
            print(f"[OK] {label}: {version or 'imported'}, providers={providers}")
            continue
        print(f"[OK] {label}: {version or 'imported'}")
    if missing:
        details = "\n  - ".join(missing)
        raise RuntimeError(f"Missing or broken Python dependencies:\n  - {details}")


def _compile_sources(root: Path) -> None:
    ok = compileall.compile_dir(str(root), quiet=1)
    if not ok:
        raise RuntimeError("compileall failed")
    print("[OK] compileall")


def _check_project_imports() -> None:
    from FlightController import FC_Controller
    from FlightController.Base import FC_Base_Uart_Comunication
    from FlightController.Components.CameraSource import CameraConfig, CameraSource
    from FlightController.Components.DeviceResolver import DeviceResolver
    from FlightController.Components.LDRadar_Driver import LD_Radar
    from FlightController.Components.LDRadar_Resolver import Radar_Package
    from FlightController.Components.MultiRadar import MultiRadar, RadarConfig
    from FlightController.Components.RealSense import T265, T265_Pose_Frame
    from FlightController.Solutions.AutonomousNavigator import (
        AutonomousNavigator,
        AutonomousNavigatorConfig,
    )
    from FlightController.Solutions.LocalPlanner import LocalPlanner, TargetObservation
    from FlightController.Solutions.Navigation import Navigation
    from FlightController.Solutions.PathPlanner import TrajectoryGenerator
    from FlightController.Solutions.TargetDetector import DetectionResult, TargetDetector

    _ = (
        FC_Base_Uart_Comunication,
        LD_Radar,
        Radar_Package,
        T265,
        Navigation,
        TrajectoryGenerator,
        TargetDetector,
        DeviceResolver,
    )

    fc = FC_Controller()
    pose = T265_Pose_Frame.get_zero()
    assert pose.tracker_confidence == 0

    camera = CameraSource(CameraConfig(device=None, warmup_frames=0))
    assert not camera.is_opened

    configs = [
        RadarConfig(name="front", index=0, mount_xy_cm=(0.0, 12.0), mount_yaw_deg=0.0),
        RadarConfig(name="rear", index=1, mount_xy_cm=(0.0, -12.0), mount_yaw_deg=180.0),
    ]
    multi_radar = MultiRadar(configs)
    assert multi_radar.get_obstacle_points_body_cm().shape == (0, 2)

    planner = LocalPlanner()
    command = planner.plan(obstacles_body_cm=np.empty((0, 2)), target=None)
    assert command.reason == "no_target"
    command = planner.plan(
        obstacles_body_cm=np.array([[30.0, 0.0]]),
        target=TargetObservation(center_px=(400.0, 240.0), image_size=(640, 480), confidence=0.9),
    )
    assert "obstacle_stop" in command.reason

    class FakeCamera:
        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

    class FakeDetector:
        def detect_best(self, frame, class_name=None):
            return DetectionResult(center=(320.0, 240.0), class_name="target", confidence=0.9)

    class FakeRadar:
        def get_obstacle_points_body_cm(self, max_distance_cm=None):
            return np.empty((0, 2), dtype=float)

    class FakeFC:
        def __init__(self):
            self.commands = []

        def send_realtime_control_data(self, *args):
            self.commands.append(args)

    fake_fc = FakeFC()
    navigator = AutonomousNavigator(
        fc=fake_fc,
        t265=None,
        multi_radar=FakeRadar(),
        camera=FakeCamera(),
        detector=FakeDetector(),
        config=AutonomousNavigatorConfig(),
    )
    nav_command = navigator.step()
    assert nav_command.reason == "target"
    assert fake_fc.commands[-1] == (20, 0, 0, 0)
    _ = fc
    print("[OK] project imports and no-hardware smoke")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the full FlightController software environment without opening hardware."
    )
    parser.add_argument(
        "--skip-pyrealsense2",
        action="store_true",
        help="Skip the pyrealsense2 import check while validating the rest of the environment.",
    )
    args = parser.parse_args()

    root = _setup_path()
    _check_python()
    _check_modules(skip_pyrealsense2=args.skip_pyrealsense2)
    _compile_sources(root)
    _check_project_imports()
    print("NO-HARDWARE ENVIRONMENT CHECK OK")


if __name__ == "__main__":
    main()
