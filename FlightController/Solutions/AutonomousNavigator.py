import threading
import time
from dataclasses import dataclass

from .LocalPlanner import LocalPlanner, TargetObservation, VelocityCommand


@dataclass
class AutonomousNavigatorConfig:
    loop_hz: float = 10.0
    target_class: str | None = None
    max_obstacle_distance_cm: float = 300.0


class AutonomousNavigator:
    def __init__(self, *, fc, multi_radar, t265=None, camera=None, detector=None, planner=None, config=None):
        self.fc = fc
        self.t265 = t265
        self.multi_radar = multi_radar
        self.camera = camera
        self.detector = detector
        if planner is None and camera is None:
            from .LocalPlanner import PlannerConfig
            planner = LocalPlanner(config=PlannerConfig(enable_free_flight=True))
        self.planner = planner or LocalPlanner()
        self.config = config or AutonomousNavigatorConfig()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def step(self):
        """Run one perception-planning-control loop and return a VelocityCommand."""
        if self.camera is None or self.detector is None:
            target = None
        else:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                command = VelocityCommand(0.0, 0.0, 0.0, 0.0, "camera_failed")
                self._send_command(command)
                return command

            detection = self.detector.detect_best(frame, class_name=self.config.target_class)
            target = None
            if detection is not None:
                target = TargetObservation(
                    center_px=detection.center,
                    image_size=(frame.shape[1], frame.shape[0]),
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                )

        obstacles = self.multi_radar.get_obstacle_points_body_cm(
            max_distance_cm=self.config.max_obstacle_distance_cm
        )
        command = self.planner.plan(obstacles_body_cm=obstacles, target=target)
        self._send_command(command)
        return command

    def start(self):
        """Start the background perception-planning-control loop."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop_task, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background loop and send zero velocity."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._send_command(VelocityCommand(0.0, 0.0, 0.0, 0.0, "stopped"))

    def _loop_task(self) -> None:
        period = 1.0 / max(self.config.loop_hz, 0.1)
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            self.step()
            elapsed = time.perf_counter() - start_time
            self._stop_event.wait(max(0.0, period - elapsed))

    def _send_command(self, command: VelocityCommand) -> None:
        self.fc.send_realtime_control_data(
            round(command.vx_cm_s),
            round(command.vy_cm_s),
            round(command.vz_cm_s),
            round(command.yaw_rate_deg_s),
        )


__all__ = ["AutonomousNavigator", "AutonomousNavigatorConfig"]
