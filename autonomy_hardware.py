"""Hardware helper functions for root-level autonomy entry points."""

from __future__ import annotations

from typing import Any


def build_dual_radar(upper_port: str, lower_port: str):
    from FlightController.Components import MultiRadar, RadarConfig

    configs = [
        RadarConfig(
            name="upper",
            index=0,
            mount_xy_cm=(0.0, 0.0),
            mount_yaw_deg=0.0,
            port=upper_port,
        ),
        RadarConfig(
            name="lower",
            index=1,
            mount_xy_cm=(0.96, 0.15),
            mount_yaw_deg=0.0,
            mount_mirror_y=True,
            port=lower_port,
        ),
    ]
    return MultiRadar(configs)


def connect_fc(fc_port: str | None = None):
    from FlightController import FC_Controller

    fc = FC_Controller()
    fc.start_listen_serial(block_until_connected=True, explicit_port=fc_port)
    fc.wait_for_connection(timeout_s=10)
    fc.set_flight_mode(2)
    fc.wait_for_last_command_done()
    return fc


def open_camera(device: str | int | None, width: int, height: int, fps: int):
    from FlightController.Components.CameraSource import CameraConfig, CameraSource

    camera = CameraSource(CameraConfig(device=device, width=width, height=height, fps=fps))
    camera.open()
    return camera


def send_fc_command(fc: Any | None, command, enable_flight: bool) -> None:
    if fc is None or not enable_flight:
        return
    vx, vy, vz, yaw = command.as_fc_tuple()
    fc.send_realtime_control_data(vx, vy, vz, yaw)


def stop_fc(fc: Any | None, enable_flight: bool) -> None:
    if fc is None:
        return
    try:
        if enable_flight:
            fc.send_realtime_control_data(0, 0, 0, 0)
    finally:
        fc.close()

