"""Compatibility hardware helpers for root-level autonomy scripts."""

from __future__ import annotations

from typing import Any


def build_dual_radar(upper_port: str, lower_port: str):
    from FlightController.Components import MultiRadar, RadarConfig

    return MultiRadar(
        [
            RadarConfig("upper", 0, (0.0, 0.0), 0.0, port=upper_port),
            RadarConfig("lower", 1, (0.96, 0.15), 0.0, port=lower_port, mount_mirror_y=True),
        ]
    )


def connect_fc(fc_port: str | None = None):
    from FlightController.Components.FCConnector import FCConnectConfig, connect_fc as _connect_fc

    return _connect_fc(FCConnectConfig(port=fc_port, mode=2, timeout_s=10.0))


def open_camera(device: str | int | None, width: int, height: int, fps: int):
    from FlightController.Components.CameraSource import CameraConfig, CameraSource

    camera = CameraSource(CameraConfig(device=device, width=width, height=height, fps=fps))
    camera.open()
    return camera


def send_fc_command(fc: Any | None, command, enable_flight: bool) -> None:
    if fc is None or not enable_flight:
        return
    fc.send_realtime_control_data(*command.as_fc_tuple())


def stop_fc(fc: Any | None) -> None:
    if fc is None:
        return
    try:
        fc.send_realtime_control_data(0, 0, 0, 0)
    finally:
        fc.close()


__all__ = ["build_dual_radar", "connect_fc", "open_camera", "send_fc_command", "stop_fc"]
