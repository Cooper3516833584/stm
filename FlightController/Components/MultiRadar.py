from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
from loguru import logger


@dataclass
class RadarConfig:
    name: str
    index: int
    mount_xy_cm: tuple[float, float]
    mount_yaw_deg: float
    port: str | None = None
    mount_mirror_y: bool = False


class MultiRadar:
    def __init__(self, configs: Iterable[RadarConfig]):
        from .LDRadar_Driver import LD_Radar

        self.configs = list(configs)
        if len(self.configs) != 2:
            raise ValueError("MultiRadar requires exactly two radar configs")
        self.radars = [
            LD_Radar(
                name=config.name,
                index=config.index,
                mount_xy_cm=config.mount_xy_cm,
                mount_yaw_deg=config.mount_yaw_deg,
                mount_mirror_y=config.mount_mirror_y,
            )
            for config in self.configs
        ]

    def start(self) -> None:
        for radar, config in zip(self.radars, self.configs):
            logger.info(f"[MultiRadar] Starting {config.name}")
            radar.start(com=config.port, radar_type="D500")

    def stop(self) -> None:
        for radar in self.radars:
            radar.stop()

    @property
    def running(self) -> bool:
        return bool(self.radars) and all(radar.running for radar in self.radars)

    @property
    def connected(self) -> bool:
        return bool(self.radars) and all(radar.connected for radar in self.radars)

    def get_health_snapshot(self, now_s: float | None = None, max_age_s: float = 0.5) -> dict[str, object]:
        if now_s is None:
            now_s = time.perf_counter()
        radar_states = [radar.get_health_snapshot(now_s=now_s) for radar in self.radars]
        required_ok = True
        for state in radar_states:
            age = state["last_frame_age_s"]
            if not state["connected"] or age is None or age > max_age_s:
                required_ok = False
        return {
            "connected": self.connected,
            "fresh": required_ok,
            "max_age_s": max_age_s,
            "radars": radar_states,
        }

    def is_fresh(self, max_age_s: float = 0.5, now_s: float | None = None) -> bool:
        if now_s is None:
            now_s = time.perf_counter()
        return all(radar.is_fresh(max_age_s=max_age_s, now_s=now_s) for radar in self.radars)

    def get_obstacle_points_body_cm(self, max_distance_cm: float | None = None) -> np.ndarray:
        point_sets = [
            radar.get_points_body_cm(max_distance_cm=max_distance_cm)
            for radar in self.radars
        ]
        point_sets = [points for points in point_sets if points.size > 0]
        if not point_sets:
            return np.empty((0, 2), dtype=float)
        return np.vstack(point_sets)


# Example upper+lower mounting convention (lower radar mounted upside-down):
# configs = [
#     RadarConfig(name="upper", index=0, mount_xy_cm=(0.0, 0.0), mount_yaw_deg=0.0),
#     RadarConfig(name="lower", index=1, mount_xy_cm=(0.96, 0.15), mount_yaw_deg=0.0, mount_mirror_y=True),
# ]


__all__ = ["MultiRadar", "RadarConfig"]
