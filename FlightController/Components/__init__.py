__all__ = [
    "CameraConfig",
    "CameraSource",
    "DeviceResolver",
    "LD_Radar",
    "Map_Circle",
    "MultiRadar",
    "Point_2D",
    "RadarConfig",
    "Radar_Package",
    "Radar_Package_Multi",
    "T265",
    "T265_Pose_Frame",
]

from .MultiRadar import MultiRadar, RadarConfig


class _LazySymbol:
    def __init__(self, module_name, symbol_name):
        self._module_name = module_name
        self._symbol_name = symbol_name
        self._symbol = None

    def _load(self):
        if self._symbol is None:
            from importlib import import_module

            module = import_module(self._module_name, __name__)
            self._symbol = getattr(module, self._symbol_name)
            globals()[self._symbol_name] = self._symbol
        return self._symbol

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __repr__(self):
        return repr(self._load())


CameraSource = _LazySymbol(".CameraSource", "CameraSource")
DeviceResolver = _LazySymbol(".DeviceResolver", "DeviceResolver")


def __getattr__(name):
    if name in ("CameraConfig", "CameraSource"):
        from .CameraSource import CameraConfig, CameraSource

        return {
            "CameraConfig": CameraConfig,
            "CameraSource": CameraSource,
        }[name]

    if name == "DeviceResolver":
        from .DeviceResolver import DeviceResolver

        return DeviceResolver

    if name == "LD_Radar":
        from .LDRadar_Driver import LD_Radar

        return LD_Radar

    if name in ("Map_Circle", "Point_2D", "Radar_Package", "Radar_Package_Multi"):
        from .LDRadar_Resolver import (
            Map_Circle,
            Point_2D,
            Radar_Package,
            Radar_Package_Multi,
        )

        return {
            "Map_Circle": Map_Circle,
            "Point_2D": Point_2D,
            "Radar_Package": Radar_Package,
            "Radar_Package_Multi": Radar_Package_Multi,
        }[name]

    if name in ("T265", "T265_Pose_Frame"):
        from .RealSense import T265, T265_Pose_Frame

        return {
            "T265": T265,
            "T265_Pose_Frame": T265_Pose_Frame,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
