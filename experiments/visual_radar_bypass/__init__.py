"""Visual road following plus side-agnostic physical-obstacle bypass."""

from .radar_bypass import (
    ObstacleBypassConfig,
    ObstacleBypassPlanner,
    ObstacleBypassState,
)


def __getattr__(name):
    # Keep planner-only benchmark/validation imports independent from camera
    # and NPU libraries while preserving the existing public package API.
    if name in {"FrozenVisualConfig", "FrozenVisualGuidance", "VisualSample"}:
        from .visual_guidance import (
            FrozenVisualConfig,
            FrozenVisualGuidance,
            VisualSample,
        )

        values = {
            "FrozenVisualConfig": FrozenVisualConfig,
            "FrozenVisualGuidance": FrozenVisualGuidance,
            "VisualSample": VisualSample,
        }
        globals().update(values)
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FrozenVisualConfig",
    "FrozenVisualGuidance",
    "VisualSample",
    "ObstacleBypassConfig",
    "ObstacleBypassPlanner",
    "ObstacleBypassState",
]
