"""Frozen-path tracker tests."""

import numpy as np

from experiments.road_contour_bypass.path_tracker import FrozenPathTracker


def _diagonal_path() -> np.ndarray:
    x = np.linspace(0.0, 220.0, 121)
    y = -70.0 * np.sin(np.pi * x / 220.0)
    return np.column_stack((x, y))


def test_tracker_moves_forward_and_laterally_on_same_cycle():
    tracker = FrozenPathTracker()
    tracker.set_path(_diagonal_path())
    result = tracker.update(local_x_cm=0.0, local_y_cm=0.0, local_yaw_deg=0.0, dt_s=0.1)

    assert result.command.vx_cm_s > 0.0
    assert result.command.vy_cm_s != 0.0


def test_path_index_is_monotonic():
    tracker = FrozenPathTracker()
    tracker.set_path(_diagonal_path())
    indices = []
    for x_cm in (0.0, 20.0, 40.0, 10.0, 60.0):
        result = tracker.update(
            local_x_cm=x_cm,
            local_y_cm=float(-70.0 * np.sin(np.pi * max(x_cm, 0.0) / 220.0)),
            local_yaw_deg=0.0,
            dt_s=0.1,
        )
        indices.append(result.nearest_index)
    assert indices == sorted(indices)


def test_tracker_keeps_an_immutable_copy_of_path():
    source = _diagonal_path()
    tracker = FrozenPathTracker()
    tracker.set_path(source)
    original = tracker.path_samples.copy()
    source[:] = 999.0

    np.testing.assert_allclose(tracker.path_samples, original)
    assert not tracker.path_samples.flags.writeable
