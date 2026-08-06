import numpy as np

from experiments.visual_radar_bypass.radar_bypass import ObstacleBypassState
from experiments.visual_radar_bypass.right_half_handoff import (
    RightHalfRadarHandoff,
)


def test_right_half_plane_is_clockwise_zero_to_180_degrees():
    points = np.asarray(
        [
            [100.0, 0.0],
            [100.0, -20.0],
            [0.0, -80.0],
            [-100.0, -20.0],
            [-100.0, 0.0],
            [100.0, 20.0],
            [0.0, 80.0],
            [-100.0, 20.0],
        ]
    )

    filtered = RightHalfRadarHandoff.filter_right_half_plane(points)

    assert filtered.tolist() == points[:5].tolist()


def test_normal_without_completed_forward_recovery_never_retires_radar():
    handoff = RightHalfRadarHandoff()

    assert not handoff.observe(
        ObstacleBypassState.NORMAL,
        ObstacleBypassState.NORMAL,
        100.0,
    )
    assert not handoff.radar_disabled


def test_completed_recovery_retires_radar_after_five_normal_seconds():
    handoff = RightHalfRadarHandoff()

    assert not handoff.observe(
        ObstacleBypassState.FORWARD_RECOVERY,
        ObstacleBypassState.NORMAL,
        10.0,
    )
    assert not handoff.observe(
        ObstacleBypassState.NORMAL,
        ObstacleBypassState.NORMAL,
        14.99,
    )
    assert handoff.observe(
        ObstacleBypassState.NORMAL,
        ObstacleBypassState.NORMAL,
        15.0,
    )
    assert handoff.radar_disabled
    assert handoff.normal_elapsed_s == 5.0
    assert not handoff.observe(
        ObstacleBypassState.NORMAL,
        ObstacleBypassState.NORMAL,
        16.0,
    )


def test_return_to_bypass_cancels_pending_visual_only_handoff():
    handoff = RightHalfRadarHandoff()

    handoff.observe(
        ObstacleBypassState.FORWARD_RECOVERY,
        ObstacleBypassState.NORMAL,
        20.0,
    )
    assert not handoff.observe(
        ObstacleBypassState.NORMAL,
        ObstacleBypassState.BYPASS_LEFT,
        22.0,
    )
    assert not handoff.observe(
        ObstacleBypassState.BYPASS_LEFT,
        ObstacleBypassState.NORMAL,
        30.0,
    )
    assert not handoff.observe(
        ObstacleBypassState.NORMAL,
        ObstacleBypassState.NORMAL,
        40.0,
    )
    assert not handoff.radar_disabled


def test_pending_first_intrusion_frame_prevents_radar_retirement():
    handoff = RightHalfRadarHandoff()
    handoff.observe(
        ObstacleBypassState.FORWARD_RECOVERY,
        ObstacleBypassState.NORMAL,
        50.0,
    )

    assert not handoff.observe(
        ObstacleBypassState.NORMAL,
        ObstacleBypassState.NORMAL,
        55.0,
        bypass_pending=True,
    )
    assert not handoff.radar_disabled
    assert handoff.normal_elapsed_s == 0.0
