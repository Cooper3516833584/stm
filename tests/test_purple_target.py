import numpy as np

from purple_target import find_purple_target_offset


def _target(height=100, width=120):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    color = (235, 0, 180)
    image[20:24, 23:68] = color
    image[56:60, 23:68] = color
    image[20:60, 23:27] = color
    image[20:60, 64:68] = color
    return image


def test_purple_target_coordinate_signs():
    assert find_purple_target_offset(_target(), max_dimension=200) == (10, 15)


def test_purple_target_accepts_bgr_frames():
    rgb = _target(80, 80)
    assert find_purple_target_offset(rgb[..., ::-1], color_order="bgr", max_dimension=200) == (0, -5)


def test_purple_target_rejects_nonpurple_frame():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 1] = 180
    assert find_purple_target_offset(image, max_dimension=200) is None
