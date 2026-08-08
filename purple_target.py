"""Low-cost purple rectangular target detection.

The detector is deliberately frame-only: camera ownership belongs to the
shared perception pipeline.  Offsets use the flight-controller convention:
positive ``x`` is upward and positive ``y`` is leftward in the image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

ImageInput = Union[np.ndarray, str, Path]
_CV2 = None
_CV2_TRIED = False


def find_purple_target_offset(
    image: ImageInput,
    *,
    color_order: str = "rgb",
    max_dimension: int = 256,
    hue_min: float = 135.0,
    hue_max: float = 179.0,
    saturation_min: float = 90.0,
    value_min: float = 60.0,
    min_area_ratio: float = 0.005,
) -> Optional[Tuple[int, int]]:
    """Return the largest purple component's image-centre offset, if found."""
    array = _load_image(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("image must have shape HxWx3")
    order = color_order.lower()
    if order not in ("rgb", "bgr"):
        raise ValueError("color_order must be 'rgb' or 'bgr'")
    if max_dimension < 32:
        raise ValueError("max_dimension must be at least 32")

    height, width = array.shape[:2]
    stride = max(1, (max(height, width) + max_dimension - 1) // max_dimension)
    sample = array[::stride, ::stride, :3]
    mask = _purple_mask(sample, order, hue_min, hue_max, saturation_min, value_min)
    minimum_pixels = max(8, int(mask.size * min_area_ratio))
    if int(np.count_nonzero(mask)) < minimum_pixels:
        return None
    bbox = _largest_component_bbox(mask, minimum_pixels)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    target_x = (left + right) * 0.5 * stride
    target_y = (top + bottom) * 0.5 * stride
    return int(round(height * 0.5 - target_y)), int(round(width * 0.5 - target_x))


def _load_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Pillow is required when image is a file path") from exc
        with Image.open(image) as pil_image:
            return np.asarray(pil_image.convert("RGB"))
    return np.asarray(image)


def _get_cv2():
    global _CV2, _CV2_TRIED
    if not _CV2_TRIED:
        _CV2_TRIED = True
        try:
            import cv2
        except ImportError:
            return None
        _CV2 = cv2
    return _CV2


def _purple_mask(image, order, hue_min, hue_max, saturation_min, value_min):
    cv2 = _get_cv2()
    if cv2 is not None:
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV if order == "bgr" else cv2.COLOR_RGB2HSV,
        )
        lower = np.array(
            [
                int(round(hue_min)),
                int(round(saturation_min)),
                int(round(value_min)),
            ],
            np.uint8,
        )
        upper = np.array([int(round(hue_max)), 255, 255], np.uint8)
        if hue_min <= hue_max:
            return cv2.inRange(hsv, lower, upper)
        first = cv2.inRange(hsv, np.array([0, lower[1], lower[2]], np.uint8), upper)
        second = cv2.inRange(hsv, lower, np.array([179, 255, 255], np.uint8))
        return first | second

    values = image.astype(np.float32, copy=False)
    channel_order = (2, 1, 0) if order == "bgr" else (0, 1, 2)
    red, green, blue = (values[..., index] for index in channel_order)
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    delta = maximum - minimum
    saturation = np.divide(
        delta * 255.0,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    hue = np.zeros_like(maximum)
    chromatic = delta > 0
    red_max = chromatic & (maximum == red)
    green_max = chromatic & (maximum == green)
    blue_max = chromatic & (maximum == blue)
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6.0
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0
    hue *= 30.0
    hue_ok = (
        (hue >= hue_min) & (hue <= hue_max)
        if hue_min <= hue_max
        else (hue >= hue_min) | (hue <= hue_max)
    )
    return (
        (hue_ok & (saturation >= saturation_min) & (maximum >= value_min)) * 255
    ).astype(np.uint8)


def _largest_component_bbox(mask: np.ndarray, minimum_pixels: int):
    cv2 = _get_cv2()
    if cv2 is not None:
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return None
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if int(stats[component, cv2.CC_STAT_AREA]) < minimum_pixels:
            return None
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        w = int(stats[component, cv2.CC_STAT_WIDTH])
        h = int(stats[component, cv2.CC_STAT_HEIGHT])
        return x, y, x + w - 1, y + h - 1
    # The target is a large frame, so supported rows and columns suppress
    # isolated purple noise when OpenCV is unavailable.
    row_support = mask.sum(axis=1)
    col_support = mask.sum(axis=0)
    row_threshold = max(2, int(mask.shape[1] * 0.003))
    col_threshold = max(2, int(mask.shape[0] * 0.003))
    active_rows = np.flatnonzero(row_support >= row_threshold)
    active_cols = np.flatnonzero(col_support >= col_threshold)
    if active_rows.size == 0 or active_cols.size == 0:
        ys, xs = np.nonzero(mask)
    else:
        region = mask[np.ix_(active_rows, active_cols)]
        ys, xs = np.nonzero(region)
        if ys.size:
            ys = active_rows[ys]
            xs = active_cols[xs]
    if ys.size < minimum_pixels:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


__all__ = ["find_purple_target_offset"]
