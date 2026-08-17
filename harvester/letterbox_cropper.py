from __future__ import annotations

import cv2
import numpy as np

_MAX_BORDER_LUMINANCE = 16
_MIN_RETAINED_AREA_RATIO = 0.5


def _content_bounds(grayscale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows_with_content = np.flatnonzero(grayscale.max(axis=1) > _MAX_BORDER_LUMINANCE)
    columns_with_content = np.flatnonzero(grayscale.max(axis=0) > _MAX_BORDER_LUMINANCE)
    return rows_with_content, columns_with_content


def crop_letterbox(frame: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows_with_content, columns_with_content = _content_bounds(grayscale)
    if rows_with_content.size == 0 or columns_with_content.size == 0:
        return frame
    cropped = frame[
        rows_with_content[0]:rows_with_content[-1] + 1,
        columns_with_content[0]:columns_with_content[-1] + 1,
    ]
    retained_area_ratio = cropped[:, :, 0].size / frame[:, :, 0].size
    if retained_area_ratio < _MIN_RETAINED_AREA_RATIO:
        return frame
    return cropped
