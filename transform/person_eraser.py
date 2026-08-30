from __future__ import annotations

import cv2
import numpy


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def dilated(mask: numpy.ndarray, dilation_ratio: float) -> numpy.ndarray:
    radius = max(3, int(round(max(mask.shape) * dilation_ratio)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(radius), _odd(radius)))
    return cv2.dilate(mask.astype(numpy.uint8), kernel)
