from __future__ import annotations

import numpy

_OPAQUE_ALPHA = 250
_FRAME_RING_PIXELS = 3


def background_removed(image: numpy.ndarray, max_opaque_ratio: float) -> bool:
    if image.shape[2] < 4:
        return False
    return float((image[:, :, 3] >= _OPAQUE_ALPHA).mean()) <= max_opaque_ratio


def cutout_clear_of_frame(image: numpy.ndarray, max_border_opaque_ratio: float) -> bool:
    if image.shape[2] < 4:
        return False
    ring = numpy.zeros(image.shape[:2], dtype=bool)
    ring[:_FRAME_RING_PIXELS, :] = ring[-_FRAME_RING_PIXELS:, :] = True
    ring[:, :_FRAME_RING_PIXELS] = ring[:, -_FRAME_RING_PIXELS:] = True
    opaque = image[:, :, 3] >= _OPAQUE_ALPHA
    return float(opaque[ring].mean()) <= max_border_opaque_ratio
