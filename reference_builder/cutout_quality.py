from __future__ import annotations

import numpy

_OPAQUE_ALPHA = 250


def background_removed(image: numpy.ndarray, max_opaque_ratio: float) -> bool:
    if image.shape[2] < 4:
        return False
    return float((image[:, :, 3] >= _OPAQUE_ALPHA).mean()) <= max_opaque_ratio
