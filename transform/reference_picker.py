from __future__ import annotations

from pathlib import Path

import cv2
import numpy

_WHITE_BACKGROUND = 255


def _flattened_on_white(image: numpy.ndarray) -> numpy.ndarray:
    if image.shape[2] != 4:
        return image
    alpha = image[:, :, 3:4].astype(numpy.float32) / 255.0
    colours = image[:, :, :3].astype(numpy.float32)
    return (colours * alpha + _WHITE_BACKGROUND * (1.0 - alpha)).astype(numpy.uint8)


def load_references(references_dir: Path, slug: str, count: int) -> tuple[numpy.ndarray, ...]:
    loaded = (cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
              for path in sorted((references_dir / slug).glob("*.png"))[:count])
    return tuple(_flattened_on_white(image) for image in loaded if image is not None)
