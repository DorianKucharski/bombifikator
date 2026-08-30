from __future__ import annotations

from sharedkernel.utils.geometry import BoundingBox

_LEFT_THIRD = 1 / 3
_RIGHT_THIRD = 2 / 3
_FOREGROUND_HEIGHT_RATIO = 0.5


def _horizontal_phrase(box: BoundingBox, image_width: int) -> str:
    centre = (box.left + box.right) / 2 / image_width
    if centre < _LEFT_THIRD:
        return "on the left"
    if centre > _RIGHT_THIRD:
        return "on the right"
    return "in the middle"


def _depth_phrase(box: BoundingBox, image_height: int) -> str:
    return "in the foreground" if box.height > _FOREGROUND_HEIGHT_RATIO * image_height else "in the background"


def locate_person(box: BoundingBox, image_width: int, image_height: int) -> str:
    return f"{_horizontal_phrase(box, image_width)}, {_depth_phrase(box, image_height)}"
