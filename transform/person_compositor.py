from __future__ import annotations

import cv2
import numpy

from sharedkernel.utils.geometry import BoundingBox

_ALPHA_CHANNEL = 3
_OPAQUE_THRESHOLD = 16


def largest_component_only(character: numpy.ndarray) -> numpy.ndarray:
    opaque = (character[:, :, _ALPHA_CHANNEL] > _OPAQUE_THRESHOLD).astype(numpy.uint8)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(opaque, connectivity=8)
    if component_count <= 2:
        return character
    largest_label = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    kept = character.copy()
    kept[:, :, _ALPHA_CHANNEL] = numpy.where(labels == largest_label, character[:, :, _ALPHA_CHANNEL], 0)
    return kept


def crop_to_alpha(character: numpy.ndarray) -> numpy.ndarray:
    opaque_rows, opaque_columns = numpy.nonzero(character[:, :, _ALPHA_CHANNEL])
    if opaque_rows.size == 0:
        return character
    return character[opaque_rows.min():opaque_rows.max() + 1, opaque_columns.min():opaque_columns.max() + 1]


def fitted_into(box: BoundingBox, character_width: int, character_height: int) -> tuple[int, int]:
    scale = min(box.width / character_width, box.height / character_height)
    return max(1, int(character_width * scale)), max(1, int(character_height * scale))


def _placement(box: BoundingBox, width: int, height: int) -> tuple[int, int]:
    return box.left + (box.width - width) // 2, box.bottom - height


def paste_character(photo: numpy.ndarray, character: numpy.ndarray,
                    box: BoundingBox) -> tuple[numpy.ndarray, numpy.ndarray]:
    trimmed = crop_to_alpha(largest_component_only(character))
    width, height = fitted_into(box, trimmed.shape[1], trimmed.shape[0])
    resized = cv2.resize(trimmed, (width, height), interpolation=cv2.INTER_AREA)
    left, top = _placement(box, width, height)
    region = photo[top:top + height, left:left + width]
    alpha = resized[:, :, _ALPHA_CHANNEL:].astype(numpy.float32) / 255.0
    blended = region.astype(numpy.float32) * (1.0 - alpha) + resized[:, :, :3].astype(numpy.float32) * alpha
    composited = photo.copy()
    composited[top:top + height, left:left + width] = numpy.clip(blended, 0, 255).astype(numpy.uint8)
    covered = numpy.zeros(photo.shape[:2], dtype=bool)
    covered[top:top + height, left:left + width] = resized[:, :, _ALPHA_CHANNEL] > _OPAQUE_THRESHOLD
    return composited, covered



def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def toned_like(plate: numpy.ndarray, photo: numpy.ndarray, outside_people: numpy.ndarray) -> numpy.ndarray:
    if not outside_people.any():
        return plate
    shift = photo[outside_people].astype(numpy.float32).mean(axis=0) \
        - plate[outside_people].astype(numpy.float32).mean(axis=0)
    return numpy.clip(plate.astype(numpy.float32) + shift, 0, 255).astype(numpy.uint8)


def with_people_removed(photo: numpy.ndarray, plate: numpy.ndarray, people: numpy.ndarray,
                        feather_pixels: int) -> numpy.ndarray:
    outside_people = ~people.astype(bool)
    toned = toned_like(plate, photo, outside_people)
    alpha = cv2.GaussianBlur(people.astype(numpy.float32), (_odd(feather_pixels), _odd(feather_pixels)), 0)
    alpha = numpy.clip(alpha, 0.0, 1.0)[:, :, None]
    blended = photo.astype(numpy.float32) * (1.0 - alpha) + toned.astype(numpy.float32) * alpha
    return numpy.clip(blended, 0, 255).astype(numpy.uint8)
