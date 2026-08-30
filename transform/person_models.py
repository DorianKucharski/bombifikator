from __future__ import annotations

from dataclasses import dataclass

import numpy

from sharedkernel.utils.geometry import BoundingBox


@dataclass(frozen=True)
class PersonInstance:
    person_index: int
    box: BoundingBox
    mask: numpy.ndarray
    confidence: float

    @property
    def pixel_area(self) -> int:
        return int(self.mask.sum())


@dataclass(frozen=True)
class CharacterAssignment:
    person_index: int
    slug: str
