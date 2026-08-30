from __future__ import annotations

from dataclasses import dataclass

from sharedkernel.utils.geometry import BoundingBox

UNKNOWN_LABEL = "UNKNOWN"


@dataclass(frozen=True)
class Detection:
    box: BoundingBox
    score: float
    phrase: str


@dataclass(frozen=True)
class CropRecord:
    crop_id: str
    episode_slug: str
    frame_file_name: str
    detection_index: int
    score: float
    phrase: str
    box: BoundingBox


@dataclass(frozen=True)
class CharacterCluster:
    cluster_id: int
    crop_ids: tuple[str, ...]
    medoid_crop_id: str


@dataclass(frozen=True)
class ClusterLabel:
    cluster_id: int
    slug: str
    confidence: float
    reasoning: str

    @property
    def is_identified(self) -> bool:
        return self.slug != UNKNOWN_LABEL


@dataclass(frozen=True)
class ReferenceCard:
    slug: str
    crop_ids: tuple[str, ...]
    cluster_ids: tuple[int, ...]
