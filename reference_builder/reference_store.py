from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy
import tomli_w

from reference_builder.reference_models import CropRecord, ReferenceCard
from sharedkernel.utils.paths import ensure_directory

REFERENCE_MANIFEST_FILE_NAME = "manifest.toml"

_REFERENCE_FILE_TEMPLATE = "%02d.png"


def _to_document(card: ReferenceCard, sources: tuple[CropRecord, ...], file_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "slug": card.slug,
        "reference_count": len(file_names),
        "cluster_ids": list(card.cluster_ids),
        "references": [
            {
                "file_name": file_name,
                "crop_id": source.crop_id,
                "episode_slug": source.episode_slug,
                "frame_file_name": source.frame_file_name,
                "score": round(source.score, 4),
            }
            for file_name, source in zip(file_names, sources)
        ],
    }


class ReferenceStore:
    def __init__(self, references_dir: Path) -> None:
        self._references_dir = references_dir

    def character_dir(self, slug: str) -> Path:
        return self._references_dir / slug

    def save_references(self, card: ReferenceCard, images: tuple[numpy.ndarray, ...]) -> tuple[str, ...]:
        character_dir = ensure_directory(self.character_dir(card.slug))
        file_names = []
        for position, image in enumerate(images):
            file_name = _REFERENCE_FILE_TEMPLATE % position
            cv2.imwrite(str(character_dir / file_name), image)
            file_names.append(file_name)
        return tuple(file_names)

    def write_manifest(self, card: ReferenceCard, sources: tuple[CropRecord, ...],
                       file_names: tuple[str, ...]) -> Path:
        manifest_path = ensure_directory(self.character_dir(card.slug)) / REFERENCE_MANIFEST_FILE_NAME
        with manifest_path.open("wb") as handle:
            tomli_w.dump(_to_document(card, sources, file_names), handle)
        return manifest_path
