from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy
import tomli_w

from reference_builder.reference_models import CropRecord
from sharedkernel.utils.geometry import BoundingBox
from sharedkernel.utils.paths import ensure_directory

CROP_MANIFEST_FILE_NAME = "crops.toml"

_CROP_FILE_SUFFIX = ".png"
_CROP_ID_SEPARATOR = "/"


def crop_id_of(episode_slug: str, frame_file_name: str, detection_index: int) -> str:
    return f"{episode_slug}{_CROP_ID_SEPARATOR}{Path(frame_file_name).stem}-{detection_index:02d}"


def _crop_file_name(crop_id: str) -> str:
    return crop_id.split(_CROP_ID_SEPARATOR, 1)[1] + _CROP_FILE_SUFFIX


def _to_document(record: CropRecord) -> dict[str, Any]:
    return {
        "crop_id": record.crop_id,
        "frame_file_name": record.frame_file_name,
        "detection_index": record.detection_index,
        "score": round(record.score, 4),
        "phrase": record.phrase,
        "box": [record.box.left, record.box.top, record.box.right, record.box.bottom],
    }


def _to_record(episode_slug: str, document: dict[str, Any]) -> CropRecord:
    left, top, right, bottom = document["box"]
    return CropRecord(
            crop_id=document["crop_id"],
            episode_slug=episode_slug,
            frame_file_name=document["frame_file_name"],
            detection_index=document["detection_index"],
            score=document["score"],
            phrase=document["phrase"],
            box=BoundingBox(left=left, top=top, right=right, bottom=bottom),
    )


class CropStore:
    def __init__(self, crops_dir: Path) -> None:
        self._crops_dir = crops_dir

    def _episode_dir(self, episode_slug: str) -> Path:
        return self._crops_dir / episode_slug

    def _manifest_path(self, episode_slug: str) -> Path:
        return self._episode_dir(episode_slug) / CROP_MANIFEST_FILE_NAME

    def contains_episode(self, episode_slug: str) -> bool:
        return self._manifest_path(episode_slug).is_file()

    def crop_path(self, crop_id: str) -> Path:
        episode_slug = crop_id.split(_CROP_ID_SEPARATOR, 1)[0]
        return self._episode_dir(episode_slug) / _crop_file_name(crop_id)

    def save_crop(self, record: CropRecord, image: numpy.ndarray) -> Path:
        path = ensure_directory(self._episode_dir(record.episode_slug)) / _crop_file_name(record.crop_id)
        cv2.imwrite(str(path), image)
        return path

    def write_manifest(self, episode_slug: str, records: tuple[CropRecord, ...]) -> Path:
        path = ensure_directory(self._episode_dir(episode_slug)) / CROP_MANIFEST_FILE_NAME
        with path.open("wb") as handle:
            tomli_w.dump({
                "episode_slug": episode_slug,
                "crop_count": len(records),
                "crops": [_to_document(record) for record in records],
            }, handle)
        return path

    def load_episode(self, episode_slug: str) -> tuple[CropRecord, ...]:
        manifest_path = self._manifest_path(episode_slug)
        if not manifest_path.is_file():
            return ()
        with manifest_path.open("rb") as handle:
            document = tomllib.load(handle)
        return tuple(_to_record(episode_slug, crop) for crop in document["crops"])

    def load_all(self) -> Iterator[CropRecord]:
        if not self._crops_dir.is_dir():
            return
        for manifest_path in sorted(self._crops_dir.glob(f"*/{CROP_MANIFEST_FILE_NAME}")):
            yield from self.load_episode(manifest_path.parent.name)
