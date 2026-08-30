from __future__ import annotations

from pathlib import Path

import numpy

from sharedkernel.utils.paths import ensure_directory

_CROP_IDS_KEY = "crop_ids"
_EMBEDDINGS_KEY = "embeddings"


def save_embeddings(embeddings_path: Path, crop_ids: tuple[str, ...], embeddings: numpy.ndarray) -> Path:
    ensure_directory(embeddings_path.parent)
    numpy.savez_compressed(
            embeddings_path,
            **{_CROP_IDS_KEY: numpy.array(crop_ids, dtype=object), _EMBEDDINGS_KEY: embeddings},
    )
    return embeddings_path


def load_embeddings(embeddings_path: Path) -> tuple[tuple[str, ...], numpy.ndarray]:
    if not embeddings_path.is_file():
        raise ValueError(f"missing embeddings file {embeddings_path}: run 'references embed' first")
    with numpy.load(embeddings_path, allow_pickle=True) as archive:
        return tuple(str(crop_id) for crop_id in archive[_CROP_IDS_KEY]), archive[_EMBEDDINGS_KEY]
