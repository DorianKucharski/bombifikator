from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy

from reference_builder.crop_store import CropStore
from reference_builder.embedding_store import load_embeddings
from reference_builder.reference_config import ClusteringConfig, ReferencePaths
from reference_builder.reference_models import CharacterCluster
from reference_builder.reference_selector import select_diverse_rows
from sharedkernel.utils.paths import ensure_directory

SHEET_FILE_TEMPLATE = "cluster-%04d.png"

_BACKGROUND_VALUE = 24


def _fitted_tile(image: numpy.ndarray, tile_pixels: int) -> numpy.ndarray:
    scale = tile_pixels / max(image.shape[0], image.shape[1])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    tile = numpy.full((tile_pixels, tile_pixels, 3), _BACKGROUND_VALUE, dtype=numpy.uint8)
    top = (tile_pixels - resized.shape[0]) // 2
    left = (tile_pixels - resized.shape[1]) // 2
    tile[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return tile


def build_contact_sheet(images: Sequence[numpy.ndarray], grid_size: int, tile_pixels: int) -> numpy.ndarray:
    sheet = numpy.full((grid_size * tile_pixels, grid_size * tile_pixels, 3), _BACKGROUND_VALUE, dtype=numpy.uint8)
    for position, image in enumerate(images[:grid_size * grid_size]):
        row, column = divmod(position, grid_size)
        top, left = row * tile_pixels, column * tile_pixels
        sheet[top:top + tile_pixels, left:left + tile_pixels] = _fitted_tile(image, tile_pixels)
    return sheet


def sheet_path_of(sheets_dir: Path, cluster_id: int) -> Path:
    return sheets_dir / (SHEET_FILE_TEMPLATE % cluster_id)


def _sheet_crop_ids(cluster: CharacterCluster, crop_id_rows: dict[str, int],
                    embeddings: numpy.ndarray, tile_count: int) -> tuple[str, ...]:
    known = tuple(crop_id for crop_id in cluster.crop_ids if crop_id in crop_id_rows)
    if not known:
        return ()
    rows = numpy.array([crop_id_rows[crop_id] for crop_id in known])
    seed_row = int(numpy.flatnonzero(rows == crop_id_rows[cluster.medoid_crop_id])[0]) \
        if cluster.medoid_crop_id in crop_id_rows else 0
    selected = select_diverse_rows(embeddings[rows], tile_count, seed_row)
    return tuple(known[row] for row in selected)


def write_cluster_sheets(paths: ReferencePaths, clusters: tuple[CharacterCluster, ...],
                         config: ClusteringConfig) -> int:
    crop_ids, embeddings = load_embeddings(paths.embeddings_path)
    crop_id_rows = {crop_id: row for row, crop_id in enumerate(crop_ids)}
    store = CropStore(paths.crops_dir)
    ensure_directory(paths.sheets_dir)
    written = 0
    for cluster in clusters:
        selected = _sheet_crop_ids(cluster, crop_id_rows, embeddings, config.sheet_grid_size ** 2)
        images = [cv2.imread(str(store.crop_path(crop_id))) for crop_id in selected]
        readable = [image for image in images if image is not None]
        if not readable:
            continue
        sheet = build_contact_sheet(readable, config.sheet_grid_size, config.sheet_tile_pixels)
        cv2.imwrite(str(sheet_path_of(paths.sheets_dir, cluster.cluster_id)), sheet)
        written += 1
    return written
