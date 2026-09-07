from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy

from codex.character_models import Character, ReferenceCoverage
from codex.character_store import CharacterStore
from reference_builder.background_remover import BackgroundRemover
from reference_builder.cutout_quality import background_removed, cutout_clear_of_frame
from reference_builder.cluster_store import read_clusters
from reference_builder.coverage_marker import coverage_of, mark_coverage
from reference_builder.crop_store import CropStore
from reference_builder.embedding_store import load_embeddings
from reference_builder.label_store import read_labels
from reference_builder.reference_config import ReferencePaths, SelectionConfig
from reference_builder.reference_models import CharacterCluster, ClusterLabel, CropRecord, ReferenceCard
from reference_builder.reference_selector import medoid_row, rows_near_medoid, select_diverse_rows
from reference_builder.reference_store import ReferenceStore
from sharedkernel.logger import get_logger

LOGGER = get_logger("reference_builder.reference_pipeline")


@dataclass(frozen=True)
class ReferenceReport:
    characters_covered: int
    references_written: int
    full_coverage: int
    thin_coverage: int
    characters_below_threshold: int
    labels_ignored: int


def _cards_by_slug(clusters: tuple[CharacterCluster, ...], labels: tuple[ClusterLabel, ...],
                   min_confidence: float) -> tuple[dict[str, ReferenceCard], int]:
    clusters_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    accepted = [
        label for label in labels
        if label.is_identified and label.confidence >= min_confidence and label.cluster_id in clusters_by_id
    ]
    crop_ids_by_slug: dict[str, list[str]] = defaultdict(list)
    cluster_ids_by_slug: dict[str, list[int]] = defaultdict(list)
    for label in accepted:
        crop_ids_by_slug[label.slug].extend(clusters_by_id[label.cluster_id].crop_ids)
        cluster_ids_by_slug[label.slug].append(label.cluster_id)
    cards = {
        slug: ReferenceCard(
                slug=slug,
                crop_ids=tuple(crop_ids),
                cluster_ids=tuple(sorted(cluster_ids_by_slug[slug])),
        )
        for slug, crop_ids in crop_ids_by_slug.items()
    }
    return cards, len(labels) - len(accepted)


def _selected_crop_ids(card: ReferenceCard, crop_id_rows: dict[str, int], embeddings: numpy.ndarray,
                       config: SelectionConfig) -> tuple[str, ...]:
    known = tuple(crop_id for crop_id in card.crop_ids if crop_id in crop_id_rows)
    if not known:
        return ()
    rows = numpy.array([crop_id_rows[crop_id] for crop_id in known])
    core_rows = rows_near_medoid(embeddings[rows], config.identity_deviation_tolerance)
    core_ids = tuple(known[row] for row in core_rows)
    core_embeddings = embeddings[rows[list(core_rows)]]
    selected = select_diverse_rows(core_embeddings, config.target_references, medoid_row(core_embeddings))
    return tuple(core_ids[row] for row in selected)


def _reference_images(crop_store: CropStore, crop_ids: tuple[str, ...], remover: BackgroundRemover | None,
                      config: SelectionConfig) -> tuple[tuple[numpy.ndarray, ...], tuple[str, ...]]:
    images = []
    readable_ids = []
    for crop_id in crop_ids:
        image = cv2.imread(str(crop_store.crop_path(crop_id)))
        if image is None:
            continue
        if remover is None:
            images.append(image)
            readable_ids.append(crop_id)
            continue
        cut = remover.cut_out(image)
        if not background_removed(cut, config.max_opaque_ratio):
            continue
        if not cutout_clear_of_frame(cut, config.max_border_opaque_ratio):
            continue
        images.append(cut)
        readable_ids.append(crop_id)
    return tuple(images), tuple(readable_ids)


def _characters_by_slug(characters_dir: Path) -> dict[str, Character]:
    return {character.slug: character for character in CharacterStore(characters_dir).load_all()}


def build_references(paths: ReferencePaths, characters_dir: Path, config: SelectionConfig) -> ReferenceReport:
    cards, labels_ignored = _cards_by_slug(
            read_clusters(paths.clusters_path), read_labels(paths.labels_path), config.min_label_confidence)
    if not cards:
        raise ValueError(f"no accepted cluster labels in {paths.labels_path}: run 'references label' first")

    crop_ids, embeddings = load_embeddings(paths.embeddings_path)
    crop_id_rows = {crop_id: row for row, crop_id in enumerate(crop_ids)}
    crop_store = CropStore(paths.crops_dir)
    crop_records: dict[str, CropRecord] = {record.crop_id: record for record in crop_store.load_all()}
    reference_store = ReferenceStore(paths.references_dir)
    character_store = CharacterStore(characters_dir)
    characters = _characters_by_slug(characters_dir)
    remover = BackgroundRemover() if config.remove_background else None

    references_written = 0
    coverage_counts: dict[ReferenceCoverage, int] = defaultdict(int)
    for slug in sorted(cards):
        card = cards[slug]
        selected = _selected_crop_ids(card, crop_id_rows, embeddings, config)
        images, readable_ids = _reference_images(crop_store, selected, remover, config)
        coverage = coverage_of(len(images), config.full_reference_threshold, config.thin_reference_threshold)
        coverage_counts[coverage] += 1
        if not images:
            LOGGER.warning("no readable crops for character: slug=%s", slug)
            continue
        sources = tuple(crop_records[crop_id] for crop_id in readable_ids)
        file_names = reference_store.save_references(card, images)
        reference_store.write_manifest(card, sources, file_names)
        references_written += len(file_names)
        if slug in characters:
            mark_coverage(character_store, characters[slug], coverage)
        else:
            LOGGER.warning("labelled slug missing from codex: slug=%s", slug)
        LOGGER.info("references built: slug=%s references=%d clusters=%d coverage=%s",
                    slug, len(file_names), len(card.cluster_ids), coverage.value)

    return ReferenceReport(
            characters_covered=len(cards),
            references_written=references_written,
            full_coverage=coverage_counts[ReferenceCoverage.FULL],
            thin_coverage=coverage_counts[ReferenceCoverage.THIN],
            characters_below_threshold=coverage_counts[ReferenceCoverage.NONE],
            labels_ignored=labels_ignored,
    )
