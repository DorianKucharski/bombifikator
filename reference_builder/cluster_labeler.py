from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import anthropic

from codex.character_models import Character
from codex.character_store import CharacterStore
from reference_builder.cluster_store import read_clusters
from reference_builder.contact_sheet import sheet_path_of
from reference_builder.label_store import LabelStore
from reference_builder.labeling_schema import CLUSTER_LABELING_SCHEMA
from reference_builder.reference_config import LabelingConfig, ReferencePaths
from reference_builder.reference_models import UNKNOWN_LABEL, CharacterCluster, ClusterLabel
from sharedkernel.clients.anthropic_client import StructuredExtractionClient
from sharedkernel.logger import get_logger

LOGGER = get_logger("reference_builder.cluster_labeler")

_SYSTEM_PROMPT_HEADER = """You identify characters from the Polish adult animated series Kapitan Bomba by \
Bartosz Walaszek. You receive a grid of frames cut from the series. Every tile in the grid is supposed to show \
the same character, taken from different shots.

Answer with the slug of the matching character from the catalogue below. Answer with {unknown} when the tiles \
show different characters, when the character is not in the catalogue, or when the tiles show scenery, text or \
an unrecognisable shape. A wrong slug poisons the training data, so prefer {unknown} over a guess.

Catalogue:
"""


@dataclass(frozen=True)
class LabelingReport:
    clusters_total: int
    clusters_labeled: int
    clusters_unknown: int
    clusters_skipped: int
    clusters_missing_sheet: int
    failed: tuple[int, ...]


def _catalogue_entry(character: Character) -> str:
    features = ", ".join(character.appearance.distinguishing[:6]) or "brak opisu"
    return (f"- {character.slug} | {character.canonical_name} | {character.species} | "
            f"{character.sex.value} | {character.appearance.build.value} | {features}")


def _build_system_prompt(characters: tuple[Character, ...]) -> str:
    return _SYSTEM_PROMPT_HEADER.format(unknown=UNKNOWN_LABEL) + "\n".join(
            _catalogue_entry(character) for character in characters)


def _load_catalogue(characters_dir: Path, prominences: tuple[str, ...]) -> tuple[Character, ...]:
    wanted = {prominence.upper() for prominence in prominences}
    characters = tuple(
            character for character in CharacterStore(characters_dir).load_all()
            if character.prominence.value in wanted
    )
    if not characters:
        raise ValueError(f"no characters with prominence {sorted(wanted)} in {characters_dir}")
    return characters


def _label_cluster(
        client: StructuredExtractionClient,
        system_prompt: str,
        known_slugs: frozenset[str],
        cluster: CharacterCluster,
        sheet_path: Path,
) -> ClusterLabel | None:
    payload = client.extract(
            system_prompt=system_prompt,
            user_prompt=f"Siatka klatek klastra {cluster.cluster_id}, {len(cluster.crop_ids)} kadrow w klastrze.",
            json_schema=CLUSTER_LABELING_SCHEMA,
            images=(sheet_path.read_bytes(),),
    )
    if payload is None:
        return None
    slug = payload["slug"] if payload["slug"] in known_slugs else UNKNOWN_LABEL
    return ClusterLabel(
            cluster_id=cluster.cluster_id,
            slug=slug,
            confidence=float(payload["confidence"]),
            reasoning=payload["reasoning"],
    )


def label_clusters(paths: ReferencePaths, characters_dir: Path, config: LabelingConfig,
                   refresh: bool) -> LabelingReport:
    clusters = read_clusters(paths.clusters_path)
    characters = _load_catalogue(characters_dir, config.prominences)
    system_prompt = _build_system_prompt(characters)
    known_slugs = frozenset(character.slug for character in characters)
    store = LabelStore(paths.labels_path)
    client = StructuredExtractionClient(
            api_key=config.api_key,
            model=config.model,
            max_tokens=config.max_tokens,
            effort=config.effort,
    )

    pending = [cluster for cluster in clusters if refresh or not store.contains(cluster.cluster_id)]
    has_sheet = {cluster.cluster_id: sheet_path_of(paths.sheets_dir, cluster.cluster_id).is_file()
                 for cluster in pending}
    labelable = [cluster for cluster in pending if has_sheet[cluster.cluster_id]]
    missing_sheet = [cluster for cluster in pending if not has_sheet[cluster.cluster_id]]
    LOGGER.info("labeling starting: clusters=%d pending=%d catalogue=%d",
                len(clusters), len(labelable), len(characters))

    def label(cluster: CharacterCluster) -> ClusterLabel | None:
        try:
            result = _label_cluster(client, system_prompt, known_slugs, cluster,
                                    sheet_path_of(paths.sheets_dir, cluster.cluster_id))
        except anthropic.APIError as error:
            LOGGER.warning("labeling failed: cluster=%d error=%s", cluster.cluster_id, error)
            return None
        if result is not None:
            store.record(result)
            LOGGER.info("cluster labeled: cluster=%d slug=%s confidence=%.2f",
                        result.cluster_id, result.slug, result.confidence)
        return result

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        results = list(executor.map(label, labelable))

    identified = [result for result in results if result is not None]
    return LabelingReport(
            clusters_total=len(clusters),
            clusters_labeled=sum(1 for result in identified if result.is_identified),
            clusters_unknown=sum(1 for result in identified if not result.is_identified),
            clusters_skipped=len(clusters) - len(pending),
            clusters_missing_sheet=len(missing_sheet),
            failed=tuple(cluster.cluster_id for cluster, result in zip(labelable, results) if result is None),
    )
