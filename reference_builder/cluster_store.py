from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from reference_builder.reference_models import CharacterCluster
from sharedkernel.utils.paths import ensure_directory


def _to_document(cluster: CharacterCluster) -> dict[str, Any]:
    return {
        "cluster_id": cluster.cluster_id,
        "medoid_crop_id": cluster.medoid_crop_id,
        "crop_count": len(cluster.crop_ids),
        "crop_ids": list(cluster.crop_ids),
    }


def _to_cluster(document: dict[str, Any]) -> CharacterCluster:
    return CharacterCluster(
            cluster_id=document["cluster_id"],
            crop_ids=tuple(document["crop_ids"]),
            medoid_crop_id=document["medoid_crop_id"],
    )


def write_clusters(clusters_path: Path, clusters: tuple[CharacterCluster, ...]) -> Path:
    ensure_directory(clusters_path.parent)
    with clusters_path.open("wb") as handle:
        tomli_w.dump({
            "cluster_count": len(clusters),
            "clusters": [_to_document(cluster) for cluster in clusters],
        }, handle)
    return clusters_path


def read_clusters(clusters_path: Path) -> tuple[CharacterCluster, ...]:
    if not clusters_path.is_file():
        raise ValueError(f"missing clusters file {clusters_path}: run 'references cluster' first")
    with clusters_path.open("rb") as handle:
        document = tomllib.load(handle)
    return tuple(_to_cluster(cluster) for cluster in document["clusters"])
