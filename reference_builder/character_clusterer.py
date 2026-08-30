from __future__ import annotations

import numpy
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from reference_builder.reference_config import ClusteringConfig
from reference_builder.reference_models import CharacterCluster
from reference_builder.reference_selector import medoid_row
from sharedkernel.logger import get_logger

LOGGER = get_logger("reference_builder.character_clusterer")

_NOISE_LABEL = -1


def _reduced(embeddings: numpy.ndarray, pca_components: int) -> numpy.ndarray:
    components = min(pca_components, embeddings.shape[1], embeddings.shape[0])
    if components >= embeddings.shape[1]:
        return embeddings
    return PCA(n_components=components, random_state=0).fit_transform(embeddings)


def _to_cluster(cluster_id: int, rows: numpy.ndarray, crop_ids: tuple[str, ...],
                embeddings: numpy.ndarray) -> CharacterCluster:
    medoid = rows[medoid_row(embeddings[rows])]
    return CharacterCluster(
            cluster_id=cluster_id,
            crop_ids=tuple(crop_ids[row] for row in rows),
            medoid_crop_id=crop_ids[medoid],
    )


def cluster_crops(crop_ids: tuple[str, ...], embeddings: numpy.ndarray,
                  config: ClusteringConfig) -> tuple[CharacterCluster, ...]:
    reduced = _reduced(embeddings, config.pca_components)
    labels = HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            metric="euclidean",
    ).fit_predict(reduced)
    cluster_ids = sorted(int(label) for label in set(labels.tolist()) if label != _NOISE_LABEL)
    LOGGER.info("clustering finished: crops=%d clusters=%d noise=%d",
                len(crop_ids), len(cluster_ids), int((labels == _NOISE_LABEL).sum()))
    return tuple(
            _to_cluster(cluster_id, numpy.flatnonzero(labels == cluster_id), crop_ids, reduced)
            for cluster_id in cluster_ids
    )
