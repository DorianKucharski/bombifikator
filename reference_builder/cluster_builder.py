from __future__ import annotations

from dataclasses import dataclass

from reference_builder.character_clusterer import cluster_crops
from reference_builder.cluster_store import write_clusters
from reference_builder.contact_sheet import write_cluster_sheets
from reference_builder.embedding_store import load_embeddings
from reference_builder.reference_config import ClusteringConfig, ReferencePaths
from sharedkernel.logger import get_logger

LOGGER = get_logger("reference_builder.cluster_builder")


@dataclass(frozen=True)
class ClusterReport:
    crops_clustered: int
    clusters_found: int
    crops_in_clusters: int
    sheets_written: int


def build_clusters(paths: ReferencePaths, config: ClusteringConfig) -> ClusterReport:
    crop_ids, embeddings = load_embeddings(paths.embeddings_path)
    clusters = cluster_crops(crop_ids, embeddings, config)
    write_clusters(paths.clusters_path, clusters)
    sheets_written = write_cluster_sheets(paths, clusters, config)
    LOGGER.info("clusters written: clusters=%d sheets=%d path=%s",
                len(clusters), sheets_written, paths.clusters_path)
    return ClusterReport(
            crops_clustered=len(crop_ids),
            clusters_found=len(clusters),
            crops_in_clusters=sum(len(cluster.crop_ids) for cluster in clusters),
            sheets_written=sheets_written,
    )
