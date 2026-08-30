from __future__ import annotations

import unittest

import numpy

from reference_builder.character_clusterer import cluster_crops
from reference_builder.reference_config import ClusteringConfig

_CONFIG = ClusteringConfig(min_cluster_size=5, min_samples=2, pca_components=64,
                           sheet_grid_size=3, sheet_tile_pixels=256)


def _two_separated_blobs() -> tuple[tuple[str, ...], numpy.ndarray]:
    generator = numpy.random.default_rng(11)
    first = generator.normal(loc=0.0, scale=0.05, size=(20, 8))
    second = generator.normal(loc=6.0, scale=0.05, size=(20, 8))
    embeddings = numpy.vstack((first, second)).astype(numpy.float32)
    crop_ids = tuple(f"odcinek-1/{index:04d}-00-00" for index in range(len(embeddings)))
    return crop_ids, embeddings


class TestCharacterClusterer(unittest.TestCase):

    def test_two_separated_blobs_become_two_clusters(self) -> None:
        crop_ids, embeddings = _two_separated_blobs()
        clusters = cluster_crops(crop_ids, embeddings, _CONFIG)
        self.assertEqual(2, len(clusters))

    def test_every_crop_id_belongs_to_at_most_one_cluster(self) -> None:
        crop_ids, embeddings = _two_separated_blobs()
        clusters = cluster_crops(crop_ids, embeddings, _CONFIG)
        clustered = [crop_id for cluster in clusters for crop_id in cluster.crop_ids]
        self.assertEqual(len(clustered), len(set(clustered)))

    def test_medoid_belongs_to_its_own_cluster(self) -> None:
        crop_ids, embeddings = _two_separated_blobs()
        for cluster in cluster_crops(crop_ids, embeddings, _CONFIG):
            self.assertIn(cluster.medoid_crop_id, cluster.crop_ids)


if __name__ == "__main__":
    unittest.main()
