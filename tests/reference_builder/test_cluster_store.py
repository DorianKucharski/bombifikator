from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reference_builder.cluster_store import read_clusters, write_clusters
from reference_builder.label_store import LabelStore, read_labels
from reference_builder.reference_models import CharacterCluster, ClusterLabel

_CLUSTER = CharacterCluster(cluster_id=3, crop_ids=("odcinek-1/0001-00-00", "odcinek-1/0002-00-00"),
                            medoid_crop_id="odcinek-1/0001-00-00")


class TestClusterStore(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self._path = Path(self._directory.name) / "clusters.toml"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_round_trip_restores_the_cluster(self) -> None:
        write_clusters(self._path, (_CLUSTER,))
        self.assertEqual((_CLUSTER,), read_clusters(self._path))

    def test_missing_file_raises_naming_the_command_to_run(self) -> None:
        with self.assertRaises(ValueError) as raised:
            read_clusters(self._path)
        self.assertIn("references cluster", str(raised.exception))


class TestLabelStore(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self._path = Path(self._directory.name) / "labels.toml"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_recorded_label_survives_a_reload(self) -> None:
        LabelStore(self._path).record(ClusterLabel(cluster_id=3, slug="kurvinox", confidence=0.9, reasoning="luski"))
        self.assertTrue(LabelStore(self._path).contains(3))
        self.assertEqual("kurvinox", read_labels(self._path)[0].slug)

    def test_unknown_label_is_not_identified(self) -> None:
        label = ClusterLabel(cluster_id=1, slug="UNKNOWN", confidence=0.2, reasoning="mieszanka postaci")
        self.assertFalse(label.is_identified)

    def test_missing_labels_file_reads_as_empty(self) -> None:
        self.assertEqual((), read_labels(self._path))


if __name__ == "__main__":
    unittest.main()
