from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from reference_builder.reference_config import (ClusteringConfig, DetectionConfig, LabelingConfig, ReferencePaths,
                                                SelectionConfig)
from sharedkernel.config_provider import ConfigProvider

_EMPTY_CONFIG = """
[logging]
level = "INFO"
"""

_OVERRIDDEN_CONFIG = """
[reference_builder.detection]
box_threshold = 0.45
device = "cuda"

[reference_builder.labeling]
prominences = ["MAIN"]
"""


class TestReferenceConfig(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _provider(self, content: str) -> ConfigProvider:
        config_path = Path(self._directory.name) / "config.toml"
        config_path.write_text(content, encoding="utf-8")
        return ConfigProvider.from_file(config_path)

    def test_paths_fall_back_to_defaults(self) -> None:
        paths = ReferencePaths.from_config_provider(self._provider(_EMPTY_CONFIG))
        self.assertEqual(Path("data/frames"), paths.frames_dir)
        self.assertEqual(Path("data/references/crops"), paths.crops_dir)
        self.assertEqual(Path("data/references/embeddings.npz"), paths.embeddings_path)

    def test_detection_reads_overrides(self) -> None:
        detection = DetectionConfig.from_config_provider(self._provider(_OVERRIDDEN_CONFIG))
        self.assertEqual(0.45, detection.box_threshold)
        self.assertEqual("cuda", detection.device)

    def test_detection_device_defaults_to_auto(self) -> None:
        detection = DetectionConfig.from_config_provider(self._provider(_EMPTY_CONFIG))
        self.assertEqual("auto", detection.device)

    def test_clustering_falls_back_to_defaults(self) -> None:
        clustering = ClusteringConfig.from_config_provider(self._provider(_EMPTY_CONFIG))
        self.assertEqual(12, clustering.min_cluster_size)
        self.assertEqual(3, clustering.sheet_grid_size)

    def test_selection_falls_back_to_defaults(self) -> None:
        selection = SelectionConfig.from_config_provider(self._provider(_EMPTY_CONFIG))
        self.assertEqual(16, selection.target_references)
        self.assertTrue(selection.remove_background)

    def test_labeling_reads_prominence_list(self) -> None:
        os.environ["BOMBIFIKATOR_ANTHROPIC_API_KEY"] = "test-key"
        try:
            labeling = LabelingConfig.from_config_provider(self._provider(_OVERRIDDEN_CONFIG))
        finally:
            del os.environ["BOMBIFIKATOR_ANTHROPIC_API_KEY"]
        self.assertEqual(("MAIN",), labeling.prominences)

    def test_labeling_without_api_key_raises_naming_the_variable(self) -> None:
        with self.assertRaises(ValueError) as raised:
            LabelingConfig.from_config_provider(self._provider(_EMPTY_CONFIG))
        self.assertIn("BOMBIFIKATOR_ANTHROPIC_API_KEY", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
