from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sharedkernel.config_provider import ConfigProvider
from transform.transform_config import DescriptionConfig, RenderingConfig, SegmentationConfig, TransformPaths

_EMPTY_CONFIG = """
[logging]
level = "INFO"
"""

_OVERRIDDEN_CONFIG = """
[transform.rendering]
inference_steps = 12
seed = 7
"""


class TestTransformConfig(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _provider(self, content: str) -> ConfigProvider:
        config_path = Path(self._directory.name) / "config.toml"
        config_path.write_text(content, encoding="utf-8")
        return ConfigProvider.from_file(config_path)

    def test_paths_fall_back_to_the_reference_cards(self) -> None:
        paths = TransformPaths.from_config_provider(self._provider(_EMPTY_CONFIG))
        self.assertEqual(Path("data/references/cards"), paths.references_dir)
        self.assertEqual(Path("data/output"), paths.output_dir)

    def test_segmentation_falls_back_to_defaults(self) -> None:
        segmentation = SegmentationConfig.from_config_provider(self._provider(_EMPTY_CONFIG))
        self.assertEqual("yolo11x-seg.pt", segmentation.model_id)
        self.assertEqual(0.3, segmentation.confidence_threshold)

    def test_rendering_reads_overrides(self) -> None:
        rendering = RenderingConfig.from_config_provider(self._provider(_OVERRIDDEN_CONFIG))
        self.assertEqual(12, rendering.inference_steps)
        self.assertEqual(7, rendering.seed)

    def test_rendering_falls_back_to_defaults(self) -> None:
        rendering = RenderingConfig.from_config_provider(self._provider(_EMPTY_CONFIG))
        self.assertEqual(1024, rendering.render_pixels)
        self.assertEqual(40, rendering.inference_steps)

    def test_description_without_api_key_raises_naming_the_variable(self) -> None:
        with self.assertRaises(ValueError) as raised:
            DescriptionConfig.from_config_provider(self._provider(_EMPTY_CONFIG))
        self.assertIn("BOMBIFIKATOR_ANTHROPIC_API_KEY", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
