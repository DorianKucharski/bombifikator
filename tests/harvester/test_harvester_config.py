from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harvester.harvester_config import FrameExtractionConfig, HarvesterConfig, VideoSource
from sharedkernel.config_provider import ConfigProvider

_CONFIG_WITH_SOURCE = """
[harvester]
episodes_dir = "data/episodes"

[[harvester.sources]]
name = "odcinki"
url = "https://www.youtube.com/playlist?list=X"
"""

_CONFIG_WITHOUT_SOURCES = """
[harvester]
episodes_dir = "data/episodes"
"""


class TestHarvesterConfig(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _provider(self, content: str) -> ConfigProvider:
        config_path = Path(self._directory.name) / "config.toml"
        config_path.write_text(content, encoding="utf-8")
        return ConfigProvider.from_file(config_path)

    def test_sources_are_read_as_video_sources(self) -> None:
        config = HarvesterConfig.from_config_provider(self._provider(_CONFIG_WITH_SOURCE))
        self.assertEqual(
                (VideoSource(name="odcinki", url="https://www.youtube.com/playlist?list=X"),),
                config.sources)

    def test_missing_sources_raise_naming_the_table(self) -> None:
        with self.assertRaises(ValueError) as raised:
            HarvesterConfig.from_config_provider(self._provider(_CONFIG_WITHOUT_SOURCES))
        self.assertIn("[[harvester.sources]]", str(raised.exception))

    def test_frame_extraction_falls_back_to_defaults(self) -> None:
        extraction = FrameExtractionConfig.from_config_provider(self._provider(_CONFIG_WITH_SOURCE))
        self.assertEqual(27.0, extraction.content_threshold)
        self.assertEqual(60.0, extraction.min_laplacian_variance)


if __name__ == "__main__":
    unittest.main()
