from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harvester.episode_catalogue import EpisodeCatalogue
from harvester.episode_models import Episode

_EPISODES = (
    Episode(video_id="YrTYv8j8T7E", title="ODC. 1", url="https://youtu.be/YrTYv8j8T7E", source_name="odcinki"),
    Episode(video_id="PRvTe4s7k9Y", title="ODC. 2", url="https://youtu.be/PRvTe4s7k9Y", source_name="odcinki"),
)


class TestEpisodeCatalogue(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self._catalogue = EpisodeCatalogue(Path(self._directory.name))

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_saved_episodes_are_loaded_back_unchanged(self) -> None:
        self._catalogue.save(_EPISODES)
        self.assertEqual(_EPISODES, self._catalogue.load())

    def test_missing_catalogue_loads_as_empty(self) -> None:
        self.assertEqual((), self._catalogue.load())


if __name__ == "__main__":
    unittest.main()
