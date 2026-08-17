from __future__ import annotations

import unittest

from harvester.episode_models import Episode


def _episode(title: str, video_id: str) -> Episode:
    return Episode(video_id=video_id, title=title, url=f"https://youtu.be/{video_id}", source_name="kapitan-bomba")


class TestEpisode(unittest.TestCase):

    def test_slug_transliterates_polish_title(self) -> None:
        episode = _episode("KAPITAN BOMBA - CZYNNOŚCI ADMINISTRACYJNE (ODC. 1)", "YrTYv8j8T7E")
        self.assertEqual("kapitan-bomba-czynnosci-administracyjne-odc-1-YrTYv8j8T7E", episode.slug)

    def test_episodes_sharing_a_title_keep_distinct_slugs(self) -> None:
        first = _episode("ZEMSTA", "aaaaaaaaaaa")
        second = _episode("ZEMSTA", "bbbbbbbbbbb")
        self.assertNotEqual(first.slug, second.slug)


if __name__ == "__main__":
    unittest.main()
