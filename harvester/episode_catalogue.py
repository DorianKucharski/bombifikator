from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w

from harvester.episode_models import Episode
from sharedkernel.utils.paths import ensure_directory

_CATALOGUE_FILE_NAME = "catalogue.toml"


def _to_document(episodes: tuple[Episode, ...]) -> dict:
    return {
        "episodes": [
            {
                "slug": episode.slug,
                "video_id": episode.video_id,
                "title": episode.title,
                "url": episode.url,
                "source_name": episode.source_name,
            }
            for episode in episodes
        ]
    }


class EpisodeCatalogue:

    def __init__(self, episodes_dir: Path) -> None:
        self._catalogue_path = episodes_dir / _CATALOGUE_FILE_NAME

    def save(self, episodes: tuple[Episode, ...]) -> None:
        ensure_directory(self._catalogue_path.parent)
        with self._catalogue_path.open("wb") as handle:
            tomli_w.dump(_to_document(episodes), handle)

    def load(self) -> tuple[Episode, ...]:
        if not self._catalogue_path.is_file():
            return ()
        with self._catalogue_path.open("rb") as handle:
            document = tomllib.load(handle)
        return tuple(
                Episode(
                        video_id=entry["video_id"],
                        title=entry["title"],
                        url=entry["url"],
                        source_name=entry["source_name"],
                )
                for entry in document.get("episodes", ())
        )
