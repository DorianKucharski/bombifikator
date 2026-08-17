from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yt_dlp import YoutubeDL

from harvester.episode_catalogue import EpisodeCatalogue
from harvester.episode_models import Episode
from harvester.harvester_config import HarvesterConfig, VideoSource
from sharedkernel.logger import get_logger
from sharedkernel.utils.paths import ensure_directory

LOGGER = get_logger("harvester.episode_downloader")

_WATCH_URL_TEMPLATE = "https://www.youtube.com/watch?v=%s"
_VIDEO_SUFFIXES = frozenset({".mp4", ".webm", ".mkv"})
_PARTIAL_SUFFIXES = frozenset({".part", ".ytdl"})
_DEFAULT_PLAYER_CLIENT = "default"

_LISTING_OPTIONS = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": "in_playlist",
    "skip_download": True,
}


def _extractor_args(player_client: str) -> dict:
    if player_client == _DEFAULT_PLAYER_CLIENT:
        return {}
    return {"youtube": {"player_client": [player_client]}}


@dataclass(frozen=True)
class DownloadReport:
    discovered: int
    downloaded: int
    skipped_existing: int
    failed: tuple[str, ...]


def _to_episode(entry: dict, source_name: str) -> Episode | None:
    video_id = entry.get("id")
    title = entry.get("title")
    if not video_id or not title:
        return None
    return Episode(
            video_id=video_id,
            title=title,
            url=entry.get("webpage_url") or _WATCH_URL_TEMPLATE % video_id,
            source_name=source_name,
    )


def _entries_of(info: dict) -> list[dict]:
    entries = info.get("entries")
    if entries is None:
        return [info]
    return [entry for entry in entries if entry is not None]


def list_episodes(source: VideoSource) -> tuple[Episode, ...]:
    with YoutubeDL(_LISTING_OPTIONS) as downloader:
        info = downloader.extract_info(source.url, download=False)
    episodes = tuple(
            episode for episode in
            (_to_episode(entry, source.name) for entry in _entries_of(info))
            if episode is not None
    )
    LOGGER.info("source listed: name=%s episodes=%d", source.name, len(episodes))
    return episodes


def video_path_of(episodes_dir: Path, episode: Episode) -> Path | None:
    candidates = sorted(
            path for path in episodes_dir.glob(f"{episode.slug}.*")
            if path.suffix.lower() in _VIDEO_SUFFIXES
    )
    return next(iter(candidates), None)


def _download_options(
        episodes_dir: Path,
        episode: Episode,
        config: HarvesterConfig,
        player_client: str,
) -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "format": config.format_selector,
        "outtmpl": str(episodes_dir / f"{episode.slug}.%(ext)s"),
        "extractor_args": _extractor_args(player_client),
    }


def _discard_partial_downloads(episodes_dir: Path, episode: Episode) -> None:
    for path in episodes_dir.glob(f"{episode.slug}.*"):
        if path.suffix.lower() in _PARTIAL_SUFFIXES:
            LOGGER.debug("discarding stale partial download: file=%s", path.name)
            path.unlink()


def _download_with_player_client(
        episodes_dir: Path,
        episode: Episode,
        config: HarvesterConfig,
        player_client: str,
) -> bool:
    _discard_partial_downloads(episodes_dir, episode)
    with YoutubeDL(_download_options(episodes_dir, episode, config, player_client)) as downloader:
        downloader.download([episode.url])
    return video_path_of(episodes_dir, episode) is not None


def _download_episode(episodes_dir: Path, episode: Episode, config: HarvesterConfig) -> bool:
    for player_client in config.player_clients:
        try:
            if _download_with_player_client(episodes_dir, episode, config, player_client):
                return True
        except Exception as error:
            LOGGER.debug("player client rejected episode: slug=%s client=%s error=%s",
                         episode.slug, player_client, error)
    return False


def download_all_sources(config: HarvesterConfig) -> DownloadReport:
    episodes_dir = ensure_directory(config.episodes_dir)
    episodes = tuple(episode for source in config.sources for episode in list_episodes(source))
    EpisodeCatalogue(episodes_dir).save(episodes)

    downloaded = 0
    skipped_existing = 0
    failed: list[str] = []
    for episode in episodes:
        if video_path_of(episodes_dir, episode) is not None:
            skipped_existing += 1
            continue
        if _download_episode(episodes_dir, episode, config):
            downloaded += 1
            LOGGER.info("episode downloaded: slug=%s", episode.slug)
        else:
            failed.append(episode.slug)
            LOGGER.warning("episode download failed for every player client: slug=%s", episode.slug)
    return DownloadReport(
            discovered=len(episodes),
            downloaded=downloaded,
            skipped_existing=skipped_existing,
            failed=tuple(failed),
    )
