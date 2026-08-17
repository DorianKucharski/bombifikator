from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2

from harvester.episode_catalogue import EpisodeCatalogue
from harvester.episode_downloader import video_path_of
from harvester.episode_models import Episode
from harvester.frame_filter import FrameVerdict, evaluate_frame
from harvester.frame_manifest import MANIFEST_FILE_NAME, FrameRecord, write_frame_manifest
from harvester.harvester_config import FrameExtractionConfig, HarvesterConfig
from harvester.letterbox_cropper import crop_letterbox
from harvester.scene_splitter import detect_shots, grab_sampled_frames
from sharedkernel.logger import get_logger
from sharedkernel.utils.paths import ensure_directory

LOGGER = get_logger("harvester.frame_harvester")

_FRAME_FILE_TEMPLATE = "%04d-%02d.png"


@dataclass(frozen=True)
class FrameReport:
    episodes_processed: int
    episodes_skipped: int
    episodes_missing_video: int
    shots_detected: int
    frames_saved: int
    rejections: tuple[tuple[str, int], ...]


def _frames_dir_of(config: HarvesterConfig, episode: Episode) -> Path:
    return config.frames_dir / episode.slug


def _already_harvested(episode_frames_dir: Path) -> bool:
    return (episode_frames_dir / MANIFEST_FILE_NAME).is_file()


def _harvest_episode(
        video_path: Path,
        episode_frames_dir: Path,
        episode: Episode,
        extraction: FrameExtractionConfig,
        rejections: Counter,
) -> tuple[int, tuple[FrameRecord, ...]]:
    shots = detect_shots(video_path, extraction)
    ensure_directory(episode_frames_dir)
    records: list[FrameRecord] = []
    for sampled in grab_sampled_frames(video_path, shots, extraction.sample_stride_seconds):
        image = crop_letterbox(sampled.image)
        verdict = evaluate_frame(image, extraction)
        if verdict is not FrameVerdict.USABLE:
            rejections[verdict.value] += 1
            continue
        file_name = _FRAME_FILE_TEMPLATE % (sampled.shot.index, sampled.sample_index)
        cv2.imwrite(str(episode_frames_dir / file_name), image)
        records.append(FrameRecord(
                file_name=file_name,
                shot_index=sampled.shot.index,
                start_seconds=sampled.shot.start_seconds,
                timestamp_seconds=sampled.timestamp_seconds,
        ))
    write_frame_manifest(episode_frames_dir, episode, tuple(records))
    return len(shots), tuple(records)


def harvest_frames(config: HarvesterConfig, extraction: FrameExtractionConfig, refresh: bool) -> FrameReport:
    episodes = EpisodeCatalogue(config.episodes_dir).load()
    if not episodes:
        raise ValueError(
                f"empty episode catalogue in {config.episodes_dir}: run 'harvester download' first")

    rejections: Counter = Counter()
    episodes_processed = 0
    episodes_skipped = 0
    episodes_missing_video = 0
    shots_detected = 0
    frames_saved = 0
    for episode in episodes:
        episode_frames_dir = _frames_dir_of(config, episode)
        if not refresh and _already_harvested(episode_frames_dir):
            episodes_skipped += 1
            continue
        video_path = video_path_of(config.episodes_dir, episode)
        if video_path is None:
            episodes_missing_video += 1
            LOGGER.warning("video missing for episode: slug=%s", episode.slug)
            continue
        shot_count, records = _harvest_episode(
                video_path, episode_frames_dir, episode, extraction, rejections)
        episodes_processed += 1
        shots_detected += shot_count
        frames_saved += len(records)
        LOGGER.info("episode harvested: slug=%s shots=%d frames=%d", episode.slug, shot_count, len(records))
    return FrameReport(
            episodes_processed=episodes_processed,
            episodes_skipped=episodes_skipped,
            episodes_missing_video=episodes_missing_video,
            shots_detected=shots_detected,
            frames_saved=frames_saved,
            rejections=tuple(sorted(rejections.items())),
    )
