from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomli_w

from harvester.episode_models import Episode
from sharedkernel.utils.paths import ensure_directory

MANIFEST_FILE_NAME = "manifest.toml"


@dataclass(frozen=True)
class FrameRecord:
    file_name: str
    shot_index: int
    start_seconds: float
    timestamp_seconds: float


def _to_document(episode: Episode, frames: tuple[FrameRecord, ...]) -> dict:
    return {
        "slug": episode.slug,
        "video_id": episode.video_id,
        "title": episode.title,
        "url": episode.url,
        "source_name": episode.source_name,
        "frame_count": len(frames),
        "frames": [
            {
                "file_name": frame.file_name,
                "shot_index": frame.shot_index,
                "start_seconds": round(frame.start_seconds, 3),
                "timestamp_seconds": round(frame.timestamp_seconds, 3),
            }
            for frame in frames
        ],
    }


def write_frame_manifest(episode_frames_dir: Path, episode: Episode, frames: tuple[FrameRecord, ...]) -> Path:
    manifest_path = ensure_directory(episode_frames_dir) / MANIFEST_FILE_NAME
    with manifest_path.open("wb") as handle:
        tomli_w.dump(_to_document(episode, frames), handle)
    return manifest_path
