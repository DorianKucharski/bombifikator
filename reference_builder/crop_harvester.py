from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy

from reference_builder.character_detector import CharacterDetector
from reference_builder.crop_geometry import pad_box
from reference_builder.crop_store import CropStore, crop_id_of
from reference_builder.reference_config import DetectionConfig, ReferencePaths
from reference_builder.reference_models import CropRecord, Detection
from sharedkernel.logger import get_logger

LOGGER = get_logger("reference_builder.crop_harvester")

_FRAME_FILE_PATTERN = "*.png"


@dataclass(frozen=True)
class CropReport:
    episodes_processed: int
    episodes_skipped: int
    frames_scanned: int
    frames_without_detection: int
    crops_saved: int


def _episode_dirs(frames_dir: Path) -> tuple[Path, ...]:
    if not frames_dir.is_dir():
        raise ValueError(f"missing frames directory {frames_dir}: run 'harvester frames' first")
    return tuple(sorted(path for path in frames_dir.iterdir() if path.is_dir()))


def _batches(frame_paths: Sequence[Path], batch_size: int) -> Iterator[tuple[Path, ...]]:
    for start in range(0, len(frame_paths), batch_size):
        yield tuple(frame_paths[start:start + batch_size])


def _crop_of(frame: numpy.ndarray, detection: Detection, padding_ratio: float) -> numpy.ndarray:
    box = pad_box(detection.box, padding_ratio, frame.shape[1], frame.shape[0])
    return frame[box.top:box.bottom, box.left:box.right]


def _save_detections(
        store: CropStore,
        episode_slug: str,
        frame_path: Path,
        frame: numpy.ndarray,
        detections: tuple[Detection, ...],
        padding_ratio: float,
) -> tuple[CropRecord, ...]:
    records = []
    for detection_index, detection in enumerate(detections):
        record = CropRecord(
                crop_id=crop_id_of(episode_slug, frame_path.name, detection_index),
                episode_slug=episode_slug,
                frame_file_name=frame_path.name,
                detection_index=detection_index,
                score=detection.score,
                phrase=detection.phrase,
                box=detection.box,
        )
        store.save_crop(record, _crop_of(frame, detection, padding_ratio))
        records.append(record)
    return tuple(records)


def _harvest_episode(
        detector: CharacterDetector,
        store: CropStore,
        episode_dir: Path,
        config: DetectionConfig,
) -> tuple[int, int, tuple[CropRecord, ...]]:
    frame_paths = sorted(episode_dir.glob(_FRAME_FILE_PATTERN))
    records: list[CropRecord] = []
    frames_without_detection = 0
    for batch in _batches(frame_paths, config.batch_size):
        frames = [cv2.imread(str(path)) for path in batch]
        readable = [(path, frame) for path, frame in zip(batch, frames) if frame is not None]
        if not readable:
            continue
        detections_per_frame = detector.detect_batch([frame for _, frame in readable])
        for (path, frame), detections in zip(readable, detections_per_frame):
            if not detections:
                frames_without_detection += 1
                continue
            records.extend(_save_detections(
                    store, episode_dir.name, path, frame, detections, config.crop_padding_ratio))
    store.write_manifest(episode_dir.name, tuple(records))
    return len(frame_paths), frames_without_detection, tuple(records)


def harvest_crops(paths: ReferencePaths, config: DetectionConfig, refresh: bool) -> CropReport:
    detector = CharacterDetector(config)
    store = CropStore(paths.crops_dir)
    episodes_processed = 0
    episodes_skipped = 0
    frames_scanned = 0
    frames_without_detection = 0
    crops_saved = 0
    for episode_dir in _episode_dirs(paths.frames_dir):
        if not refresh and store.contains_episode(episode_dir.name):
            episodes_skipped += 1
            continue
        frame_count, without_detection, records = _harvest_episode(detector, store, episode_dir, config)
        episodes_processed += 1
        frames_scanned += frame_count
        frames_without_detection += without_detection
        crops_saved += len(records)
        LOGGER.info("episode cropped: slug=%s frames=%d crops=%d empty_frames=%d",
                    episode_dir.name, frame_count, len(records), without_detection)
    return CropReport(
            episodes_processed=episodes_processed,
            episodes_skipped=episodes_skipped,
            frames_scanned=frames_scanned,
            frames_without_detection=frames_without_detection,
            crops_saved=crops_saved,
    )
