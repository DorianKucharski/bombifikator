from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video

from harvester.episode_models import Shot
from harvester.harvester_config import FrameExtractionConfig
from sharedkernel.logger import get_logger

LOGGER = get_logger("harvester.scene_splitter")


def _video_duration_seconds(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        frames_per_second = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if frames_per_second <= 0:
            return 0.0
        return frame_count / frames_per_second
    finally:
        capture.release()


def _whole_video_as_single_shot(video_path: Path) -> tuple[Shot, ...]:
    duration_seconds = _video_duration_seconds(video_path)
    if duration_seconds <= 0:
        return ()
    return (Shot(index=1, start_seconds=0.0, end_seconds=duration_seconds),)


def detect_shots(video_path: Path, config: FrameExtractionConfig) -> tuple[Shot, ...]:
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=config.content_threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scenes = scene_manager.get_scene_list()
    if not scenes:
        return _whole_video_as_single_shot(video_path)
    shots = tuple(
            Shot(index=index, start_seconds=start.get_seconds(), end_seconds=end.get_seconds())
            for index, (start, end) in enumerate(scenes, start=1)
    )
    return tuple(shot for shot in shots if shot.duration_seconds >= config.min_shot_seconds)


@dataclass(frozen=True)
class SampledFrame:
    shot: Shot
    sample_index: int
    timestamp_seconds: float
    image: np.ndarray


def grab_sampled_frames(
        video_path: Path,
        shots: tuple[Shot, ...],
        stride_seconds: float,
) -> Iterator[SampledFrame]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        for shot in shots:
            for sample_index, timestamp_seconds in enumerate(shot.sample_timestamps(stride_seconds), start=1):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
                was_read, frame = capture.read()
                if not was_read:
                    LOGGER.debug("frame unreadable: video=%s shot=%d sample=%d",
                                 video_path.name, shot.index, sample_index)
                    continue
                yield SampledFrame(
                        shot=shot,
                        sample_index=sample_index,
                        timestamp_seconds=timestamp_seconds,
                        image=frame,
                )
    finally:
        capture.release()
