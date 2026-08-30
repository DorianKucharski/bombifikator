from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from reference_builder.crop_geometry import clamp_box, covers_whole_frame, suppress_overlapping
from reference_builder.reference_config import DetectionConfig
from reference_builder.reference_models import Detection
from reference_builder.torch_device import resolve_device
from sharedkernel.logger import get_logger

LOGGER = get_logger("reference_builder.character_detector")


def _to_pil(frame: numpy.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _phrases_of(result: dict[str, Any]) -> list[str]:
    labels = result["text_labels"] if "text_labels" in result else result["labels"]
    return [str(label) for label in labels]


class CharacterDetector:
    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._device = resolve_device(config.device)
        self._processor = AutoProcessor.from_pretrained(config.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(config.model_id).to(self._device).eval()
        LOGGER.info("detector loaded: model=%s device=%s", config.model_id, self._device)

    def _keeps(self, detection: Detection, frame_width: int, frame_height: int) -> bool:
        if detection.box.shorter_side < self._config.min_crop_pixels:
            return False
        return not covers_whole_frame(
                detection.box, frame_width, frame_height, self._config.max_frame_coverage_ratio)

    def _to_detections(self, result: dict[str, Any], frame_width: int, frame_height: int) -> tuple[Detection, ...]:
        phrases = _phrases_of(result)
        detections = [
            Detection(
                    box=clamp_box(*boxes.tolist(), frame_width=frame_width, frame_height=frame_height),
                    score=float(score),
                    phrase=phrase,
            )
            for boxes, score, phrase in zip(result["boxes"].cpu(), result["scores"].cpu(), phrases)
        ]
        kept = [detection for detection in detections if self._keeps(detection, frame_width, frame_height)]
        kept.sort(key=lambda detection: detection.score, reverse=True)
        distinct = suppress_overlapping(tuple(kept), self._config.duplicate_iou_threshold)
        return distinct[:self._config.max_crops_per_frame]

    @torch.inference_mode()
    def detect_batch(self, frames: Sequence[numpy.ndarray]) -> tuple[tuple[Detection, ...], ...]:
        images = [_to_pil(frame) for frame in frames]
        inputs = self._processor(
                images=images,
                text=[self._config.text_prompt] * len(images),
                return_tensors="pt",
                padding=True,
        ).to(self._device)
        outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self._config.box_threshold,
                text_threshold=self._config.text_threshold,
                target_sizes=[image.size[::-1] for image in images],
        )
        return tuple(
                self._to_detections(result, frame.shape[1], frame.shape[0])
                for result, frame in zip(results, frames)
        )
