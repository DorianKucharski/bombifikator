from __future__ import annotations

import cv2
import numpy
from ultralytics import YOLO

from reference_builder.torch_device import resolve_device
from sharedkernel.logger import get_logger
from sharedkernel.utils.geometry import BoundingBox
from transform.person_models import PersonInstance
from transform.transform_config import SegmentationConfig

LOGGER = get_logger("transform.person_detector")

_PERSON_CLASS_NAME = "person"


def _to_full_size_mask(raw_mask: numpy.ndarray, height: int, width: int) -> numpy.ndarray:
    resized = cv2.resize(raw_mask.astype(numpy.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _to_box(raw_box: numpy.ndarray, width: int, height: int) -> BoundingBox:
    left, top, right, bottom = (int(round(value)) for value in raw_box)
    return BoundingBox(
            left=max(0, left),
            top=max(0, top),
            right=min(width, right),
            bottom=min(height, bottom),
    )


class PersonDetector:
    def __init__(self, config: SegmentationConfig) -> None:
        self._config = config
        self._device = resolve_device(config.device)
        self._model = YOLO(config.model_id)
        LOGGER.info("segmentation model loaded: model=%s device=%s", config.model_id, self._device)

    def detect(self, image: numpy.ndarray) -> tuple[PersonInstance, ...]:
        height, width = image.shape[:2]
        result = self._model.predict(
                image, conf=self._config.confidence_threshold, device=self._device, verbose=False)[0]
        if result.masks is None:
            return ()
        class_names = result.names
        minimum_area = self._config.min_person_area_ratio * height * width
        people = []
        for raw_mask, raw_box, class_id, confidence in zip(
                result.masks.data.cpu().numpy(),
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
        ):
            if class_names[int(class_id)] != _PERSON_CLASS_NAME:
                continue
            mask = _to_full_size_mask(raw_mask, height, width)
            if mask.sum() < minimum_area:
                continue
            people.append(PersonInstance(
                    person_index=len(people),
                    box=_to_box(raw_box, width, height),
                    mask=mask,
                    confidence=float(confidence),
            ))
        people.sort(key=lambda person: person.pixel_area, reverse=True)
        return tuple(
                PersonInstance(person_index=index, box=person.box, mask=person.mask, confidence=person.confidence)
                for index, person in enumerate(people)
        )
