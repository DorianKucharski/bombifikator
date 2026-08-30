from __future__ import annotations

import cv2
import numpy
import rembg

from sharedkernel.logger import get_logger

LOGGER = get_logger("reference_builder.background_remover")


class BackgroundRemover:
    def __init__(self) -> None:
        self._session = rembg.new_session()
        LOGGER.info("background remover ready")

    def cut_out(self, image: numpy.ndarray) -> numpy.ndarray:
        cut = rembg.remove(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), session=self._session)
        return cv2.cvtColor(numpy.asarray(cut), cv2.COLOR_RGBA2BGRA)
