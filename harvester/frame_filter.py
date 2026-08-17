from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from harvester.harvester_config import FrameExtractionConfig


class FrameVerdict(str, Enum):
    USABLE = "USABLE"
    TOO_DARK = "TOO_DARK"
    TOO_FLAT = "TOO_FLAT"
    TOO_BLURRY = "TOO_BLURRY"


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _mean_luminance(grayscale: np.ndarray) -> float:
    return float(grayscale.mean())


def _pixel_standard_deviation(grayscale: np.ndarray) -> float:
    return float(grayscale.std())


def _laplacian_variance(grayscale: np.ndarray) -> float:
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def evaluate_frame(frame: np.ndarray, config: FrameExtractionConfig) -> FrameVerdict:
    grayscale = _to_grayscale(frame)
    if _mean_luminance(grayscale) < config.min_mean_luminance:
        return FrameVerdict.TOO_DARK
    if _pixel_standard_deviation(grayscale) < config.min_pixel_standard_deviation:
        return FrameVerdict.TOO_FLAT
    if _laplacian_variance(grayscale) < config.min_laplacian_variance:
        return FrameVerdict.TOO_BLURRY
    return FrameVerdict.USABLE
