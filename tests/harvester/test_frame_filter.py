from __future__ import annotations

import unittest

import numpy as np

from harvester.frame_filter import FrameVerdict, evaluate_frame
from harvester.harvester_config import FrameExtractionConfig

_CONFIG = FrameExtractionConfig(
        content_threshold=27.0,
        min_shot_seconds=0.4,
        sample_stride_seconds=3.0,
        min_mean_luminance=12.0,
        min_laplacian_variance=60.0,
        min_pixel_standard_deviation=18.0,
)


def _noisy_frame(brightness: int) -> np.ndarray:
    generator = np.random.default_rng(seed=7)
    noise = generator.integers(0, 120, size=(64, 64, 3), dtype=np.int16)
    return np.clip(noise + brightness, 0, 255).astype(np.uint8)


def _uniform_frame(brightness: int) -> np.ndarray:
    return np.full((64, 64, 3), brightness, dtype=np.uint8)


def _smooth_gradient_frame() -> np.ndarray:
    column = np.linspace(0, 255, 64, dtype=np.uint8)
    return np.repeat(np.tile(column, (64, 1))[:, :, np.newaxis], 3, axis=2)


class TestFrameFilter(unittest.TestCase):

    def test_detailed_bright_frame_is_usable(self) -> None:
        self.assertEqual(FrameVerdict.USABLE, evaluate_frame(_noisy_frame(brightness=90), _CONFIG))

    def test_dark_frame_is_rejected_before_other_checks(self) -> None:
        self.assertEqual(FrameVerdict.TOO_DARK, evaluate_frame(_uniform_frame(brightness=5), _CONFIG))

    def test_uniform_title_card_is_rejected_as_flat(self) -> None:
        self.assertEqual(FrameVerdict.TOO_FLAT, evaluate_frame(_uniform_frame(brightness=200), _CONFIG))

    def test_smooth_gradient_without_edges_is_rejected_as_blurry(self) -> None:
        self.assertEqual(FrameVerdict.TOO_BLURRY, evaluate_frame(_smooth_gradient_frame(), _CONFIG))


if __name__ == "__main__":
    unittest.main()
