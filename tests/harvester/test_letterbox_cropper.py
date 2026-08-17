from __future__ import annotations

import unittest

import numpy as np

from harvester.letterbox_cropper import crop_letterbox


def _frame_with_black_bars(bar_height: int) -> np.ndarray:
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    frame[:bar_height, :, :] = 0
    frame[-bar_height:, :, :] = 0
    return frame


class TestLetterboxCropper(unittest.TestCase):

    def test_black_bars_are_removed(self) -> None:
        cropped = crop_letterbox(_frame_with_black_bars(bar_height=60))
        self.assertEqual((360, 640, 3), cropped.shape)

    def test_frame_without_bars_is_returned_unchanged(self) -> None:
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        self.assertEqual(frame.shape, crop_letterbox(frame).shape)

    def test_mostly_black_frame_is_not_cropped_away(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[230:250, 310:330, :] = 200
        self.assertEqual(frame.shape, crop_letterbox(frame).shape)


if __name__ == "__main__":
    unittest.main()
