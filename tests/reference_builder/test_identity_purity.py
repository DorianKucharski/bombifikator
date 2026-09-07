from __future__ import annotations

import unittest

import numpy

from reference_builder.cutout_quality import background_removed, cutout_clear_of_frame
from reference_builder.reference_selector import dominant_look_rows


class TestDominantLookRows(unittest.TestCase):

    def _two_looks(self) -> numpy.ndarray:
        helmet = numpy.array([[1.0, 0.0], [0.99, 0.1], [0.98, 0.12], [0.99, 0.08]])
        blond = numpy.array([[0.0, 1.0], [0.1, 0.99]])
        return numpy.vstack([helmet, blond])

    def test_the_smaller_look_is_dropped(self) -> None:
        self.assertEqual((0, 1, 2, 3), dominant_look_rows(self._two_looks(), 0.45))

    def test_one_look_keeps_every_row(self) -> None:
        embeddings = numpy.array([[1.0, 0.0], [0.99, 0.1], [0.98, 0.12]])
        self.assertEqual((0, 1, 2), dominant_look_rows(embeddings, 0.45))

    def test_two_rows_are_too_few_to_call_either_one_a_second_look(self) -> None:
        embeddings = numpy.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual((0, 1), dominant_look_rows(embeddings, 0.45))

    def test_a_threshold_wide_enough_merges_the_looks(self) -> None:
        self.assertEqual(6, len(dominant_look_rows(self._two_looks(), 2.0)))


class TestBackgroundRemoved(unittest.TestCase):

    def _cut(self, opaque_ratio: float) -> numpy.ndarray:
        image = numpy.zeros((10, 10, 4), dtype=numpy.uint8)
        image[:int(10 * opaque_ratio), :, 3] = 255
        return image

    def test_a_cut_out_that_kept_the_whole_frame_is_a_failure(self) -> None:
        self.assertFalse(background_removed(self._cut(1.0), 0.9))

    def test_a_cut_out_that_left_a_silhouette_passes(self) -> None:
        self.assertTrue(background_removed(self._cut(0.4), 0.9))

    def test_an_image_without_an_alpha_channel_is_a_failure(self) -> None:
        self.assertFalse(background_removed(numpy.zeros((10, 10, 3), dtype=numpy.uint8), 0.9))


class TestCutoutClearOfFrame(unittest.TestCase):

    def _cut_touching_frame(self, ring_opaque_ratio: float) -> numpy.ndarray:
        image = numpy.zeros((20, 20, 4), dtype=numpy.uint8)
        image[6:14, 6:14, 3] = 255
        opaque_columns = int(round(20 * ring_opaque_ratio))
        image[:3, :opaque_columns, 3] = 255
        return image

    def test_a_cut_out_that_kept_the_scenery_touching_the_frame_is_rejected(self) -> None:
        image = numpy.zeros((20, 20, 4), dtype=numpy.uint8)
        image[:, :, 3] = 255
        self.assertFalse(cutout_clear_of_frame(image, 0.35))

    def test_a_silhouette_away_from_the_frame_passes(self) -> None:
        image = numpy.zeros((20, 20, 4), dtype=numpy.uint8)
        image[6:14, 6:14, 3] = 255
        self.assertTrue(cutout_clear_of_frame(image, 0.35))

    def test_feet_reaching_the_bottom_edge_still_pass(self) -> None:
        self.assertTrue(cutout_clear_of_frame(self._cut_touching_frame(0.2), 0.35))

    def test_an_image_without_an_alpha_channel_is_a_failure(self) -> None:
        self.assertFalse(cutout_clear_of_frame(numpy.zeros((10, 10, 3), dtype=numpy.uint8), 0.35))


if __name__ == "__main__":
    unittest.main()
