from __future__ import annotations

import unittest

import numpy

from reference_builder.cutout_quality import background_removed
from reference_builder.reference_selector import rows_near_medoid


class TestRowsNearMedoid(unittest.TestCase):

    def test_an_intruder_far_from_the_rest_is_dropped(self) -> None:
        embeddings = numpy.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.2, 0.1], [40.0, 40.0]])
        self.assertNotIn(4, rows_near_medoid(embeddings, 2.0))

    def test_a_tight_group_keeps_every_row(self) -> None:
        embeddings = numpy.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
        self.assertEqual(4, len(rows_near_medoid(embeddings, 2.0)))

    def test_two_rows_are_too_few_to_call_either_one_an_intruder(self) -> None:
        embeddings = numpy.array([[0.0, 0.0], [40.0, 40.0]])
        self.assertEqual((0, 1), rows_near_medoid(embeddings, 2.0))

    def test_a_tolerance_wide_enough_keeps_the_intruder(self) -> None:
        embeddings = numpy.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.2, 0.1], [40.0, 40.0]])
        self.assertIn(4, rows_near_medoid(embeddings, 10000.0))


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


if __name__ == "__main__":
    unittest.main()
