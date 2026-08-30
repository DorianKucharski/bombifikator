from __future__ import annotations

import unittest

import numpy

from reference_builder.reference_selector import medoid_row, select_diverse_rows


class TestReferenceSelector(unittest.TestCase):

    def test_medoid_is_the_most_central_row(self) -> None:
        embeddings = numpy.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        self.assertEqual(1, medoid_row(embeddings))

    def test_single_row_is_its_own_medoid(self) -> None:
        self.assertEqual(0, medoid_row(numpy.array([[3.0, 4.0]])))

    def test_selection_prefers_the_farthest_rows(self) -> None:
        embeddings = numpy.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [0.0, 5.0]])
        self.assertEqual({0, 2, 3}, set(select_diverse_rows(embeddings, 3, seed_row=0)))

    def test_selection_never_returns_more_rows_than_available(self) -> None:
        embeddings = numpy.array([[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(2, len(select_diverse_rows(embeddings, 10, seed_row=0)))

    def test_selection_never_repeats_a_row(self) -> None:
        embeddings = numpy.random.default_rng(7).normal(size=(20, 8))
        selected = select_diverse_rows(embeddings, 12, seed_row=3)
        self.assertEqual(len(selected), len(set(selected)))


if __name__ == "__main__":
    unittest.main()
