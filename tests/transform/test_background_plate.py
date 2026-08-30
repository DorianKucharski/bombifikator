from __future__ import annotations

import unittest

import numpy

from transform.person_compositor import toned_like, with_people_removed


class TestBackgroundPlate(unittest.TestCase):

    def setUp(self) -> None:
        self._photo = numpy.full((60, 60, 3), 10, dtype=numpy.uint8)
        self._people = numpy.zeros((60, 60), dtype=numpy.uint8)
        self._people[20:40, 20:40] = 1
        self._plate = numpy.full((60, 60, 3), 10, dtype=numpy.uint8)
        self._plate[15:45, 15:45] = 200

    def test_the_middle_of_the_person_comes_from_the_plate(self) -> None:
        removed = with_people_removed(self._photo, self._plate, self._people, 5)
        self.assertGreater(int(removed[30, 30, 0]), int(self._photo[30, 30, 0]))

    def test_far_from_the_person_the_photograph_is_untouched(self) -> None:
        removed = with_people_removed(self._photo, self._plate, self._people, 5)
        self.assertEqual(10, int(removed[0, 0, 0]))

    def test_the_seam_is_feathered_not_hard(self) -> None:
        removed = with_people_removed(self._photo, self._plate, self._people, 15)
        edge = int(removed[20, 30, 0])
        self.assertTrue(10 < edge < int(removed[30, 30, 0]))

    def test_toning_shifts_the_plate_towards_the_photograph(self) -> None:
        outside = ~self._people.astype(bool)
        bright_plate = numpy.full((60, 60, 3), 200, dtype=numpy.uint8)
        self.assertLess(int(toned_like(bright_plate, self._photo, outside)[0, 0, 0]), 200)

    def test_toning_without_any_reference_area_leaves_the_plate(self) -> None:
        outside = numpy.zeros((60, 60), dtype=bool)
        self.assertTrue(numpy.array_equal(self._plate, toned_like(self._plate, self._photo, outside)))


if __name__ == "__main__":
    unittest.main()
