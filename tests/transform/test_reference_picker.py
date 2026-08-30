from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy

from transform.reference_picker import load_references


class TestReferencePicker(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self._references_dir = Path(self._directory.name)
        character_dir = self._references_dir / "kurvinox"
        character_dir.mkdir()
        for index in range(4):
            card = numpy.zeros((10, 10, 4), dtype=numpy.uint8)
            card[:, :, 3] = 255
            cv2.imwrite(str(character_dir / f"{index:02d}.png"), card)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_only_the_requested_number_is_loaded(self) -> None:
        self.assertEqual(2, len(load_references(self._references_dir, "kurvinox", 2)))

    def test_transparent_pixels_are_flattened_onto_white(self) -> None:
        card = numpy.zeros((10, 10, 4), dtype=numpy.uint8)
        cv2.imwrite(str(self._references_dir / "kurvinox" / "00.png"), card)
        loaded = load_references(self._references_dir, "kurvinox", 1)[0]
        self.assertEqual(255, int(loaded[0, 0, 0]))

    def test_loaded_references_have_three_channels(self) -> None:
        self.assertEqual(3, load_references(self._references_dir, "kurvinox", 1)[0].shape[2])

    def test_character_without_a_directory_gives_nothing(self) -> None:
        self.assertEqual((), load_references(self._references_dir, "nieznany", 2))


if __name__ == "__main__":
    unittest.main()
