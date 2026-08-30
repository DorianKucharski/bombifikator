from __future__ import annotations

import unittest

import numpy

from reference_builder.contact_sheet import build_contact_sheet


def _solid_image(height: int, width: int, value: int) -> numpy.ndarray:
    return numpy.full((height, width, 3), value, dtype=numpy.uint8)


class TestContactSheet(unittest.TestCase):

    def test_sheet_has_grid_sized_canvas(self) -> None:
        sheet = build_contact_sheet([_solid_image(40, 20, 255)], grid_size=3, tile_pixels=64)
        self.assertEqual((192, 192, 3), sheet.shape)

    def test_missing_tiles_stay_background(self) -> None:
        sheet = build_contact_sheet([_solid_image(40, 20, 255)], grid_size=2, tile_pixels=32)
        self.assertEqual(24, int(sheet[-1, -1, 0]))

    def test_tall_image_is_letterboxed_not_stretched(self) -> None:
        sheet = build_contact_sheet([_solid_image(64, 16, 255)], grid_size=1, tile_pixels=64)
        self.assertEqual(24, int(sheet[32, 0, 0]))
        self.assertEqual(255, int(sheet[32, 32, 0]))

    def test_extra_images_beyond_the_grid_are_dropped(self) -> None:
        sheet = build_contact_sheet([_solid_image(8, 8, 255)] * 9, grid_size=2, tile_pixels=16)
        self.assertEqual((32, 32, 3), sheet.shape)


if __name__ == "__main__":
    unittest.main()
