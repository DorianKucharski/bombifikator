from __future__ import annotations

import unittest

import numpy

from sharedkernel.utils.geometry import BoundingBox
from transform.character_prompt import build_character_prompt
from transform.person_compositor import (crop_to_alpha, fitted_into, largest_component_only,
                                          paste_character)
from transform.person_eraser import dilated
from transform.render_size import render_size
from codex.character_models import (AgeGroup, AppearanceTraits, Build, Character, PersonDescription, Prominence, Sex)

_CHARACTER = Character(
        slug="kurvinox",
        canonical_name="Kurvinox",
        species="kurvinox",
        sex=Sex.MALE,
        age_group=AgeGroup.ADULT,
        prominence=Prominence.RECURRING,
        appearance=AppearanceTraits(build=Build.REPTILIAN, distinguishing=("dlugi ogon",), outfit="mundur"),
)

_DESCRIPTION = PersonDescription(
        person_index=0,
        sex=Sex.MALE,
        age_group=AgeGroup.ADULT,
        build_description="wysoki mezczyzna",
        outfit="czarna kurtka",
        pose="stoi bokiem",
        facing="w lewo",
)


def _opaque_character(height: int, width: int, value: int = 200) -> numpy.ndarray:
    character = numpy.full((height, width, 4), value, dtype=numpy.uint8)
    character[:, :, 3] = 255
    return character


class TestPersonCompositor(unittest.TestCase):

    def test_character_is_scaled_to_fit_inside_the_box(self) -> None:
        self.assertEqual((100, 200), fitted_into(BoundingBox(0, 0, 100, 200), 200, 400))

    def test_wider_character_is_limited_by_the_box_width(self) -> None:
        self.assertEqual((100, 50), fitted_into(BoundingBox(0, 0, 100, 200), 400, 200))

    def test_pasting_covers_the_box_bottom(self) -> None:
        photo = numpy.zeros((200, 200, 3), dtype=numpy.uint8)
        pasted, _ = paste_character(photo, _opaque_character(100, 50), BoundingBox(50, 50, 100, 150))
        self.assertEqual(200, int(pasted[149, 75, 0]))

    def test_pasting_leaves_the_rest_of_the_photo_untouched(self) -> None:
        photo = numpy.zeros((200, 200, 3), dtype=numpy.uint8)
        pasted, _ = paste_character(photo, _opaque_character(100, 50), BoundingBox(50, 50, 100, 150))
        self.assertEqual(0, int(pasted[10, 10, 0]))

    def test_transparent_character_changes_nothing(self) -> None:
        photo = numpy.full((200, 200, 3), 33, dtype=numpy.uint8)
        character = _opaque_character(100, 50)
        character[:, :, 3] = 0
        pasted, _ = paste_character(photo, character, BoundingBox(50, 50, 100, 150))
        self.assertTrue(numpy.array_equal(photo, pasted))


class TestLargestComponent(unittest.TestCase):

    def test_detached_blob_is_dropped(self) -> None:
        character = numpy.zeros((100, 100, 4), dtype=numpy.uint8)
        character[10:60, 10:60, 3] = 255
        character[80:90, 80:90, 3] = 255
        kept = largest_component_only(character)
        self.assertEqual(255, int(kept[30, 30, 3]))
        self.assertEqual(0, int(kept[85, 85, 3]))

    def test_single_figure_is_left_alone(self) -> None:
        character = numpy.zeros((100, 100, 4), dtype=numpy.uint8)
        character[10:60, 10:60, 3] = 255
        self.assertTrue(numpy.array_equal(character, largest_component_only(character)))

    def test_fully_transparent_character_survives(self) -> None:
        character = numpy.zeros((10, 10, 4), dtype=numpy.uint8)
        self.assertTrue(numpy.array_equal(character, largest_component_only(character)))


class TestAlphaCropping(unittest.TestCase):

    def test_transparent_margins_are_trimmed(self) -> None:
        character = numpy.zeros((100, 100, 4), dtype=numpy.uint8)
        character[40:60, 45:55, 3] = 255
        self.assertEqual((20, 10), crop_to_alpha(character).shape[:2])

    def test_fully_transparent_character_is_left_alone(self) -> None:
        character = numpy.zeros((10, 10, 4), dtype=numpy.uint8)
        self.assertEqual((10, 10), crop_to_alpha(character).shape[:2])

    def test_trimmed_character_fills_the_box(self) -> None:
        photo = numpy.zeros((200, 200, 3), dtype=numpy.uint8)
        character = numpy.zeros((100, 100, 4), dtype=numpy.uint8)
        character[40:60, 40:60] = 255
        pasted, _ = paste_character(photo, character, BoundingBox(50, 50, 100, 100))
        self.assertEqual(255, int(pasted[99, 75, 0]))


class TestPersonEraser(unittest.TestCase):

    def test_dilation_grows_the_mask(self) -> None:
        mask = numpy.zeros((64, 64), dtype=bool)
        mask[30:34, 30:34] = True
        self.assertGreater(dilated(mask, 0.1).sum(), mask.sum())

    def test_dilation_keeps_the_mask_shape(self) -> None:
        mask = numpy.zeros((64, 64), dtype=bool)
        mask[30:34, 30:34] = True
        self.assertEqual((64, 64), dilated(mask, 0.1).shape)


class TestRenderSize(unittest.TestCase):

    def test_render_size_is_a_multiple_of_sixteen(self) -> None:
        width, height = render_size(333, 777, 1024)
        self.assertEqual(0, width % 16)
        self.assertEqual(0, height % 16)

    def test_tall_box_renders_tall(self) -> None:
        width, height = render_size(200, 600, 1024)
        self.assertGreater(height, width)

    def test_small_box_is_still_rendered_large_enough(self) -> None:
        width, height = render_size(40, 90, 1024)
        self.assertGreaterEqual(min(width, height), 256)


class TestCharacterPrompt(unittest.TestCase):

    def test_prompt_starts_with_the_trained_token(self) -> None:
        self.assertTrue(build_character_prompt(_CHARACTER, _DESCRIPTION, "bmb16").startswith("bmb16,"))

    def test_prompt_carries_the_pose(self) -> None:
        self.assertIn("stoi bokiem", build_character_prompt(_CHARACTER, _DESCRIPTION, "bmb16"))

    def test_prompt_carries_the_facing(self) -> None:
        self.assertIn("w lewo", build_character_prompt(_CHARACTER, _DESCRIPTION, "bmb16"))

    def test_prompt_demands_a_plain_background(self) -> None:
        self.assertIn("plain white background", build_character_prompt(_CHARACTER, _DESCRIPTION, "bmb16"))

    def test_prompt_forbids_invented_props(self) -> None:
        self.assertIn("no props", build_character_prompt(_CHARACTER, _DESCRIPTION, "bmb16"))


if __name__ == "__main__":
    unittest.main()
