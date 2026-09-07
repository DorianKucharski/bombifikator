from __future__ import annotations

import unittest

from codex.character_models import (AgeGroup, AppearanceTraits, Build, Character, Prominence, ReferenceCoverage, Sex)
from training.token_preview import build_preview_prompt, slugs_without_character


def _character(slug: str) -> Character:
    return Character(
            slug=slug,
            canonical_name=slug.title(),
            species="kurvinox",
            sex=Sex.MALE,
            age_group=AgeGroup.ADULT,
            prominence=Prominence.MAIN,
            appearance=AppearanceTraits(build=Build.REPTILIAN, skin="niebieskie luski",
                                        distinguishing=("dlugi ogon",), outfit="mundur"),
            reference_coverage=ReferenceCoverage.FULL,
    )


class TestBuildPreviewPrompt(unittest.TestCase):
    def test_prompt_starts_with_the_token(self) -> None:
        self.assertTrue(build_preview_prompt("bmb07").startswith("bmb07, "))

    def test_prompt_leaves_the_token_as_the_only_anchor_of_identity(self) -> None:
        self.assertNotIn("kurvinox", build_preview_prompt("bmb07"))

    def test_prompt_asks_for_a_full_figure_so_the_crop_bias_does_not_win(self) -> None:
        self.assertIn("full figure", build_preview_prompt("bmb07"))

    def test_prompt_carries_the_style_the_lora_was_trained_with(self) -> None:
        self.assertIn("flat cel shaded cartoon character", build_preview_prompt("bmb07"))

    def test_prompt_forbids_props_that_would_break_the_cutout(self) -> None:
        prompt = build_preview_prompt("bmb07")
        self.assertIn("no props", prompt)
        self.assertIn("no scenery", prompt)


class TestSlugsWithoutCharacter(unittest.TestCase):
    def test_slug_missing_from_the_codex_is_reported(self) -> None:
        characters = {"kurvinox": _character(slug="kurvinox")}
        self.assertEqual(("zaginiony",),
                         slugs_without_character({"kurvinox": "bmb01", "zaginiony": "bmb02"}, characters))

    def test_token_map_covered_by_the_codex_reports_nothing(self) -> None:
        characters = {"kurvinox": _character(slug="kurvinox")}
        self.assertEqual((), slugs_without_character({"kurvinox": "bmb01"}, characters))


if __name__ == "__main__":
    unittest.main()
