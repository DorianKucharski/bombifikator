from __future__ import annotations

import unittest

from codex.character_matcher import (acceptable_choices, assign_without_repeats, sex_allows, usable_characters)
from codex.character_models import (AgeGroup, AppearanceTraits, Build, Character, PersonDescription, Prominence,
                                    ReferenceCoverage, Sex)


def _character(slug: str, sex: Sex, coverage: ReferenceCoverage = ReferenceCoverage.FULL) -> Character:
    return Character(
            slug=slug,
            canonical_name=slug.title(),
            species="kurvinox",
            sex=sex,
            age_group=AgeGroup.ADULT,
            prominence=Prominence.MAIN,
            appearance=AppearanceTraits(build=Build.HUMANOID, distinguishing=("mundur",), outfit="mundur"),
            reference_coverage=coverage,
    )


_DESCRIPTION = PersonDescription(
        person_index=0,
        sex=Sex.MALE,
        age_group=AgeGroup.ADULT,
        build_description="krepy mezczyzna",
        outfit="mundur wojskowy",
        pose="stoi na wprost",
        facing="do kamery",
        notable_features=("broda",),
)

_CATALOGUE = {
    "male-one": _character("male-one", Sex.MALE),
    "female-one": _character("female-one", Sex.FEMALE),
    "robot": _character("robot", Sex.UNKNOWN),
}


class TestCharacterMatcher(unittest.TestCase):

    def test_female_character_is_rejected_for_a_male_person(self) -> None:
        self.assertFalse(sex_allows(_character("andzela", Sex.FEMALE), _DESCRIPTION))

    def test_character_of_unknown_sex_fits_anyone(self) -> None:
        self.assertTrue(sex_allows(_character("robot", Sex.UNKNOWN), _DESCRIPTION))

    def test_characters_without_references_are_not_usable(self) -> None:
        characters = (_character("a", Sex.MALE), _character("b", Sex.MALE, ReferenceCoverage.NONE))
        self.assertEqual(("a",), tuple(c.slug for c in usable_characters(characters, allow_thin_coverage=True)))

    def test_thin_coverage_can_be_excluded(self) -> None:
        characters = (_character("a", Sex.MALE, ReferenceCoverage.THIN),)
        self.assertEqual((), usable_characters(characters, allow_thin_coverage=False))

    def test_choices_of_the_wrong_sex_are_dropped(self) -> None:
        chosen = acceptable_choices(("female-one", "male-one", "robot"), _CATALOGUE, _DESCRIPTION)
        self.assertEqual(("male-one", "robot"), chosen)

    def test_choices_outside_the_catalogue_are_dropped(self) -> None:
        self.assertEqual(("male-one",), acceptable_choices(("wymyslony", "male-one"), _CATALOGUE, _DESCRIPTION))

    def test_the_order_of_the_choices_is_kept(self) -> None:
        self.assertEqual(("robot", "male-one"), acceptable_choices(("robot", "male-one"), _CATALOGUE, _DESCRIPTION))


class TestAssignmentWithoutRepeats(unittest.TestCase):

    def test_second_person_gets_the_next_free_character(self) -> None:
        choices = {0: ("kurvinox", "skurwol"), 1: ("kurvinox", "skurwol")}
        self.assertEqual({0: "kurvinox", 1: "skurwol"}, assign_without_repeats(choices))

    def test_person_without_any_free_character_is_skipped(self) -> None:
        self.assertEqual({0: "kurvinox"}, assign_without_repeats({0: ("kurvinox",), 1: ("kurvinox",)}))

    def test_person_without_any_choice_is_skipped(self) -> None:
        self.assertEqual({}, assign_without_repeats({0: ()}))


if __name__ == "__main__":
    unittest.main()
