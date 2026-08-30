from __future__ import annotations

import unittest
from typing import Any, Sequence

import numpy

from codex.character_models import (AgeGroup, AppearanceTraits, Build, Character, Prominence, Sex)
from sharedkernel.utils.geometry import BoundingBox
from transform.person_describer import build_system_prompt, describe_person
from transform.person_models import PersonInstance

_KURVINOX = Character(
        slug="kurvinox",
        canonical_name="Kurvinox",
        species="kurvinox",
        sex=Sex.MALE,
        age_group=AgeGroup.ADULT,
        prominence=Prominence.RECURRING,
        appearance=AppearanceTraits(build=Build.REPTILIAN, distinguishing=("dlugi ogon",), outfit="mundur"),
)

_PAYLOAD = {
    "sex": "MALE",
    "age_group": "ADULT",
    "build_description": "krepy mezczyzna",
    "outfit": "czarna kurtka",
    "pose": "stoi bokiem",
    "facing": "w lewo",
    "notable_features": ["broda"],
    "character_slugs": ["kurvinox", "skurwol", "nadesral"],
}


class _StubClient:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self._payload = payload
        self.images: Sequence[bytes] = ()

    def extract(self, system_prompt: str, user_prompt: str, json_schema: dict[str, Any],
                images: Sequence[bytes] = ()) -> dict[str, Any] | None:
        self.images = images
        return self._payload


def _person() -> PersonInstance:
    mask = numpy.zeros((60, 60), dtype=bool)
    mask[10:50, 10:50] = True
    return PersonInstance(person_index=0, box=BoundingBox(10, 10, 50, 50), mask=mask, confidence=0.9)


class TestPersonDescriber(unittest.TestCase):

    def setUp(self) -> None:
        self._image = numpy.zeros((60, 60, 3), dtype=numpy.uint8)

    def test_description_is_read_from_the_payload(self) -> None:
        description, _ = describe_person(_StubClient(_PAYLOAD), "system", self._image, _person())
        self.assertIs(Sex.MALE, description.sex)
        self.assertEqual("stoi bokiem", description.pose)

    def test_character_choices_are_returned_in_order(self) -> None:
        _, choices = describe_person(_StubClient(_PAYLOAD), "system", self._image, _person())
        self.assertEqual(("kurvinox", "skurwol", "nadesral"), choices)

    def test_refused_response_produces_nothing(self) -> None:
        self.assertIsNone(describe_person(_StubClient(None), "system", self._image, _person()))

    def test_both_the_photograph_and_the_close_up_are_sent(self) -> None:
        client = _StubClient(_PAYLOAD)
        describe_person(client, "system", self._image, _person())
        self.assertEqual(2, len(client.images))

    def test_system_prompt_lists_the_catalogue(self) -> None:
        prompt = build_system_prompt((_KURVINOX,))
        self.assertIn("kurvinox", prompt)
        self.assertIn("dlugi ogon", prompt)


if __name__ == "__main__":
    unittest.main()
