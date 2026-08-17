from __future__ import annotations

import unittest

from codex.character_extractor import extract_character
from codex.wiki_client import WikiPage


def _payload(canonical_name: str) -> dict:
    return {
        "is_character": True,
        "rejection_reason": "",
        "canonical_name": canonical_name,
        "species": "kurvinox",
        "aliases": [],
        "sex": "MALE",
        "age_group": "ADULT",
        "prominence": "EPISODIC",
        "appearance": {
            "build": "HUMANOID",
            "height_meters": 1.8,
            "skin": "zielona",
            "distinguishing": ["rogi"],
            "palette": [],
            "outfit": "pancerz",
        },
        "role_tags": [],
        "summary": "",
    }


class _StubClient:

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def extract(self, system_prompt: str, user_prompt: str, json_schema: dict) -> dict:
        return self._payload


def _page(title: str) -> WikiPage:
    return WikiPage(
            site_name="bombaverse",
            title=title,
            url=f"https://example.invalid/wiki/{title}",
            wikitext="tekst",
            categories=(),
    )


class TestCharacterExtractor(unittest.TestCase):

    def _extract_slug(self, title: str, canonical_name: str) -> str:
        character = extract_character(_StubClient(_payload(canonical_name)), _page(title), 1000)
        self.assertIsNotNone(character)
        return character.slug

    def test_slug_comes_from_page_title_not_canonical_name(self) -> None:
        self.assertEqual("bogdan-inny", self._extract_slug("Bogdan (inny)", "Bogdan"))

    def test_pages_sharing_a_canonical_name_keep_distinct_slugs(self) -> None:
        first = self._extract_slug("Domino", "Domino")
        second = self._extract_slug("Domino (kantyna, laser)", "Domino")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
