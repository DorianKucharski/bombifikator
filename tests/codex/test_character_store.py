from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codex.character_models import AgeGroup, AppearanceTraits, Build, Character, Prominence, ReferenceCoverage, Sex
from codex.character_store import CharacterStore

_KURVINOX = Character(
        slug="kurvinox",
        canonical_name="Kurvinox",
        species="kurvinox",
        aliases=("Kurwinox", "Kurwinoks"),
        sex=Sex.MALE,
        age_group=AgeGroup.ADULT,
        prominence=Prominence.RECURRING,
        appearance=AppearanceTraits(
                build=Build.REPTILIAN,
                height_meters=2.0,
                skin="luski",
                distinguishing=("dlugi ogon", "para klow pod oczami", "ostre pazury"),
                palette=("#3B7DD8", "#7A3BD8"),
                outfit="brak",
        ),
        role_tags=("kosmita", "antagonista"),
        summary="Humanoidalna jaszczurka, najpospolitsza rasa w uniwersum.",
        source_urls=("https://bombaversearchive.fandom.com/pl/wiki/Kurvinox",),
        reference_coverage=ReferenceCoverage.NONE,
)


class TestCharacterStore(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self._store = CharacterStore(Path(self._directory.name))

    def tearDown(self):
        self._directory.cleanup()

    def test_character_survives_a_save_and_load_round_trip(self):
        self._store.save(_KURVINOX)

        self.assertEqual([_KURVINOX], list(self._store.load_all()))

    def test_character_without_height_survives_a_round_trip(self):
        without_height = Character(
                slug="glus",
                canonical_name="Michal Glus",
                species="czlowiek",
                appearance=AppearanceTraits(build=Build.HUMANOID),
        )

        self._store.save(without_height)

        loaded = next(iter(self._store.load_all()))
        self.assertIsNone(loaded.appearance.height_meters)

    def test_loading_from_an_absent_directory_yields_nothing(self):
        absent = CharacterStore(Path(self._directory.name) / "absent")

        self.assertEqual([], list(absent.load_all()))

    def test_character_is_usable_for_transform_only_with_full_coverage(self):
        self.assertFalse(_KURVINOX.is_usable_for_transform)


if __name__ == "__main__":
    unittest.main()
