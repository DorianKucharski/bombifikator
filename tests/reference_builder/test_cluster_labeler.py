from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from codex.character_models import AgeGroup, AppearanceTraits, Build, Character, Prominence, Sex
from codex.character_store import CharacterStore
from reference_builder.cluster_labeler import _build_system_prompt, _label_cluster, _load_catalogue
from reference_builder.reference_models import UNKNOWN_LABEL, CharacterCluster

_CLUSTER = CharacterCluster(cluster_id=1, crop_ids=("odcinek-1/0001-00-00",), medoid_crop_id="odcinek-1/0001-00-00")

_KURVINOX = Character(
        slug="kurvinox",
        canonical_name="Kurvinox",
        species="kurvinox",
        sex=Sex.MALE,
        age_group=AgeGroup.ADULT,
        prominence=Prominence.RECURRING,
        appearance=AppearanceTraits(build=Build.REPTILIAN, skin="luski", distinguishing=("dlugi ogon",)),
)

_MIETEK = Character(
        slug="mietek-zolw",
        canonical_name="Mietek",
        species="zolw",
        sex=Sex.MALE,
        age_group=AgeGroup.ADULT,
        prominence=Prominence.EPISODIC,
        appearance=AppearanceTraits(build=Build.ANIMAL),
)


class _StubClient:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self._payload = payload
        self.images: Sequence[bytes] = ()

    def extract(self, system_prompt: str, user_prompt: str, json_schema: dict[str, Any],
                images: Sequence[bytes] = ()) -> dict[str, Any] | None:
        self.images = images
        return self._payload


class TestClusterLabeler(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self._sheet_path = Path(self._directory.name) / "cluster-0001.png"
        self._sheet_path.write_bytes(b"png-bytes")

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _label_with(self, payload: dict[str, Any] | None):
        return _label_cluster(_StubClient(payload), "system", frozenset({"kurvinox"}), _CLUSTER, self._sheet_path)

    def test_known_slug_is_kept(self) -> None:
        label = self._label_with({"slug": "kurvinox", "confidence": 0.91, "reasoning": "luski i ogon"})
        self.assertEqual("kurvinox", label.slug)
        self.assertTrue(label.is_identified)

    def test_slug_outside_the_catalogue_falls_back_to_unknown(self) -> None:
        label = self._label_with({"slug": "wymyslona-postac", "confidence": 0.9, "reasoning": "zgadywanie"})
        self.assertEqual(UNKNOWN_LABEL, label.slug)

    def test_refused_response_produces_no_label(self) -> None:
        self.assertIsNone(self._label_with(None))

    def test_sheet_is_sent_as_an_image(self) -> None:
        client = _StubClient({"slug": "kurvinox", "confidence": 0.5, "reasoning": "ogon"})
        _label_cluster(client, "system", frozenset({"kurvinox"}), _CLUSTER, self._sheet_path)
        self.assertEqual((b"png-bytes",), tuple(client.images))

    def test_catalogue_keeps_only_the_configured_prominences(self) -> None:
        store = CharacterStore(Path(self._directory.name) / "characters")
        store.save(_KURVINOX)
        store.save(_MIETEK)
        catalogue = _load_catalogue(Path(self._directory.name) / "characters", ("MAIN", "RECURRING"))
        self.assertEqual(("kurvinox",), tuple(character.slug for character in catalogue))

    def test_system_prompt_lists_slug_and_features(self) -> None:
        prompt = _build_system_prompt((_KURVINOX,))
        self.assertIn("kurvinox", prompt)
        self.assertIn("dlugi ogon", prompt)
        self.assertIn(UNKNOWN_LABEL, prompt)

    def test_empty_catalogue_raises_naming_the_directory(self) -> None:
        characters_dir = Path(self._directory.name) / "empty"
        with self.assertRaises(ValueError) as raised:
            _load_catalogue(characters_dir, ("MAIN",))
        self.assertIn(str(characters_dir), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
