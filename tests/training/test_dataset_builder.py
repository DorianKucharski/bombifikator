from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy

from codex.character_models import (AgeGroup, AppearanceTraits, Build, Character, Prominence, ReferenceCoverage, Sex)
from training.caption_builder import build_caption, character_token
from training.dataset_builder import (build_dataset, flattened_on_white, squared_on_white, trainable_characters)
from training.training_config import DatasetConfig


def _character(slug: str, coverage: ReferenceCoverage = ReferenceCoverage.FULL) -> Character:
    return Character(
            slug=slug,
            canonical_name=slug.title(),
            species="kurvinox",
            sex=Sex.MALE,
            age_group=AgeGroup.ADULT,
            prominence=Prominence.MAIN,
            appearance=AppearanceTraits(build=Build.REPTILIAN, skin="niebieskie luski",
                                        distinguishing=("dlugi ogon",), outfit="mundur"),
            reference_coverage=coverage,
    )


class TestCaptionBuilder(unittest.TestCase):

    def test_token_replaces_dashes_with_underscores(self) -> None:
        self.assertEqual("bmb_tytus_bomba", character_token("bmb", "tytus-bomba"))

    def test_caption_starts_with_the_token(self) -> None:
        self.assertTrue(build_caption(_character("kurvinox"), "bmb").startswith("bmb_kurvinox,"))

    def test_caption_carries_appearance_and_style(self) -> None:
        caption = build_caption(_character("kurvinox"), "bmb")
        self.assertIn("niebieskie luski", caption)
        self.assertIn("dlugi ogon", caption)
        self.assertIn("mundur", caption)
        self.assertIn("flat cel shaded", caption)


class TestImagePreparation(unittest.TestCase):

    def test_transparent_pixels_become_white(self) -> None:
        card = numpy.zeros((10, 10, 4), dtype=numpy.uint8)
        self.assertEqual(255, int(flattened_on_white(card)[0, 0, 0]))

    def test_opaque_pixels_keep_their_colour(self) -> None:
        card = numpy.zeros((10, 10, 4), dtype=numpy.uint8)
        card[:, :, 3] = 255
        self.assertEqual(0, int(flattened_on_white(card)[0, 0, 0]))

    def test_square_canvas_has_the_requested_size(self) -> None:
        self.assertEqual((64, 64, 3), squared_on_white(numpy.zeros((40, 20, 3), dtype=numpy.uint8), 64).shape)

    def test_tall_image_is_padded_not_stretched(self) -> None:
        image = numpy.zeros((40, 10, 3), dtype=numpy.uint8)
        squared = squared_on_white(image, 64)
        self.assertEqual(255, int(squared[32, 0, 0]))
        self.assertEqual(0, int(squared[32, 32, 0]))


class TestTrainableCharacters(unittest.TestCase):

    def test_only_full_coverage_is_trainable(self) -> None:
        characters = (_character("a"), _character("b", ReferenceCoverage.THIN),
                      _character("c", ReferenceCoverage.NONE))
        self.assertEqual(("a",), tuple(c.slug for c in trainable_characters(characters)))


class TestDatasetBuilder(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        self._config = DatasetConfig(
                references_dir=root / "cards",
                characters_dir=root / "characters",
                dataset_dir=root / "dataset",
                token_prefix="bmb",
                image_size=64,
                minimum_references=2,
        )
        from codex.character_store import CharacterStore
        CharacterStore(self._config.characters_dir).save(_character("kurvinox"))
        character_dir = self._config.references_dir / "kurvinox"
        character_dir.mkdir(parents=True)
        for index in range(3):
            card = numpy.zeros((20, 10, 4), dtype=numpy.uint8)
            card[:, :, 3] = 255
            cv2.imwrite(str(character_dir / f"{index:02d}.png"), card)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_every_reference_becomes_an_image_and_a_caption(self) -> None:
        report = build_dataset(self._config)
        self.assertEqual(3, report.images_written)
        self.assertEqual(3, len(list(self._config.dataset_dir.glob("*.png"))))
        self.assertEqual(3, len(list(self._config.dataset_dir.glob("*.txt"))))

    def test_caption_file_carries_the_token(self) -> None:
        build_dataset(self._config)
        caption = (self._config.dataset_dir / "bmb_kurvinox_00.txt").read_text(encoding="utf-8")
        self.assertTrue(caption.startswith("bmb_kurvinox,"))

    def test_character_below_the_minimum_is_skipped(self) -> None:
        config = DatasetConfig(**{**self._config.__dict__, "minimum_references": 10})
        report = build_dataset(config)
        self.assertEqual(("kurvinox",), report.characters_skipped)

    def test_empty_codex_raises_naming_the_directory(self) -> None:
        config = DatasetConfig(**{**self._config.__dict__, "characters_dir": Path(self._directory.name) / "empty"})
        with self.assertRaises(ValueError) as raised:
            build_dataset(config)
        self.assertIn("empty", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
