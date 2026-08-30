from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy

from reference_builder.crop_store import CropStore, crop_id_of
from reference_builder.reference_models import CropRecord
from sharedkernel.utils.geometry import BoundingBox

_RECORD = CropRecord(
        crop_id="odcinek-1/0007-01-00",
        episode_slug="odcinek-1",
        frame_file_name="0007-01.png",
        detection_index=0,
        score=0.7123,
        phrase="a cartoon character",
        box=BoundingBox(10, 20, 110, 220),
)


class TestCropStore(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self._store = CropStore(Path(self._directory.name))

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_crop_id_carries_episode_and_detection_index(self) -> None:
        self.assertEqual("odcinek-1/0007-01-02", crop_id_of("odcinek-1", "0007-01.png", 2))

    def test_manifest_round_trip_restores_the_record(self) -> None:
        self._store.write_manifest("odcinek-1", (_RECORD,))
        self.assertEqual((_RECORD,), self._store.load_episode("odcinek-1"))

    def test_episode_without_manifest_is_not_marked_as_done(self) -> None:
        self.assertFalse(self._store.contains_episode("odcinek-1"))
        self._store.write_manifest("odcinek-1", ())
        self.assertTrue(self._store.contains_episode("odcinek-1"))

    def test_saved_crop_lands_under_its_episode_directory(self) -> None:
        path = self._store.save_crop(_RECORD, numpy.zeros((8, 8, 3), dtype=numpy.uint8))
        self.assertEqual(Path(self._directory.name) / "odcinek-1" / "0007-01-00.png", path)
        self.assertEqual(path, self._store.crop_path(_RECORD.crop_id))

    def test_load_all_walks_every_episode(self) -> None:
        self._store.write_manifest("odcinek-1", (_RECORD,))
        self._store.write_manifest("odcinek-2", ())
        self.assertEqual([_RECORD], list(self._store.load_all()))


if __name__ == "__main__":
    unittest.main()
