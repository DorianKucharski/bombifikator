from __future__ import annotations

import unittest

from reference_builder.crop_geometry import (clamp_box, covers_whole_frame, intersection_over_union, pad_box,
                                             suppress_overlapping)
from reference_builder.reference_models import Detection
from sharedkernel.utils.geometry import BoundingBox


class TestCropGeometry(unittest.TestCase):

    def test_padding_grows_the_box_by_the_shorter_side_ratio(self) -> None:
        padded = pad_box(BoundingBox(100, 100, 200, 300), 0.1, 640, 480)
        self.assertEqual(BoundingBox(90, 90, 210, 310), padded)

    def test_padding_stops_at_frame_edges(self) -> None:
        padded = pad_box(BoundingBox(5, 5, 105, 105), 0.5, 120, 120)
        self.assertEqual(BoundingBox(0, 0, 120, 120), padded)

    def test_clamping_keeps_the_box_inside_the_frame(self) -> None:
        clamped = clamp_box(-12.4, -0.6, 999.9, 800.2, frame_width=640, frame_height=360)
        self.assertEqual(BoundingBox(0, 0, 640, 360), clamped)

    def test_box_covering_almost_the_whole_frame_is_rejected(self) -> None:
        self.assertTrue(covers_whole_frame(BoundingBox(0, 0, 640, 360), 640, 360, 0.85))

    def test_half_frame_box_is_kept(self) -> None:
        self.assertFalse(covers_whole_frame(BoundingBox(0, 0, 320, 360), 640, 360, 0.85))


class TestOverlapSuppression(unittest.TestCase):

    def test_identical_boxes_have_full_overlap(self) -> None:
        box = BoundingBox(0, 0, 100, 100)
        self.assertEqual(1.0, intersection_over_union(box, box))

    def test_disjoint_boxes_have_no_overlap(self) -> None:
        self.assertEqual(0.0, intersection_over_union(BoundingBox(0, 0, 10, 10), BoundingBox(20, 20, 30, 30)))

    def test_same_character_found_under_two_phrases_is_kept_once(self) -> None:
        detections = (
            Detection(box=BoundingBox(0, 0, 100, 200), score=0.58, phrase="a monster"),
            Detection(box=BoundingBox(2, 1, 103, 203), score=0.42, phrase="a cartoon character"),
        )
        distinct = suppress_overlapping(detections, 0.55)
        self.assertEqual(("a monster",), tuple(detection.phrase for detection in distinct))

    def test_two_characters_side_by_side_both_survive(self) -> None:
        detections = (
            Detection(box=BoundingBox(0, 0, 100, 200), score=0.58, phrase="a monster"),
            Detection(box=BoundingBox(120, 0, 220, 200), score=0.42, phrase="a person"),
        )
        self.assertEqual(2, len(suppress_overlapping(detections, 0.55)))


if __name__ == "__main__":
    unittest.main()
