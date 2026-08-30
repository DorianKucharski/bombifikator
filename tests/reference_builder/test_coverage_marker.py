from __future__ import annotations

import unittest

from codex.character_models import ReferenceCoverage
from reference_builder.coverage_marker import coverage_of


class TestCoverageMarker(unittest.TestCase):

    def test_reaching_the_target_is_full_coverage(self) -> None:
        self.assertIs(ReferenceCoverage.FULL, coverage_of(16, target_references=16, thin_threshold=10))

    def test_above_the_thin_threshold_is_thin_coverage(self) -> None:
        self.assertIs(ReferenceCoverage.THIN, coverage_of(11, target_references=16, thin_threshold=10))

    def test_below_the_thin_threshold_is_no_coverage(self) -> None:
        self.assertIs(ReferenceCoverage.NONE, coverage_of(9, target_references=16, thin_threshold=10))


if __name__ == "__main__":
    unittest.main()
