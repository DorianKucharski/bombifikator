from __future__ import annotations

import unittest

from harvester.episode_models import Shot


class TestShotSampling(unittest.TestCase):

    def test_short_shot_yields_one_frame_near_its_middle(self) -> None:
        shot = Shot(index=1, start_seconds=10.0, end_seconds=12.0)
        self.assertEqual((11.0,), shot.sample_timestamps(stride_seconds=3.0))

    def test_long_shot_yields_evenly_spread_frames_inside_its_bounds(self) -> None:
        shot = Shot(index=1, start_seconds=0.0, end_seconds=12.0)
        timestamps = shot.sample_timestamps(stride_seconds=3.0)
        self.assertEqual(4, len(timestamps))
        for expected, actual in zip((2.4, 4.8, 7.2, 9.6), timestamps):
            self.assertAlmostEqual(expected, actual)
        self.assertTrue(all(shot.start_seconds < timestamp < shot.end_seconds for timestamp in timestamps))


if __name__ == "__main__":
    unittest.main()
