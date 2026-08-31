from __future__ import annotations

import unittest

from training.token_preview import build_preview_prompt


class TestBuildPreviewPrompt(unittest.TestCase):
    def test_prompt_starts_with_the_token(self) -> None:
        self.assertTrue(build_preview_prompt("bmb07").startswith("bmb07, "))

    def test_prompt_asks_for_a_full_figure_so_the_crop_bias_does_not_win(self) -> None:
        self.assertIn("full figure", build_preview_prompt("bmb07"))

    def test_prompt_carries_the_style_the_lora_was_trained_with(self) -> None:
        self.assertIn("flat cel shaded cartoon character", build_preview_prompt("bmb07"))

    def test_prompt_forbids_props_that_would_break_the_cutout(self) -> None:
        prompt = build_preview_prompt("bmb07")
        self.assertIn("no props", prompt)
        self.assertIn("no scenery", prompt)


if __name__ == "__main__":
    unittest.main()
