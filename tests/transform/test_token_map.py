from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from transform.token_map import read_token_map


class TestTokenMap(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self._tokens_path = Path(self._directory.name) / "tokens.toml"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_tokens_are_read_by_slug(self) -> None:
        self._tokens_path.write_text('[tokens]\nkurvinox = "bmb16"\n', encoding="utf-8")
        self.assertEqual({"kurvinox": "bmb16"}, read_token_map(self._tokens_path))

    def test_missing_file_raises_naming_the_command_to_run(self) -> None:
        with self.assertRaises(ValueError) as raised:
            read_token_map(self._tokens_path)
        self.assertIn("training dataset", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
