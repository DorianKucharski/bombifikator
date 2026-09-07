from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.token_vocabulary import read_token_vocabulary


def _vocabulary_file(directory: Path, body: str) -> Path:
    vocabulary_path = directory / "identity_tokens.toml"
    vocabulary_path.write_text(body, encoding="utf-8")
    return vocabulary_path


class TestReadTokenVocabulary(unittest.TestCase):

    def test_tokens_are_read_in_file_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _vocabulary_file(Path(directory), 'tokens = ["zqylan", "murgash"]')
            self.assertEqual(("zqylan", "murgash"), read_token_vocabulary(path))

    def test_missing_file_names_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as raised:
                read_token_vocabulary(Path(directory) / "absent.toml")
            self.assertIn("absent.toml", str(raised.exception))

    def test_empty_vocabulary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _vocabulary_file(Path(directory), "tokens = []")
            with self.assertRaises(ValueError):
                read_token_vocabulary(path)

    def test_repeated_token_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _vocabulary_file(Path(directory), 'tokens = ["zqylan", "zqylan"]')
            with self.assertRaises(ValueError) as raised:
                read_token_vocabulary(path)
            self.assertIn("zqylan", str(raised.exception))

    def test_tokens_sharing_a_prefix_are_rejected_because_that_is_what_broke_v3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _vocabulary_file(Path(directory), 'tokens = ["bmb01", "bmb02"]')
            with self.assertRaises(ValueError) as raised:
                read_token_vocabulary(path)
            self.assertIn("bmb", str(raised.exception))

    def test_shipped_vocabulary_covers_every_trainable_character(self) -> None:
        self.assertGreaterEqual(len(read_token_vocabulary(Path("training/configs/identity_tokens.toml"))), 32)


if __name__ == "__main__":
    unittest.main()
