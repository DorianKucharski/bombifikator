from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from sharedkernel.env_file import load_env_file


class TestEnvFile(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self._env_path = Path(self._directory.name) / ".env"
        self._touched_names: list[str] = []

    def tearDown(self) -> None:
        for name in self._touched_names:
            os.environ.pop(name, None)
        self._directory.cleanup()

    def _write(self, content: str, *names: str) -> None:
        self._env_path.write_text(content, encoding="utf-8")
        self._touched_names.extend(names)

    def test_assignment_lands_in_environment(self) -> None:
        self._write("BOMBIFIKATOR_TEST_SECRET=value\n", "BOMBIFIKATOR_TEST_SECRET")
        load_env_file(self._env_path)
        self.assertEqual("value", os.environ["BOMBIFIKATOR_TEST_SECRET"])

    def test_quotes_and_comments_are_ignored(self) -> None:
        self._write("# komentarz\nBOMBIFIKATOR_TEST_QUOTED=\"quoted value\"\n", "BOMBIFIKATOR_TEST_QUOTED")
        load_env_file(self._env_path)
        self.assertEqual("quoted value", os.environ["BOMBIFIKATOR_TEST_QUOTED"])

    def test_existing_environment_variable_wins(self) -> None:
        self._touched_names.append("BOMBIFIKATOR_TEST_EXISTING")
        os.environ["BOMBIFIKATOR_TEST_EXISTING"] = "from-shell"
        self._write("BOMBIFIKATOR_TEST_EXISTING=from-file\n")
        load_env_file(self._env_path)
        self.assertEqual("from-shell", os.environ["BOMBIFIKATOR_TEST_EXISTING"])

    def test_missing_file_is_not_an_error(self) -> None:
        load_env_file(Path(self._directory.name) / "brak.env")


if __name__ == "__main__":
    unittest.main()
