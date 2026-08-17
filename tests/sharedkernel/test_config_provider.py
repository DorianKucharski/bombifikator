from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from sharedkernel.config_provider import ConfigProvider

_CONFIG_BODY = """
[codex]
raw_dir = "data/codex/raw"
max_category_depth = 3
requests_per_second = 1.0
"""


class TestConfigProvider(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self._config_path = Path(self._directory.name) / "config.toml"
        self._config_path.write_text(_CONFIG_BODY, encoding="utf-8")
        self._provider = ConfigProvider.from_file(self._config_path)

    def tearDown(self):
        self._directory.cleanup()
        os.environ.pop("BOMBIFIKATOR_CODEX_MAX_CATEGORY_DEPTH", None)

    def test_nested_key_is_read_from_toml(self):
        self.assertEqual(3, self._provider.get_int("codex.max_category_depth"))

    def test_environment_variable_overrides_toml(self):
        os.environ["BOMBIFIKATOR_CODEX_MAX_CATEGORY_DEPTH"] = "7"

        self.assertEqual(7, self._provider.get_int("codex.max_category_depth"))

    def test_missing_required_key_names_both_sources(self):
        with self.assertRaises(ValueError) as raised:
            self._provider.get("anthropic_api_key")

        message = str(raised.exception)
        self.assertIn("BOMBIFIKATOR_ANTHROPIC_API_KEY", message)
        self.assertIn("anthropic_api_key", message)

    def test_default_is_used_when_key_is_absent(self):
        self.assertEqual(30.0, self._provider.get_float("codex.timeout_seconds", 30.0))

    def test_missing_config_file_fails_fast(self):
        with self.assertRaises(ValueError):
            ConfigProvider.from_file(Path(self._directory.name) / "absent.toml")


if __name__ == "__main__":
    unittest.main()
