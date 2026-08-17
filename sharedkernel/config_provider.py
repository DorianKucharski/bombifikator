from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

_MISSING = object()


def _truthy(value: str) -> bool:
    return value.strip().lower() in _TRUTHY_VALUES


def _env_var_name(key: str) -> str:
    return "BOMBIFIKATOR_" + key.replace(".", "_").upper()


class ConfigProvider:
    def __init__(self, config: dict[str, Any], config_path: Path) -> None:
        self._config = config
        self._config_path = config_path

    @classmethod
    def from_file(cls, config_path: Path) -> "ConfigProvider":
        if not config_path.is_file():
            raise ValueError(f"config file not found: {config_path}")
        with config_path.open("rb") as handle:
            return cls(tomllib.load(handle), config_path)

    def _from_toml(self, key: str) -> Any:
        node: Any = self._config
        for segment in key.split("."):
            if not isinstance(node, dict) or segment not in node:
                return _MISSING
            node = node[segment]
        return node

    def get(self, key: str, default: Any = _MISSING) -> Any:
        env_value = os.environ.get(_env_var_name(key))
        if env_value is not None:
            return env_value
        toml_value = self._from_toml(key)
        if toml_value is not _MISSING:
            return toml_value
        if default is not _MISSING:
            return default
        raise ValueError(
                f"missing required configuration '{key}': "
                f"set environment variable {_env_var_name(key)} or key '{key}' in {self._config_path}"
        )

    def get_str(self, key: str, default: Any = _MISSING) -> str:
        return str(self.get(key, default))

    def get_int(self, key: str, default: Any = _MISSING) -> int:
        return int(self.get(key, default))

    def get_float(self, key: str, default: Any = _MISSING) -> float:
        return float(self.get(key, default))

    def get_bool(self, key: str, default: Any = _MISSING) -> bool:
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        return _truthy(str(value))

    def get_path(self, key: str, default: Any = _MISSING) -> Path:
        return Path(self.get_str(key, default)).expanduser()

    def get_str_list(self, key: str, default: Any = _MISSING) -> list[str]:
        value = self.get(key, default)
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item) for item in value]

    def get_secret(self, key: str) -> str:
        value = self.get(key)
        if not str(value).strip():
            raise ValueError(
                    f"empty secret '{key}': set environment variable {_env_var_name(key)}"
            )
        return str(value)
