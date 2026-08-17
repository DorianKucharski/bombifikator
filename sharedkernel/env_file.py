from __future__ import annotations

import os
from pathlib import Path

_COMMENT_PREFIX = "#"
_ASSIGNMENT_SEPARATOR = "="


def _stripped_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _assignments(content: str) -> list[tuple[str, str]]:
    assignments = []
    for line in content.splitlines():
        statement = line.strip()
        if not statement or statement.startswith(_COMMENT_PREFIX) or _ASSIGNMENT_SEPARATOR not in statement:
            continue
        name, raw_value = statement.split(_ASSIGNMENT_SEPARATOR, 1)
        assignments.append((name.strip(), _stripped_quotes(raw_value.strip())))
    return assignments


def load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for name, value in _assignments(env_path.read_text(encoding="utf-8")):
        os.environ.setdefault(name, value)
