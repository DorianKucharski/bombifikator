from __future__ import annotations

import tomllib
from pathlib import Path

_SHARED_PREFIX_LENGTH = 3


def _repeated(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if values.count(value) > 1}))


def read_token_vocabulary(vocabulary_path: Path) -> tuple[str, ...]:
    if not vocabulary_path.is_file():
        raise ValueError(f"token vocabulary file not found: {vocabulary_path}")
    with vocabulary_path.open("rb") as handle:
        tokens = tuple(tomllib.load(handle).get("tokens", ()))
    if not tokens:
        raise ValueError(f"token vocabulary is empty: {vocabulary_path}")

    repeated_tokens = _repeated(tokens)
    if repeated_tokens:
        raise ValueError(f"token vocabulary {vocabulary_path} repeats tokens: {', '.join(repeated_tokens)}")

    repeated_prefixes = _repeated(tuple(token[:_SHARED_PREFIX_LENGTH] for token in tokens))
    if repeated_prefixes:
        raise ValueError(f"token vocabulary {vocabulary_path} has tokens sharing a prefix: "
                         f"{', '.join(repeated_prefixes)}")
    return tokens
