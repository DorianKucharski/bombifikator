from __future__ import annotations

import tomllib
from pathlib import Path


def read_token_map(tokens_path: Path) -> dict[str, str]:
    if not tokens_path.is_file():
        raise ValueError(f"missing token map {tokens_path}: run 'training dataset' first")
    with tokens_path.open("rb") as handle:
        return dict(tomllib.load(handle)["tokens"])
