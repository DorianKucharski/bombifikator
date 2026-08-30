from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sharedkernel.config_provider import ConfigProvider

_TOKEN_PREFIX = "bmb"


@dataclass(frozen=True)
class DatasetConfig:
    references_dir: Path
    characters_dir: Path
    dataset_dir: Path
    token_prefix: str
    image_size: int
    minimum_references: int

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "DatasetConfig":
        return cls(
                references_dir=config_provider.get_path("reference_builder.references_dir", "data/references/cards"),
                characters_dir=config_provider.get_path("codex.characters_dir", "data/codex/characters"),
                dataset_dir=config_provider.get_path("training.dataset_dir", "data/training/identity"),
                token_prefix=config_provider.get_str("training.token_prefix", _TOKEN_PREFIX),
                image_size=config_provider.get_int("training.image_size", 1024),
                minimum_references=config_provider.get_int("training.minimum_references", 10),
        )
