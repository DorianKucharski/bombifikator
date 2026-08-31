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


@dataclass(frozen=True)
class PreviewConfig:
    tokens_path: Path
    previews_dir: Path
    samples_per_token: int
    sheet_grid_size: int
    sheet_tile_pixels: int
    preview_pixels: int
    seed: int

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "PreviewConfig":
        return cls(
                tokens_path=config_provider.get_path("training.tokens_path", "data/training/tokens.toml"),
                previews_dir=config_provider.get_path("training.previews_dir", "data/training/previews"),
                samples_per_token=config_provider.get_int("training.preview.samples_per_token", 4),
                sheet_grid_size=config_provider.get_int("training.preview.sheet_grid_size", 2),
                sheet_tile_pixels=config_provider.get_int("training.preview.sheet_tile_pixels", 512),
                preview_pixels=config_provider.get_int("training.preview.preview_pixels", 1024),
                seed=config_provider.get_int("training.preview.seed", 42),
        )
