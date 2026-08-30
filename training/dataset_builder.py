from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy
import tomli_w

from codex.character_models import Character, ReferenceCoverage
from codex.character_store import CharacterStore
from sharedkernel.logger import get_logger
from sharedkernel.utils.paths import ensure_directory
from training.caption_builder import build_caption, tokens_by_slug
from training.training_config import DatasetConfig

LOGGER = get_logger("training.dataset_builder")

TOKEN_MAP_FILE_NAME = "tokens.toml"

_WHITE = 255


@dataclass(frozen=True)
class DatasetReport:
    characters_written: int
    images_written: int
    characters_skipped: tuple[str, ...]
    tokens_path: Path


def flattened_on_white(image: numpy.ndarray) -> numpy.ndarray:
    if image.shape[2] != 4:
        return image
    alpha = image[:, :, 3:4].astype(numpy.float32) / 255.0
    return (image[:, :, :3].astype(numpy.float32) * alpha + _WHITE * (1.0 - alpha)).astype(numpy.uint8)


def squared_on_white(image: numpy.ndarray, size: int) -> numpy.ndarray:
    scale = size / max(image.shape[0], image.shape[1])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    canvas = numpy.full((size, size, 3), _WHITE, dtype=numpy.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def trainable_characters(characters: tuple[Character, ...]) -> tuple[Character, ...]:
    return tuple(character for character in characters if character.reference_coverage is ReferenceCoverage.FULL)


def _write_character(config: DatasetConfig, character: Character, token: str) -> int:
    reference_paths = sorted((config.references_dir / character.slug).glob("*.png"))
    if len(reference_paths) < config.minimum_references:
        return 0
    caption = build_caption(character, token)
    written = 0
    for index, reference_path in enumerate(reference_paths):
        image = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        squared = squared_on_white(flattened_on_white(image), config.image_size)
        stem = f"{token}_{index:02d}"
        cv2.imwrite(str(config.dataset_dir / f"{stem}.png"), squared)
        (config.dataset_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
        written += 1
    return written


def build_dataset(config: DatasetConfig) -> DatasetReport:
    characters = trainable_characters(tuple(CharacterStore(config.characters_dir).load_all()))
    if not characters:
        raise ValueError(f"no character with full reference coverage in {config.characters_dir}")
    ensure_directory(config.dataset_dir)

    tokens = tokens_by_slug(config.token_prefix, characters)
    images_written = 0
    written_characters = 0
    skipped = []
    for character in characters:
        token = tokens[character.slug]
        written = _write_character(config, character, token)
        if written == 0:
            skipped.append(character.slug)
            continue
        images_written += written
        written_characters += 1
        LOGGER.info("character written: slug=%s token=%s images=%d", character.slug, token, written)
    tokens_path = write_token_map(config.dataset_dir.parent / TOKEN_MAP_FILE_NAME, tokens)
    return DatasetReport(
            characters_written=written_characters,
            images_written=images_written,
            characters_skipped=tuple(skipped),
            tokens_path=tokens_path,
    )


def write_token_map(tokens_path: Path, tokens: dict[str, str]) -> Path:
    ensure_directory(tokens_path.parent)
    with tokens_path.open("wb") as handle:
        tomli_w.dump({"tokens": dict(sorted(tokens.items()))}, handle)
    return tokens_path
