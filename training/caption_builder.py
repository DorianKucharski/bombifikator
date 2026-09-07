from __future__ import annotations

from codex.character_models import Character
from codex.drawing_style import STYLE_PHRASE


def tokens_by_slug(vocabulary: tuple[str, ...], characters: tuple[Character, ...]) -> dict[str, str]:
    if len(vocabulary) < len(characters):
        raise ValueError(f"token vocabulary holds {len(vocabulary)} tokens for {len(characters)} characters")
    ordered = sorted(characters, key=lambda character: character.slug)
    return {character.slug: vocabulary[position] for position, character in enumerate(ordered)}


def build_caption(token: str) -> str:
    return f"{token}, {STYLE_PHRASE}"
