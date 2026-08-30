from __future__ import annotations

from codex.character_models import Character

_STYLE_PHRASE = "flat cel shaded cartoon character with heavy black outlines on a plain white background"


def character_token(token_prefix: str, position: int) -> str:
    return f"{token_prefix}{position:02d}"


def tokens_by_slug(token_prefix: str, characters: tuple[Character, ...]) -> dict[str, str]:
    ordered = sorted(characters, key=lambda character: character.slug)
    return {character.slug: character_token(token_prefix, position)
            for position, character in enumerate(ordered, start=1)}


def build_caption(character: Character, token: str) -> str:
    return f"{token}, a {character.species}, {_STYLE_PHRASE}"
