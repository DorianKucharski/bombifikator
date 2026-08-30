from __future__ import annotations

from codex.character_models import Character

_MAX_FEATURES = 6
_STYLE_PHRASE = "flat cel shaded cartoon character with heavy black outlines on a plain white background"


def character_token(token_prefix: str, slug: str) -> str:
    return f"{token_prefix}_{slug.replace('-', '_')}"


def _feature_phrase(character: Character) -> str:
    features = ", ".join(character.appearance.distinguishing[:_MAX_FEATURES])
    return f", {features}" if features else ""


def _outfit_phrase(character: Character) -> str:
    return f", wearing {character.appearance.outfit}" if character.appearance.outfit else ""


def _skin_phrase(character: Character) -> str:
    return f", {character.appearance.skin}" if character.appearance.skin else ""


def build_caption(character: Character, token_prefix: str) -> str:
    return (f"{character_token(token_prefix, character.slug)}, {character.canonical_name}, "
            f"a {character.species}"
            f"{_skin_phrase(character)}{_feature_phrase(character)}{_outfit_phrase(character)}, "
            f"{_STYLE_PHRASE}")
