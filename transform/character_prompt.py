from __future__ import annotations

from codex.character_models import Character, PersonDescription

_STYLE_PHRASE = "flat cel shaded cartoon character with heavy black outlines on a plain white background"


def build_character_prompt(character: Character, description: PersonDescription, token: str) -> str:
    return (f"{token}, a {character.species}, {description.pose}, facing {description.facing}, "
            f"full figure, empty hands, no props, no scenery, {_STYLE_PHRASE}")
