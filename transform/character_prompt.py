from __future__ import annotations

from codex.character_models import Character, PersonDescription
from codex.drawing_style import STYLE_PHRASE


def build_character_prompt(character: Character, description: PersonDescription, token: str) -> str:
    return (f"{token}, a {character.species}, {description.pose}, facing {description.facing}, "
            f"full figure, empty hands, no props, no scenery, {STYLE_PHRASE}")
