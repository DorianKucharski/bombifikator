from __future__ import annotations

from codex.character_models import PersonDescription
from codex.drawing_style import STYLE_PHRASE


def build_character_prompt(description: PersonDescription, token: str) -> str:
    return (f"{token}, {description.pose}, facing {description.facing}, "
            f"full figure, empty hands, no props, no scenery, {STYLE_PHRASE}")
