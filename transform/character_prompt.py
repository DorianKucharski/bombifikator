from __future__ import annotations

from codex.character_models import Character, PersonDescription

_MAX_FEATURES = 6


def _feature_phrase(character: Character) -> str:
    features = ", ".join(character.appearance.distinguishing[:_MAX_FEATURES])
    return f", with {features}" if features else ""


def _outfit_phrase(character: Character) -> str:
    return f" wearing {character.appearance.outfit}" if character.appearance.outfit else ""


def build_character_prompt(character: Character, description: PersonDescription) -> str:
    return (
        f"Redraw the character from the image as one full figure on a plain flat white background, alone. "
        f"Keep the face, the head shape, the colours and every feature exactly as in the image: "
        f"{character.canonical_name}, a {character.species}"
        f"{_feature_phrase(character)}{_outfit_phrase(character)}. "
        f"Put the character in this pose: {description.pose}, facing {description.facing}. "
        f"Empty hands, no props, no objects, no scenery, no text, no shadow on the background. "
        f"Flat cel shading with heavy black outlines."
    )
