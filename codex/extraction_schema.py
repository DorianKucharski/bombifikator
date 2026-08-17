from __future__ import annotations

from typing import Any

from codex.character_models import AgeGroup, Build, Prominence, Sex


def _enum_values(enum_class: type) -> list[str]:
    return [member.value for member in enum_class]


def _string_array(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": description}


CHARACTER_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "is_character",
        "rejection_reason",
        "canonical_name",
        "species",
        "aliases",
        "sex",
        "age_group",
        "prominence",
        "role_tags",
        "summary",
        "appearance",
    ],
    "properties": {
        "is_character": {
            "type": "boolean",
            "description": "True only for an individual character or a sapient species that could stand in "
                           "for a human on a photograph. False for episodes, planets, objects, food, "
                           "organisations, weapons, locations and meta pages.",
        },
        "rejection_reason": {
            "type": "string",
            "description": "When is_character is false, one short phrase naming what the page actually is. "
                           "Empty string otherwise.",
        },
        "canonical_name": {"type": "string"},
        "species": {
            "type": "string",
            "description": "Species or race in Polish, lowercase, for example kurvinox, czlowiek, chujew. "
                           "Use 'nieznany' when the source does not say.",
        },
        "aliases": _string_array("Alternative spellings and nicknames used in the source."),
        "sex": {"type": "string", "enum": _enum_values(Sex)},
        "age_group": {"type": "string", "enum": _enum_values(AgeGroup)},
        "prominence": {
            "type": "string",
            "enum": _enum_values(Prominence),
            "description": "MAIN for the handful of leads, RECURRING for characters across several episodes, "
                           "EPISODIC for one-off appearances.",
        },
        "role_tags": _string_array("Short Polish role tags, for example zolnierz, kosmita, antagonista."),
        "summary": {
            "type": "string",
            "description": "Two or three sentences in Polish covering who the character is and how they look.",
        },
        "appearance": {
            "type": "object",
            "additionalProperties": False,
            "required": ["build", "height_meters", "skin", "distinguishing", "palette", "outfit"],
            "properties": {
                "build": {"type": "string", "enum": _enum_values(Build)},
                "height_meters": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "skin": {
                    "type": "string",
                    "description": "Skin, scales, fur or casing, described in Polish. Empty string when unknown.",
                },
                "distinguishing": _string_array(
                        "Visual features a generative model would need: horns, tail, uniform, scar, glasses."),
                "palette": _string_array("Dominant colours as hex values, for example #3B7DD8."),
                "outfit": {"type": "string", "description": "Typical clothing or armour, in Polish."},
            },
        },
    },
}
