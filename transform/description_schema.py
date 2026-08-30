from __future__ import annotations

from typing import Any

from codex.character_models import AgeGroup, Sex


def _enum_values(enum_class: type) -> list[str]:
    return [member.value for member in enum_class]


PERSON_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sex", "age_group", "build_description", "outfit", "pose", "facing", "notable_features",
                 "character_slugs"],
    "properties": {
        "sex": {"type": "string", "enum": _enum_values(Sex)},
        "age_group": {"type": "string", "enum": _enum_values(AgeGroup)},
        "build_description": {
            "type": "string",
            "description": "Body build and height impression in Polish, for example krepy niski mezczyzna.",
        },
        "outfit": {"type": "string", "description": "Clothing in Polish, colours included."},
        "pose": {
            "type": "string",
            "description": "Body posture in Polish and nothing else: the stance, what the arms and legs do, "
                           "how the head is turned. Never name furniture, vehicles, tables or anything the "
                           "person sits on, holds or leans against. Write 'siedzi z rekami na kolanach', "
                           "never 'siedzi w fotelu samochodowym'.",
        },
        "facing": {
            "type": "string",
            "description": "Which way the person faces, in Polish: do kamery, w lewo, w prawo, tylem.",
        },
        "notable_features": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Visually distinctive traits in Polish: broda, okulary, czapka, tatuaz.",
        },
        "character_slugs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exactly three slugs from the catalogue, best match first. Judge by how well the "
                           "character "
                           "would stand in for this person: build, sex, age, hair, facial hair, headgear and "
                           "clothing. Slugs only, exactly as spelled in the catalogue.",
        },
    },
}
