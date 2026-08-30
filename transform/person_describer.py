from __future__ import annotations

import cv2
import numpy

from codex.character_models import AgeGroup, Character, PersonDescription, Sex
from sharedkernel.clients.anthropic_client import StructuredExtractionClient
from sharedkernel.logger import get_logger
from transform.description_schema import PERSON_DESCRIPTION_SCHEMA
from transform.person_models import PersonInstance

LOGGER = get_logger("transform.person_describer")

_SYSTEM_PROMPT_HEADER = """You describe one person on a photograph and pick the character from the Polish \
animated series Kapitan Bomba that should replace them.

You receive the whole photograph and then a close-up of the person. Describe only that person, the one shown in \
the close-up. Write every free-text field in Polish.

Judge only what you can see. Never guess an identity, never name a real person, never describe anything the \
photograph does not show. Prefer a short empty answer over an invented detail.

Then pick three characters from the catalogue below, best match first. A good match shares the build, the sex, \
the age impression and the visible traits: hair, facial hair, headgear, glasses, clothing. Return slugs only.

Catalogue:
"""


def _catalogue_entry(character: Character) -> str:
    features = ", ".join(character.appearance.distinguishing[:6]) or "brak opisu"
    return (f"- {character.slug} | {character.canonical_name} | {character.species} | {character.sex.value} | "
            f"{character.age_group.value} | {character.appearance.build.value} | {character.appearance.outfit} | "
            f"{features}")


def build_system_prompt(characters: tuple[Character, ...]) -> str:
    return _SYSTEM_PROMPT_HEADER + "\n".join(_catalogue_entry(character) for character in characters)

_CROP_PADDING_RATIO = 0.12


def _padded_person_crop(image: numpy.ndarray, person: PersonInstance) -> numpy.ndarray:
    padding = int(round(person.box.shorter_side * _CROP_PADDING_RATIO))
    top = max(0, person.box.top - padding)
    bottom = min(image.shape[0], person.box.bottom + padding)
    left = max(0, person.box.left - padding)
    right = min(image.shape[1], person.box.right + padding)
    return image[top:bottom, left:right]


def _to_png_bytes(image: numpy.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


def describe_person(
        client: StructuredExtractionClient,
        system_prompt: str,
        image: numpy.ndarray,
        person: PersonInstance,
) -> tuple[PersonDescription, tuple[str, ...]] | None:
    payload = client.extract(
            system_prompt=system_prompt,
            user_prompt=f"Pierwszy obraz to cale zdjecie, drugi to zblizenie osoby numer {person.person_index}. "
                        f"Opisz osobe ze zblizenia.",
            json_schema=PERSON_DESCRIPTION_SCHEMA,
            images=(_to_png_bytes(image), _to_png_bytes(_padded_person_crop(image, person))),
    )
    if payload is None:
        LOGGER.warning("description refused: person=%d", person.person_index)
        return None
    description = PersonDescription(
            person_index=person.person_index,
            sex=Sex(payload["sex"]),
            age_group=AgeGroup(payload["age_group"]),
            build_description=payload["build_description"],
            outfit=payload["outfit"],
            pose=payload["pose"],
            facing=payload["facing"],
            notable_features=tuple(payload["notable_features"]),
    )
    return description, tuple(payload["character_slugs"])
