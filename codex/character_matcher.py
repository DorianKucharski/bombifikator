from __future__ import annotations

from codex.character_models import Character, PersonDescription, ReferenceCoverage, Sex

_NEUTRAL_SEX_VALUES = frozenset({Sex.UNKNOWN, Sex.NONBINARY})


def sex_allows(character: Character, description: PersonDescription) -> bool:
    if character.sex in _NEUTRAL_SEX_VALUES or description.sex in _NEUTRAL_SEX_VALUES:
        return True
    return character.sex is description.sex


def usable_characters(characters: tuple[Character, ...], allow_thin_coverage: bool) -> tuple[Character, ...]:
    accepted = {ReferenceCoverage.FULL} | ({ReferenceCoverage.THIN} if allow_thin_coverage else set())
    return tuple(character for character in characters if character.reference_coverage in accepted)


def acceptable_choices(choices: tuple[str, ...], characters: dict[str, Character],
                       description: PersonDescription) -> tuple[str, ...]:
    return tuple(
            slug for slug in choices
            if slug in characters and sex_allows(characters[slug], description)
    )


def assign_without_repeats(choices_by_person: dict[int, tuple[str, ...]]) -> dict[int, str]:
    taken: set[str] = set()
    assignments: dict[int, str] = {}
    for person_index in sorted(choices_by_person):
        free = next((slug for slug in choices_by_person[person_index] if slug not in taken), None)
        if free is None:
            continue
        taken.add(free)
        assignments[person_index] = free
    return assignments
