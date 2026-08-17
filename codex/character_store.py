from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Iterator

import tomli_w

from codex.character_models import AgeGroup, AppearanceTraits, Build, Character, Prominence, ReferenceCoverage, Sex
from sharedkernel.utils.paths import ensure_directory


def _to_document(character: Character) -> dict[str, Any]:
    document: dict[str, Any] = {
        "slug": character.slug,
        "canonical_name": character.canonical_name,
        "species": character.species,
        "aliases": list(character.aliases),
        "sex": character.sex.value,
        "age_group": character.age_group.value,
        "prominence": character.prominence.value,
        "role_tags": list(character.role_tags),
        "summary": character.summary,
        "source_urls": list(character.source_urls),
        "reference_coverage": character.reference_coverage.value,
        "appearance": {
            "build": character.appearance.build.value,
            "skin": character.appearance.skin,
            "distinguishing": list(character.appearance.distinguishing),
            "palette": list(character.appearance.palette),
            "outfit": character.appearance.outfit,
        },
    }
    if character.appearance.height_meters is not None:
        document["appearance"]["height_meters"] = character.appearance.height_meters
    return document


def _to_character(document: dict[str, Any]) -> Character:
    appearance = document["appearance"]
    return Character(
            slug=document["slug"],
            canonical_name=document["canonical_name"],
            species=document["species"],
            aliases=tuple(document["aliases"]),
            sex=Sex(document["sex"]),
            age_group=AgeGroup(document["age_group"]),
            prominence=Prominence(document["prominence"]),
            appearance=AppearanceTraits(
                    build=Build(appearance["build"]),
                    height_meters=appearance.get("height_meters"),
                    skin=appearance["skin"],
                    distinguishing=tuple(appearance["distinguishing"]),
                    palette=tuple(appearance["palette"]),
                    outfit=appearance["outfit"],
            ),
            role_tags=tuple(document["role_tags"]),
            summary=document["summary"],
            source_urls=tuple(document["source_urls"]),
            reference_coverage=ReferenceCoverage(document["reference_coverage"]),
    )


class CharacterStore:
    def __init__(self, characters_dir: Path) -> None:
        self._characters_dir = characters_dir

    def _character_path(self, slug: str) -> Path:
        return ensure_directory(self._characters_dir) / f"{slug}.toml"

    def save(self, character: Character) -> Path:
        path = self._character_path(character.slug)
        path.write_text(tomli_w.dumps(_to_document(character)), encoding="utf-8")
        return path

    def contains(self, slug: str) -> bool:
        return self._character_path(slug).is_file()

    def load_all(self) -> Iterator[Character]:
        if not self._characters_dir.is_dir():
            return
        for path in sorted(self._characters_dir.glob("*.toml")):
            with path.open("rb") as handle:
                yield _to_character(tomllib.load(handle))
