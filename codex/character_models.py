from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Sex(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    NONBINARY = "NONBINARY"
    UNKNOWN = "UNKNOWN"


class Build(str, Enum):
    HUMANOID = "HUMANOID"
    REPTILIAN = "REPTILIAN"
    ANIMAL = "ANIMAL"
    BLOB = "BLOB"
    MECHANICAL = "MECHANICAL"
    UNKNOWN = "UNKNOWN"


class AgeGroup(str, Enum):
    CHILD = "CHILD"
    YOUNG_ADULT = "YOUNG_ADULT"
    ADULT = "ADULT"
    ELDERLY = "ELDERLY"
    UNKNOWN = "UNKNOWN"


class ReferenceCoverage(str, Enum):
    NONE = "NONE"
    THIN = "THIN"
    FULL = "FULL"


class Prominence(str, Enum):
    MAIN = "MAIN"
    RECURRING = "RECURRING"
    EPISODIC = "EPISODIC"


@dataclass(frozen=True)
class AppearanceTraits:
    build: Build = Build.UNKNOWN
    height_meters: float | None = None
    skin: str = ""
    distinguishing: tuple[str, ...] = ()
    palette: tuple[str, ...] = ()
    outfit: str = ""


@dataclass(frozen=True)
class Character:
    slug: str
    canonical_name: str
    species: str
    aliases: tuple[str, ...] = ()
    sex: Sex = Sex.UNKNOWN
    age_group: AgeGroup = AgeGroup.UNKNOWN
    prominence: Prominence = Prominence.EPISODIC
    appearance: AppearanceTraits = field(default_factory=AppearanceTraits)
    role_tags: tuple[str, ...] = ()
    summary: str = ""
    source_urls: tuple[str, ...] = ()
    reference_coverage: ReferenceCoverage = ReferenceCoverage.NONE

    @property
    def is_usable_for_transform(self) -> bool:
        return self.reference_coverage is ReferenceCoverage.FULL


@dataclass(frozen=True)
class PersonDescription:
    person_index: int
    sex: Sex
    age_group: AgeGroup
    build_description: str
    outfit: str
    pose: str
    facing: str
    notable_features: tuple[str, ...] = ()

    def as_matching_text(self) -> str:
        features = ", ".join(self.notable_features)
        return " ".join(part for part in (
            self.build_description,
            self.outfit,
            self.pose,
            features,
        ) if part).strip()
