from __future__ import annotations

from dataclasses import replace

from codex.character_models import Character, ReferenceCoverage
from codex.character_store import CharacterStore


def coverage_of(reference_count: int, full_threshold: int, thin_threshold: int) -> ReferenceCoverage:
    if reference_count >= full_threshold:
        return ReferenceCoverage.FULL
    if reference_count >= thin_threshold:
        return ReferenceCoverage.THIN
    return ReferenceCoverage.NONE


def mark_coverage(store: CharacterStore, character: Character, coverage: ReferenceCoverage) -> Character:
    marked = replace(character, reference_coverage=coverage)
    store.save(marked)
    return marked
