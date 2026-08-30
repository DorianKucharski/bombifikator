from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy

from codex.character_matcher import acceptable_choices, assign_without_repeats, usable_characters
from codex.character_models import Character, PersonDescription
from codex.character_store import CharacterStore
from reference_builder.background_remover import BackgroundRemover
from sharedkernel.clients.anthropic_client import StructuredExtractionClient
from sharedkernel.logger import get_logger
from sharedkernel.utils.paths import ensure_directory
from transform.character_prompt import build_character_prompt
from transform.background_plate import render_plate
from transform.character_renderer import render_character
from transform.person_compositor import paste_character, with_people_removed
from transform.person_describer import build_system_prompt, describe_person
from transform.person_detector import PersonDetector
from transform.person_eraser import dilated
from transform.person_models import CharacterAssignment, PersonInstance
from transform.qwen_editor import QwenEditor
from transform.reference_picker import load_references
from transform.transform_config import (DescriptionConfig, MatchingConfig, RenderingConfig,
                                        SegmentationConfig, TransformPaths)

LOGGER = get_logger("transform.transform_pipeline")


@dataclass(frozen=True)
class TransformReport:
    output_path: Path
    people_detected: int
    people_described: int
    people_replaced: int
    assignments: tuple[CharacterAssignment, ...]


def _describe_all(client: StructuredExtractionClient, system_prompt: str, image: numpy.ndarray,
                  people: tuple[PersonInstance, ...]) -> dict[int, tuple[PersonDescription, tuple[str, ...]]]:
    described = {}
    for person in people:
        answer = describe_person(client, system_prompt, image, person)
        if answer is not None:
            described[person.person_index] = answer
            LOGGER.info("person described: person=%d sex=%s outfit=%s choices=%s",
                        person.person_index, answer[0].sex.value, answer[0].outfit, ", ".join(answer[1]))
    return described


def _assign_characters(characters_by_slug: dict[str, Character],
                       described: dict[int, tuple[PersonDescription, tuple[str, ...]]],
                       ) -> tuple[CharacterAssignment, ...]:
    choices = {
        person_index: acceptable_choices(slugs, characters_by_slug, description)
        for person_index, (description, slugs) in described.items()
    }
    return tuple(
            CharacterAssignment(person_index=person_index, slug=slug)
            for person_index, slug in sorted(assign_without_repeats(choices).items())
    )


def _farthest_first(assignments: tuple[CharacterAssignment, ...],
                    people_by_index: dict[int, PersonInstance]) -> tuple[CharacterAssignment, ...]:
    return tuple(sorted(assignments, key=lambda assignment: people_by_index[assignment.person_index].pixel_area))


def transform_photo(
        image_path: Path,
        output_path: Path,
        paths: TransformPaths,
        segmentation: SegmentationConfig,
        description: DescriptionConfig,
        matching: MatchingConfig,
        rendering: RenderingConfig,
) -> TransformReport:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"unreadable photograph {image_path}")

    people = PersonDetector(segmentation).detect(image)
    LOGGER.info("people detected: count=%d", len(people))
    if not people:
        raise ValueError(f"no person found on {image_path}")

    client = StructuredExtractionClient(
            api_key=description.api_key,
            model=description.model,
            max_tokens=description.max_tokens,
            effort=description.effort,
    )
    characters = usable_characters(
            tuple(CharacterStore(paths.characters_dir).load_all()), matching.allow_thin_coverage)
    if not characters:
        raise ValueError(f"no character with reference coverage in {paths.characters_dir}: run 'references build'")
    characters_by_slug = {character.slug: character for character in characters}

    described = _describe_all(client, build_system_prompt(characters), image, people)
    assignments = _assign_characters(characters_by_slug, described)
    people_by_index = {person.person_index: person for person in people}

    editor = QwenEditor(rendering)
    remover = BackgroundRemover()
    replaceable = tuple(people_by_index[assignment.person_index] for assignment in assignments)
    people_mask = numpy.zeros(image.shape[:2], dtype=numpy.uint8)
    for person in replaceable:
        people_mask |= dilated(person.mask, rendering.erase_dilation_ratio)
    composited = with_people_removed(
            image, render_plate(editor, image, rendering.seed), people_mask, rendering.plate_feather_pixels)
    replaced = 0
    for assignment in _farthest_first(assignments, people_by_index):
        references = load_references(paths.references_dir, assignment.slug, rendering.reference_images)
        if not references:
            LOGGER.warning("no reference image for character: slug=%s", assignment.slug)
            continue
        person = people_by_index[assignment.person_index]
        rendered = render_character(
                editor,
                references,
                build_character_prompt(characters_by_slug[assignment.slug],
                                       described[assignment.person_index][0]),
                person.box.width,
                person.box.height,
                rendering.seed + assignment.person_index,
        )
        composited, _ = paste_character(composited, remover.cut_out(rendered), person.box)
        replaced += 1
        LOGGER.info("person replaced: person=%d slug=%s box=%dx%d",
                    assignment.person_index, assignment.slug, person.box.width, person.box.height)

    ensure_directory(output_path.parent)
    cv2.imwrite(str(output_path), composited)
    return TransformReport(
            output_path=output_path,
            people_detected=len(people),
            people_described=len(described),
            people_replaced=replaced,
            assignments=assignments,
    )
