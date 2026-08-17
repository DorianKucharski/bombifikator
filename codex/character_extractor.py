from __future__ import annotations

from codex.character_models import AgeGroup, AppearanceTraits, Build, Character, Prominence, ReferenceCoverage, Sex
from codex.extraction_schema import CHARACTER_EXTRACTION_SCHEMA
from codex.wiki_client import WikiPage
from sharedkernel.clients.anthropic_client import StructuredExtractionClient
from sharedkernel.logger import get_logger
from sharedkernel.utils.paths import slugify

LOGGER = get_logger("codex.character_extractor")

_SYSTEM_PROMPT = """You build a character codex for the Polish adult animated series Kapitan Bomba by Bartosz \
Walaszek. The codex feeds an image pipeline that replaces people in real photographs with characters from that \
universe, so every field is judged by whether it helps reproduce the character visually.

You receive one page of wikitext from a fan wiki. Decide first whether the page describes an individual character \
or a sapient species that could plausibly stand in for a human on a photograph. Episodes, planets, spacecraft, \
weapons, food, drinks, organisations, locations, in-universe media and meta pages are not characters.

Rules for the fields:
- Write every free-text field in Polish, matching the vocabulary the series itself uses.
- Never invent a detail the page does not support. Prefer an empty string or an empty list over a guess.
- Vulgar names are canonical here; reproduce them exactly as the source spells them.
- appearance.distinguishing is the most important field. List concrete, drawable features only.
- appearance.palette holds hex values you infer from colour words in the text. Leave it empty when the page \
names no colours."""


def _to_appearance(payload: dict) -> AppearanceTraits:
    return AppearanceTraits(
            build=Build(payload["build"]),
            height_meters=payload["height_meters"],
            skin=payload["skin"],
            distinguishing=tuple(payload["distinguishing"]),
            palette=tuple(payload["palette"]),
            outfit=payload["outfit"],
    )


def _build_user_prompt(page: WikiPage, max_characters: int) -> str:
    categories = ", ".join(page.categories) if page.categories else "brak"
    return (
        f"Tytul strony: {page.title}\n"
        f"Kategorie: {categories}\n"
        f"Zrodlo: {page.url}\n\n"
        f"Wikitekst:\n{page.wikitext[:max_characters]}"
    )


def extract_character(
        client: StructuredExtractionClient,
        page: WikiPage,
        max_wikitext_characters: int,
) -> Character | None:
    payload = client.extract(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(page, max_wikitext_characters),
            json_schema=CHARACTER_EXTRACTION_SCHEMA,
    )
    if payload is None:
        return None
    if not payload["is_character"]:
        LOGGER.debug("page rejected: title=%s reason=%s", page.title, payload["rejection_reason"])
        return None

    canonical_name = payload["canonical_name"] or page.title
    return Character(
            slug=slugify(page.title),
            canonical_name=canonical_name,
            species=payload["species"],
            aliases=tuple(payload["aliases"]),
            sex=Sex(payload["sex"]),
            age_group=AgeGroup(payload["age_group"]),
            prominence=Prominence(payload["prominence"]),
            appearance=_to_appearance(payload["appearance"]),
            role_tags=tuple(payload["role_tags"]),
            summary=payload["summary"],
            source_urls=(page.url,),
            reference_coverage=ReferenceCoverage.NONE,
    )
