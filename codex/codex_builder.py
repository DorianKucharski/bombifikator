from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import anthropic

from codex.character_extractor import extract_character
from codex.character_store import CharacterStore
from codex.codex_config import CodexConfig, ExtractionConfig
from codex.processed_ledger import ProcessedPageLedger
from codex.raw_page_store import RawPageStore
from codex.wiki_client import WikiPage
from sharedkernel.clients.anthropic_client import StructuredExtractionClient
from sharedkernel.logger import get_logger

LOGGER = get_logger("codex.codex_builder")


@dataclass(frozen=True)
class ExtractionReport:
    total_pages: int
    extracted: int
    rejected: int
    skipped_processed: int
    failed: tuple[str, ...]


def _process_page(
        page: WikiPage,
        client: StructuredExtractionClient,
        store: CharacterStore,
        ledger: ProcessedPageLedger,
        max_wikitext_characters: int,
) -> str:
    try:
        character = extract_character(client, page, max_wikitext_characters)
    except anthropic.APIError as error:
        LOGGER.warning("extraction failed: title=%s error=%s", page.title, error)
        return "failed"
    if character is None:
        ledger.record(page.site_name, page.title, None)
        return "rejected"
    store.save(character)
    ledger.record(page.site_name, page.title, character.slug)
    LOGGER.info("character extracted: slug=%s species=%s prominence=%s",
                character.slug, character.species, character.prominence.value)
    return "extracted"


def build_codex(config: CodexConfig, extraction: ExtractionConfig, refresh: bool) -> ExtractionReport:
    pages = list(RawPageStore(config.raw_dir).load_all())
    ledger = ProcessedPageLedger(config.ledger_path)
    store = CharacterStore(config.characters_dir)
    client = StructuredExtractionClient(
            api_key=extraction.api_key,
            model=extraction.model,
            max_tokens=extraction.max_tokens,
            effort=extraction.effort,
    )

    pending = [page for page in pages if refresh or not ledger.contains(page.site_name, page.title)]
    LOGGER.info("extraction starting: pages=%d pending=%d concurrency=%d",
                len(pages), len(pending), extraction.concurrency)

    with ThreadPoolExecutor(max_workers=extraction.concurrency) as executor:
        outcomes = list(executor.map(
                lambda page: _process_page(page, client, store, ledger, extraction.max_wikitext_characters),
                pending,
        ))

    return ExtractionReport(
            total_pages=len(pages),
            extracted=outcomes.count("extracted"),
            rejected=outcomes.count("rejected"),
            skipped_processed=len(pages) - len(pending),
            failed=tuple(page.title for page, outcome in zip(pending, outcomes) if outcome == "failed"),
    )
