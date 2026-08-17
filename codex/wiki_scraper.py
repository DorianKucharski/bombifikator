from __future__ import annotations

from dataclasses import dataclass

import httpx

from codex.codex_config import CodexConfig
from codex.raw_page_store import RawPageStore
from codex.wiki_client import MediaWikiClient
from sharedkernel.logger import get_logger

LOGGER = get_logger("codex.wiki_scraper")


@dataclass(frozen=True)
class ScrapeReport:
    discovered: int
    downloaded: int
    skipped_existing: int
    skipped_too_short: int
    failed: tuple[str, ...]


def scrape_all_sites(config: CodexConfig, refresh: bool) -> ScrapeReport:
    store = RawPageStore(config.raw_dir)
    discovered = 0
    downloaded = 0
    skipped_existing = 0
    skipped_too_short = 0
    failed: list[str] = []

    for site in config.sites:
        with MediaWikiClient(site, config.requests_per_second, config.timeout_seconds) as client:
            titles = client.all_article_titles()
            discovered += len(titles)
            LOGGER.info("discovery finished: site=%s articles=%d", site.name, len(titles))

            for title in titles:
                if not refresh and store.contains(site.name, title):
                    skipped_existing += 1
                    continue
                try:
                    page = client.page_wikitext(title)
                except httpx.HTTPError as error:
                    LOGGER.warning("download failed: site=%s title=%s error=%s", site.name, title, error)
                    failed.append(f"{site.name}/{title}")
                    continue
                if page is None:
                    failed.append(f"{site.name}/{title}")
                    continue
                if len(page.wikitext) < config.min_wikitext_length:
                    skipped_too_short += 1
                    continue
                store.save(page)
                downloaded += 1

    return ScrapeReport(
            discovered=discovered,
            downloaded=downloaded,
            skipped_existing=skipped_existing,
            skipped_too_short=skipped_too_short,
            failed=tuple(failed),
    )
