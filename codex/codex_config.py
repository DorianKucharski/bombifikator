from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codex.wiki_client import WikiSite
from sharedkernel.config_provider import ConfigProvider


@dataclass(frozen=True)
class ExtractionConfig:
    model: str
    max_tokens: int
    effort: str
    concurrency: int
    max_wikitext_characters: int
    api_key: str

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "ExtractionConfig":
        return cls(
                model=config_provider.get_str("codex.extraction.model", "claude-opus-5"),
                max_tokens=config_provider.get_int("codex.extraction.max_tokens", 4000),
                effort=config_provider.get_str("codex.extraction.effort", "low"),
                concurrency=config_provider.get_int("codex.extraction.concurrency", 6),
                max_wikitext_characters=config_provider.get_int(
                        "codex.extraction.max_wikitext_characters", 24000),
                api_key=config_provider.get_secret("anthropic_api_key"),
        )


def _to_wiki_site(raw_site: dict[str, Any]) -> WikiSite:
    return WikiSite(
            name=raw_site["name"],
            api_url=raw_site["api_url"],
            article_base_url=raw_site["article_base_url"],
            category_namespace=raw_site.get("category_namespace", "Kategoria"),
    )


@dataclass(frozen=True)
class CodexConfig:
    raw_dir: Path
    characters_dir: Path
    ledger_path: Path
    requests_per_second: float
    timeout_seconds: float
    min_wikitext_length: int
    sites: tuple[WikiSite, ...]

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "CodexConfig":
        raw_sites = config_provider.get("codex.sites")
        if not raw_sites:
            raise ValueError("missing required configuration 'codex.sites': define at least one [[codex.sites]] table")
        return cls(
                raw_dir=config_provider.get_path("codex.raw_dir", "data/codex/raw"),
                characters_dir=config_provider.get_path("codex.characters_dir", "data/codex/characters"),
                ledger_path=config_provider.get_path("codex.ledger_path", "data/codex/processed.json"),
                requests_per_second=config_provider.get_float("codex.requests_per_second", 1.0),
                timeout_seconds=config_provider.get_float("codex.timeout_seconds", 30.0),
                min_wikitext_length=config_provider.get_int("codex.min_wikitext_length", 200),
                sites=tuple(_to_wiki_site(raw_site) for raw_site in raw_sites),
        )
