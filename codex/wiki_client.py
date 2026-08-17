from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import quote

import httpx

from sharedkernel.logger import get_logger

LOGGER = get_logger("codex.wiki_client")

_USER_AGENT = "bombifikator/0.1 (private research; contact via repository owner)"


@dataclass(frozen=True)
class WikiSite:
    name: str
    api_url: str
    article_base_url: str
    category_namespace: str = "Kategoria"


@dataclass(frozen=True)
class WikiPage:
    site_name: str
    title: str
    url: str
    wikitext: str
    categories: tuple[str, ...] = ()


class MediaWikiClient:
    def __init__(self, site: WikiSite, requests_per_second: float, timeout_seconds: float) -> None:
        self._site = site
        self._minimum_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._last_request_at = 0.0
        self._client = httpx.Client(
                timeout=timeout_seconds,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
        )

    def __enter__(self) -> "MediaWikiClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._minimum_interval:
            time.sleep(self._minimum_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _query(self, params: dict[str, str]) -> dict[str, Any]:
        self._throttle()
        response = self._client.get(self._site.api_url, params={**params, "format": "json"})
        response.raise_for_status()
        return response.json()

    def _paginated_query(self, params: dict[str, str]) -> Iterator[dict[str, Any]]:
        continuation: dict[str, str] = {}
        while True:
            payload = self._query({**params, **continuation})
            yield payload
            next_continuation = payload.get("continue")
            if not next_continuation:
                return
            continuation = {key: str(value) for key, value in next_continuation.items()}

    def all_article_titles(self) -> list[str]:
        titles: list[str] = []
        for payload in self._paginated_query({
            "action": "query",
            "list": "allpages",
            "apnamespace": "0",
            "apfilterredir": "nonredirects",
            "aplimit": "500",
        }):
            titles.extend(page["title"] for page in payload["query"]["allpages"])
        return sorted(titles)

    def page_wikitext(self, title: str) -> WikiPage | None:
        payload = self._query({"action": "parse", "page": title, "prop": "wikitext|categories"})
        if "error" in payload:
            LOGGER.warning("parse failed: site=%s title=%s code=%s",
                           self._site.name, title, payload["error"].get("code"))
            return None
        parsed = payload["parse"]
        return WikiPage(
                site_name=self._site.name,
                title=title,
                url=self.article_url(title),
                wikitext=parsed["wikitext"]["*"],
                categories=tuple(
                        category["*"].replace("_", " ")
                        for category in parsed.get("categories", [])
                        if "hidden" not in category
                ),
        )

    def article_url(self, title: str) -> str:
        return self._site.article_base_url + quote(title.replace(" ", "_"))
