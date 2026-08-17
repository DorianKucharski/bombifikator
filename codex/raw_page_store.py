from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from codex.wiki_client import WikiPage
from sharedkernel.utils.paths import ensure_directory, slugify


class RawPageStore:
    def __init__(self, raw_dir: Path) -> None:
        self._raw_dir = raw_dir

    def _page_path(self, site_name: str, title: str) -> Path:
        return ensure_directory(self._raw_dir / site_name) / f"{slugify(title)}.json"

    def save(self, page: WikiPage) -> Path:
        path = self._page_path(page.site_name, page.title)
        path.write_text(
                json.dumps({
                    "site_name": page.site_name,
                    "title": page.title,
                    "url": page.url,
                    "wikitext": page.wikitext,
                    "categories": list(page.categories),
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
        )
        return path

    def contains(self, site_name: str, title: str) -> bool:
        return self._page_path(site_name, title).is_file()

    def load_all(self) -> Iterator[WikiPage]:
        for path in sorted(self._raw_dir.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            yield WikiPage(
                    site_name=payload["site_name"],
                    title=payload["title"],
                    url=payload["url"],
                    wikitext=payload["wikitext"],
                    categories=tuple(payload.get("categories", [])),
            )
