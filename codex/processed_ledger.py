from __future__ import annotations

import json
import threading
from pathlib import Path

from sharedkernel.utils.paths import ensure_directory


class ProcessedPageLedger:
    def __init__(self, ledger_path: Path) -> None:
        self._ledger_path = ledger_path
        self._lock = threading.Lock()
        self._entries: dict[str, str | None] = (
            json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
        )

    @staticmethod
    def _key(site_name: str, title: str) -> str:
        return f"{site_name}/{title}"

    def contains(self, site_name: str, title: str) -> bool:
        return self._key(site_name, title) in self._entries

    def record(self, site_name: str, title: str, slug: str | None) -> None:
        with self._lock:
            self._entries[self._key(site_name, title)] = slug
            ensure_directory(self._ledger_path.parent)
            self._ledger_path.write_text(
                    json.dumps(self._entries, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
            )
