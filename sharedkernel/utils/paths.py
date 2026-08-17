from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_POLISH_TRANSLITERATION = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
    "ó": "o", "ś": "s", "ź": "z", "ż": "z",
})

_NON_SLUG_CHARACTERS = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    lowered = text.strip().lower().translate(_POLISH_TRANSLITERATION)
    ascii_only = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")
    return _NON_SLUG_CHARACTERS.sub("-", ascii_only).strip("-")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
