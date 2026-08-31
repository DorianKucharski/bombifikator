from __future__ import annotations

from pathlib import Path

from codex.drawing_style import STYLE_PHRASE

_PREVIEW_POSE = "standing upright, arms at the sides"


def build_preview_prompt(token: str) -> str:
    return f"{token}, {_PREVIEW_POSE}, full figure, empty hands, no props, no scenery, {STYLE_PHRASE}"


def preview_path_of(previews_dir: Path, slug: str) -> Path:
    return previews_dir / f"{slug}.png"
