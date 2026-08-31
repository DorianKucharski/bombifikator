from __future__ import annotations

import cv2

from codex.drawing_style import NEGATIVE_STYLE_PHRASE
from reference_builder.contact_sheet import build_contact_sheet
from sharedkernel.logger import get_logger
from sharedkernel.utils.paths import ensure_directory
from training.token_preview import build_preview_prompt, preview_path_of
from training.training_config import PreviewConfig
from transform.qwen_editor import QwenCharacterPainter
from transform.token_map import read_token_map

LOGGER = get_logger("training.preview_renderer")


def write_token_previews(painter: QwenCharacterPainter, config: PreviewConfig) -> int:
    tokens = read_token_map(config.tokens_path)
    ensure_directory(config.previews_dir)
    written = 0
    for slug, token in sorted(tokens.items()):
        prompt = build_preview_prompt(token)
        images = [painter.paint(prompt, NEGATIVE_STYLE_PHRASE, config.preview_pixels, config.preview_pixels,
                                config.seed + offset)
                  for offset in range(config.samples_per_token)]
        cv2.imwrite(str(preview_path_of(config.previews_dir, slug)),
                    build_contact_sheet(images, config.sheet_grid_size, config.sheet_tile_pixels))
        written += 1
        LOGGER.info("preview written: slug=%s token=%s samples=%d", slug, token, len(images))
    return written
