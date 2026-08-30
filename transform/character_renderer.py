from __future__ import annotations

import numpy

from transform.qwen_editor import QwenCharacterPainter

_NEGATIVE_PROMPT = ("photorealistic skin, human face, photograph, scenery, background objects, several characters, "
                    "props, furniture, vehicles, blurry, deformed, extra limbs, watermark, text")


def paint_character(painter: QwenCharacterPainter, prompt: str, target_width: int, target_height: int,
                    seed: int) -> numpy.ndarray:
    return painter.paint(prompt, _NEGATIVE_PROMPT, target_width, target_height, seed)
