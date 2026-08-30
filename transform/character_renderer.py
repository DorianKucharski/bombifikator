from __future__ import annotations

from typing import Sequence

import numpy

from transform.qwen_editor import QwenEditor

_NEGATIVE_PROMPT = ("photorealistic skin, human face, photograph, scenery, background objects, several characters, "
                    "blurry, deformed, extra limbs, watermark, text")


def render_character(editor: QwenEditor, references: Sequence[numpy.ndarray], prompt: str,
                     target_width: int, target_height: int, seed: int) -> numpy.ndarray:
    return editor.edit(references, prompt, _NEGATIVE_PROMPT, target_width, target_height, seed)
