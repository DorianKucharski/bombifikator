from __future__ import annotations

import cv2
import numpy

from transform.qwen_editor import QwenEditor

_PLATE_PROMPT = ("Remove every person from the photograph. Leave the place empty: the same room or street, "
                 "the same walls, floor, furniture, vehicles and objects, the same camera angle, the same "
                 "lighting and the same colours. Fill in whatever the people were covering so the scene looks "
                 "natural and complete. Change nothing else. Keep it a photograph.")
_NEGATIVE_PROMPT = "people, person, face, hands, cartoon, drawing, text, watermark"


def render_plate(editor: QwenEditor, photo: numpy.ndarray, seed: int) -> numpy.ndarray:
    plate = editor.edit((photo,), _PLATE_PROMPT, _NEGATIVE_PROMPT, photo.shape[1], photo.shape[0], seed)
    return cv2.resize(plate, (photo.shape[1], photo.shape[0]))
