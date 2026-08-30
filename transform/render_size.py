from __future__ import annotations

_SIZE_MULTIPLE = 16
_MINIMUM_SIDE = 256


def render_size(target_width: int, target_height: int, render_pixels: int) -> tuple[int, int]:
    scale = render_pixels / max(target_width, target_height)
    scaled = (max(_MINIMUM_SIDE, int(target_width * scale)), max(_MINIMUM_SIDE, int(target_height * scale)))
    return tuple(value - value % _SIZE_MULTIPLE for value in scaled)
