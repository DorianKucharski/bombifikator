from __future__ import annotations

import torch

from sharedkernel.logger import get_logger

LOGGER = get_logger("reference_builder.torch_device")

_AUTOMATIC_DEVICE = "auto"


def resolve_device(configured_device: str) -> str:
    if configured_device != _AUTOMATIC_DEVICE:
        return configured_device
    if torch.cuda.is_available():
        return "cuda"
    LOGGER.warning("cuda unavailable, falling back to cpu")
    return "cpu"
