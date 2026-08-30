from __future__ import annotations

from typing import Sequence

import cv2
import numpy
import torch
from diffusers import QwenImageEditPlusPipeline
from huggingface_hub import hf_hub_download
from nunchaku import NunchakuQwenImageTransformer2DModel
from PIL import Image

from reference_builder.torch_device import resolve_device
from sharedkernel.logger import get_logger
from transform.render_size import render_size
from transform.transform_config import RenderingConfig

LOGGER = get_logger("transform.qwen_editor")


def _to_pil(image: numpy.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _to_bgr(image: Image.Image) -> numpy.ndarray:
    return cv2.cvtColor(numpy.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


class QwenEditor:
    def __init__(self, config: RenderingConfig) -> None:
        self._config = config
        self._device = resolve_device(config.device)
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
                hf_hub_download(config.transformer_repo_id, config.transformer_file))
        self._pipeline = QwenImageEditPlusPipeline.from_pretrained(
                config.model_id, transformer=transformer, torch_dtype=torch.bfloat16).to(self._device)
        self._pipeline.set_progress_bar_config(disable=True)
        LOGGER.info("editor loaded: model=%s transformer=%s device=%s",
                    config.model_id, config.transformer_file, self._device)

    def edit(self, images: Sequence[numpy.ndarray], prompt: str, negative_prompt: str,
             target_width: int, target_height: int, seed: int) -> numpy.ndarray:
        width, height = render_size(target_width, target_height, self._config.render_pixels)
        edited = self._pipeline(
                image=[_to_pil(image) for image in images],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=self._config.inference_steps,
                true_cfg_scale=self._config.true_cfg_scale,
                generator=torch.Generator(device=self._device).manual_seed(seed),
        ).images[0]
        return _to_bgr(edited)
