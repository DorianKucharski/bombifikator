from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy
import torch
from diffusers import QwenImageEditPlusPipeline, QwenImagePipeline
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


def _quantized_transformer(repo_id: str, transformer_file: str) -> NunchakuQwenImageTransformer2DModel:
    return NunchakuQwenImageTransformer2DModel.from_pretrained(hf_hub_download(repo_id, transformer_file))


class QwenEditor:
    def __init__(self, config: RenderingConfig) -> None:
        self._config = config
        self._device = resolve_device(config.device)
        self._pipeline = QwenImageEditPlusPipeline.from_pretrained(
                config.model_id,
                transformer=_quantized_transformer(config.transformer_repo_id, config.transformer_file),
                torch_dtype=torch.bfloat16,
        ).to(self._device)
        self._pipeline.set_progress_bar_config(disable=True)
        LOGGER.info("editor loaded: model=%s device=%s", config.model_id, self._device)

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


class QwenCharacterPainter:
    def __init__(self, config: RenderingConfig, lora_path: Path) -> None:
        self._config = config
        self._device = resolve_device(config.device)
        transformer = _quantized_transformer(config.base_transformer_repo_id, config.base_transformer_file)
        transformer.load_lora_adapter(str(lora_path))
        self._pipeline = QwenImagePipeline.from_pretrained(
                config.base_model_id, transformer=transformer, torch_dtype=torch.bfloat16).to(self._device)
        self._pipeline.set_progress_bar_config(disable=True)
        LOGGER.info("painter loaded: model=%s lora=%s device=%s", config.base_model_id, lora_path, self._device)

    def paint(self, prompt: str, negative_prompt: str, target_width: int, target_height: int,
              seed: int) -> numpy.ndarray:
        width, height = render_size(target_width, target_height, self._config.render_pixels)
        painted = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=self._config.inference_steps,
                true_cfg_scale=self._config.true_cfg_scale,
                generator=torch.Generator(device=self._device).manual_seed(seed),
        ).images[0]
        return _to_bgr(painted)
