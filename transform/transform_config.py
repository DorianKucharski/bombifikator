from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sharedkernel.config_provider import ConfigProvider

_SEGMENTATION_MODEL = "yolo11x-seg.pt"
_RENDERING_MODEL = "Qwen/Qwen-Image-Edit-2509"
_QUANTIZED_TRANSFORMER_REPO = "nunchaku-ai/nunchaku-qwen-image-edit-2509"
_QUANTIZED_TRANSFORMER_FILE = "svdq-fp4_r128-qwen-image-edit-2509.safetensors"
_BASE_MODEL = "Qwen/Qwen-Image"


@dataclass(frozen=True)
class TransformPaths:
    references_dir: Path
    characters_dir: Path
    tokens_path: Path
    output_dir: Path
    weights_dir: Path

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "TransformPaths":
        return cls(
                references_dir=config_provider.get_path("reference_builder.references_dir", "data/references/cards"),
                characters_dir=config_provider.get_path("codex.characters_dir", "data/codex/characters"),
                tokens_path=config_provider.get_path("training.tokens_path", "data/training/tokens.toml"),
                output_dir=config_provider.get_path("transform.output_dir", "data/output"),
                weights_dir=config_provider.get_path("transform.weights_dir", "data/weights"),
        )


@dataclass(frozen=True)
class SegmentationConfig:
    model_id: str
    confidence_threshold: float
    min_person_area_ratio: float
    device: str

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "SegmentationConfig":
        return cls(
                model_id=config_provider.get_str("transform.segmentation.model_id", _SEGMENTATION_MODEL),
                confidence_threshold=config_provider.get_float("transform.segmentation.confidence_threshold", 0.3),
                min_person_area_ratio=config_provider.get_float("transform.segmentation.min_person_area_ratio", 0.002),
                device=config_provider.get_str("transform.segmentation.device", "auto"),
        )


@dataclass(frozen=True)
class DescriptionConfig:
    model: str
    max_tokens: int
    effort: str
    api_key: str

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "DescriptionConfig":
        return cls(
                model=config_provider.get_str("transform.description.model", "claude-opus-5"),
                max_tokens=config_provider.get_int("transform.description.max_tokens", 2000),
                effort=config_provider.get_str("transform.description.effort", "medium"),
                api_key=config_provider.get_secret("anthropic_api_key"),
        )


@dataclass(frozen=True)
class MatchingConfig:
    allow_thin_coverage: bool

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "MatchingConfig":
        return cls(
                allow_thin_coverage=config_provider.get_bool("transform.matching.allow_thin_coverage", True),
        )


@dataclass(frozen=True)
class RenderingConfig:
    model_id: str
    transformer_repo_id: str
    transformer_file: str
    base_model_id: str
    lora_path: Path
    device: str
    inference_steps: int
    true_cfg_scale: float
    seed: int
    render_pixels: int
    erase_dilation_ratio: float
    plate_feather_pixels: int

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "RenderingConfig":
        return cls(
                model_id=config_provider.get_str("transform.rendering.model_id", _RENDERING_MODEL),
                transformer_repo_id=config_provider.get_str(
                        "transform.rendering.transformer_repo_id", _QUANTIZED_TRANSFORMER_REPO),
                transformer_file=config_provider.get_str(
                        "transform.rendering.transformer_file", _QUANTIZED_TRANSFORMER_FILE),
                base_model_id=config_provider.get_str("transform.rendering.base_model_id", _BASE_MODEL),
                lora_path=config_provider.get_path("transform.rendering.lora_path", "data/loras/identity.safetensors"),
                device=config_provider.get_str("transform.rendering.device", "auto"),
                inference_steps=config_provider.get_int("transform.rendering.inference_steps", 40),
                true_cfg_scale=config_provider.get_float("transform.rendering.true_cfg_scale", 4.0),
                seed=config_provider.get_int("transform.rendering.seed", 42),
                render_pixels=config_provider.get_int("transform.rendering.render_pixels", 1024),
                erase_dilation_ratio=config_provider.get_float("transform.rendering.erase_dilation_ratio", 0.02),
                plate_feather_pixels=config_provider.get_int("transform.rendering.plate_feather_pixels", 25),
        )
