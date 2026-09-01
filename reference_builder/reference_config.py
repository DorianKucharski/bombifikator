from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sharedkernel.config_provider import ConfigProvider

_DETECTION_TEXT_PROMPT = "a cartoon character. a person. a monster."
_DETECTION_MODEL = "IDEA-Research/grounding-dino-base"
_EMBEDDING_MODEL = "facebook/dinov2-base"


@dataclass(frozen=True)
class ReferencePaths:
    frames_dir: Path
    crops_dir: Path
    sheets_dir: Path
    references_dir: Path
    embeddings_path: Path
    clusters_path: Path
    labels_path: Path

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "ReferencePaths":
        return cls(
                frames_dir=config_provider.get_path("harvester.frames_dir", "data/frames"),
                crops_dir=config_provider.get_path("reference_builder.crops_dir", "data/references/crops"),
                sheets_dir=config_provider.get_path("reference_builder.sheets_dir", "data/references/sheets"),
                references_dir=config_provider.get_path("reference_builder.references_dir", "data/references/cards"),
                embeddings_path=config_provider.get_path(
                        "reference_builder.embeddings_path", "data/references/embeddings.npz"),
                clusters_path=config_provider.get_path(
                        "reference_builder.clusters_path", "data/references/clusters.toml"),
                labels_path=config_provider.get_path("reference_builder.labels_path", "data/references/labels.toml"),
        )


@dataclass(frozen=True)
class DetectionConfig:
    model_id: str
    text_prompt: str
    box_threshold: float
    text_threshold: float
    batch_size: int
    device: str
    crop_padding_ratio: float
    min_crop_pixels: int
    max_crops_per_frame: int
    max_frame_coverage_ratio: float
    duplicate_iou_threshold: float

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "DetectionConfig":
        return cls(
                model_id=config_provider.get_str("reference_builder.detection.model_id", _DETECTION_MODEL),
                text_prompt=config_provider.get_str("reference_builder.detection.text_prompt", _DETECTION_TEXT_PROMPT),
                box_threshold=config_provider.get_float("reference_builder.detection.box_threshold", 0.3),
                text_threshold=config_provider.get_float("reference_builder.detection.text_threshold", 0.25),
                batch_size=config_provider.get_int("reference_builder.detection.batch_size", 8),
                device=config_provider.get_str("reference_builder.detection.device", "auto"),
                crop_padding_ratio=config_provider.get_float("reference_builder.detection.crop_padding_ratio", 0.08),
                min_crop_pixels=config_provider.get_int("reference_builder.detection.min_crop_pixels", 96),
                max_crops_per_frame=config_provider.get_int("reference_builder.detection.max_crops_per_frame", 6),
                max_frame_coverage_ratio=config_provider.get_float(
                        "reference_builder.detection.max_frame_coverage_ratio", 0.85),
                duplicate_iou_threshold=config_provider.get_float(
                        "reference_builder.detection.duplicate_iou_threshold", 0.55),
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str
    batch_size: int
    device: str

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "EmbeddingConfig":
        return cls(
                model_id=config_provider.get_str("reference_builder.embedding.model_id", _EMBEDDING_MODEL),
                batch_size=config_provider.get_int("reference_builder.embedding.batch_size", 64),
                device=config_provider.get_str("reference_builder.embedding.device", "auto"),
        )


@dataclass(frozen=True)
class ClusteringConfig:
    min_cluster_size: int
    min_samples: int
    pca_components: int
    sheet_grid_size: int
    sheet_tile_pixels: int

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "ClusteringConfig":
        return cls(
                min_cluster_size=config_provider.get_int("reference_builder.clustering.min_cluster_size", 12),
                min_samples=config_provider.get_int("reference_builder.clustering.min_samples", 4),
                pca_components=config_provider.get_int("reference_builder.clustering.pca_components", 64),
                sheet_grid_size=config_provider.get_int("reference_builder.clustering.sheet_grid_size", 3),
                sheet_tile_pixels=config_provider.get_int("reference_builder.clustering.sheet_tile_pixels", 256),
        )


@dataclass(frozen=True)
class LabelingConfig:
    model: str
    max_tokens: int
    effort: str
    concurrency: int
    prominences: tuple[str, ...]
    api_key: str

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "LabelingConfig":
        return cls(
                model=config_provider.get_str("reference_builder.labeling.model", "claude-opus-5"),
                max_tokens=config_provider.get_int("reference_builder.labeling.max_tokens", 2000),
                effort=config_provider.get_str("reference_builder.labeling.effort", "medium"),
                concurrency=config_provider.get_int("reference_builder.labeling.concurrency", 6),
                prominences=tuple(config_provider.get_str_list(
                        "reference_builder.labeling.prominences", ("MAIN", "RECURRING"))),
                api_key=config_provider.get_secret("anthropic_api_key"),
        )


@dataclass(frozen=True)
class SelectionConfig:
    target_references: int
    full_reference_threshold: int
    thin_reference_threshold: int
    min_label_confidence: float
    remove_background: bool
    identity_deviation_tolerance: float
    max_opaque_ratio: float

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "SelectionConfig":
        return cls(
                target_references=config_provider.get_int("reference_builder.selection.target_references", 16),
                full_reference_threshold=config_provider.get_int(
                        "reference_builder.selection.full_reference_threshold", 16),
                thin_reference_threshold=config_provider.get_int(
                        "reference_builder.selection.thin_reference_threshold", 10),
                min_label_confidence=config_provider.get_float(
                        "reference_builder.selection.min_label_confidence", 0.6),
                remove_background=config_provider.get_bool("reference_builder.selection.remove_background", True),
                identity_deviation_tolerance=config_provider.get_float(
                        "reference_builder.selection.identity_deviation_tolerance", 2.0),
                max_opaque_ratio=config_provider.get_float("reference_builder.selection.max_opaque_ratio", 0.9),
        )
