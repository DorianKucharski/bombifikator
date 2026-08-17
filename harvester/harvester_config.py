from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sharedkernel.config_provider import ConfigProvider

_DECODABLE_FORMAT_SELECTOR = "bv*[vcodec^=avc1][ext=mp4]/bv*[ext=mp4]/bv*/b"
_PLAYER_CLIENT_FALLBACK_CHAIN = ("default", "android", "tv_simply")


@dataclass(frozen=True)
class VideoSource:
    name: str
    url: str


@dataclass(frozen=True)
class FrameExtractionConfig:
    content_threshold: float
    min_shot_seconds: float
    sample_stride_seconds: float
    min_mean_luminance: float
    min_laplacian_variance: float
    min_pixel_standard_deviation: float

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "FrameExtractionConfig":
        return cls(
                content_threshold=config_provider.get_float("harvester.frames.content_threshold", 27.0),
                min_shot_seconds=config_provider.get_float("harvester.frames.min_shot_seconds", 0.4),
                sample_stride_seconds=config_provider.get_float("harvester.frames.sample_stride_seconds", 3.0),
                min_mean_luminance=config_provider.get_float("harvester.frames.min_mean_luminance", 12.0),
                min_laplacian_variance=config_provider.get_float("harvester.frames.min_laplacian_variance", 60.0),
                min_pixel_standard_deviation=config_provider.get_float(
                        "harvester.frames.min_pixel_standard_deviation", 18.0),
        )


def _to_video_source(raw_source: dict[str, Any]) -> VideoSource:
    return VideoSource(name=raw_source["name"], url=raw_source["url"])


@dataclass(frozen=True)
class HarvesterConfig:
    episodes_dir: Path
    frames_dir: Path
    format_selector: str
    player_clients: tuple[str, ...]
    sources: tuple[VideoSource, ...]

    @classmethod
    def from_config_provider(cls, config_provider: ConfigProvider) -> "HarvesterConfig":
        raw_sources = config_provider.get("harvester.sources", ())
        if not raw_sources:
            raise ValueError(
                    "missing required configuration 'harvester.sources': "
                    "define at least one [[harvester.sources]] table")
        return cls(
                episodes_dir=config_provider.get_path("harvester.episodes_dir", "data/episodes"),
                frames_dir=config_provider.get_path("harvester.frames_dir", "data/frames"),
                format_selector=config_provider.get_str("harvester.format_selector", _DECODABLE_FORMAT_SELECTOR),
                player_clients=tuple(config_provider.get_str_list(
                        "harvester.player_clients", _PLAYER_CLIENT_FALLBACK_CHAIN)),
                sources=tuple(_to_video_source(raw_source) for raw_source in raw_sources),
        )
