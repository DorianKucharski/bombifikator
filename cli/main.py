from __future__ import annotations

from pathlib import Path

import typer

from codex.character_store import CharacterStore
from codex.codex_config import CodexConfig, ExtractionConfig
from harvester.harvester_config import FrameExtractionConfig, HarvesterConfig
from reference_builder.reference_config import (ClusteringConfig, DetectionConfig, EmbeddingConfig, LabelingConfig,
                                                ReferencePaths, SelectionConfig)
from training.training_config import DatasetConfig, PreviewConfig
from transform.transform_config import (DescriptionConfig, MatchingConfig, RenderingConfig,
                                        SegmentationConfig, TransformPaths)
from sharedkernel.config_provider import ConfigProvider
from sharedkernel.env_file import load_env_file
from sharedkernel.logger import get_logger, set_log_level

LOGGER = get_logger("cli")

DEFAULT_CONFIG_PATH = Path("config/config.toml")
ENV_FILE_NAME = ".env"

app = typer.Typer(add_completion=False, help="Bombifikator: podmiana ludzi na postacie z uniwersum Kapitana Bomby")
codex_app = typer.Typer(add_completion=False, help="Slownik uniwersum")
app.add_typer(codex_app, name="codex")
harvester_app = typer.Typer(add_completion=False, help="Pozyskiwanie klatek z odcinkow")
app.add_typer(harvester_app, name="harvester")
references_app = typer.Typer(add_completion=False, help="Karty referencyjne postaci")
app.add_typer(references_app, name="references")
transform_app = typer.Typer(add_completion=False, help="Podmiana osob na postacie")
app.add_typer(transform_app, name="transform")
training_app = typer.Typer(add_completion=False, help="Trening LoRA")
app.add_typer(training_app, name="training")


def _config_provider(config_path: Path) -> ConfigProvider:
    load_env_file(config_path.parent / ENV_FILE_NAME)
    config_provider = ConfigProvider.from_file(config_path)
    set_log_level(config_provider.get_str("logging.level", "INFO"))
    return config_provider


@codex_app.command("scrape")
def codex_scrape(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
        refresh: bool = typer.Option(False, "--refresh", help="Pobierz ponownie strony juz obecne na dysku"),
) -> None:
    from codex.wiki_scraper import scrape_all_sites

    config = CodexConfig.from_config_provider(_config_provider(config_path))
    report = scrape_all_sites(config, refresh)
    LOGGER.info("scrape finished: discovered=%d downloaded=%d skipped_existing=%d skipped_too_short=%d failed=%d",
                report.discovered, report.downloaded, report.skipped_existing,
                report.skipped_too_short, len(report.failed))
    if report.failed:
        LOGGER.warning("failed pages: %s", ", ".join(report.failed))


@codex_app.command("extract")
def codex_extract(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
        refresh: bool = typer.Option(False, "--refresh", help="Przetworz ponownie strony obecne w rejestrze"),
) -> None:
    from codex.codex_builder import build_codex

    config_provider = _config_provider(config_path)
    report = build_codex(
            CodexConfig.from_config_provider(config_provider),
            ExtractionConfig.from_config_provider(config_provider),
            refresh,
    )
    LOGGER.info("extraction finished: pages=%d extracted=%d rejected=%d skipped_processed=%d failed=%d",
                report.total_pages, report.extracted, report.rejected,
                report.skipped_processed, len(report.failed))
    if report.failed:
        LOGGER.warning("failed pages: %s", ", ".join(report.failed))


@codex_app.command("list")
def codex_list(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
        prominence: str = typer.Option("", "--prominence", help="MAIN, RECURRING albo EPISODIC"),
) -> None:
    config = CodexConfig.from_config_provider(_config_provider(config_path))
    characters = [
        character for character in CharacterStore(config.characters_dir).load_all()
        if not prominence or character.prominence.value == prominence.upper()
    ]
    for character in sorted(characters, key=lambda item: item.slug):
        typer.echo(f"{character.slug:32} {character.prominence.value:10} {character.species:16} "
                   f"{', '.join(character.appearance.distinguishing[:4])}")
    typer.echo(f"\nrazem: {len(characters)}")


@harvester_app.command("download")
def harvester_download(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
) -> None:
    from harvester.episode_downloader import download_all_sources

    config = HarvesterConfig.from_config_provider(_config_provider(config_path))
    report = download_all_sources(config)
    LOGGER.info("download finished: discovered=%d downloaded=%d skipped_existing=%d failed=%d",
                report.discovered, report.downloaded, report.skipped_existing, len(report.failed))
    if report.failed:
        LOGGER.warning("failed episodes: %s", ", ".join(report.failed))


@harvester_app.command("frames")
def harvester_frames(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
        refresh: bool = typer.Option(False, "--refresh", help="Przetworz ponownie odcinki majace juz manifest"),
) -> None:
    from harvester.frame_harvester import harvest_frames

    config_provider = _config_provider(config_path)
    report = harvest_frames(
            HarvesterConfig.from_config_provider(config_provider),
            FrameExtractionConfig.from_config_provider(config_provider),
            refresh,
    )
    LOGGER.info("frames finished: episodes=%d skipped=%d missing_video=%d shots=%d saved=%d",
                report.episodes_processed, report.episodes_skipped, report.episodes_missing_video,
                report.shots_detected, report.frames_saved)
    for verdict, count in report.rejections:
        LOGGER.info("frames rejected: verdict=%s count=%d", verdict, count)


@references_app.command("detect")
def references_detect(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
        refresh: bool = typer.Option(False, "--refresh", help="Przetworz ponownie odcinki majace juz manifest kadrow"),
) -> None:
    from reference_builder.crop_harvester import harvest_crops

    config_provider = _config_provider(config_path)
    report = harvest_crops(
            ReferencePaths.from_config_provider(config_provider),
            DetectionConfig.from_config_provider(config_provider),
            refresh,
    )
    LOGGER.info("detect finished: episodes=%d skipped=%d frames=%d empty_frames=%d crops=%d",
                report.episodes_processed, report.episodes_skipped, report.frames_scanned,
                report.frames_without_detection, report.crops_saved)


@references_app.command("embed")
def references_embed(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
) -> None:
    from reference_builder.embedding_extractor import build_embeddings

    config_provider = _config_provider(config_path)
    report = build_embeddings(
            ReferencePaths.from_config_provider(config_provider),
            EmbeddingConfig.from_config_provider(config_provider),
    )
    LOGGER.info("embed finished: crops=%d unreadable=%d dimensions=%d",
                report.crops_embedded, report.crops_unreadable, report.dimensions)


@references_app.command("cluster")
def references_cluster(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
) -> None:
    from reference_builder.cluster_builder import build_clusters

    config_provider = _config_provider(config_path)
    report = build_clusters(
            ReferencePaths.from_config_provider(config_provider),
            ClusteringConfig.from_config_provider(config_provider),
    )
    LOGGER.info("cluster finished: crops=%d clusters=%d clustered_crops=%d sheets=%d",
                report.crops_clustered, report.clusters_found, report.crops_in_clusters, report.sheets_written)


@references_app.command("label")
def references_label(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
        refresh: bool = typer.Option(False, "--refresh", help="Zetykietuj ponownie klastry obecne w labels.toml"),
) -> None:
    from reference_builder.cluster_labeler import label_clusters

    config_provider = _config_provider(config_path)
    report = label_clusters(
            ReferencePaths.from_config_provider(config_provider),
            CodexConfig.from_config_provider(config_provider).characters_dir,
            LabelingConfig.from_config_provider(config_provider),
            refresh,
    )
    LOGGER.info("label finished: clusters=%d labeled=%d unknown=%d skipped=%d missing_sheet=%d failed=%d",
                report.clusters_total, report.clusters_labeled, report.clusters_unknown,
                report.clusters_skipped, report.clusters_missing_sheet, len(report.failed))


@references_app.command("build")
def references_build(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
) -> None:
    from reference_builder.reference_pipeline import build_references

    config_provider = _config_provider(config_path)
    report = build_references(
            ReferencePaths.from_config_provider(config_provider),
            CodexConfig.from_config_provider(config_provider).characters_dir,
            SelectionConfig.from_config_provider(config_provider),
    )
    LOGGER.info("build finished: characters=%d references=%d full=%d thin=%d below_threshold=%d ignored_labels=%d",
                report.characters_covered, report.references_written, report.full_coverage,
                report.thin_coverage, report.characters_below_threshold, report.labels_ignored)


@transform_app.command("photo")
def transform_photo_command(
        image_path: Path = typer.Argument(..., help="Zdjecie wejsciowe"),
        out: Path = typer.Option(None, "--out", help="Plik wyjsciowy, domyslnie data/output/<nazwa>"),
        seed: int = typer.Option(None, "--seed", help="Ziarno generatora, nadpisuje konfiguracje"),
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
) -> None:
    from dataclasses import replace

    from transform.transform_pipeline import transform_photo

    config_provider = _config_provider(config_path)
    paths = TransformPaths.from_config_provider(config_provider)
    rendering = RenderingConfig.from_config_provider(config_provider)
    report = transform_photo(
            image_path,
            out or paths.output_dir / image_path.name,
            paths,
            SegmentationConfig.from_config_provider(config_provider),
            DescriptionConfig.from_config_provider(config_provider),
            MatchingConfig.from_config_provider(config_provider),
            rendering if seed is None else replace(rendering, seed=seed),
    )
    LOGGER.info("transform finished: detected=%d described=%d replaced=%d output=%s",
                report.people_detected, report.people_described, report.people_replaced, report.output_path)
    for assignment in report.assignments:
        LOGGER.info("assignment: person=%d slug=%s", assignment.person_index, assignment.slug)


@training_app.command("dataset")
def training_dataset(
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
) -> None:
    from training.dataset_builder import build_dataset

    report = build_dataset(DatasetConfig.from_config_provider(_config_provider(config_path)))
    LOGGER.info("dataset finished: characters=%d images=%d skipped=%d",
                report.characters_written, report.images_written, len(report.characters_skipped))
    if report.characters_skipped:
        LOGGER.warning("characters without enough references: %s", ", ".join(report.characters_skipped))


@training_app.command("preview")
def training_preview(
        lora: Path = typer.Option(..., "--lora"),
        config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
) -> None:
    from training.preview_renderer import write_token_previews
    from transform.qwen_editor import QwenCharacterPainter

    config_provider = _config_provider(config_path)
    painter = QwenCharacterPainter(RenderingConfig.from_config_provider(config_provider), lora)
    written = write_token_previews(painter, PreviewConfig.from_config_provider(config_provider))
    LOGGER.info("previews finished: characters=%d", written)


if __name__ == "__main__":
    app()
