from __future__ import annotations

from pathlib import Path

import typer

from codex.character_store import CharacterStore
from codex.codex_builder import build_codex
from codex.codex_config import CodexConfig, ExtractionConfig
from codex.wiki_scraper import scrape_all_sites
from harvester.episode_downloader import download_all_sources
from harvester.frame_harvester import harvest_frames
from harvester.harvester_config import FrameExtractionConfig, HarvesterConfig
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


if __name__ == "__main__":
    app()
