# Bombifikator

A Python pipeline that replaces people in real photos with characters from the Polish animated series *Kapitan Bomba*, while keeping the original composition (background, pose, lighting and number of people).

This is a personal, experimental project for private use. See [LICENSE](LICENSE) for what is and is not covered.

## How it works

The pipeline is split into stages, each exposed as a CLI sub-command:

1. **Codex** (`codex`) - scrapes the series' wiki (MediaWiki API) and uses Claude to extract a structured character dictionary (one TOML file per character).
2. **Harvester** (`harvester`) - downloads episodes with `yt-dlp`, splits them into shots and samples quality-filtered frames.
3. **Reference builder** (`references`) - detects characters in frames, removes backgrounds, computes embeddings, clusters crops, labels clusters with Claude and selects reference cards per character.
4. **Training** (`training`) - builds LoRA training datasets (full cast or a single character) and renders token previews. Training itself runs on a rented cloud GPU with ai-toolkit, using the YAML configs in `training/configs/`.
5. **Transform** (`transform`) - detects people in an input photo, describes each person and picks a matching character with Claude, rebuilds the background and paints the character in with Qwen-Image-Edit (Nunchaku FP4 transformer) plus the trained LoRA.

## Tech stack

- Python, Typer (CLI), `unittest`
- Anthropic API (structured outputs) for extraction, labeling and person description
- yt-dlp, PySceneDetect, OpenCV
- PyTorch, Transformers, timm, scikit-learn, Ultralytics, rembg / onnxruntime
- Diffusers (pinned to 0.36.0) + Nunchaku for Qwen-Image-Edit inference

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Secrets are read from `config/.env` (gitignored) or from environment variables. Copy the template and fill it in:

```bash
cp config/.env.example config/.env
```

It expects `BOMBIFIKATOR_ANTHROPIC_API_KEY` and `BOMBIFIKATOR_HUGGINGFACE_TOKEN`. All other settings live in `config/config.toml`; any key can be overridden with a `BOMBIFIKATOR_<DOTTED_KEY_WITH_UNDERSCORES>` environment variable (e.g. `codex.max_category_depth` -> `BOMBIFIKATOR_CODEX_MAX_CATEGORY_DEPTH`).

## Usage

Run the CLI from the repository root:

```bash
python -m cli.main --help

python -m cli.main codex scrape
python -m cli.main codex extract
python -m cli.main codex list

python -m cli.main harvester download
python -m cli.main harvester frames

python -m cli.main references detect
python -m cli.main references embed
python -m cli.main references cluster
python -m cli.main references label
python -m cli.main references build

python -m cli.main training dataset
python -m cli.main training single-dataset --slug <character-slug> --out <dataset-dir>
python -m cli.main training preview --lora <path-to-lora.safetensors>

python -m cli.main transform photo <image> [--out <file>] [--seed <n>]
```

All generated data (wiki pages, frames, reference cards, datasets, LoRA weights, outputs) is written to `data/`, which is gitignored.

### Cloud training helpers

`scripts/` contains shell helpers for a remote GPU pod reachable over SSH (host alias configured via the `POD` variable, default `runpod`):

- `bootstrap_pod.sh` - set up the environment on a fresh pod
- `sync_pod.sh` - upload code, secrets, input photos and reference cards
- `start_training.sh` - upload a training config and start a run (`RUN_NAME`, `CONFIG`)
- `pull_training.sh` - download checkpoints and samples (`RUN_NAME`)

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Project structure

```
cli/                Typer CLI entry point
codex/              wiki scraping and character extraction
harvester/          episode download and frame sampling
reference_builder/  detection, clustering, labeling and reference selection
training/           dataset building, captions, token vocabulary, training configs
transform/          per-photo person replacement pipeline
sharedkernel/       config provider, env loading, logging, Anthropic client, utilities
config/             config.toml and .env.example
scripts/            remote GPU pod helpers
tests/              unit tests mirroring the source tree
```

## License

Source code is MIT-licensed. *Kapitan Bomba* material and anything derived from it (scraped text, frames, reference images, trained weights, generated images) is not included and not covered by the license - see [LICENSE](LICENSE).
