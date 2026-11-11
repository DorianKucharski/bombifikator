# Bombifikator

Pipeline w Pythonie, który podmienia ludzi na prawdziwych zdjęciach na postacie z serialu animowanego *Kapitan Bomba*, zachowując oryginalną kompozycję (tło, pozę, oświetlenie i liczbę osób).

To osobisty, eksperymentalny projekt do prywatnego użytku. Co obejmuje licencja, a czego nie, opisuje [LICENSE](LICENSE).

## Jak to działa

Pipeline jest podzielony na etapy, a każdy z nich to osobna podkomenda CLI:

1. **Codex** (`codex`) - pobiera wiki serialu (MediaWiki API) i za pomocą Claude buduje ustrukturyzowany słownik postaci (jeden plik TOML na postać).
2. **Harvester** (`harvester`) - ściąga odcinki przez `yt-dlp`, dzieli je na ujęcia i wybiera klatki przechodzące filtr jakości.
3. **Reference builder** (`references`) - wykrywa postacie na klatkach, usuwa tło, liczy embeddingi, klastruje wycinki, opisuje klastry przez Claude i wybiera karty referencyjne dla każdej postaci.
4. **Training** (`training`) - buduje zbiory treningowe LoRA (cała obsada albo pojedyncza postać) i renderuje podglądy tokenów. Sam trening idzie na wynajętym GPU w chmurze przez ai-toolkit, z konfiguracjami YAML z `training/configs/`.
5. **Transform** (`transform`) - wykrywa ludzi na zdjęciu wejściowym, opisuje każdą osobę i dobiera do niej postać przez Claude, odtwarza tło i domalowuje postać przez Qwen-Image-Edit (transformer Nunchaku FP4) z wytrenowaną LoRA.

## Technologie

- Python, Typer (CLI), `unittest`
- Anthropic API (structured outputs) do ekstrakcji, opisywania klastrów i opisu osób
- yt-dlp, PySceneDetect, OpenCV
- PyTorch, Transformers, timm, scikit-learn, Ultralytics, rembg / onnxruntime
- Diffusers (przypięte do 0.36.0) + Nunchaku do inferencji Qwen-Image-Edit

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sekrety są czytane z `config/.env` (w `.gitignore`) albo ze zmiennych środowiskowych. Skopiuj szablon i uzupełnij go:

```bash
cp config/.env.example config/.env
```

Potrzebne są `BOMBIFIKATOR_ANTHROPIC_API_KEY` i `BOMBIFIKATOR_HUGGINGFACE_TOKEN`. Pozostałe ustawienia są w `config/config.toml`; każdy klucz można nadpisać zmienną środowiskową `BOMBIFIKATOR_<KLUCZ_Z_KROPKAMI_ZAMIENIONYMI_NA_PODKREŚLNIKI>` (np. `codex.max_category_depth` -> `BOMBIFIKATOR_CODEX_MAX_CATEGORY_DEPTH`).

## Użycie

CLI uruchamia się z katalogu głównego repozytorium:

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
python -m cli.main training single-dataset --slug <slug-postaci> --out <katalog-zbioru>
python -m cli.main training preview --lora <ścieżka-do-lora.safetensors>

python -m cli.main transform photo <zdjęcie> [--out <plik>] [--seed <n>]
```

Wszystkie wygenerowane dane (strony wiki, klatki, karty referencyjne, zbiory treningowe, wagi LoRA, wyniki) trafiają do `data/`, które jest w `.gitignore`.

### Skrypty do treningu w chmurze

W `scripts/` są skrypty powłoki do zdalnego poda GPU dostępnego przez SSH (alias hosta ustawia zmienna `POD`, domyślnie `runpod`):

- `bootstrap_pod.sh` - przygotowanie środowiska na świeżym podzie
- `sync_pod.sh` - wysłanie kodu, sekretów, zdjęć wejściowych i kart referencyjnych
- `start_training.sh` - wysłanie konfiguracji treningu i start przebiegu (`RUN_NAME`, `CONFIG`)
- `pull_training.sh` - pobranie checkpointów i próbek (`RUN_NAME`)

## Testy

```bash
python -m unittest discover -s tests -t .
```

## Struktura projektu

```
cli/                punkt wejścia CLI (Typer)
codex/              pobieranie wiki i ekstrakcja postaci
harvester/          pobieranie odcinków i wybór klatek
reference_builder/  detekcja, klastrowanie, opisywanie i wybór referencji
training/           budowa zbiorów, podpisy, słownik tokenów, konfiguracje treningu
transform/          pipeline podmiany osób na pojedynczym zdjęciu
sharedkernel/       konfiguracja, ładowanie env, logowanie, klient Anthropic, narzędzia
config/             config.toml i .env.example
scripts/            skrypty do zdalnego poda GPU
tests/              testy jednostkowe odwzorowujące strukturę źródeł
```

## Licencja

Kod źródłowy jest na licencji MIT. Materiały z *Kapitana Bomby* i wszystko, co z nich powstało (pobrany tekst, klatki, obrazy referencyjne, wytrenowane wagi, wygenerowane obrazy), nie są dołączone do repozytorium i nie są objęte licencją - szczegóły w [LICENSE](LICENSE).
