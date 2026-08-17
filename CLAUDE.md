# Bombifikator

Pipeline zamieniający ludzi na zdjęciach na postacie z uniwersum Kapitana Bomby.
Wejście: prawdziwe zdjęcie. Wyjście: to samo zdjęcie z każdą osobą podmienioną na postać z serialu,
przy zachowaniu kompozycji (tło, poza, oświetlenie, liczba osób).

Plan całości: `/home/dorian/.claude/plans/mutable-strolling-otter.md` (zatwierdzony 2026-08-16).

## Ustalenia (nie pytaj o nie ponownie)

- GPU lokalnie 8-16 GB. Trening LoRA idzie do chmury (RunPod A100 80GB, ~$1.5-2/h), inferencja lokalnie na 4-bit Nunchaku.
- Dobór postaci do osoby jest automatyczny: VLM opisuje osobę, matcher wybiera postać ze słownika.
- Priorytet: zachowanie kompozycji oryginału, nie globalny restyling na kreskę Walaszka.
- Dostępne klucze: Anthropic API, Hugging Face. Klucz Anthropic NIE jest jeszcze ustawiony w środowisku.
- Czas realizacji nie jest ograniczeniem, liczy się jakość.
- Materiału źródłowego nie ma lokalnie, wszystko trzeba pozyskać z sieci.
- Kwestia prawna spisana w `LICENSE`: kod MIT, dane i wagi nie są dystrybuowane, użytek prywatny.

## Stan na 2026-08-17

Etap 0 (słownik) i etap 1 (harvester) zrobione i uruchomione. Następny w kolejce: etap 2 `reference_builder/`.

Etap 1, wyniki przebiegu:
- `data/episodes/` - 137 ze 142 pozycji, 4.0 GB. Brakujące 5 ma blokadę wiekową na YouTube
  i wymaga ciasteczek zalogowanego konta. Nie ma jeszcze opcji `cookies_from_browser` w konfiguracji.
- `data/frames/` - 16071 klatek z 10129 ujęć, 7.4 GB, manifest TOML w każdym katalogu odcinka.
  Odrzucone: 1300 `TOO_DARK`, 150 `TOO_FLAT`, 115 `TOO_BLURRY`.
- Rozdzielczości mieszane: 640x360, 1280x720 i 1920x1080. Dwa filmy pełnometrażowe w 1080p
  dają razem 2535 klatek, to najlepszy materiał referencyjny.
- `harvester/` - `harvester_config`, `episode_models`, `episode_catalogue`, `episode_downloader`,
  `scene_splitter`, `letterbox_cropper`, `frame_filter`, `frame_manifest`, `frame_harvester`.
- `cli/main.py` - doszły `harvester download` i `harvester frames`.
- Testy: 35, wszystkie przechodzą.

Etap 0, wyniki przebiegu: 499 stron wiki, 175 postaci w `data/codex/characters/` (8 MAIN, 32 RECURRING,
135 EPISODIC), 324 strony odrzucone jako nie-postacie. Ręczny przegląd wciąż nierobiony. Znane wady:
model miesza polskie znaki z ich brakiem, trafiają się literówki, `prominence` jest niestabilne
między przebiegami.

Zrobione wcześniej:
- `sharedkernel/` - `config_provider` (env → TOML → default, `ValueError` z nazwą zmiennej i klucza), `logger`,
  `clients/anthropic_client.py` (structured outputs przez `output_config.format`), `utils/paths.py` (slugify z polskimi znakami).
- `codex/` - `wiki_client` (MediaWiki API, rate limit 1 req/s), `wiki_scraper`, `raw_page_store`,
  `character_models` (frozen dataclasses + enumy), `extraction_schema`, `character_extractor`,
  `character_store` (TOML per postać), `processed_ledger` (wznawianie), `codex_builder` (ThreadPoolExecutor).
- `cli/main.py` - `codex scrape`, `codex extract`, `codex list`.
- Sekrety: `config/.env`, ładowany przez `sharedkernel/env_file.py` (nie nadpisuje zmiennych z shella).
  W ENV wyłącznie sekrety, cała reszta konfiguracji w `config/config.toml`.

Testy: `.venv/bin/python -m unittest discover -s tests -t .`

Git: repozytorium zainicjowane, ZERO commitów. Nic nie commitowałem, bo użytkownik nie prosił.

## Decyzje techniczne i dlaczego

- Discovery przez `list=allpages`, nie przez chodzenie po kategoriach. Wiki jest niekonsekwentna:
  strona `Kurvinox` nie ma żadnej sensownej kategorii, więc crawl kategorii gubił kluczowe strony.
  Kategorie pobierane są mimo to razem z wikitekstem, jako sygnał pomocniczy przy ekstrakcji.
  `codex/category_crawler.py` istniał i został usunięty, nie przywracaj go.
- `species` to zwykły string, nie enum. Zbiór ras w uniwersum jest otwarty.
- Model ekstrakcji: `claude-opus-5`, effort `low`. Zadanie jest mechaniczne, `low` wystarcza.
- Wybrany model edycji obrazu na etap 3: Qwen-Image-Edit 2509 (natywne wejście wieloobrazowe
  osoba+osoba i wbudowany ControlNet). FLUX.1 Kontext odrzucony, bo lepiej trzyma tożsamość twarzy,
  a my tę tożsamość chcemy zniszczyć.
- Prompt cache na system promptcie ekstraktora jest już ustawiony (`cache_control` na bloku system).
- Slug postaci bierze się z tytułu strony wiki, nie z `canonical_name`. Trzy pary różnych postaci
  dzielą imię (Bogdan, Domino, Mateo) i nadpisywały się nawzajem.
- `harvester.player_clients` to łańcuch fallbacku dla yt-dlp. Żaden pojedynczy klient nie działa
  na wszystkim: domyślny gubi część odcinków na 403, `android` ogranicza filmy do 360p.
- `format_selector` preferuje `avc1`. OpenCV nie dekoduje AV1, a YouTube podaje go jako domyślny
  dla materiałów HD - oba filmy dały zero klatek, zanim to wymusiłem.
- `ffmpeg` nie jest potrzebny. Bierzemy pojedynczy strumień wideo bez audio, nic się nie muksuje.
- Klatki próbkowane co ~3 s wewnątrz ujęcia, nie jedna na ujęcie. Jedna na ujęcie dawała ~1400 klatek
  na całą serię, za mało na klastrowanie 175 postaci.
- `letterbox_cropper` obcina czarne pasy przed oceną jakości. Bez tego dobre klatki wypadały jako `TOO_DARK`.

## Kolejne kroki, w tej kolejności

1. Przejrzeć ręcznie `data/codex/characters/*.toml`, zacząć od 40 postaci MAIN i RECURRING,
   dopisać brakujące. Reszta i tak nie trafi do referencji.
2. Etap 2 `reference_builder/`: GroundingDINO + DINOv2 + HDBSCAN, etykietowanie klastrów Claude'em,
   wybór 10-20 referencji na postać, `rembg` na tło. Postacie bez 10 dobrych klatek dostają `reference_coverage: THIN`.
3. Etap 3 `transform/`: YOLO11-seg → opis osób Claude'em → matcher → inpainting Qwen-Image-Edit per osoba.
   Punkt decyzyjny: 30 zdjęć testowych, ocena 1-5 na trzech osiach. Poniżej 3.5 otwiera etap 4.
4. Etap 4 `training/`: LoRA stylu, potem LoRA tożsamości postaci, `ai-toolkit` na RunPodzie. Tylko jeśli etap 3 nie dowozi.

## Konwencje w tym repo

Obowiązuje `~/.claude/CLAUDE.md` w całości. Dodatkowo:

- Zero komentarzy w kodzie. Nazwa metody albo wyekstrahowana zmienna zamiast komentarza.
- `from __future__ import annotations` na górze każdego modułu, pełne type hinty łącznie z `-> None`.
- Frozen dataclasses na wszystko, co nie musi mutować. Bez pydantic.
- `unittest`, nie pytest. `tests/` odzwierciedla drzewo źródeł. Nazwy metod czytają się jak zdania.
- Konfiguracja wyłącznie przez `from_config_provider`. Brak wymaganej wartości podnosi `ValueError` od razu.
- Logowanie leniwe, `%s`, nigdy f-stringi w wywołaniu loggera.
- `data/` jest w `.gitignore`. Nic z niego nie trafia do repo.
- Wirtualne środowisko: `.venv/`, zależności w `requirements.txt`.
