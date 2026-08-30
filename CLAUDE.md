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

## Stan na 2026-08-30

Etapy 0, 1, 2 i 3 zrobione i uruchomione. Punkt decyzyjny etapu 3 przeszedł na 7 zdjęciach:
wyniki poniżej progu, więc etap 4 (LoRA) jest otwarty.

Etap 3, architektura po ośmiu wariantach. Obowiązuje wariant per-osoba:
1. YOLO11-seg wykrywa osoby.
2. Claude opisuje każdą osobę i od razu wybiera trzy pasujące postacie z katalogu w system promptcie.
3. Qwen robi płytę tła: to samo zdjęcie z usuniętymi ludźmi. Bierzemy z niej piksele tylko tam,
   gdzie stali ludzie, po dopasowaniu tonu do fotografii i z wtopioną krawędzią. Reszta kadru
   zostaje oryginalną fotografią co do piksela.
4. Qwen rysuje każdą postać osobno na białym tle, `rembg` ją wycina, zostaje największa spójna
   składowa alfa (odcina dorysowane rekwizyty), wklejamy w bounding box osoby od najdalszej.
Kompozycja jest z definicji zachowana, bo poza maskami osób zdjęcie nie jest dotykane.

Warianty odrzucone, nie wracaj do nich:
1. Edycja całego zdjęcia per osoba i wklejanie po masce YOLO. Duchy: Qwen resyntezuje kadr,
   więc postać ląduje gdzie indziej niż sylwetka z maski.
2. Sekwencyjne edycje całego zdjęcia, osoba po osobie. Dryf: z pięciu osób na grupowym zdjęciu
   został jeden stwór na korytarzu.
3. Jedno wywołanie na zdjęcie z wymienionymi wszystkimi osobami. Kompozycja trzyma, ale model
   restyluje ludzi zamiast ich podmienić, dorysowuje osoby i prostuje kadrowanie.
4. Wycinek osoby jako drugi obraz, jako referencja pozy. Model dorysował butelkę i walizkę,
   pozy nie skopiował.
5. Wypełniona czarna sylwetka jako drugi obraz. Poza lepsza, ale tożsamość znika,
   zostaje generyczny brodacz w kolorze postaci.
6. Sam kontur sylwetki. Ciemna bezkształtna plama bez głowy.
7. Dwie karty referencyjne tej samej postaci. Postać bez twarzy z dwiema puszkami piwa.
Wniosek: 2509 traktuje każdy kolejny obraz jako podmiot do zmieszania, nigdy jako strukturę.
Sterowanie pozą przez drugi obraz albo przez prompt nie działa.

Co zostaje nierozwiązane: poza postaci nie odpowiada pozie człowieka, bo opis tekstowy to za słaby
sygnał, a każda próba podania struktury drugim obrazem psuje tożsamość. To wymaga postaci w wagach,
czyli LoRA tożsamości plus ControlNet na pozę.

Czas: ok. 2,5 minuty na zdjęcie z jedną osobą, minuta na każdą kolejną.

Etap 3, stan:
- `transform/` - `transform_config`, `person_models`, `person_detector` (YOLO11-seg), `description_schema`,
  `person_describer`, `matching_encoder`, `person_locator`, `character_prompt`, `character_renderer`,
  `render_size`, `person_eraser`, `person_compositor`, `reference_picker`, `transform_pipeline`.
  Matcher siedzi w `codex/character_matcher.py`.
- `cli/main.py` - doszło `transform photo IMG --out OUT --seed N`.
- Inferencja: Qwen-Image-Edit-2509 w Nunchaku FP4 (`svdq-fp4_r128`), 40 kroków na RTX PRO 6000.
- Testy: 120, wszystkie przechodzą.

Etap 2, wyniki przebiegu (RunPod, RTX PRO 6000 Blackwell 96 GB):
- `data/references/crops/` - 28673 kadry z 16071 klatek, 1026 klatek bez żadnej detekcji, 4,4 GB.
  Detekcja 0,081 s na klatkę, cały przebieg 25 minut.
- `data/references/embeddings.npz` - 28673 x 768, DINOv2-base, 82 MB.
- 511 klastrów HDBSCAN, 13234 kadry w klastrach, 15439 jako szum.
- `data/references/labels.toml` - 293 klastry rozpoznane, 218 `UNKNOWN`, zero błędów API.
- `data/references/cards/` - 33 postacie, 525 referencji z wyciętym tłem, 32 `FULL` i jedna `THIN`
  (`chujew`, 13 kadrów). 289 etykiet odrzuconych jako `UNKNOWN` albo poniżej progu pewności 0,6.
- `reference_builder/` - `reference_config`, `reference_models`, `crop_geometry`, `crop_store`,
  `character_detector`, `crop_harvester`, `torch_device`, `embedding_extractor`, `embedding_store`,
  `character_clusterer`, `cluster_store`, `cluster_builder`, `contact_sheet`, `cluster_labeler`,
  `labeling_schema`, `label_store`, `reference_selector`, `background_remover`, `reference_store`,
  `coverage_marker`, `reference_pipeline`.
- `cli/main.py` - doszły `references detect|embed|cluster|label|build`.
- Testy: 83, wszystkie przechodzą.

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
- GroundingDINO zwraca tę samą postać po kilka razy pod różnymi frazami promptu, dlatego
  `crop_geometry.suppress_overlapping` robi class-agnostic NMS po IoU. Bez tego jeden bohater
  dawał trzy niemal identyczne kadry i zaburzał klastry.
- Praca na RunPodzie idzie na dysku kontenera (`/root/bombifikator`), nie na `/workspace`.
  Wolumen sieciowy ma limit ~50 GB i instalacja torcha wywalała się na `Disk quota exceeded`.
- Etykietowanie klastrów odpalamy lokalnie, nie na podzie, żeby nie wysyłać `config/.env` na maszynę.
  Na poda idą tylko `labels.toml`, na dół wracają `cards/` i zaktualizowane karty postaci.
- Katalog postaci siedzi w system promptcie etykietera, więc prompt cache łapie go raz na cały przebieg.
- `scripts/bootstrap_pod.sh` odtwarza całe środowisko na świeżym podzie, `scripts/sync_pod.sh` wysyła kod,
  sekrety, zdjęcia i karty. Pod znika razem z danymi, więc wagi ściągamy przy każdym nowym podzie.
- Matcher na `sentence-transformers` był zepsuty: cosine wszystkich opisów siedział w paśmie 0,44-0,63,
  więc ranking był płaski i `dominik-jachas` trafiał na cztery zdjęcia z siedmiu. Wybór postaci robi teraz
  Claude w tym samym wywołaniu, w którym opisuje osobę, i zwraca ranking trzech. Zależność wyleciała.
- Structured outputs nie przyjmują `minItems` innego niż 0 albo 1. Liczbę elementów tablicy opisuje się
  słownie w `description`.
- Nie zamalowuj dwa razy. Płyta tła wypełnia całą maskę osoby, więc dokładanie `cv2.inpaint` na resztki
  rozmazywało to, co model poprawnie odtworzył, i zostawiało szarą aureolę wokół postaci.
- Płytę trzeba wtopić i stonować. Twarda krawędź maski zostawiała widoczny obrys człowieka nawet przy
  dobrze odtworzonym tle.
- Pełny Qwen-Image-Edit-2509 w bf16 to 57,7 GB, czyli więcej niż dysk kontenera (30 GB) i niż wolumen
  sieciowy RunPoda. Dlatego transformer idzie z Nunchaku FP4 (~12 GB), a z repo bazowego bierzemy tylko
  text encoder, VAE i scheduler. FP4, nie INT4, bo RTX PRO 6000 to Blackwell sm_120.
- `diffusers` przypięty na 0.36.0. Nunchaku 1.2.1 woła `QwenEmbedRope.forward(video_fhw, txt_seq_lens, device)`,
  a 0.40 zmienił tę sygnaturę i pipeline wywala się na `got multiple values for argument 'device'`.
- `BoundingBox` mieszka w `sharedkernel/utils/geometry.py`, bo używają go i `reference_builder`, i `transform`.

## Kolejne kroki, w tej kolejności

1. Etap 3 `transform/`: YOLO11-seg → opis osób Claude'em → matcher → inpainting Qwen-Image-Edit per osoba.
   Punkt decyzyjny: 30 zdjęć testowych, ocena 1-5 na trzech osiach. Poniżej 3.5 otwiera etap 4.
2. Etap 4 `training/`: LoRA stylu, potem LoRA tożsamości postaci, `ai-toolkit` na RunPodzie. Tylko jeśli etap 3 nie dowozi.

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
