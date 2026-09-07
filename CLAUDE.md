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

## Stan na 2026-09-07, tu wracamy

Tożsamość wchodzi w wagi. Rozstrzygnęła to LoRA jednej postaci: Kapitan Bomba, token `klombu`,
dwadzieścia ręcznie wybranych kadrów, rank 32, 2000 kroków, zapis co 200.
Najlepszy jest **krok 1400**, `data/loras/klombu_v1/bombifikator_klombu_lora_v1_000001400.safetensors`.
Trzyma srebrny hełm, brązową brodę, biały mundur z czerwoną odznaką i czarne buty na wszystkich
czterech promptach naraz, także na pustyni i na czerwonej kanapie. Od 1600 w górę idzie
przetrenowanie: broda znika, twarz robi się jednolicie pomarańczowa, hełm zielenieje.
Próbki ze wszystkich jedenastu zapisów leżą w `data/training/samples_klombu_v1`.

**Co to znaczy.** Wąskim gardłem jest jakość kart referencyjnych, nie tokenizacja, nie format podpisu
i nie liczba kroków. Dwadzieścia dobranych ręcznie kadrów dało to, czego 1414 zebranych automatycznie
nie dało przez cztery przebiegi. Ręczny przegląd wszystkich 60 kart `tytus-bomba` zostawił 20 sztuk.
Odpadło 40: nieusunięte tło (kanapa, pustynia, ciemne i niebieskie sceny, wyblakłe kadry z napisami),
blondyn bez hełmu jako osobny wygląd tej samej postaci oraz obca postać w czarnym hełmie z napisem PKS.
Zapis, które kadry zostały, jest w `data/training/single_klombu_clean/kept_from_cards.txt`.

**Trening jednej postaci.** `training single-dataset --slug <slug> --out <katalog>` buduje zbiór
dla jednej postaci, tokeny liczy po pełnej obsadzie, więc slug dostaje ten sam token co zawsze,
a mapa tokenów ląduje w katalogu zbioru, żeby nie nadpisać wspólnego `data/training/tokens.toml`.
Konfiguracja przebiegu to `training/configs/single_character_lora.yaml`.
`scripts/start_training.sh` bierze teraz nazwę pliku konfiguracji z `CONFIG`, więc
`RUN_NAME=... CONFIG=training/configs/single_character_lora.yaml scripts/start_training.sh`.

**Dlaczego v5 nie wystarczył.** Gołe podpisy `{token}, {styl}` niczego nie naprawiły. Cztery prompty
próbek na wspólnym ziarnie dają na kroku 6000 jeden archetyp, a rozjazd pikseli spada z 58 na kroku 0
do 14 na 6000, przy minimum 3,9 między dwoma tokenami. Trening zbliżał tokeny do siebie zamiast je
rozdzielać. Checkpointy leżą w `data/loras/v5`, próbki w `data/training/samples_v5`.

**Kolejne kroki.** Przenieść kurację na resztę obsady: `background_remover` przepuszcza kadr,
w którym `rembg` wyciął postać razem z tłem, a klastry mieszają różne wyglądy tej samej postaci
pod jednym slugiem. Dopiero czyste karty mają sens jako wejście do LoRA na całą obsadę.

## Stan na 2026-09-06

Trwa trening v5, pierwszy z gołymi podpisami, puszczony od nowa. Pierwsze podejście z 5 września
przepadło razem z podem: po restarcie `/root` przyszedł pusty, bez ai-toolkit, bez cache modeli,
bez zbioru i bez ani jednego checkpointu. Ściągaj checkpointy w trakcie przebiegu, nie na końcu.

Pod: `69.8.146.165`, port `11188`, alias `runpod` w `~/.ssh/config` (klucz `~/.ssh/dorian`,
nie `id_ed25519`). Karta to RTX PRO 6000 Blackwell w plastrze MIG 48 GB, dysk kontenera 130 GB.
Host i port zmieniają się po każdym restarcie poda.

**Co robić po powrocie.** Sprawdź, na którym kroku jest trening:
`ssh runpod 'tail -c 250 /root/logs/bombifikator_identity_lora_v5.log | tr "\r" "\n" | tail -1; df -h /root | tail -1'`.
Tempo to ok. 3,3 s na krok, czyli 6000 kroków plus dwanaście serii próbek wychodzi na jakieś sześć godzin.
Przebieg wystartował 6 września ok. 12:20.
Checkpointy leżą w `/root/training/output/bombifikator_identity_lora_v5/`, dwanaście zapisów co 500 kroków,
i trzeba je ściągnąć razem z `samples/` (`RUN_NAME=bombifikator_identity_lora_v5 scripts/pull_training.sh`),
zanim pod zniknie.

**Dlaczego v4 nie wystarczył.** Rozdzielne tokeny naprawiły tokenizację, ale nie tożsamość.
Ocena kroku 4500 na wszystkich 32 tokenach jest w `data/training/previews_v4_4500/`, a rozstrzygająca
kontrola w `data/training/control_v4_4500/`. Każdy plik kontrolny to trzy rzędy: podpis pełny,
sam opis wyglądu i sam token. Między postaciami różnica pikseli wynosi 30-43 na 255 dla podpisu pełnego,
32-43 dla samego opisu i tylko 6-8 dla samego tokenu. Czyli całą tożsamość niósł tekstowy opis wyglądu
doklejony do podpisu, a token dalej nie znaczył nic. To był dokładnie ten opis, który v4 dodał,
żeby pomóc.

**Co zmienia v5.** Podpis to `{token}, {styl}` i nic więcej, `build_caption` nie przyjmuje już postaci.
Skoro opis wyglądu odbierał tokenowi pracę, token dostaje ją z powrotem, bo nie ma alternatywy.
Zbiór przebudowany, `data/training/identity` to te same 1414 obrazów z nowymi podpisami,
poprzednie leżą w `data/training/identity_v4_old`. `walk_seed` zbite na `false`, żeby cztery prompty
próbek dzieliły ziarno i dało się je porównywać między tokenami, bo chodzące ziarno myliło mnie
przy v3 i v3b.

**Stawianie poda od zera, kolejność, która działa.** `scripts/bootstrap_pod.sh` stawia środowisko
do inferencji, nie do treningu, i nie tknie ai-toolkit. Trening wymaga ręcznie:
`apt-get install -y rsync` (obrazy RunPoda go nie mają, a wszystkie skrypty na nim stoją),
`git clone --depth 1 https://github.com/ostris/ai-toolkit.git /root/ai-toolkit`,
własny venv z `torch==2.8.0 torchvision torchaudio==2.8.0` z indeksu `cu128` (bez przypiętego
torchaudio dociąga się 2.11 i nie ładuje rozszerzenia), potem `pip install -r requirements.txt`.
Python na obrazie to 3.11, więc wheel Nunchaku cp312 tu nie wejdzie, ale do treningu nie jest potrzebny.
Zbiór idzie na górę przez `tar czf - -C data/training identity | ssh runpod 'tar xzf - -C /root/training'`,
zanim rsync w ogóle istnieje na podzie, albo rsyncem po jego instalacji, ale wtedy katalogi docelowe
trzeba założyć wcześniej (`mkdir -p /root/training/identity /root/training/configs /root/logs`),
bo rsync nie tworzy brakujących rodziców. Qwen-Image ai-toolkit ściąga sam przy pierwszym
uruchomieniu, ok. 46 GB i kilkanaście minut. `HF_HOME=/root/hf` ze `scripts/start_training.sh`
jest martwe, wagi lądują w `/root/ai-toolkit`. Ten sam dysk, więc nie boli.

**Dlaczego v2, v3 i v3b były do wyrzucenia.** Nie z powodu liczby kroków, tylko dlatego, że tożsamość
nigdy się nie nauczyła. Ocena v3b krok 4500 na wszystkich 32 tokenach (`training preview --lora`)
pokazała, że prawie wszystkie tokeny zwijają się do trzech archetypów, a `bmb31` przy 60 referencjach
daje na czterech ziarnach cztery różne postacie. Kontrola rozstrzygnęła sprawę: `bmb31` i `bmb08`
w formacie podpisu treningowego, przy tych samych ziarnach, dają praktycznie ten sam obraz
(średnia różnica pikseli 9 na 255). Token nie wpływał na wynik w ogóle, decydowało ziarno.
LoRA nauczyła się stylu serialu i uśrednienia zbioru.

Winne były dwie rzeczy naraz. Tokeny `bmb01` do `bmb32` dzieliły prefiks i różniły się dwiema cyframi,
więc po tokenizacji text encoder widział dla wszystkich 32 postaci niemal identyczne warunkowanie.
Podpisy poza tokenem były dosłownie takie same (`{token}, a {species}, {styl}`), a kodeks miał
opisy wyglądu, których nikt nie używał.

**Co zmienia v4.** Tokeny to rozdzielne pseudosłowa z `training/configs/identity_tokens.toml`
(`klombu` to Kapitan Bomba, `tuvvel` to Torpeda), przypisywane po kolejności slugów.
`training/token_vocabulary.py` odrzuca duplikaty i tokeny dzielące trzyznakowy prefiks, więc błąd v3
nie wróci po cichu. Podpisy niosły karnację, strój i trzy pierwsze cechy szczególne z kodeksu,
składane przez `codex/character_appearance.py`. To właśnie ten opis okazał się szkodliwy i v5 go
z podpisu wyrzucił, ale `character_appearance` dalej obsługuje podgląd (`training/token_preview.py`)
i inferencję (`transform/character_prompt.py`).
Zbiór to 1414 obrazów, 32 postacie, `max_step_saves_to_keep: 12` zamiast 3, bo dysk wreszcie pozwala
i wybór checkpointu przestaje być loterią.

**Cztery pułapki przy uruchamianiu, wszystkie już trafione.**
Pierwsza: `nohup python ... &` wewnątrz `ssh POD 'bash -c ...'` nie przeżywa zamknięcia sesji,
bo backgroundowany jest cały łańcuch, nie sam python. Dlatego `setsid nohup ... < /dev/null &`.
Druga: dysk kontenera. Jeden checkpoint to 2,36 GB, `optimizer.pt` 2,39 GB, dwanaście zapisów
to ok. 31 GB. Obecny pod ma 130 GB, poprzedni miał 30 GB i dwa przebiegi na tym padły.
Trzecia: wolumen sieciowy nie jest gwarantowany. Po jednym restarcie `/workspace` przyszedł jako pusty
dysk 20 GB, bez ai-toolkit, bez cache modeli i bez zbioru. Dlatego wszystko idzie teraz na `/root`,
a `scripts/start_training.sh` używa `/root/training/configs`, `/root/logs` i `HF_HOME=/root/hf`.
Czwarta: obraz poda potrafi mieć torcha, który nie ma kerneli na Blackwella. Ten przyszedł
z 2.4.1+cu124, `torch.cuda.is_available()` zwracało `True`, a każdy `matmul` na GPU się wywalał.
Trzeba postawić własny venv z torchem 2.8.0+cu128, do tego `torchaudio==2.8.0` przypięty z tego samego
indeksu, bo bez wersji dociąga się 2.11 i nie ładuje rozszerzenia. `rsync` na tym obrazie też nie ma.

**Mapa tokenów.** Obowiązuje `data/training/tokens.toml` z przebiegu, który faktycznie trenował.
Przy v4 generuje ją `training dataset` razem ze zbiorem, więc jest spójna z podpisami.

**Ocena LoRA wymaga poda z dyskiem kontenera co najmniej 120 GB**, bo transformer bf16 waży 40,9 GB.
`training preview --lora <plik>` renderuje po cztery próbki na każdy z 32 tokenów, ok. 5 minut
na postać, czyli niecałe 3 godziny na komplet. To jedyna ocena, która odpowiada na pytanie o tożsamość;
cztery prompty próbek z chodzącym ziarnem wprowadziły mnie w błąd przy v3 i v3b.

**Wyniki przebudowy etapu 2:** 28676 kadrów z 16071 klatek, 509 klastrów, 13200 kadrów w klastrach,
223 klastry nazwane, 35 postaci, 1499 referencji, z tego 32 postacie z pokryciem `FULL`.
Kapitan Bomba ma 60 referencji zamiast 16 i jego zestaw jest jednolity tożsamościowo.

**Cztery błędy naprawione po drodze**, wszystkie zacommitowane:
- `medoid_row` materializował tablicę N×N×768, czyli 5,8 GB dla jednej postaci. Teraz przez macierz
  Grama, 0,2 s zamiast minut.
- `rembg[gpu]` ciągnie `onnxruntime-gpu` zbudowany pod CUDA 13, a pod ma 12.8, więc leciało po CPU,
  21 s na kadr zamiast 0,27 s. Wersja przypięta na `onnxruntime-gpu==1.22.0`.
- `coverage_of` porównywał liczbę referencji z `target_references`, więc podniesienie targetu na 64
  zepchnęło prawie wszystkie postacie na `THIN` i `training dataset` brał tylko dwie. Próg pełnego
  pokrycia jest teraz osobny (`full_reference_threshold`, 16).
- `QwenCharacterPainter` ładował transformer bazowy z Nunchaku, który LoRA nie przyjmie. Teraz bierze
  bf16 wprost z `Qwen/Qwen-Image` z `enable_model_cpu_offload`.

**Co zostaje nierozwiązane:** na ok. jednej czwartej kart tło nie jest usunięte (kanapa, pustynia,
ciemne sceny). `background_removed` łapie tylko całkowitą porażkę `rembg`, a tu model wycina blob
razem z tłem, więc alfa nie jest w pełni nieprzezroczysta. Do poprawy przed ewentualnym v5.

## Stan na 2026-08-31

Etap 4, LoRA tożsamości v2 wytrenowana do końca: rank 128, alpha 128, 6000 kroków, ok. 7 godzin
na RTX PRO 6000. Plik lokalnie w `data/loras/identity_v2.safetensors`. Nie jest jeszcze oceniona
end to end, bo po drodze wyszły dwa problemy.

Problem pierwszy, blokujący inferencję. Nunchaku nie umie wczytać LoRA do Qwen-Image. Konwertery
LoRA w Nunchaku istnieją wyłącznie dla FLUX, także na gałęzi głównej. PEFT nie wstrzyknie adaptera
w skwantyzowane warstwy (`ValueError: Target module AWQW4A16Linear ... is not supported`).
Do tego klucze z `ai-toolkit` mają prefiks `diffusion_model`, nie `transformer`, więc
`load_lora_adapter` bez `prefix=` cicho nic nie robi i renderuje czysty model bazowy.
Wniosek: malowanie postaci z LoRA wymaga transformera bf16 (40,9 GB z `Qwen/Qwen-Image`),
a ten nie mieści się na podzie z dyskiem kontenera 30 GB i wolumenem 50 GB.
Plik `qwen_image_fp8mixed.safetensors`, na którym uczył `ai-toolkit`, nie jest zamiennikiem:
diffusers odrzuca przy wczytywaniu tensory `weight_scale` i render wychodzi czystym szumem.
Kolejny pod pod inferencję z LoRA musi mieć dysk kontenera co najmniej 120 GB.

Problem drugi, ważniejszy: za mało referencji na postać. `target_references` było 16, a Kapitan Bomba
ma 385 dostępnych kadrów, czyli 369 szło do kosza. Przy 16 referencjach rozłożonych na dwa różne
looki tej samej postaci (hełm w stylu Robocopa i mundur kontra podkoszulek w domu) na każdy wygląd
zostaje po osiem kadrów, za mało, żeby token złapał którykolwiek. Stąd rozpad `bmb30` po kroku 3500
w bezkształtną plamę, podczas gdy tokeny postaci o jednolitym wyglądzie (`bmb16`, `bmb21`, `bmb26`)
poprawiały się do końca. `target_references` podniesione do 64, co daje dataset 1597 obrazów zamiast
525 i 18 z 33 postaci na pełnym targecie.

Uwaga na slugi, żeby nie powtórzyć mojej pomyłki: `tytus-bomba` to Kapitan Bomba. Slug bierze się
z tytułu strony wiki, a `canonical_name` to "Kapitan Tytus Bomba". Etykieter przypisywał ten slug
poprawnie, jego 23 klastry to jedna postać w różnych strojach, nie kilka postaci.
Podobnie `kurvinox` z 82 klastrami: `species = "kurvinox"` i alias "kurvinoxy" znaczą, że to cała rasa,
a nie pojedynczy osobnik, więc token dla niej z definicji uśrednia różnych przedstawicieli.
To samo dotyczy `naukowcy` i `c-qrwozaurscy-kardynalowie`.

Poprawki dodatkowe: `rows_near_medoid` odsiewa kadry odstające od mediany odległości do medoidu
(mediana plus tolerancja razy MAD) zanim zadziała dobór różnorodności, a `background_removed`
odrzuca kadry, na których `rembg` zawiódł i zostało nieprzezroczyste tło (widziałem kanapę, pustynię).
Kolejność ma znaczenie: `select_diverse_rows` celowo bierze kadry najbardziej odległe,
więc bez filtra czystości preferował kadry najmniej reprezentatywne.

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

1. Dokończyć trening v5 i ściągnąć wszystkie dwanaście checkpointów razem z próbkami, zanim pod zniknie.
2. Ocenić przez `training preview --lora` na całych 32 tokenach, plus kontrola z trzema rzędami podpisu.
   Kryterium jest jedno: czy sam token przy tym samym ziarnie daje różne postacie. Przy v4 różnica
   wynosiła 6-8 na 255, więc dopiero wyraźnie wyższa liczba znaczy, że tożsamość weszła w wagi.
3. Wybrany checkpoint podłożyć w `data/loras/` jako `identity.safetensors`, potem `transform photo`
   na zdjęciach z `data/input`.
4. Jeśli tożsamość dalej nie trzyma, następną dźwignią jest osobna LoRA na postać. Angielski opis
   wyglądu w kodeksie odpada, bo v4 pokazał, że opis w podpisie właśnie jest problemem, nie lekarstwem.
5. Do poprawy przed ewentualnym v6: na ok. jednej czwartej kart tło nie jest usunięte.

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
