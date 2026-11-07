# D5 Cyrillic <-> Traditional-Script Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `rinchen.script_conv` package: validator, metrics, a lexicon-first rule baseline measured against a held-out reference, a trainable Eole character-level converter (tiny profile trained and exported, full profile rendered), a CTranslate2 runtime wrapper and the `rinchen convert-script` CLI.

**Architecture:** Leaf modules (`alphabet`, `validate`, `segments`, `eval`) feed the rule baseline and the corpus builder; Eole configs are rendered by code and validated; training, export and parity run as hardened subprocesses; `runtime.ScriptConverter` loads the CT2 export. Only the CLI touches the manifest.

**Tech Stack:** Python 3.14.10, Eole 0.6.4 (CPU training), CTranslate2 4.8.4, sentencepiece 0.2.1, pydantic v2, typer, pytest + hypothesis, polars not needed.

**Spec:** `docs/superpowers/specs/2026-11-07-d5-script-converter-design.md` (binding; rulings 1-13). Master plan: `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`, section D5.

## Global Constraints

- CPython 3.14.x only; every script entrypoint asserts `sys.version_info[:2] == (3, 14)`. Use `uv run`. No other env.
- Train every Rinchen model from random init. No fine-tuning, LoRA or continued training of any pretrained model.
- `position_encoding_type: SinusoidalInterleaved` in every Eole model; never Relative, rotary, ALiBi, sliding window or GQA.
- Never use Inner Mongolian data, varieties, datasets or tools, FLORES+ `khk_Mong`, or Latin-script informal Mongolian ("galig"). Write nothing about licensing or legal topics, in code, comments, docs or commit messages.
- The converter never emits U+200C or U+200D; NNBSP (U+202F) before separated suffixes, MVS (U+180E) only where lexical; `normalize` stays as D3 left it. Never apply NFKC/NFKD to `mn_Mong`.
- Word segmentation of traditional-script text splits on U+0020 only; never `str.split()` with no argument (it splits at U+202F).
- Record every resolved package version and checkpoint hash in `manifests/` (CLI commands write manifests last; the UCD table gets `manifests/ucd_standardized_variants.json`).
- Tests before code; numeric gates are gates: a missed gate means more steps or a fix, never a lower threshold. Never weaken, skip or delete a failing test.
- Nothing written under `data/` or `models/`. Training and export work in `tmp_path` or the scratchpad, never in the repo tree.
- Commit subjects start with `D5: `; trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Gate: `uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -m "not network" -q`. No file mode changes. Match local style (see `src/rinchen/mine/`, `src/rinchen/cli/filter.py`).
- Eole facts measured in the spike (spec "Evidence"): every corpus with the `prefix` transform needs `tgt_prefix: ""`; `embeddings.word_vec_size` must equal `hidden_size`; tiny runs need `bucket_size: 1024`, `num_workers: 1`, `prefetch_factor: 8`, `OMP_NUM_THREADS=2`; Noam `learning_rate: 0.5`, `warmup_steps: 100`; `eole_ct2` export needs `CUDA_VISIBLE_DEVICES=""` (segfault otherwise); CTranslate2 source tokens are `[tag, *sp.encode(text, out_type=str)]`.

## Review Focus

Inputs the spec implies but no single task's happy path exercises, most likely to bite first:

1. A traditional-script word with an NNBSP-separated suffix is cut in two by `.split()` or by a segmenter that treats U+202F as a space. Pinned in Task 2 (segments) and Task 4 (data), end to end in Task 8.
2. Capitalized or upper-case Cyrillic (`Монгол`, `МОНГОЛ`) misses a lower-case lexicon key. Pinned in Task 3.
3. Cyrillic words glued to digits, Latin or hyphens (`2024оны`, `н-р`, `iPhone-ыг`). Pinned in Task 2 and Task 8.
4. Empty and whitespace-only input, and a word longer than the decoding limit, at the runtime and CLI boundary. Pinned in Task 8 and Task 9.
5. A machine with a visible GPU: export must not segfault, training must stay on CPU. Pinned in Task 7.

---

### Task 1: Alphabet and validator

**Files:**
- Create: `src/rinchen/script_conv/__init__.py` (empty), `src/rinchen/script_conv/alphabet.py`, `src/rinchen/script_conv/validate.py`, `src/rinchen/script_conv/standardized_variants.tsv`, `manifests/ucd_standardized_variants.json`
- Test: `tests/unit/test_script_conv_alphabet.py`, `tests/unit/test_script_conv_validate.py`, `tests/property/test_script_conv_validate_property.py`

**Interfaces:**
- Produces (`alphabet`): `LETTERS: frozenset[str]` (U+1820..U+1842), `FVS: frozenset[str]` (U+180B, U+180C, U+180D, U+180F), `MVS = "᠎"`, `NNBSP = " "`, `NIRUGU = "᠊"`, `JOINERS: frozenset[str]` (U+200C, U+200D), `CYRILLIC_TO_SCRIPT_PUNCTUATION: Mapping[str, str]` (`"."` -> U+1803, `","` -> U+1802), `SCRIPT_TO_CYRILLIC_PUNCTUATION` (its inverse).
- Produces (`validate`): `@dataclass(frozen=True) Violation(index: int, code: str, detail: str)`, `validate_traditional(text: str) -> tuple[Violation, ...]`, `VIOLATION_CODES: frozenset[str] = {"joiner", "foreign_character", "orphan_selector", "misplaced_nnbsp", "invalid_variant"}`, `VARIANT_TABLE: Mapping[str, frozenset[str]]` (base letter -> FVS characters the UCD lists for it).

- [ ] **Step 1: Write failing tests.** `test_alphabet.py`: `LETTERS` has 35 members (U+1820..U+1842) and contains `"ᠠ"`; `FVS` has 4 members; `CYRILLIC_TO_SCRIPT_PUNCTUATION["."] == "᠃"`; the inverse round-trips both keys. `test_validate.py`, one failing and one passing example per code in `VIOLATION_CODES`: `"ᠠ‍ᠠ"` -> `joiner` at index 1; `"ᠠX9"` passes (Latin run and digits allowed) but `"ᠠа"` (Cyrillic а) -> `foreign_character`; `"᠋ᠠ"` and `"ᠠ᠎ "` -> `orphan_selector`; `"ᠠ  "` and `" ᠠ"` -> `misplaced_nnbsp` while `"ᠮᠣᠩᠭᠣᠯ ᠤᠨ"` passes; a letter followed by an FVS the table does not list for it -> `invalid_variant`; a listed pair passes. The empty string returns `()`. Also: every entry of `VARIANT_TABLE` has a key in `LETTERS` and values in `FVS`; `standardized_variants.tsv` header names source URL, UCD version 18.0.0 and sha256, and `manifests/ucd_standardized_variants.json` carries the same sha256s. Property test: `validate_traditional(text)` never raises for `st.text()` and returns violations sorted by index.
- [ ] **Step 2: Run them and see them fail** (`uv run pytest tests/unit/test_script_conv_alphabet.py tests/unit/test_script_conv_validate.py tests/property/test_script_conv_validate_property.py -q`).
- [ ] **Step 3: Implement `alphabet.py` and `validate.py`.** Allowed characters outside the script: ASCII letters, digits, ASCII punctuation, U+0020, and the two mapped marks. The table file `standardized_variants.tsv` is derived once from `StandardizedVariants.txt` of UCD 18.0.0 (fetch `https://www.unicode.org/Public/18.0.0/ucd/StandardizedVariants.txt`; if that path is absent, find the 18.0.0 file under `https://www.unicode.org/Public/` and record the real URL): keep lines whose base is a Mongolian letter (U+1820..U+1842) and whose selector is U+180B, U+180C, U+180D or U+180F, one `base_hex<TAB>fvs_hex` per line after a `#` header with the URL, retrieval date and the sha256 of the downloaded file; also write `manifests/ucd_standardized_variants.json` with the source URL, UCD version, sha256 of source and of the derived TSV. Load the table with `importlib.resources`.
- [ ] **Step 4: Run the tests and see them pass, then the full gate.**
- [ ] **Step 5: Commit** (`D5: script_conv alphabet, validator and UCD variant table`).

### Task 2: Segments and metrics

**Files:**
- Create: `src/rinchen/script_conv/segments.py`, `src/rinchen/script_conv/eval.py`
- Test: `tests/unit/test_script_conv_segments.py`, `tests/unit/test_script_conv_eval.py`, `tests/property/test_script_conv_segments_property.py`, `tests/property/test_script_conv_eval_property.py`

**Interfaces:**
- Consumes: `alphabet` constants.
- Produces (`segments`): `@dataclass(frozen=True) Segment(text: str, kind: Literal["cyrillic", "mongolian", "other"])`, `segment(text: str) -> tuple[Segment, ...]`.
- Produces (`eval`): `edit_distance(a: str, b: str) -> int`, `cer(references: Sequence[str], hypotheses: Sequence[str]) -> float`, `word_accuracy(references, hypotheses) -> float`, `@dataclass(frozen=True) ConversionScores(cer: float, word_accuracy: float, n: int)`, `score_conversions(references, hypotheses) -> ConversionScores`, `length_ratio_quantiles(pairs: Sequence[tuple[str, str]], quantiles: Sequence[float]) -> dict[float, float]` (script length over Cyrillic length per pair, in characters).

- [ ] **Step 1: Write failing tests.** Segments: `segment("")` is `()`; `segment("Монгол улс")` is cyrillic, other (" "), cyrillic; `segment("ᠮᠣᠩᠭᠣᠯ ᠤᠨ ᠨᠣᠮ")` has exactly two mongolian segments and the first contains the NNBSP; a leading or trailing NNBSP is `other`; `segment("2024оны")` is other, cyrillic; `segment("н-р")` is cyrillic, other, cyrillic; `segment("iPhone-ыг")` is other, cyrillic; `"‍"` between two letters stays inside the mongolian segment; Cyrillic with a combining mark (`"а́"`) stays one segment. Property: `"".join(s.text for s in segment(t)) == t` for `st.text()` and for text drawn from the Cyrillic, Mongolian and ASCII blocks plus U+202F; adjacent segments never share a kind except `other`... (assert adjacent segments of kind cyrillic or mongolian are never equal-kind neighbours). Eval: `edit_distance("", "abc") == 3`, `("kitten", "sitting") == 3`, symmetric, `edit_distance(x, x) == 0`; `cer(["abc"], ["abc"]) == 0.0`; `cer(["abcd"], ["abxd"]) == 0.25`; `cer` over several pairs is total edits over total reference characters (not the mean of ratios); `cer([], [])`, mismatched lengths and an all-empty reference set raise `ValueError`; `word_accuracy` counts exact matches; `length_ratio_quantiles` on a planted list returns the exact values. Property: `edit_distance` symmetric, `<= max(len)`, zero only for equal strings, and the triangle inequality.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** `segment`: one pass classifying each character (Cyrillic letter by `unicodedata`/`regex` script, Mongolian run characters from `alphabet`), attach NNBSP to a mongolian run only when the next character continues it. `edit_distance`: two-row dynamic programming over code points.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D5: script segments and conversion metrics`).

### Task 3: Reference fixtures, lexicon and rule baseline

**Files:**
- Create: `scripts/build_script_conv_fixtures.py`, `tests/fixtures/script_conv/wiktionary_reference.tsv`, `tests/fixtures/script_conv/rules_golden.tsv`, `tests/fixtures/script_conv/tiny_words.tsv`, `src/rinchen/script_conv/lexicon.tsv`, `src/rinchen/script_conv/rules.py`
- Test: `tests/unit/test_script_conv_rules.py`, `tests/unit/test_script_conv_fixtures.py`

**Interfaces:**
- Consumes: `segment`, `validate_traditional`, `alphabet`, `cer`/`word_accuracy`.
- Produces: `WordResult(text: str, source: Literal["lexicon", "fallback"])` (frozen dataclass), `LEXICON: Mapping[str, str]`, `transliterate_word(word: str) -> WordResult`, `transliterate_fallback(word: str) -> str` (the lexicon-free path, used for held-out measurement), `transliterate(text: str) -> str`, `development_split(word: str) -> bool` (True when `int(sha256(word.encode()).hexdigest(), 16) % 2 == 0`).

**Inputs:** the reference harvest TSV the controller supplies at `<scratchpad>/d5-wikt-ref/reference.tsv` (columns `cyrillic`, `traditional`, `part_of_speech`, `page_url`, `revision`, `has_fvs_mvs_nnbsp`; revision ids per page; Mongolian section only). It is committed unchanged as `tests/fixtures/script_conv/wiktionary_reference.tsv`.

- [ ] **Step 1: Write failing tests.** Fixtures: `build_script_conv_fixtures.py` is deterministic (run twice, identical bytes); `lexicon.tsv` equals exactly the development-split rows of the reference; `rules_golden.tsv` has exactly 100 rows, 30 with `path=lexicon` (development split) and 70 with `path=fallback` (held-out split, fallback output equals the reference); no `fallback` golden word is a lexicon key; `tiny_words.tsv` words are all from the reference and number between 400 and 700. Rules: the 100 golden rows convert exactly (`transliterate_word(word).text == expected` and `.source == path`); `transliterate_word("МОНГОЛ")` equals `transliterate_word("монгол")` (case-insensitive lookup); every lexicon value and every `transliterate_fallback` output over the whole reference passes `validate_traditional`; `transliterate("Монгол, улс.")` ends the first word before U+1802 and ends with U+1803, keeps the space as U+0020; `transliterate("2024оны н-р iPhone-ыг")` keeps digits, hyphen and Latin run unchanged and converts only the Cyrillic parts; a word with an uppercase letter outside the lexicon falls back; `transliterate("")` is `""`; text already in script passes through unchanged. Reference accuracy: a test computes exact-match rate and CER of `transliterate_fallback` on the held-out split and asserts both are within 0.02 of the constants `HELD_OUT_EXACT_MATCH` and `HELD_OUT_CER` defined in the test file (set from the one-time measurement, Step 5).
- [ ] **Step 2: Run them and see them fail** (the fixture files do not exist yet).
- [ ] **Step 3: Implement the builder script and the rules.** The builder reads the reference, drops rows whose script form fails `validate_traditional`, splits by `development_split`, writes `lexicon.tsv` from the development rows, selects the golden rows (first 30 development rows by sorted Cyrillic word whose script contains a letter other than the plain letter map, if any, else first 30; then the first 70 held-out rows, in sorted order, on which `transliterate_fallback` equals the reference), and writes `tiny_words.tsv` (the first 500 reference rows by sorted word). The fallback is table-driven: one Cyrillic-letter to script-letter map (letters a, e, i, o, u, ö, ü and consonants per the W3C Mongolian orthography notes at `https://r12a.github.io/scripts/mong/mn`; ы -> i, я/ю/е/ё as j + vowel, harmony class of the word choosing the ю and ё vowel, ж and з -> ᠵ, ц and ч -> ᠴ, нг -> ᠩᠭ is allowed only if it improves the development split) plus at most five contextual rules. Develop the rules against the DEVELOPMENT split only: a rule is kept only if it lifts the development exact-match rate by at least 0.5 percentage points and can be stated in one sentence for the runbook. Do not read held-out accuracy until the rules are frozen.
- [ ] **Step 4: Run the tests and see them pass** (with `HELD_OUT_*` placeholders set to the measured values from Step 5).
- [ ] **Step 5: Measure once.** After freezing the rules, run the held-out measurement a single time, write the three numbers (exact match, CER, count) into the report file and set the two test constants. Report them in the task report; do not tune after seeing them.
- [ ] **Step 6: Run the gate; commit** (`D5: reference fixtures, lexicon and rule baseline`).

### Task 4: Converter corpus builder

**Files:**
- Create: `src/rinchen/script_conv/data.py`
- Test: `tests/unit/test_script_conv_data.py`

**Interfaces:**
- Consumes: `rules.transliterate_word`, `validate_traditional`, `rinchen.text.normalize.normalize`, `rinchen.utils.part_file.unique_part_path`.
- Produces: `WordPair(cyrillic: str, script: str, origin: Literal["reference", "rule", "parallel"])` (frozen), `align_words(cyrillic_line: str, script_line: str) -> tuple[WordPair, ...] | None`, `rule_pairs(words: Iterable[str], *, allow_fallback: bool = False) -> tuple[WordPair, ...]`, `read_pairs_tsv(path: Path) -> tuple[WordPair, ...]`, `@dataclass(frozen=True) CorpusFiles(directory: Path, files: Mapping[str, Path], rows: Mapping[str, int], dropped: Mapping[str, int])`, `build_corpus(pairs: Sequence[WordPair], out_dir: Path, *, valid_fraction: float, seed: int) -> CorpusFiles`. File names: `cyrl2mong.{train,valid}.{src,tgt}` and `mong2cyrl.{train,valid}.{src,tgt}`.

- [ ] **Step 1: Write failing tests.** `align_words("Монгол улс", "ᠮᠣᠩᠭᠣᠯ ᠤᠯᠤᠰ")` returns two pairs with origin `parallel`; unequal counts return `None`; a script word with NNBSP stays one word (`align_words("Монголын", "ᠮᠣᠩᠭᠣᠯ ᠤᠨ")` is one pair); empty lines return `None`. `rule_pairs` keeps only lexicon words unless `allow_fallback`. `build_corpus` on a planted list of 40 pairs (3 exact duplicates, 1 pair whose script side holds ZWJ, 1 pair with an empty side, 1 pair whose script side holds a Cyrillic letter) returns exact `rows` and `dropped` counts (`duplicate: 3`, `joiner: 1`, `empty: 1`, `foreign_character: 1`); train and valid are disjoint by Cyrillic word; the same seed gives identical files and a different seed a different split; every `.src` and `.tgt` line count matches per split and direction; `mong2cyrl` files are the swapped columns of `cyrl2mong`; files are UTF-8 with LF; an NNBSP and an FVS in the planted data survive unchanged; `valid_fraction` outside (0, 0.5] raises `ConfigError`; no pairs after dropping raises `DataError`. Normalization: a Cyrillic word with a Latin homoglyph is fixed by `normalize(..., "khk_Cyrl")`; a script side with ZWJ is dropped as `joiner` BEFORE normalization strips it (the validator runs on the raw script side).
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Write each file through `unique_part_path` then rename, as in `rinchen.corpus.writers`. Dedup key is the (normalized Cyrillic, normalized script) pair. Split by `sha256(f"{seed}:{cyrillic}")` below `valid_fraction`.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D5: converter corpus builder`).

### Task 5: Character SentencePiece model

**Files:**
- Create: `src/rinchen/script_conv/char_model.py`, `configs/spm/converter_char.yaml`
- Test: `tests/unit/test_script_conv_char_model.py`

**Interfaces:**
- Produces: `CONVERTER_TAGS = ("<2mn_Mong>", "<2mn_Cyrl>")`, `train_char_model(texts: Iterable[str], prefix: Path, *, vocab_size: int) -> Path` (returns `<prefix>.model`), `check_char_model(model: Path, texts: Iterable[str]) -> None` (raises `DataError` with the offending piece or character).

- [ ] **Step 1: Write failing tests.** The trainer receives exactly `model_type="char"`, `character_coverage=1.0`, `byte_fallback=False`, `normalization_rule_name="identity"`, `add_dummy_prefix=True`, `user_defined_symbols=["<2mn_Mong>", "<2mn_Cyrl>"]`, `hard_vocab_limit=False` (monkeypatch `SentencePieceTrainer.train` and compare the options). A real run on a small fixture (the Cyrillic and script columns of `tests/fixtures/script_conv/tiny_words.tsv`) produces a model for which both tags are single pieces, `sp.decode(sp.encode(word))` returns the word for every training word, and the NNBSP, MVS and each FVS character round-trip; `check_char_model` raises `DataError` when a tag is split (a model trained without the symbols) and when a character does not round-trip; trainer failure becomes `UpstreamError` with the cause chained. `configs/spm/converter_char.yaml` loads and has `vocab_size`, `output_prefix`, `seed`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement,** writing the texts to a `.part` file next to the prefix as `train_spm` does; keep the trainer's log level at 2.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D5: converter character SentencePiece model`).

### Task 6: Eole converter configuration

**Files:**
- Create: `src/rinchen/script_conv/config.py`, `configs/eole/converter.yaml`, `configs/eole/converter_tiny.yaml`
- Test: `tests/unit/test_script_conv_config.py`

**Interfaces:**
- Produces: `render_converter_config(profile: Literal["tiny", "full"], *, data_dir: Path, spm_model: Path, run_dir: Path) -> dict[str, Any]`, `validate_converter_config(cfg: Mapping[str, Any]) -> None` (raises `ConfigError` naming the key), `dump_config(cfg, path: Path) -> None` (YAML, key order stable), `FORBIDDEN_KEYS`.

- [ ] **Step 1: Write failing tests.** Tiny rendering has encoder 2, decoder 2, `hidden_size` 64, `word_vec_size` 64, heads 4, `transformer_ff` 128, `learning_rate` 0.5, `warmup_steps` 100, `train_steps` 800, `bucket_size` 1024, `num_workers` 1, `prefetch_factor` 8, `compute_dtype` `fp32`, `gpu_ranks: []`, `world_size: 1`; full has 6/6, 512, heads 8, ff 2048, `compute_dtype` `bf16` and no bucket override; both set `position_encoding_type: SinusoidalInterleaved`, `share_vocab: true`, two training corpora and every corpus carries `tgt_prefix: ""`. The tag is the TARGET script: the corpus that converts Cyrillic to script (`cyrl2mong`) uses `src_prefix: "<2mn_Mong>"`, and `mong2cyrl` uses `src_prefix: "<2mn_Cyrl>"`. Eole takes one validation corpus named `valid`, and the two validation files must not be concatenated (their prefixes differ). Check against the installed Eole 0.6.4 whether it accepts a second validation corpus; if it does, render one per direction, otherwise render `valid` for the Cyrillic-to-script direction only and say so in a code comment and in the test. Validation rejects: missing `tgt_prefix`, `word_vec_size != hidden_size`, `position_encoding_type` other than `SinusoidalInterleaved`, every key in `FORBIDDEN_KEYS` (`rotary_interleave`, `rotary_theta`, `rotary_dim`, `max_relative_positions` other than 0, `alibi`, `sliding_window`, `num_kv`/`heads_kv` less than heads), an unknown top-level key, a missing direction corpus. The checked-in `configs/eole/converter_tiny.yaml` and `converter.yaml` equal `dump_config(render_converter_config(...))` rendered with the placeholder paths `data/script_conv/corpus`, `data/script_conv/converter_char.model`, `data/script_conv/run` (the test fails on any difference). Rendered tiny config loads through Eole's own pydantic config class without error (`eole.config.run.TrainConfig` or the class `eole train` uses; find it in `eole/bin/run/train.py`).
  (Superseded by ruling K: the tiny profile trains 1000 steps with dropout 0; see the design's amendments.)
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Build the dict from small per-profile override dictionaries over one base; keys follow the spike config exactly.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D5: Eole converter configuration`).

### Task 7: Training, export and parity

**Files:**
- Create: `src/rinchen/script_conv/train.py`, `src/rinchen/script_conv/export.py`
- Test: `tests/unit/test_script_conv_train.py`, `tests/unit/test_script_conv_export.py`, `tests/integration/__init__.py`, `tests/integration/test_script_conv_tiny.py`

**Interfaces:**
- Consumes: `data.build_corpus`/`CorpusFiles`, `char_model.train_char_model`, `config.render_converter_config`/`dump_config`, `eval.cer`.
- Produces (`train`): `training_environment(threads: int) -> dict[str, str]`, `run_training(cfg_path: Path, run_dir: Path, *, threads: int = 2) -> Path` (checkpoint directory), `predict(ckpt_dir: Path, src_file: Path, *, tag: str, spm_model: Path, out_file: Path) -> Path`.
- Produces (`export`): `export_environment() -> dict[str, str]` (sets `CUDA_VISIBLE_DEVICES=""`), `export_ct2(ckpt_dir: Path, out_dir: Path) -> Path`, `@dataclass(frozen=True) ParityReport(agree: int, total: int, mismatches: tuple[tuple[str, str, str], ...])` with `agreement` property, `parity(ct2_dir: Path, spm_model: Path, src_lines: Sequence[str], eole_lines: Sequence[str], *, tag: str) -> ParityReport`.

- [ ] **Step 1: Write failing unit tests (no subprocess).** `training_environment(2)` sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to `"2"` and keeps `PATH`; `export_environment()["CUDA_VISIBLE_DEVICES"] == ""` even when the parent has it set; `run_training` builds the two commands `eole build_vocab -config <cfg> -n_sample -1` and `eole train -config <cfg>` (monkeypatch `subprocess.run`, assert argv and env), returns `<run_dir>/ckpt`, and raises `UpstreamError` whose message ends with the last 20 lines of the log on a non-zero exit; `export_ct2` raises `UpstreamError` when `model.bin` is missing or empty even with exit 0; `parity` on stub translators counts agreement and returns the mismatches, and `agreement >= 0.99` semantics are the caller's. A test that sets `CUDA_VISIBLE_DEVICES=0` in the parent and asserts the child env is empty pins Review Focus 5.
- [ ] **Step 2: Write the failing integration test** (`@pytest.mark.subprocess`, module-scoped fixture `trained` using `tmp_path_factory`): build the corpus from `tiny_words.tsv` (`origin="reference"`), train the char model (vocab 200), render the tiny config, run training, predict both directions on the TRAIN sources, and compute CER per direction. Assertions: `cer_cyrl2mong <= 0.15` and `cer_mong2cyrl <= 0.15`; wall time of `run_training` under 150 s; NNBSP and FVS counts in the predicted script lines equal the counts in the reference for at least 90% of the lines that contain them (pins Review Focus 1 through the model). Second test: `export_ct2` then `parity` over the training sources in the Cyrillic-to-script direction, `agreement >= 0.99`; a third, skipped unless `RINCHEN_CT2_CUDA_PROBE=1`, runs the converter WITHOUT hiding CUDA and asserts it fails, documenting the segfault.
- [ ] **Step 3: Run them and see them fail.**
- [ ] **Step 4: Implement `train.py` and `export.py`.** Subprocesses via `subprocess.run([...], capture_output=True, text=True, env=...)` with the interpreter's own `eole` entry (`[sys.executable, "-m", "eole"]` if that works on this Eole, else the venv `eole` script found via `shutil.which` next to `sys.executable`; verify which one runs). Predict writes a small YAML config as in the spike (`transforms: [sentencepiece, prefix]`, `beam_size: 1`, `gpu_ranks: []`). `parity` loads `ctranslate2.Translator(ct2_dir, device="cpu", intra_threads=1)` and decodes with the spike's token convention. If the tiny gate is missed: raise `train_steps` in the tiny profile (and the config test constant), up to 1500, never the threshold; if it is still missed, report BLOCKED with the measured CERs.
- [ ] **Step 5: Run the tests and the gate;** report the CERs, steps, wall time and parity count.
- [ ] **Step 6: Commit** (`D5: converter training, CTranslate2 export and parity check`).

### Task 8: Runtime wrapper

**Files:**
- Create: `src/rinchen/script_conv/runtime.py`
- Test: `tests/unit/test_script_conv_runtime.py`, `tests/integration/test_script_conv_runtime.py`

**Interfaces:**
- Consumes: `segment`, `rules.transliterate_word`, `validate_traditional`, `alphabet` punctuation maps, `ctranslate2`, `sentencepiece`.
- Produces: `Conversion(text: str, rule_fallbacks: int)` (frozen), `ScriptConverter(ct2_dir: Path, spm_model: Path, *, device: str = "cpu")`, `ScriptConverter.convert(text: str, target: Literal["Mong", "Cyrl"]) -> Conversion`, plus a seam `ScriptConverter.from_translator(translator, spm, ...)` used by unit tests with a stub.

- [ ] **Step 1: Write failing unit tests with a stub translator.** Words are deduplicated: three occurrences of one word cause one translation; the output keeps spaces, digits, Latin runs and unmapped punctuation unchanged and maps `.`/`,` to U+1803/U+1802 for `"Mong"` and back for `"Cyrl"`; a stub output containing U+200D is replaced by `rules.transliterate_word(word).text` and `rule_fallbacks == 1`; a stub output that is not valid for `"Cyrl"` stays and is counted; `convert("", "Mong")` returns `Conversion("", 0)`; whitespace-only text and text with no segment of the source script are returned unchanged with no translator call; an unknown `target` raises `ConfigError`; a Cyrillic word longer than 80 characters is converted by the rule baseline (counted) and never sent to the model; source tokens passed to the translator are `[tag, *pieces]` with `<2mn_Mong>` for `"Mong"` and `<2mn_Cyrl>` for `"Cyrl"`; the decoded hypothesis drops the leading `▁` correctly (no leading space in the word); mixed-case input (`МОНГОЛ`) is lower-cased before translation. Integration test (`subprocess` marker, reusing the module-scoped `trained` fixture and a module-scoped export from `tests/integration/test_script_conv_tiny.py`, moved to `tests/integration/conftest.py` if needed): `ScriptConverter(...).convert(line_of_fixture_words, "Mong").text` equals the reference script line for at least 90% of fixture lines, contains no U+200C or U+200D, and an NNBSP-bearing fixture word comes back as one word.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Batch all distinct words of one call in a single `translate_batch(..., beam_size=1, max_decoding_length=min(256, 4 * len(word) + 8))`; keep a per-instance dict cache; run segments through `segment`.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D5: ScriptConverter runtime wrapper`).

### Task 9: CLI

**Files:**
- Modify: `src/rinchen/cli/convert_script.py` (replace the stub), `src/rinchen/cli/main.py` only if the registration needs a change
- Test: `tests/unit/test_cli_convert_script.py`

**Interfaces:**
- Consumes: everything above, `rinchen.cli.errors.rinchen_errors_as_exit_codes`, `rinchen.manifest.stage` (`begin_commit`, `build_inputs`, `output_file`, `tool_versions`, `code_version`, `refuse_foreign_manifest`), `rinchen.manifest.store.write`, `rinchen.utils.part_file`.
- Produces: commands `build-data`, `train`, `export`, `apply`, `eval` under `rinchen convert-script`, with the options in the spec.

- [ ] **Step 1: Write failing tests.** Follow `tests/unit/test_cli_filter.py` for style (CliRunner, `monkeypatch` seams). `--help` for the group and for each command. `build-data --pairs fixture.tsv --out-dir d` writes the eight corpus files and a manifest listing inputs and outputs with sha256 and the drop counts; `--words` builds pairs through `rule_pairs`; neither or both of `--words` and `--pairs` exit 2. `train --profile tiny --dry-run` prints the two commands and the rendered config path and runs nothing; `train --profile full` without `--confirm-full` and without `--dry-run` exits 2; a missing `--data-dir` corpus exits 3 (`DataError`). `export` with a monkeypatched `export_ct2` failing leaves NO manifest and removes `.part` files; success writes the manifest last. `apply --model-dir m --target Mong` reads stdin and writes converted text to stdout (monkeypatched `ScriptConverter`), empty input exits 0 with no output, `--target Latn` exits 2. `eval` prints CER, word accuracy and n from a pairs TSV (monkeypatched converter) and exits 3 on an empty file. A foreign manifest in the output directory is refused, also under `--dry-run`. Output and input collision (`--out-dir` equal to the directory of `--pairs`) exits 2.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Thin commands: parse, call library functions, write manifests; no business rules in the CLI. Reuse D4's commit protocol exactly.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D5: rinchen convert-script CLI`).

### Task 10: Runbook and records

**Files:**
- Create: `docs/runbooks/05_script_converter.md`
- Modify: `STATE.md`, `DECISIONS.md` (CRLF file: single-line edits or append with CRLF), `CLAUDE.md` (repo layout and current phase), `docs/VERIFY.md` (new row `D-CT2-EOLE-CUDA`), `docs/runbooks/04_filtering_mining.md` only if a `mn_Mong` length-ratio note belongs there

**Interfaces:** none.

- [ ] **Step 1: Measure the figures the runbook needs** with the installed code: held-out rule exact match and CER (already measured in Task 3), `length_ratio_quantiles` (0.01, 0.05, 0.5, 0.95, 0.99) of script over Cyrillic length on the reference pairs, tiny training time and CERs (Task 7), parity count, `apply` throughput in words per second on the tiny model.
- [ ] **Step 2: Write `05_script_converter.md`.** Sections: what the converter is and is not (the scaffold framing), commands for every CLI step with expected output, the Eole facts found in the spike with the symptom of each, thresholds table (CER gates, parity), measured numbers, the open gates and limitations of the spec, how to replace the lexicon and reference with real data, the full-profile training command (`--confirm-full`) as an operator step, the CUDA segfault and its workaround. No licensing or legal text.
- [ ] **Step 3: Update the records.** `STATE.md`: D5 done on branch `d5-script-converter` (not merged), evidence, counts, open gates. `DECISIONS.md`: entry `2026-11-07: D5 script converter - the rulings, made without a review round` with rulings 1-13 and their costs (CRLF preserved). `CLAUDE.md`: add the `script_conv/` package and `cli/convert_script.py` to the layout, update "Current phase". `docs/VERIFY.md`: the CUDA row.
- [ ] **Step 4: Run the full gate** and the network tests that exist, `rg -n -i "licen|legal"` over every file D5 added returns nothing, and `git status` shows only intended files.
- [ ] **Step 5: Commit** (`D5: runbook, state and decision records`).
