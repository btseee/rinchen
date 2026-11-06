# D3 — Normalization & SentencePiece Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Rinchen's text normalizer, sentence splitter, script detector, 10-language registry and SentencePiece train/check pipeline, proven on a tiny synthetic fixture.

**Architecture:** Small single-purpose modules, each tested before it is written: `lang/scripts.py` (script ratios) → `config/languages.py` (registry) → `text/normalize.py`, `split/rules.py`, `spm/symbols.py` (all validate language codes against the registry). `spm/corpus.py` samples balanced text via `utils/sampling.py`; `spm/train.py` and `spm/check.py` wrap `sentencepiece`; `cli/spm.py` exposes them. Only the `tiny_1k` model is trained in D3; `student_64k` / `teacher_128k` ship as validated configs.

**Tech Stack:** Python 3.14, `regex`, `sentencepiece==0.2.1`, pydantic v2, typer, pytest + hypothesis (all already pinned; **no new dependencies**).

**Spec:** `docs/superpowers/specs/2026-11-02-d3-normalization-sentencepiece-design.md` (read it first; master-plan D3 text is `docs/research/2026-10-29-rinchen-dev-agent-prompt.md` lines 345-357, B5 at 261-267).

## Global Constraints

- CPython 3.14.x only; every script entrypoint starts with `assert sys.version_info[:2] == (3, 14)`. uv only; no new dependencies, no `uv lock` changes.
- Tests before code (Three Laws): write the test, watch it fail, then implement. Numeric/golden gates are zero-tolerance.
- NEVER galig/Latin-transliterated Mongolian, Inner Mongolian data/varieties, or licensing/legal prose. Fixture and golden Mongolian text must be genuine Cyrillic (U+0400–U+04FF) or traditional-script codepoints; tests enforce it mechanically.
- SentencePiece policy invariants (constants in `spm/train.py`, not config): `model_type=unigram`, `byte_fallback=True`, `split_digits=True`, `normalization_rule_name=identity`. Config-carried values: `vocab_size` 1000 / 64000 / 128000, `character_coverage` 0.9999, `temperature` 0.3, `input_sentence_size` per profile.
- 118 user-defined symbols = 10 `<2xx>` tags (one per `configs/languages.yaml` entry) + `<bt> <ft> <t> </t> <r> </r> <doc> <sep>` + `⟦0⟧`..`⟦99⟧`; all unique.
- Errors: `ConfigError` (bad config / unknown language), `DataError` (bad corpus), `UpstreamError` (sentencepiece failure, cause chained). No bare `except`.
- Typing: full annotations; `@dataclass(frozen=True, slots=True)` for value objects; pydantic only at I/O boundaries. Files UTF-8 with explicit `encoding="utf-8"`; never rely on locale or CWD (resolve from `REPO_ROOT`).
- Test layout: `tests/unit/test_<pkg>_<module>.py`, `tests/property/`, `tests/golden/`, fixtures under `tests/fixtures/`. Every new `tests/` subdirectory gets an `__init__.py`.
- Commits: subject `D3: <what>`; every message ends with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (steps below show subjects only). Before each commit `git diff --summary` must show no `mode change` lines and `git status` no stray files.
- Gate = `uv run ruff format --check .` + `uv run ruff check .` + `uv run mypy src` + `uv run pytest -m "not network" -q` (`just` is not installed on this Linux box; run the four commands).

## Verified Facts (2026-11-06, Python 3.14.10, Linux, installed libs)

- `sentencepiece` 0.2.1: `SentencePieceTrainer.train(input=, model_prefix=, vocab_size=, model_type=, character_coverage=, byte_fallback=, split_digits=, normalization_rule_name=, user_defined_symbols=list[str], input_sentence_size=, shuffle_input_sentence=, num_threads=, minloglevel=)` works. Failures raise `RuntimeError` (e.g. `Vocabulary size too high (1000). Please set it to a value <= 587`).
- With default `add_dummy_prefix`, encoding a bare symbol returns `['▁', '<2khk_Cyrl>']`, **not** one piece. "Atomic" therefore means `pieces == [sym]` or `pieces == ['▁', sym]`. `⟦0⟧` inside text encodes as the single piece `⟦0⟧`.
- Byte fallback detection without protobuf: `sp.is_byte(sp.piece_to_id("<0xNN>"))` for all 256 values (on: 256 byte pieces; off: `piece_to_id` returns 0 = `<unk>`). `sp.is_user_defined` does not exist. `unk_id=0, bos_id=1, eos_id=2, pad_id=-1`.
- A repetitive corpus (30 distinct words) fails the hard vocab limit at 1000; a pseudo-word corpus of 375 lines trained to exactly 1000 in 0.23 s with all 118 symbols. Digits absent from the corpus fall back to bytes, so the fixture must contain digits.
- `regex`: `\p{Script=Cyrillic|Mongolian|Latin|Han|Hangul|Hiragana|Katakana}` all work. `\p{L}` matches Mongolian letters U+1820–U+18AA but not U+1802/1803/1809 (punctuation) or U+180B–180F (marks); `str.isspace()` and `\s` are true for U+202F, false for U+200B/200C/200D/2060/FEFF/180E.
- Ѳ U+0472/ѳ U+0473 (Fita), Ү/ү U+04AE/04AF, Ө/ө U+04E8/04E9, Ұ U+04B0 are all NFC-stable single codepoints. Ұ/ұ is a distinct Kazakh letter: never fold it.
- `mypy src` fails with `import-untyped` for `regex` and `sentencepiece` until a `[[tool.mypy.overrides]]` entry exists (Task 1 adds it).
- Hypothesis profile `settings.register_profile("rinchen", derandomize=True, deadline=None, max_examples=200)` works; `pytest-timeout` is installed.

## Review Focus

Inputs the spec implies but its tests do not cover; each has a test in the owning task.

1. **Empty / whitespace-only text** → `normalize` returns `""`, `split` returns `[]`, `script_of` returns `{}` (Tasks 1, 4, 6).
2. **Unknown language code** (typo `mn_mong`) → `ConfigError`, never a silent no-op that would strip traditional-script FVS (Tasks 4, 6).
3. **Tokens with no letters** (`12.5`, `—`) and letterless text must not hit a division by zero in homoglyph gating or `script_of` (Tasks 1, 4).
4. **Sentences without a terminator, with stacked terminators (`?!`), ellipsis, decimals, or a trailing abbreviation** → no empty parts, no lost text (Tasks 6, 7, 8).
5. **Degenerate SPM inputs**: missing/empty/unknown-stem corpus files and `temperature_weights` with `[]`, zero sizes, `T<=0` → a clear `ConfigError`/`DataError`/`ValueError` naming the culprit, not a sentencepiece crash (Tasks 3, 12, 13).

---

### Task 0: Preflight (no product code)

**Files:** none created; commits the already-approved spec and the earlier STATE/CLAUDE edits.

- [ ] **Step 1:** `uv sync --frozen --group dev` → exit 0. Then `uv run python -c "import sys; assert sys.version_info[:2]==(3,14)"` → no output.
- [ ] **Step 2:** If `git diff --summary` shows `mode change 100755 => 100644` for `scripts/build_pyonmttok/*.py`, run `chmod +x scripts/build_pyonmttok/*.py` (the index already records 755; this checkout lost the bit and ruff `EXE001` fails the gate). Re-check: `git diff --summary` prints nothing.
- [ ] **Step 3:** Baseline gate. Expected: `93 files already formatted`, `All checks passed!`, `Success: no issues found in 50 source files`, `161 passed, 6 deselected`.
- [ ] **Step 4:** Commit `D3: add approved design and plan; sync STATE/CLAUDE for D2 CI result`:
  `git add docs/superpowers/specs/2026-11-02-d3-normalization-sentencepiece-design.md docs/superpowers/plans/2026-11-06-d3-normalization-sentencepiece.md CLAUDE.md STATE.md`

---

### Task 1: Script detection

**Files:**
- Create: `src/rinchen/lang/__init__.py` (docstring only), `src/rinchen/lang/scripts.py`
- Modify: `pyproject.toml` (append mypy override below)
- Test: `tests/unit/test_lang_scripts.py`

**Interfaces:**
- Produces:
  ```python
  class Script(str, Enum):  # value == the `regex` \p{Script=...} name
      CYRILLIC = "Cyrillic"; MONGOLIAN = "Mongolian"; LATIN = "Latin"
      HAN = "Han"; HANGUL = "Hangul"; HIRAGANA = "Hiragana"; KATAKANA = "Katakana"
  def script_of(text: str) -> dict[Script, float]
  ```
  Ratio = letters (`\p{L}`) of that script ÷ **all** letters in `text`; digits, punctuation, marks, whitespace excluded. Letters of a script outside the enum (e.g. Greek) count in the denominator only, so the sum is ≤ 1.0 (== 1.0 when every letter is in the enum). No letters → `{}`. Only scripts with count > 0 appear as keys.

- [ ] **Step 1: Write failing tests** in `tests/unit/test_lang_scripts.py`:
  - `test_pure_script_is_ratio_one` — parametrized (script, text): CYRILLIC `"Монгол"`, MONGOLIAN `"\u1820\u1821\u1822"`, LATIN `"Mongol"`, HAN `"中文"`, HANGUL `"한국어"`, HIRAGANA `"ひらがな"`, KATAKANA `"カタカナ"` → `script_of(text) == {script: 1.0}`.
  - `test_mixed_text_gives_proportions`: `script_of("abМо") == {Script.LATIN: 0.5, Script.CYRILLIC: 0.5}`; `script_of("日本語のカタカナ") == {HAN: 0.375, HIRAGANA: 0.125, KATAKANA: 0.5}`.
  - `test_non_letters_are_ignored`: `script_of("12 ,.— Мо") == {Script.CYRILLIC: 1.0}`; `script_of("\u1820\u180b\u202f\u1803") == {Script.MONGOLIAN: 1.0}` (FVS, NNBSP, U+1803 not counted).
  - `test_letterless_text_returns_empty` — parametrized `""`, `"   "`, `"123 !?"` → `{}`.
  - `test_letters_of_unlisted_scripts_count_in_denominator`: `script_of("Мо" + "αβ") == {Script.CYRILLIC: 0.5}` (Greek letters are in the denominator only).
- [ ] **Step 2:** `uv run pytest tests/unit/test_lang_scripts.py -v` → FAIL (`ModuleNotFoundError: rinchen.lang`).
- [ ] **Step 3:** Implement `Script` and `script_of` in `src/rinchen/lang/scripts.py`: one compiled `regex` pattern per script (`\p{Script=X}`), iterate `\p{L}` matches once, classify each letter. Append to `pyproject.toml`:
  ```toml
  [[tool.mypy.overrides]]
  # Neither ships usable type information (no py.typed / stubs not installed).
  module = ["regex", "sentencepiece"]
  ignore_missing_imports = true
  ```
- [ ] **Step 4:** Tests pass; `uv run mypy src` → `Success`.
- [ ] **Step 5:** Commit `D3: add script_of Unicode script ratios`.

---

### Task 2: Language registry

**Files:**
- Create: `configs/languages.yaml`, `src/rinchen/config/languages.py`
- Test: `tests/unit/test_config_languages.py`

**Interfaces:**
- Consumes: `Script` (Task 1), `ConfigError`.
- Produces:
  ```python
  class LanguageEntry(BaseModel):  # extra="forbid", frozen=True
      code: str; scripts: list[Script]; direction_tag: str
  LANGUAGES_PATH: Path  # REPO_ROOT / "configs" / "languages.yaml"
  def load_languages(path: Path = LANGUAGES_PATH) -> list[LanguageEntry]  # ConfigError on any violation
  def known_language_codes() -> frozenset[str]  # @functools.cache over the default path
  ```
  Reuse `REPO_ROOT` from `rinchen.config.loader`; do not define a third copy.
  **Builder's call:** the spec says a single `script`; `jpn_Jpan` spans three scripts, so the field is `scripts: list[Script]` (non-empty).

`configs/languages.yaml` — top-level `languages:` list, this order, `direction_tag` always `<2{code}>`:

| code | scripts |
|---|---|
| khk_Cyrl | Cyrillic |
| mn_Mong | Mongolian |
| eng_Latn | Latin |
| rus_Cyrl | Cyrillic |
| zho_Hans | Han |
| kor_Hang | Hangul |
| jpn_Jpan | Han, Hiragana, Katakana |
| deu_Latn | Latin |
| kaz_Cyrl | Cyrillic |
| tur_Latn | Latin |

**Builder's call (spec says "9" but lists 10):** the registry has 10 entries (8 partners + `khk_Cyrl` + `mn_Mong`); `mn_Mong` is an output variant, not a partner, so directions = 8 partners × 2 = 16 and tags = 10.

- [ ] **Step 1: Write failing tests:** `test_default_registry_has_ten_entries_in_spec_order`; `test_direction_tag_is_2_plus_code` (all entries); `test_codes_and_tags_are_unique`; `test_sixteen_directions` (partners = codes other than `khk_Cyrl`/`mn_Mong` → `len == 8`, `2 * 8 == 16`); `test_known_language_codes_matches_registry`; `test_jpn_has_three_scripts`; `test_malformed_registry_raises_config_error` parametrized over YAML strings written to `tmp_path`: missing field, extra field, unknown script name, empty `scripts`, duplicate `code`, `direction_tag != "<2{code}>"`, top level a list, file missing.
- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement per the pattern of `src/rinchen/acquire/registry.py` (`yaml.safe_load`, `TypeAdapter`/model validation wrapped in `ConfigError(... ) from exc`, top-level mapping check). Duplicate-code and tag-equality checks are pydantic validators / a list-level check.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `D3: add 10-language registry (configs/languages.yaml)`.

---

### Task 3: Temperature sampling weights

**Files:**
- Create: `src/rinchen/utils/sampling.py`
- Test: `tests/unit/test_utils_sampling.py`

**Interfaces:**
- Produces: `def temperature_weights(sizes: Sequence[int], temperature: float) -> list[float]` — `size**T / Σ size**T`. `[]` → `[]`. A size of 0 gets weight `0.0`. `ValueError` for any negative size, `temperature <= 0`, or all sizes zero (message names the problem).

- [ ] **Step 1: Write failing tests:** `test_weights_sum_to_one` (`[10, 1000, 50]`, T 0.3, `pytest.approx(1.0)`); `test_temperature_one_is_proportional` (`[100, 300]` → `[0.25, 0.75]`); `test_low_temperature_gives_small_corpus_more_share` (`[10, 1000]`: `w(T=0.3)[0] > w(T=1.0)[0]`); `test_very_low_temperature_approaches_uniform` (`[10, 1000]`, T=0.01 → each within 0.02 of 0.5); `test_high_temperature_exaggerates` (`[10, 1000]`: `w(T=2)[0] < w(T=1)[0]`); `test_zero_size_gets_zero_weight` (`[0, 5, 5]` → `[0.0, 0.5, 0.5]`); `test_empty_returns_empty`; `test_invalid_inputs_raise_value_error` parametrized: `([1, -1], 0.3)`, `([1, 2], 0.0)`, `([1, 2], -0.5)`, `([0, 0], 0.3)`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the formula above.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `D3: add temperature_weights sampling utility`.

---

### Task 4: `normalize`

**Files:**
- Create: `src/rinchen/text/normalize.py`, `tests/property/__init__.py`
- Modify: `tests/conftest.py` (register hypothesis profile)
- Test: `tests/unit/test_text_normalize.py`, `tests/property/test_normalize_properties.py`

**Interfaces:**
- Consumes: `known_language_codes()` (Task 2), `ConfigError`.
- Produces: `def normalize(text: str, lang: str) -> str` — `ConfigError` if `lang` is not a registry code.

**Pipeline (pinned values).** *Builder's call:* the spec's order puts zero-width cleanup last, but then `"a \u200b b"` becomes `"a  b"` (two spaces), violating the spec's own idempotency property; and the Ү/Ө fold or homoglyph step can create a composable pair. So the order is: **1 NFC → 2 zero-width → 3 whitespace → 4 quotes → 5 homoglyph → 6 Ү/Ө fold → 7 NFC again.** Results are identical to the spec's order except for those cases.

| Step | Rule |
|---|---|
| 2 zero-width | Strip U+200B, U+200C, U+200D, U+2060, U+FEFF for every language. Also strip U+180B–U+180F **unless** `lang == "mn_Mong"` (there they are preserved verbatim). *Spec-literal:* ZWNJ/ZWJ are stripped in `mn_Mong` too. |
| 3 whitespace | Every whitespace run → one `" "`; strip leading/trailing `" "`. For `mn_Mong`, U+202F is **not** whitespace (survives, is never collapsed or stripped); elsewhere it collapses like any space. Tabs/newlines collapse too. |
| 4 quotes | `\u201c \u201d \u201e \u201f \u00ab \u00bb` (U+201C, 201D, 201E, 201F, 00AB, 00BB) → `"`; `\u2018 \u2019 \u201a \u201b` (U+2018, 2019, 201A, 201B) → `'`. All languages. |
| 5 homoglyph | Token = maximal run of non-space chars. If > 50% of the token's `str.isalpha()` chars are Cyrillic (`\p{Script=Cyrillic}`), map Latin `a e o p c x y A E O P C X Y` → Cyrillic `\u0430 \u0435 \u043e \u0440 \u0441 \u0445 \u0443 \u0410 \u0415 \u041e \u0420 \u0421 \u0425 \u0423` (U+0430, 0435, 043E, 0440, 0441, 0445, 0443, 0410, 0415, 041E, 0420, 0421, 0425, 0423). No alphabetic chars → untouched. *Builder's call:* uppercase variants included. |
| 6 fold | Ѳ U+0472 → Ө U+04E8; ѳ U+0473 → ө U+04E9. Nothing else (Ұ/ұ and Greek Θ/θ are never touched). |

- [ ] **Step 1: Write failing unit tests** in `tests/unit/test_text_normalize.py` (BUILD-OPERATE-CHECK, escapes keep invisible chars visible):
  - `test_nfc_composes`: `normalize("e\u0301", "eng_Latn") == "\u00e9"`.
  - `test_whitespace_collapses_and_strips`: `normalize("  a \t\n b\u00a0\u3000c  ", "eng_Latn") == "a b c"`.
  - `test_nnbsp_collapses_outside_mn_mong`: `normalize("a\u202fb", "eng_Latn") == "a b"`.
  - `test_nnbsp_preserved_in_mn_mong`: `s = "\u1820\u202f\u1821"`; `normalize(s, "mn_Mong") == s`.
  - `test_quotes_canonicalized`: `normalize("\u201cHi\u201d \u2018x\u2019 \u00abY\u00bb \u201eZ\u201f", "eng_Latn") == "\"Hi\" 'x' \"Y\" \"Z\""`.
  - `test_zero_width_stripped_without_double_space`: `normalize("a \u200b b", "eng_Latn") == "a b"`; `normalize("a\u200bb\ufeff\u2060", "eng_Latn") == "ab"`.
  - `test_fvs_preserved_only_in_mn_mong`: `s = "\u1820\u180b\u1821"`; `normalize(s, "mn_Mong") == s`; `normalize(s, "khk_Cyrl") == "\u1820\u1821"`.
  - `test_homoglyph_repairs_latin_inside_cyrillic_token`: `normalize("\u041co\u043d\u0433o\u043b", "khk_Cyrl") == "\u041c\u043e\u043d\u0433\u043e\u043b"` (Latin `o` at two positions).
  - `test_homoglyph_uppercase`: `normalize("P\u043e\u0441\u0441\u0438\u044f", "rus_Cyrl") == "\u0420\u043e\u0441\u0441\u0438\u044f"`.
  - `test_homoglyph_gated_by_majority` — parametrized `(text, expected)`: `("pop", "pop")`, `("\u041cop", "\u041cop")` (1/3 Cyrillic), `("\u041co", "\u041co")` (exactly 50% — unchanged), `("\u041c\u043eo", "\u041c\u043e\u043e")` (2/3 — fixed), `("\u0430\u04301o", "\u0430\u04301\u043e")` (digit not counted).
  - `test_fita_folds_to_o_umlaut_letters`: `normalize("\u0472\u0473", "khk_Cyrl") == "\u04e8\u04e9"`; `test_other_u_o_variants_untouched`: `"\u04ae\u04af\u04e8\u04e9"` and Kazakh `"\u04b0\u04b1"` unchanged.
  - `test_final_nfc_pass_keeps_idempotence`: `normalize("\u0472\u0308", "khk_Cyrl") == "\u04ea"`; `normalize("\u0443\u0443y\u0306", "rus_Cyrl") == "\u0443\u0443\u045e"`.
  - `test_mn_mong_letters_untouched`: all letters U+1820..U+1842 joined → unchanged for `"mn_Mong"`.
  - `test_empty_and_whitespace_only`: `normalize("", l) == ""`; `normalize(" \u00a0\t ", "eng_Latn") == ""`; `normalize("\u202f", "khk_Cyrl") == ""`; `normalize(" \u202f ", "mn_Mong") == "\u202f"`.
  - `test_letterless_tokens_do_not_crash`: `normalize("12.5 \u2014 !?", "khk_Cyrl") == "12.5 \u2014 !?"`.
  - `test_unknown_language_raises_config_error`: `pytest.raises(ConfigError)` for `"mn_mong"` and `""`.
- [ ] **Step 2: Write failing property tests** in `tests/property/test_normalize_properties.py`. In `tests/conftest.py` add `from hypothesis import settings; settings.register_profile("rinchen", derandomize=True, deadline=None, max_examples=200); settings.load_profile("rinchen")`. Strategy alphabet = `st.sampled_from` over: Latin `a e o p c x y A E O P C X Y b d`, Cyrillic `\u0430 \u043e \u04e8 \u04e9 \u04ae \u04af \u0472 \u04b1 \u043c \u043d`, digits, `. , ! ? \u2014`, space/tab/newline/U+00A0/U+202F, U+200B/200C/200D/FEFF, U+180B/180E/180F, quotes `\u201c \u201d \u2018 \u2019 \u00ab \u00bb`, combining U+0301/0306/0308, Mongolian letters U+1820–U+182F. Properties: (a) `test_idempotent` for `lang` drawn from `known_language_codes()`; (b) `test_mn_mong_letters_never_altered`: the subsequence of `\p{L}` chars in U+1800–U+18AF is identical before/after for `lang="mn_Mong"`; (c) `test_homoglyph_only_fires_in_cyrillic_majority_tokens`: for a token built only from Latin and Cyrillic letters with Cyrillic ≤ 50% of letters, `normalize(token, "eng_Latn") == token`.
- [ ] **Step 3:** Run both files → FAIL.
- [ ] **Step 4:** Implement `normalize` in `src/rinchen/text/normalize.py`: one private function per step, tables as module-level constants, `normalize` only sequences them. If a property test finds a counterexample, fix the implementation, not the property.
- [ ] **Step 5:** Run both files → PASS (property tests deterministic via the profile).
- [ ] **Step 6:** Commit `D3: add text.normalize with unit and property tests`.

---

### Task 5: Golden normalization file (STOP-gate part 1)

**Files:**
- Create: `tests/fixtures/golden/normalize_200.tsv`, `tests/golden/__init__.py`, `tests/golden/test_normalize_golden.py`

**File format.** UTF-8, LF, pure ASCII: every non-ASCII or control char is written `\uXXXX` / `\UXXXXXXXX`, a literal backslash is `\\`. Lines starting `#` or blank are ignored; each data line is `input<TAB>expected<TAB>lang` (real tabs). `# category: <name>` comments group lines. **200 data lines exactly**, quotas: `nfc` 15, `zero_width` 20, `whitespace` 30, `quotes` 25, `homoglyph` 40, `fold` 20, `mn_mong_preservation` 30, `passthrough` 20 (already-clean text, every one of the 10 languages at least once). Example rows:

```
e\u0301<TAB>\u00e9<TAB>eng_Latn
a \u200b b<TAB>a b<TAB>eng_Latn
\u00a0 \u041c\u043e\u043d\u0433\u043e\u043b\u3000<TAB>\u041c\u043e\u043d\u0433\u043e\u043b<TAB>khk_Cyrl
\u201c\u0421\u0430\u0439\u043d\u201d<TAB>"\u0421\u0430\u0439\u043d"<TAB>khk_Cyrl
\u041co\u043d\u0433o\u043b<TAB>\u041c\u043e\u043d\u0433\u043e\u043b<TAB>khk_Cyrl
\u0472\u0473<TAB>\u04e8\u04e9<TAB>khk_Cyrl
\u1820\u180b\u202f\u1821<TAB>\u1820\u180b\u202f\u1821<TAB>mn_Mong
```

**Authoring rule:** derive every `expected` by hand from the Task 4 rule table. Never run `normalize` and paste its output. Mongolian lines use real Cyrillic words (Монгол, улс, нийслэл, Улаанбаатар, өдөр, үнэ …) or traditional-script codepoints; no Latin transliteration.

- [ ] **Step 1: Write the golden test** `tests/golden/test_normalize_golden.py` with a small `_decode` helper (regex over `\\u`, `\\U`, `\\\\`), parametrized by data line, so a failure names the file line number: `test_golden_line` → `normalize(decode(input), lang) == decode(expected)`. Plus `test_golden_has_exactly_200_data_lines`; `test_every_category_meets_its_quota` (counts from the `# category:` headers); `test_langs_are_registry_codes`; `test_cyrillic_labeled_lines_are_cyrillic` (every `khk_Cyrl`/`rus_Cyrl`/`kaz_Cyrl` line whose expected output has letters has `script_of(expected)[Script.CYRILLIC] >= 0.5` — mechanical galig guard).
- [ ] **Step 2:** Author the 200 lines. Run `uv run pytest tests/golden -q`. Failures mean either a hand-derivation error or an implementation bug: re-derive from the Task 4 table and fix whichever is wrong; never edit `expected` merely to match output.
- [ ] **Step 3:** All 200 pass, zero tolerance.
- [ ] **Step 4:** Commit `D3: add 200-line golden normalization file and test`.

---

### Task 6: `split` — dispatcher, `mn_Mong`, CJK

**Files:**
- Create: `src/rinchen/split/__init__.py` (re-exports `split`, nothing else), `src/rinchen/split/rules.py`
- Test: `tests/unit/test_split_rules.py`

**Interfaces:**
- Consumes: `ConfigError`.
- Produces: `def split(text: str, lang: str) -> list[str]`. Contract: parts are stripped, non-empty, in order; concatenating them equals the input minus whitespace; empty/whitespace-only → `[]`. Family by language: Cyrillic = `khk_Cyrl rus_Cyrl kaz_Cyrl`; `mn_Mong`; CJK = `zho_Hans jpn_Jpan`; Latin = `eng_Latn deu_Latn tur_Latn kor_Hang`. Any other `lang` → `ConfigError`.
- Closing characters (kept with the sentence they close): `" ' » ” ’ ) ] ） 」 』`.

Rules in this task: **`mn_Mong`** splits after a run of `U+1803 U+1809 ? !` (plus closers), whitespace or not, if more text follows; U+1802 never splits. **CJK** splits after a run of `。！？` (plus closers `） 」 』 ” "`), whitespace or not.

- [ ] **Step 1: Write failing tests** (use `S1 = "\u1820\u1821"`, `S2 = "\u1822\u1823"` for traditional script):
  - `test_mn_mong_splits_on_full_stops_and_question`: `split(f"{S1}\u1803 {S2}\u1809 {S1}?", "mn_Mong") == [f"{S1}\u1803", f"{S2}\u1809", f"{S1}?"]`.
  - `test_mn_mong_comma_never_splits`: `split(f"{S1}\u1802 {S2}\u1803", "mn_Mong") == [f"{S1}\u1802 {S2}\u1803"]`.
  - `test_mn_mong_splits_without_following_space`: `split(f"{S1}\u1803{S2}\u1803", "mn_Mong") == [f"{S1}\u1803", f"{S2}\u1803"]`; stacked `?!`: `split(f"{S1}?! {S2}", "mn_Mong") == [f"{S1}?!", S2]`.
  - `test_cjk_splits_after_fullwidth_terminators`: `split("今天天气很好。你好吗？我很好！", "zho_Hans") == ["今天天气很好。", "你好吗？", "我很好！"]`; `split("今日は晴れです。明日は雨ですか？はい。", "jpn_Jpan") == ["今日は晴れです。", "明日は雨ですか？", "はい。"]`.
  - `test_cjk_closers_stay_with_their_sentence`: `split("他说：「你好。」然后走了。", "zho_Hans") == ["他说：「你好。」", "然后走了。"]`.
  - `test_no_terminator_returns_single_part` (parametrized over `mn_Mong`, `zho_Hans`): text without terminator → `[text]`.
  - `test_empty_and_whitespace_only_return_empty_list` (parametrized over `mn_Mong`, `zho_Hans`, `jpn_Jpan`; Task 7 extends it to all 10 registry codes): `split("", l) == []`, `split("  \n ", l) == []`.
  - `test_unknown_language_raises_config_error`: `"mn_mong"`, `""`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the dispatcher (a `dict[str, Callable[[str], list[str]]]` holding only the families built so far: `mn_Mong`, `zho_Hans`, `jpn_Jpan`) and the two simple families. A shared helper turns boundary offsets into stripped non-empty parts. Until Task 7, the Latin and Cyrillic language codes are simply absent from the dict and raise `ConfigError` like any unknown code.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `D3: add sentence splitter dispatcher with mn_Mong and CJK rules`.

---

### Task 7: `split` — Latin and Cyrillic families

**Files:**
- Modify: `src/rinchen/split/rules.py`, `tests/unit/test_split_rules.py`

**Rules (pinned).** A boundary candidate is a terminator run `[.!?…]+` + optional closers, then whitespace, then a start character. **Latin family** (`eng_Latn deu_Latn tur_Latn kor_Hang`): start = any letter (Hangul has no case). **Cyrillic family** (`khk_Cyrl rus_Cyrl kaz_Cyrl`): start = uppercase Cyrillic letter (incl. Ө Ү Ё), an opening quote (`" ' «`), or a dash (`— – -`). A candidate whose terminator is `.` is suppressed when the token before the dot (lowercased, final dot removed) is in the family's abbreviation set, or is a single uppercase letter (initial); for `deu_Latn` also when the token is all digits (ordinal, `3. Oktober`). Decimals/dates (`1.5`, `12.03.2024`) have no whitespace after the dot, so they never become candidates. `kor_Hang` and `tur_Latn` use the base rule with neither an abbreviation set nor the initial exemption; `eng_Latn`, `deu_Latn` and the Cyrillic family use both. Abbreviation sets are module-level `frozenset` constants (spec: no config file). Required minimum entries: en `mr mrs ms dr prof st vs etc e.g i.e fig inc`; de `z.b d.h u.a bzw usw dr prof nr str ca vgl`; Cyrillic `т.д т.е т.п др см ул г гг им проф тыс млн руб рис стр`. *Builder's call:* the single-initial exemption for `eng_Latn`/`deu_Latn` (the spec names initials only for Cyrillic). Add only entries you can confirm are real abbreviations.

- [ ] **Step 1: Write failing tests** (parametrized `(lang, text, expected)` so each row is one case):
  - eng: `("Hello world. How are you? Fine!")` → `["Hello world.", "How are you?", "Fine!"]`; `"Dr. Smith arrived. He sat."` → `["Dr. Smith arrived.", "He sat."]`; `"J. K. Rowling wrote it. Then left."` → `["J. K. Rowling wrote it.", "Then left."]`; `"Pi is 3.14 today. Ok."` → `["Pi is 3.14 today.", "Ok."]`; `"Wait… Then go."` → `["Wait…", "Then go."]`; `"Really?! Yes."` → `["Really?!", "Yes."]`; `'He said "stop." Then left.'` → `['He said "stop."', "Then left."]`; `"no terminator here"` → `["no terminator here"]`; `"See also Fig."` → `["See also Fig."]` (trailing abbreviation, nothing follows).
  - deu: `"Am 3. Oktober feiern wir. Das ist schön."` → `["Am 3. Oktober feiern wir.", "Das ist schön."]`; `"Das ist z.B. ein Test. Gut."` → `["Das ist z.B. ein Test.", "Gut."]`.
  - tur: `"Bugün hava güzel. Yarın yağmur yağacak."` → two parts. kor: `"오늘은 날씨가 좋습니다. 내일은 비가 옵니다."` → two parts.
  - khk_Cyrl: `"Монгол Улсын нийслэл Улаанбаатар хот юм. Энэ хот Туул голын хөндийд оршдог."` → two parts; `"Чи хаанаас ирсэн бэ? Би Монголоос ирсэн."` → two parts; `"Б. Баяр, Д. Дорж нар ирлээ. Дараа нь явав."` → `["Б. Баяр, Д. Дорж нар ирлээ.", "Дараа нь явав."]`; `"Үнэ 1.5 сая төгрөг. Огноо 12.03.2024 байна. Сайн."` → `["Үнэ 1.5 сая төгрөг.", "Огноо 12.03.2024 байна.", "Сайн."]`; `"Тэр ирлээ. — Сайн уу? — гэв."` → `["Тэр ирлээ.", "— Сайн уу?", "— гэв."]`; `'Би "за" гэв. "Явъя" гэв.'` → `['Би "за" гэв.', '"Явъя" гэв.']`; `"Тэр ирлээ... дараа нь явав."` → single part (lowercase after ellipsis).
  - rus_Cyrl: `"В 2020 г. Москва выросла. Это хорошо."` → `["В 2020 г. Москва выросла.", "Это хорошо."]`; `"Он живёт на ул. Ленина. Это далеко."` → `["Он живёт на ул. Ленина.", "Это далеко."]`.
  - kaz_Cyrl: `"Қазақстан — үлкен ел. Астана — оның астанасы."` → two parts.
  - Also: extend Task 6's empty/whitespace test to all 10 registry codes, and add `test_every_registry_language_has_a_split_family` (`split("x", code) == ["x"]` for every code in `known_language_codes()`), which fails loudly if a language is added to the registry without a splitter family.
- [ ] **Step 2:** Run → new tests FAIL.
- [ ] **Step 3:** Implement the two families in `rules.py` (regex candidate finder + a private `_is_suppressed(prefix_token, family)` predicate); register them in the dispatcher.
- [ ] **Step 4:** All `test_split_rules.py` tests pass.
- [ ] **Step 5:** Commit `D3: add Latin and Cyrillic sentence-splitting rules with abbreviation lists`.

---

### Task 8: Splitter property tests

**Files:**
- Create: `tests/property/test_split_properties.py`

- [ ] **Step 1: Write the property tests**, parametrized over one representative language per family (`khk_Cyrl`, `mn_Mong`, `zho_Hans`, `eng_Latn`), each with a strategy alphabet of that family's letters plus digits, space/newline, `. ! ? …`, the family's own terminators (U+1803, U+1809, `。！？`), closers, quotes and dashes: (a) `test_parts_reassemble_to_input_modulo_whitespace`: `"".join("".join(p.split()) for p in parts) == "".join(text.split())`; (b) `test_no_empty_or_unstripped_parts`; (c) `test_each_part_is_a_fixed_point`: `split(part, lang) == [part]` for every part; (d) `test_whitespace_only_text_gives_empty_list`.
- [ ] **Step 2:** Run. If hypothesis finds a counterexample, fix the rule in `rules.py` (and add the minimized example as a case in Task 7's parametrized table), never loosen the property.
- [ ] **Step 3:** Property tests pass.
- [ ] **Step 4:** Commit `D3: add splitter property tests across all four rule families`.

---

### Task 9: Reserved symbols

**Files:**
- Create: `src/rinchen/spm/__init__.py` (docstring only), `src/rinchen/spm/symbols.py`
- Test: `tests/unit/test_spm_symbols.py`

**Interfaces:**
- Consumes: `load_languages()` (Task 2).
- Produces: `def user_defined_symbols() -> list[str]` — the 10 registry `direction_tag`s in registry order, then `STRUCTURAL_SYMBOLS = ("<bt>", "<ft>", "<t>", "</t>", "<r>", "</r>", "<doc>", "<sep>")`, then `⟦0⟧`..`⟦99⟧` (`"⟦{i}⟧"`). Raises `ConfigError` if the composed list has duplicates (guards a future registry edit that re-adds a tag).

- [ ] **Step 1: Write failing tests:** `test_total_is_118`; `test_all_symbols_unique`; `test_language_tags_come_first_in_registry_order` (first 10 == `[e.direction_tag for e in load_languages()]`); `test_every_direction_tag_present` (each partner's `<2code>` and `<2khk_Cyrl>`, `<2mn_Mong>` appear exactly once); `test_structural_symbols_exact_and_ordered` (positions 10..17); `test_placeholders_zero_to_ninety_nine_only` (`⟦0⟧`, `⟦99⟧` present, `⟦100⟧` absent); `test_duplicate_symbols_raise_config_error` (monkeypatch `load_languages` to return a registry with a repeated entry).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `D3: add spm.symbols user_defined_symbols (118 reserved symbols)`.

---

### Task 10: SentencePiece configs

**Files:**
- Create: `src/rinchen/config/spm.py`, `configs/spm/tiny_1k.yaml`, `configs/spm/student_64k.yaml`, `configs/spm/teacher_128k.yaml`
- Delete: `configs/spm/.gitkeep`
- Test: `tests/unit/test_config_spm.py`

**Interfaces:**
- Produces:
  ```python
  class SpmConfig(BaseModel):  # extra="forbid", frozen=True; every field required
      vocab_size: int               # > 0
      input_sentence_size: int      # > 0
      character_coverage: float     # 0 < x <= 1
      temperature: float            # > 0
      seed: int
      corpus_dir: Path              # holds <lang_code>.txt files
      output_prefix: Path           # trained model is <output_prefix>.model
  def load_spm_config(path: Path) -> SpmConfig  # relative paths resolved against REPO_ROOT; ConfigError on any violation
  ```

| config | vocab_size | input_sentence_size | corpus_dir | output_prefix |
|---|---|---|---|---|
| tiny_1k | 1000 | 100000 | `tests/fixtures/spm/tiny_corpus` | `data/tiny/spm/tiny_1k` |
| student_64k | 64000 | 10000000 | `data/clean/spm_corpus` | `models/spm/student_64k` |
| teacher_128k | 128000 | 20000000 | `data/clean/spm_corpus` | `models/spm/teacher_128k` |

All three: `character_coverage: 0.9999`, `temperature: 0.3`, `seed: 1234`. The `input_sentence_size` values are the builder's call (master plan says only "per profile"). `converter_char.yaml` is D5's.

- [ ] **Step 1: Write failing tests:** `test_shipped_configs_load` (parametrized name → vocab size 1000 / 64000 / 128000; `corpus_dir` and `output_prefix` are absolute and under `REPO_ROOT`); `test_relative_paths_resolve_against_repo_root_and_absolute_stay`; `test_malformed_config_raises_config_error` parametrized over YAML strings: missing field, extra field, `vocab_size: 0`, `temperature: 0`, `character_coverage: 1.5`, top level a list; `test_missing_file_raises_config_error`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement and write the three YAMLs; remove `.gitkeep`.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `D3: add SpmConfig schema and tiny/student/teacher SPM configs`.

---

### Task 11: Tiny synthetic corpus fixture

**Files:**
- Create: `scripts/build_spm_fixture.py`, `tests/fixtures/spm/tiny_corpus/{khk_Cyrl,mn_Mong,eng_Latn,rus_Cyrl}.txt`
- Test: `tests/unit/test_spm_fixture.py`

**Interfaces:**
- Produces: `def build(out_dir: Path) -> None` in the script (importable: `pyproject.toml` already has `pythonpath = ["scripts"]`); `if __name__ == "__main__":` asserts Python 3.14 and writes to `tests/fixtures/spm/tiny_corpus` resolved from `__file__`.

**Generator (deterministic, `random.Random(1234)`, obviously synthetic pseudo-words, same style as `scripts/build_sample_warc_fixture.py`).** Words = 1–4 syllables, syllable = consonant+vowel from per-language inventories: khk_Cyrl `бвгдзклмнпрстхцчшй` × `аэиоуөүя`; rus_Cyrl `бвгджзклмнпрстфхцчшщ` × `аэиоуыяеёю`; eng_Latn `bcdfghjklmnprstvwz` × `aeiou`; mn_Mong two-letter pairs `chr(a)+chr(b)`, `a ∈ range(0x1820, 0x1842, 3)`, `b ∈ (0x1820, 0x1822, 0x1824, 0x1826)`. A line = 4–10 words + terminator (`.`, or U+1803 for mn_Mong); ~15% of non-`mn_Mong` lines get a digit token (`2024`, `3.5`) inserted so digits are in the vocabulary. Lines per file: khk_Cyrl 400, rus_Cyrl 200, eng_Latn 100, mn_Mong 100. Files: UTF-8, LF, NFC, no blank lines. Verified: ~375 such lines trained to exactly 1000 pieces; if Task 13 later fails with `Vocabulary size too high`, raise the word-inventory sizes or line counts here.

- [ ] **Step 1: Write failing tests:** `test_build_reproduces_committed_files` (build into `tmp_path`; bytes equal the committed files after `\r\n`→`\n`); `test_files_are_registry_languages` (stems ⊆ `known_language_codes()`); `test_each_file_is_script_pure` (sum of `script_of(text)` ratios over the language's registry `scripts` ≥ 0.99 — mechanical galig guard); `test_lines_are_nonblank_nfc_and_tab_free`; `test_files_contain_digits` (khk_Cyrl, rus_Cyrl, eng_Latn).
- [ ] **Step 2:** Run → FAIL (script and files missing).
- [ ] **Step 3:** Write the script, run `uv run python scripts/build_spm_fixture.py`, review the generated files.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `D3: add tiny synthetic SPM corpus fixture and its builder`.

---

### Task 12: Corpus discovery and balanced sampling

**Files:**
- Create: `src/rinchen/spm/corpus.py`
- Test: `tests/unit/test_spm_corpus.py`

**Interfaces:**
- Consumes: `known_language_codes()`, `temperature_weights()`, `ConfigError`, `DataError`.
- Produces:
  ```python
  def discover_corpus(corpus_dir: Path) -> dict[str, Path]
  def read_lines(path: Path) -> list[str]
  def sample_balanced(lines_by_lang: Mapping[str, list[str]], *, input_sentence_size: int, temperature: float, seed: int) -> list[str]
  ```
  `discover_corpus`: `{code: path}` for each `*.txt` in `corpus_dir` (other extensions ignored), sorted by code. `ConfigError` if the directory is missing or a `.txt` stem is not a registry code (message names the file); `DataError` if there are no `.txt` files or any file has no non-blank line (message names the language). `read_lines`: UTF-8, strips `\r\n`, skips blank lines. `sample_balanced`: if the total line count ≤ `input_sentence_size` return every line (mapping order, then file order); otherwise per-language quota `floor(w_i × input_sentence_size)` with `w = temperature_weights(line_counts, temperature)`, capped at the language's line count, unused quota **not** redistributed; each language sampled without replacement via `random.Random(seed)` and kept in original relative order.

- [ ] **Step 1: Write failing tests:** discovery — `test_discover_returns_sorted_code_to_path_ignoring_other_extensions`; `test_unknown_stem_raises_config_error_naming_file` (`eng_latn.txt`); `test_missing_directory_raises_config_error`; `test_no_txt_files_raises_data_error`; `test_blank_only_file_raises_data_error_naming_language`; `test_read_lines_strips_crlf_and_skips_blanks`. Sampling — `test_all_lines_returned_when_under_budget`; `test_quotas_follow_temperature` (sizes `[1000, 10]`, budget 100, T 0.3 → counts `[79, 20]`; the 10-line language's share is above its proportional share); `test_quota_capped_by_language_size_without_redistribution` (sizes `[1000, 5]`, budget 100 → `[83, 5]`); `test_same_seed_same_sample_different_seed_differs`; `test_sample_preserves_within_language_order`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `D3: add SPM corpus discovery and temperature-balanced sampling`.

---

### Task 13: `train_spm`

**Files:**
- Create: `src/rinchen/spm/train.py`
- Test: `tests/unit/test_spm_train.py`

**Interfaces:**
- Consumes: `SpmConfig`, `discover_corpus`, `read_lines`, `sample_balanced`, `user_defined_symbols`, `UpstreamError`.
- Produces: `def train_spm(cfg: SpmConfig) -> Path` — returns `Path(f"{cfg.output_prefix}.model")`. Steps: discover → read → sample → write `<output_prefix>.sampled.txt` (UTF-8, LF; `output_prefix.parent` created) → `sentencepiece.SentencePieceTrainer.train(...)` with `input`, `model_prefix`, `vocab_size`, `character_coverage`, `input_sentence_size` from config; module constants `unigram` / `byte_fallback=True` / `split_digits=True` / `normalization_rule_name="identity"`; `user_defined_symbols=user_defined_symbols()`, `shuffle_input_sentence=True`, `minloglevel=2`. A `RuntimeError` from sentencepiece becomes `UpstreamError(...) from exc`. Do not pass `hard_vocab_limit=False` (the tiny model must really have 1000 pieces) and do not pass `num_threads`.

- [ ] **Step 1: Write failing tests** with a `tiny_cfg(tmp_path)` fixture = `load_spm_config(<tiny_1k.yaml>).model_copy(update={"output_prefix": tmp_path / "tiny"})`: `test_train_tiny_returns_model_with_1000_pieces` (path ends `.model`, exists, `SentencePieceProcessor(model_file=...).get_piece_size() == 1000`); `test_sampled_training_file_written` (exists, line count ≤ `input_sentence_size`); `test_tiny_training_finishes_under_ten_seconds` (`time.perf_counter()` around `train_spm` `< 10.0`); `test_vocab_too_large_raises_upstream_error_with_cause` (`vocab_size=50000` → `UpstreamError`, `__cause__` is `RuntimeError`); `test_missing_corpus_dir_raises_config_error`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `D3: add train_spm`.

---

### Task 14: `check_spm`

**Files:**
- Create: `src/rinchen/spm/check.py`
- Test: `tests/unit/test_spm_check.py`

**Interfaces:**
- Consumes: `user_defined_symbols()`, `read_lines`, `sentencepiece.SentencePieceProcessor`.
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class LanguageCoverage:
      lang: str; pieces: int; fallback_pieces: int
      @property
      def coverage(self) -> float: ...  # 1 - fallback_pieces / pieces; 1.0 when pieces == 0
  @dataclass(frozen=True, slots=True)
  class SpmCheckReport:
      non_atomic_symbols: tuple[str, ...]; byte_fallback_on: bool; coverage: tuple[LanguageCoverage, ...]
      @property
      def ok(self) -> bool: ...  # no non-atomic symbols and byte_fallback_on
  def check_spm(model_path: Path, corpus_files: Mapping[str, Path], *, sample_lines: int = 1000) -> SpmCheckReport
  ```
  A symbol is atomic iff its `encode(..., out_type=str)` equals `[symbol]` or `["▁", symbol]` (default dummy prefix, see Verified Facts). Byte fallback is on iff all 256 `<0xNN>` pieces exist and `is_byte`. Coverage per language over its first `sample_lines` lines: a piece "falls back" if `is_unknown` or `is_byte`. Failed validation is a report value, not an exception.

- [ ] **Step 1: Write failing tests** (module-scoped fixture trains the tiny model once into `tmp_path_factory`): `test_tiny_model_passes_all_checks` (`report.ok`, `non_atomic_symbols == ()`, `byte_fallback_on`, four languages each `coverage >= 0.95`); `test_detects_byte_fallback_off` (train a variant straight through `sentencepiece.SentencePieceTrainer.train(byte_fallback=False, ...)` on the sampled file → `byte_fallback_on is False`, `not ok`); `test_detects_non_atomic_symbols` (model trained with no `user_defined_symbols` → `"<2eng_Latn>"` and `"⟦0⟧"` listed, `not ok`); `test_out_of_vocabulary_script_counts_as_fallback` (temp file of 中文 lines for `zho_Hans` → `coverage < 0.5`); `test_sample_lines_limits_work` (`sample_lines=5` → `pieces` smaller than with the default).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `D3: add check_spm symbol, byte-fallback and coverage validation`.

---

### Task 15: `rinchen spm` CLI

**Files:**
- Modify: `src/rinchen/cli/spm.py` (replace the stub docstring/body)
- Test: `tests/unit/test_cli_spm.py`

**Interfaces:**
- Produces: `rinchen spm train --config PATH` (prints `trained <model path>`), `rinchen spm check --model PATH --corpus-dir DIR [--sample-lines N]` (corpus files found via `discover_corpus`; prints one `<lang>\t<coverage>` line per language, `byte_fallback: on|off`, and each non-atomic symbol; exits **3** when `not report.ok`, the `DataError` code from the master plan). *Builder's call:* `--config` rather than `--profile` (SPM config names are `tiny_1k`, not profile names); no manifest is written in D3 (D1/D2 precedent; STATE.md notes it).

- [ ] **Step 1: Write failing tests** with `CliRunner`: `test_train_then_check_round_trip` (write a YAML into `tmp_path` that points `corpus_dir` at the committed fixture and `output_prefix` into `tmp_path`; `train` exits 0 and output contains `.model`; `check` on that model exits 0 and output contains `khk_Cyrl` and `byte_fallback: on`); `test_check_exits_three_for_a_model_without_symbols` (train a symbol-less model directly; exit code 3, output lists `<2eng_Latn>`). `tests/unit/test_cli_help.py` must still pass.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement both commands; presentation only, all logic stays in `spm/`.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `D3: add rinchen spm train and check commands`.

---

### Task 16: Runbook, records, STOP gate

**Files:**
- Create: `docs/runbooks/03_tokenizer.md`
- Modify: `DECISIONS.md` (append), `STATE.md`, `CLAUDE.md`

- [ ] **Step 1:** Write `docs/runbooks/03_tokenizer.md` (match `02_human_testsets.md`'s style): retraining each profile (`uv run rinchen spm train --config configs/spm/{tiny_1k,student_64k,teacher_128k}.yaml`, then `spm check`; student/teacher read `<lang_code>.txt` from `data/clean/spm_corpus` that later milestones produce); adding a language (registry entry → symbols regenerate automatically → add a `split` family mapping, golden-file lines, fixture/corpus file, retrain; existing models become invalid because the symbol set changes); the temperature formula and why `T=0.3` boosts small languages; the golden-update procedure (edit `normalize_200.tsv` by hand with diff review, re-derive from the rule table, keep exactly 200 lines and the category quotas; no `just golden-update` recipe exists in D3).
- [ ] **Step 2:** Append a dated `DECISIONS.md` entry recording: the 10-entry registry and `scripts` list; the reordered normalize pipeline and uppercase homoglyphs; ZWNJ/ZWJ stripped in `mn_Mong` (spec-literal, flagged for review against Mongolian shaping needs); FVS stripped outside `mn_Mong`; Fita fold table; abbreviation lists as module constants (master plan's `abbrev/*.txt` layout not used, per the approved spec); policy constants in `train.py`; default `add_dummy_prefix` (a tagged source encodes with a leading standalone `▁`; the check accepts it; revisit before D8); no manifest from `spm train`.
- [ ] **Step 3: STOP gate, run for real:** `uv run pytest tests/golden -q` → 200 golden lines pass; `time uv run rinchen spm train --config configs/spm/tiny_1k.yaml && uv run rinchen spm check --model data/tiny/spm/tiny_1k.model --corpus-dir tests/fixtures/spm/tiny_corpus` → real time under 10 s total, exit 0 (`data/tiny/spm/` is gitignored).
- [ ] **Step 4:** Full gate (four commands). Record the actual counts. Expected: format clean, `All checks passed!`, mypy `Success`, pytest all passed with 6 deselected.
- [ ] **Step 5:** Update `STATE.md` (D3 paragraph with evidence: commit range, gate counts, STOP-gate timings; `Next step: D4`) and `CLAUDE.md` (repo layout gains `lang`, `split`, `spm`, `tests/{property,golden}`, `configs/languages.yaml`, `scripts/build_spm_fixture.py`; "Current phase" → D3 done, next D4). State precisely what was and was not exercised (only `tiny_1k` trained; student/teacher configs validated only).
- [ ] **Step 6:** Commit `D3: add tokenizer runbook, record decisions, update STATE and CLAUDE`.
