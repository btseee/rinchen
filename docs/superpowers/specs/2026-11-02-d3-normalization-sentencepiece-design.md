# D3 — Normalization & SentencePiece: Design

**Status:** approved by user, 2026-11-02. See master plan
`docs/research/2026-10-29-rinchen-dev-agent-prompt.md` lines 345-357 for
D3's one-paragraph spec and STOP gate, and lines 261-267 for the B5
sentence-splitter spec it references. This document fills in the
implementation detail that one-liner leaves open.

## Goal

Build the text-normalization, sentence-splitting, script-detection, and
SentencePiece training/validation pipeline that every later milestone's
text handling depends on: `text/normalize.py`, `split/rules.py`,
`lang/scripts.py`, `spm/{train,symbols,check}.py`, a new 16-direction
language registry (`configs/languages.yaml`), and the 3 SentencePiece
training configs D3 owns (tiny/student/teacher — `converter_char.yaml` is
D5's to add later).

## Scope boundary: tiny profile only, no real corpus data yet

D1 shipped `configs/data/sources.yaml` deliberately empty (6 of 7
candidate real corpora failed a 100-sample quality review; Rinchen builds
its own corpus from scratch as a separate, ongoing effort — see
`DECISIONS.md`'s 2026-10-30 entry). D3 therefore has no real multi-language
text to train the `student_64k`/`teacher_128k` SentencePiece models on
yet. Those two configs are built and schema-validated as real,
structurally-complete YAML (so later milestones can use them once real
corpus data exists), but **only the `tiny_1k` config is actually exercised
end-to-end in D3**, trained on the new hand-authored fixture described
below. This mirrors D1's own precedent: build the machinery and prove it
works on synthetic fixtures now, populate it with real data in a later,
separate effort.

## Language registry: `configs/languages.yaml`

9 languages, derived directly from the corpus schema's own language-code
list (master plan lines 204-205): `khk_Cyrl` (the Mongolian hub),
`mn_Mong` (Mongolian traditional-script *output variant*, not a
translation partner — it shares `khk_Cyrl`'s content, just rendered in
traditional script via the `<2mn_Mong>` tag), `eng_Latn`, `rus_Cyrl`,
`zho_Hans`, `kor_Hang`, `jpn_Jpan`, `deu_Latn`, `kaz_Cyrl`, `tur_Latn`.

8 partner languages × 2 directions (into and out of `khk_Cyrl`) = the
master plan's "**16 directions**". Each direction only needs *one* tag
(the target language's tag, per the corpus schema's "target tag `<2xx>`
prepended to SOURCE" convention) — so 9 distinct `<2xx>` tags (one per
language entry, including `<2khk_Cyrl>`) cover all 16 directions; no
distinct tag set of size 16 is needed.

`src/rinchen/config/languages.py`: pydantic schema + loader, `ConfigError`
on malformed entries (matching D1's registry pattern). Each entry: `code`
(e.g. `eng_Latn`), `script` (one of `lang.scripts.Script`'s values),
`direction_tag` (e.g. `<2eng_Latn>`).

## `text/normalize.py`

`normalize(text: str, lang: str) -> str`, six steps in order:

1. **NFC** Unicode normalization.
2. **Whitespace**: collapse runs to a single space, strip leading/trailing,
   normalize Unicode space variants to a regular space — except U+202F
   NNBSP, which passes through untouched when `lang == "mn_Mong"` (the
   master plan's literal scoping; NNBSP in any other language's text, if
   it appears, collapses like ordinary whitespace).
3. **Quotes**: canonicalize curly/typographic quote variants (`" " ' ' «
   » „ "`) to straight ASCII `"` and `'` — standard MT-preprocessing
   convention, reduces vocabulary fragmentation.
4. **Cyrillic homoglyph fix**: Latin `a e o p c x y` → Cyrillic `а е о р с
   х у`, applied per-token, gated by: **a token is "Cyrillic-majority" if
   more than 50% of its alphabetic characters (digits/punctuation
   excluded from the count) are already Cyrillic.** This makes the step a
   structural no-op on non-Cyrillic text (an English loanword embedded in
   Mongolian text stays untouched), so it runs unconditionally with no
   per-language branching needed.
5. **Mongolian Ү/ү Ө/ө canonical forms**: fold any confusable/decomposed
   Unicode variants of these four letters to their single canonical
   precomposed codepoints. Also self-gating (only matches
   Mongolian-Cyrillic-specific codepoints), runs unconditionally.
6. **Zero-width cleanup**: strip U+200B (ZWSP), U+200C/U+200D
   (ZWNJ/ZWJ) — except U+180B–U+180F (Mongolian Free Variation
   Selectors), preserved verbatim when `lang == "mn_Mong"`.

Property tests (hypothesis): idempotent (`normalize(normalize(x)) ==
normalize(x)`); never alters a `mn_Mong`-lang text's U+1800–U+18AF
*letters* themselves (FVS/NNBSP preserved, letters untouched); the
homoglyph fix never fires on a token whose alphabetic majority isn't
Cyrillic.

Plus a **200-line golden file** (`tests/fixtures/golden/normalize_200.tsv`
— `input\texpected_output\tlang` per line): every line's
`normalize(input, lang) == expected_output`, byte-exact, zero tolerance
(matches `CLAUDE.md`'s "numeric acceptance thresholds are gates" and
D1/D2's zero-tolerance precedent for fixture correctness — this is also
D3's literal STOP-gate requirement).

## `split/rules.py` (B5)

`split(text: str, lang: str) -> list[str]`, dispatching to 4 rule
families:

- **Cyrillic** (`khk_Cyrl`, `rus_Cyrl`, `kaz_Cyrl`): split on `[.!?…]` +
  space + uppercase Cyrillic/quote/dash, with small hand-authored
  abbreviation lists, single-initial exceptions (`А. Б.`), and
  number-pattern exceptions (`1.5`, `12.03.2024`).
- **`mn_Mong`**: split on U+1803 (᠃) and U+1809 (᠉) and `?!`; U+1802 (᠂
  comma) never triggers a split.
- **CJK** (`zho_Hans`, `jpn_Jpan`): split after `。！？` and full-width
  closers `）」』"`.
- **Latin** (`eng_Latn`, `deu_Latn`, `tur_Latn`, and **`kor_Hang`** —
  Korean folded in here rather than CJK, since Korean text conventionally
  uses ASCII `.!?` punctuation, not the CJK full-width characters that
  rule actually matches; the master plan's B5 spec doesn't assign Korean
  anywhere explicitly, this is this design's resolution): split on ASCII
  `[.!?…]` + space + letter, with abbreviation lists for `en`/`de`
  (German additionally exempts ordinals like `3. Oktober`); `kor_Hang`
  and `tur_Latn` use the base rule with no abbreviation-list exception.

Abbreviation lists are small, hand-authored, module-level constants in
`rules.py` — no separate config file, matching YAGNI for a short, rarely
changing list.

Property tests (hypothesis): `"".join(split(text, lang)) == text` modulo
whitespace; no empty parts; idempotent when applied to an already-single
sentence. Run across all 4 rule families.

## `lang/scripts.py`

`script_of(text: str) -> dict[Script, float]`, implemented via the
`regex` module's Unicode `\p{Script=...}` property matching (already a
pinned dependency — stdlib `unicodedata` doesn't expose script property
directly). Counts letters per script among *all* letter characters in the
text (digits/punctuation/whitespace excluded from both numerator and
denominator), returns ratios summing to 1.0. Returns `{}` for text with no
letters at all.

`Script` enum: `CYRILLIC`, `MONGOLIAN` (the traditional U+1800–U+18AF
block), `LATIN`, `HAN`, `HANGUL`, `HIRAGANA`, `KATAKANA` — kept as the
real distinct Unicode script properties rather than merging
Hiragana/Katakana/Han into a composite "Japanese" category; a later
milestone's LID step can combine them for language-level inference,
`script_of` itself stays script-level.

## `utils/sampling.py`

`temperature_weights(sizes: list[int], temperature: float) -> list[float]`
— a shared utility, since the master plan's D7 ablation milestone needs
the identical "weights computed from corpus sizes with temperature T"
math later; building it once now avoids D7 re-deriving or duplicating it.

**Formula** (the XLM-R/mBERT "exponential smoothing" convention — chosen
over the inverse softmax-style convention because it's the one where
`temperature < 1.0` gives smaller corpora *more* relative weight than
proportional sampling, matching the master plan's literal "balanced... T
0.3" framing):

```text
weight_i = size_i ** temperature / sum(size_j ** temperature for all j)
```

`temperature = 1.0` → pure proportional-to-size sampling. `temperature →
0` → approaches uniform weighting regardless of size. `temperature > 1.0`
→ exaggerates size disparity (not used by D3, included for D7's future
flexibility). D3 calls this with `temperature = 0.3` per the master plan.

Unit tests: weights sum to 1.0; a low-count language's weight under T=0.3
is strictly greater than its weight would be under T=1.0 (proportional) —
this is the concrete, testable definition of "balanced."

## SentencePiece pipeline

- **`spm/symbols.py`** — `user_defined_symbols() -> list[str]`: one
  `<2xx>` tag per `languages.yaml` entry (9, covering all 16 directions),
  plus `<2mn_Mong>`, plus the 8 structural tags `<bt> <ft> <t> </t> <r>
  </r> <doc> <sep>`, plus 100 placeholder tokens `⟦0⟧..⟦99⟧` — **118
  symbols total**. (The placeholder tokens are reserved here only — the actual
  masking engine that uses them, `infer/placeholders.py`, is a later
  milestone; D3's job is just to make sure the tokenizer treats
  `⟦0⟧`..`⟦99⟧` as atomic single pieces.)
- **`spm/train.py`** — `train_spm(cfg) -> Path`: reads per-language
  corpus files, computes `temperature_weights` (T=0.3) from each
  language's line count, samples lines proportionally up to
  `input_sentence_size`, writes a combined training file, calls
  `sentencepiece.SentencePieceTrainer.train(...)` with `model_type=unigram`,
  `character_coverage=0.9999`, `byte_fallback=True`, `split_digits=True`,
  `normalization_rule_name=identity` (the project normalizes itself via
  `text/normalize.py` before SPM ever sees the text), and the 118
  `user_defined_symbols`; returns the trained `.model` path.
- **`spm/check.py`** — validates: every one of the 118 user-defined
  symbols encodes as exactly one piece; `byte_fallback` is on in the
  trained model; prints a per-language coverage report (share of each
  language's sample text that doesn't fall back to `<unk>`/byte-fallback).
- **`configs/spm/{tiny_1k,student_64k,teacher_128k}.yaml`**:
  pydantic-validated (`ConfigError` on malformed, matching D1's pattern),
  `vocab_size` 1000/64000/128000 respectively, everything else shared
  across the three. `configs/spm/converter_char.yaml` is explicitly out of
  scope — D5 (Cyrillic↔traditional converter) adds it later.

The exact `sentencepiece==0.2.1` Python API (the real `train()` call
signature, how to check a model's `byte_fallback` flag, how to confirm a
symbol tokenizes to exactly one piece) will be empirically pre-verified
during the implementation-plan phase, against the real installed library,
before being written into the plan — same practice as D1 and D2.

## New fixture: SPM training corpus

A new, small, hand-authored, obviously-synthetic plain-text fixture
(distinct from D1's acquisition-pipeline fixtures, which are too small and
serve a different purpose) mixing `khk_Cyrl`, `mn_Mong`, and at least one
or two of the 8 partner languages — enough real running text to train a
meaningful tiny SentencePiece model on, while staying fast (<10s) and
clearly synthetic per this project's established fixture convention (D1's
QA review, D2's galig-text correction).

## Testing strategy

- `normalize`: hypothesis property tests (idempotent, never alters
  `mn_Mong` letters, homoglyph fix self-gated correctly) + the 200-line
  golden file (byte-exact, zero tolerance).
- `split`: hypothesis property tests (join-equals-original modulo
  whitespace, no empty parts, idempotent on a single sentence) across all
  4 rule families.
- `script_of`: unit tests per script category plus a mixed-script
  fixture.
- `temperature_weights`: unit tests confirming weights sum to 1.0 and
  that `T=0.3` gives a low-count language more share than `T=1.0`
  (proportional) would.
- `spm/check.py`: tested against the tiny 1k model trained on the new
  fixture — all 118 symbols single-piece, `byte_fallback` on, coverage
  report runs without error.

## STOP gate

Verbatim from the master plan: golden normalization file (200 lines)
passes; tiny SentencePiece model trained on the new fixture and checked,
in under 10 seconds.

## Runbook (`docs/runbooks/03_tokenizer.md`)

Covers: how to retrain each profile's (`tiny`/`student`/`teacher`) SPM
model; how to add a new language to `languages.yaml` and what downstream
symbol/config regeneration that requires; the temperature-sampling
rationale and formula; the golden-file update procedure (what to do when
a legitimate normalization change requires updating expected outputs).
