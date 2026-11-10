# 12 - Robustness

## Scope

D12 measures how a system of ours treats the inputs that break translation quietly: numbers, dates and amounts; the numbered placeholders (`⟦0⟧` to `⟦99⟧`); names; glossary
terms; the script of the output; and a source written in the traditional Mongolian script. It builds six metrics as pure functions of (source, reference, hypothesis) triples
(`rinchen.eval.robust_*`); the suite file that carries the triples and their annotations; a runner that asks an engine for the hypotheses or reads them from a file; the result
`robustness.json` with one outcome per row; the command `rinchen eval robustness` with its manifest; a narrow `report.json` whose `robustness` block is what D13's and D15's reports embed;
a deterministic toy that a tiny model is trained on in 190 s so that the metrics are measured on a real binary; and the gate. The design, with the rulings T1 to T20, the amendment
C1 and a section of the amendments made while building (A1 to A21), is `docs/superpowers/specs/2026-11-10-d12-robustness-design.md`; the decisions are condensed in the entries
"2026-11-10: D12 robustness: the rulings" and "2026-11-10: D12 review - corrections and fixes" of `DECISIONS.md`; the upstream facts are in `docs/VERIFY.md` (section "D12 findings"). The
facts about the engine, the toy and the metrics come from a spike made before the design (`.superpowers/sdd/2026-11-10-d12-robustness/spike-findings.md`; Eole 0.6.4,
CTranslate2 4.8.4, sacrebleu 2.6.0, CPU, toy data).

**What D12 is.** An instrument, tested on files and on one real tiny binary. Given a suite and either an export of ours (the engine translates) or a file of hypotheses (any system's
output enters through it), it says how many numbers, placeholders, names and glossary terms survived, how pure the script of the output is and how much of the Cyrillic quality a
model keeps when the Khalkha source is written in traditional script. It measures: it has no threshold, no pass and no fail, and it exits 0 whenever it measured. The floors are the
gate's.

**What D12 is NOT.** Nothing below is a statement about the quality of any translation.

- **No mechanism.** D12 builds no placeholder masking or unmasking, no terminology engine (constrained decoding or replacement), no `terminology` field in the service's request and no
  training row that teaches a model to copy a placeholder or use a term marker. The engine, the service and the reserved-symbol rule are untouched; the request schema keeps refusing
  `terminology`. This amends D11's spec (T22: "the request schema gains `terminology` there (D12)") and the sentences of runbook 11 that said D12 owns them: nobody owns the mechanisms
  after D12 (spec section 6, item 1). What D12 leaves is the instrument that tells a later builder whether a mechanism works.
- **No GPU was used.** The machine has a GeForce RTX 5070 Ti that the owner did not authorise for a run. Every Eole process, the converter and every CTranslate2 translator of D12 (the
  spike, the tests, this runbook's measurements) ran with `CUDA_VISIBLE_DEVICES=""`, `gpu_ranks: []` and `device="cpu"`; `auto` is refused; `--device cuda` exists for the operator, is
  validated and was never run. Everything GPU-side (CTranslate2 on a device, `float16`, `int8_float16`) is **UNMEASURED**. No container was started and no network was used.
- **No real model, corpus, glossary or reference.** The toy is templated (37 nouns in a few frames, numbers in seven kinds, ten placeholder indices, twenty names), so every number tests
  machinery and ranks nothing. The conventions table, the 4-letter case suffix, the currency symbols and the 0.98 purity floor are priors.
- **No fine-tuning.** The toy's model is trained from random initialisation, as every Rinchen model; an external model enters only through a hypotheses file, as a system under test.
- **Departures from the master plan** (`eval/robustness.py`, a robustness block of four keys in `rinchen.eval/1`): the work is split over `rinchen.eval.robust_*` (see "Architecture"); the
  block has SIX keys (amendment C1: `entity_acc` and `mong_source_retention` are added to the master plan's four, because its own D12 item lists the two suites); no `result.json`, no
  runner over many systems, no significance, no model card (D13); no generic suite builder from real parallel text.

Tags used below: **measured** (run for this runbook on 2026-11-10 with the code at `ae4e816`; the drivers are read-only scripts in `scratchpad/d12-t9/`: `train.py`, `measure.py`,
`vectors.py`, `toy_digest.py`, `example/make.py`), **gate** (printed by `tests/integration/test_robustness_gate.py`, in the run of the fix wave or of Task 8), **spike** (run in the
design spike on the CPU), **source-read** (read from the code, never run), **ARITHMETIC** (a formula applied to values that were measured), **UNMEASURED** (nothing was run). Every
number is from a test, the code, the spike, the ledger of the build (`.superpowers/sdd/2026-11-10-d12-robustness/progress.md`) or a measurement made for this runbook ("Measured
numbers"); a number that was not obtained is not given. A TIME moves by a third from one run to the next on this shared machine (load average 1 to 5 while these were taken, 58 to 130
during the fix wave's gate); a count or a score does not.

What was checked: about 1,765 new tests (13,543 passed after the fix wave against 11,778 at the end of D11; unit, property, golden and integration), each new test mutation-checked by hand
during the build (63, 68, 57, 93, 122, 68 + 5, 93 and six gate mutants for Tasks 1 to 8, with the first-pass survivors each given a test or an analysis; 13 + 18 + 12 for the fix
wave), differential fuzzes against independent references (300,000 random cases of the numbers metric against a brute-force unit-aware matching, 0 mismatches; the placeholder,
span and script metrics against their specifications, 0 mismatches), the real-binary gate (two modules, 31 tests), a combined review of Tasks 1 to 8 (`37634d1..b2466ef`, one
independent read-only reviewer: ready with fixes, 0 Critical, 4 Important and 9 Minor findings), a fix wave of five commits (`b2466ef..ae4e816`) with the full gate at its end (13,543 passed,
1 skipped, 10 deselected, run in five parallel parts), and the real binary for this runbook. **A scoped re-review of the fix wave (`b2466ef..ae4e816`) was RUNNING when this runbook was
written and its verdict is NOT recorded here; a follow-up edit adds it** (see "The review, its fix wave and the re-review").

## Architecture

```
suite file (JSON)  --Suite.from_bytes-->  requests  --Engine.translate (ONE call)-->  hypotheses  --compute_robustness-->  RobustnessResult
                                                    \--hypotheses file (JSONL)--/                                          |
                                                                                            robustness.json, report.json, hypotheses.jsonl, manifest.json (LAST)
```

| Module | Job | Lines |
|---|---|---:|
| `rinchen.text.numerals` | `fold_numerals`: every Unicode decimal digit to ASCII, every run of CJK numerals to its digits (moved out of D4's `filter/numbers.py`, behaviour identical) | 69 |
| `rinchen.eval.robust_numbers` | `numbers_in`, `score_numbers`, `numbers_agree`, the conventions, the 64-token cap | 325 |
| `rinchen.eval.robust_placeholders` | `placeholders_in`, `damaged_brackets`, `score_placeholders`, `placeholders_agree` | 98 |
| `rinchen.eval.robust_spans` | `span_present`, `EntityTable`, `score_entities`, `score_terms` (the `regex` module) | 115 |
| `rinchen.eval.robust_script` | `target_scripts`, `score_script` (`infer.script_mask` purity) | 66 |
| `rinchen.eval.robust_mong` | `score_mong`: the arms, the retention, the chrF signature | 88 |
| `rinchen.eval.robust_suite`, `robust_suite_rules` | the suite file: strict reader and writer, `Request`, `suite_requests` | 348 + 407 |
| `rinchen.eval.robust_hypotheses` | `Hypothesis`, the JSONL file, `run_requests` (one call), `Translates` | 275 |
| `rinchen.eval.robust_outcomes`, `robust_sections` | `RowOutcome`, `SectionResult`; the scoring of each section | 211 + 375 |
| `rinchen.eval.robustness` | `compute_robustness`, `RobustnessResult`, `robustness_block`, `robustness_coverage` | 255 |
| `rinchen.eval.robust_report` | `report.json` (`rinchen.report.robustness/1`) | 70 |
| `rinchen.cli.eval_robustness`, `_plan`, `_output` | the command: the order of the steps, the plan of the options, the lines and the manifest | 398 + 119 + 209 |
| tests only: `tests/support/robust_{toy,toy_text,suites,gate,golden,cli,suite_documents}.py`, `tests/fixtures/robustness/` | the toy, the suites, the gate stages, the golden files | |

`rinchen.eval` imports nothing of `train`, `synth`, `serve`, `cli`, `annotate`, `script_conv`, `filter` and none of `ctranslate2`, `sentencepiece`, `eole` or `torch` (an AST scan:
`tests/unit/test_eval_layering.py`); `import rinchen.eval.robustness` takes 0.164 s (**measured**) because it loads no model and no polars. The command may import `script_conv.runtime` (the
converter) and `cli.train_common`. The only edits to existing source are `src/rinchen/filter/numbers.py` (the folding moved, D4 unchanged) and `src/rinchen/cli/eval.py` (the stub
registers the command); the only edits to existing tests are additive session fixtures in `tests/integration/conftest.py`, the extension of `tests/support/error_messages.py` and of the
layering and message scans, and the race fix of one D8 test (see "The review, its fix wave and the re-review").

## What the six metrics mean

Each value is a share, a mean or a ratio over the units that were SCORED. A unit that cannot be scored is EXCLUDED and counted under a reason, never dropped and never counted as a pass.
A value is `null` (shown `n/a`, never 0 or 1) when the suite has no such section or nothing in it was scored. Read a value together with its `coverage` (see "Reading the results").

| Key | Unit | Headline | The question it answers |
|---|---|---|---|
| `numbers_acc` | row | exact rows over scored rows | did every number of the reference survive, by value? |
| `placeholder_acc` | row | exact rows over scored rows | did every `⟦n⟧` of the source survive, once each, undamaged? |
| `entity_acc` | mention | names found over names expected | is each name the row expects present in the target's spelling? |
| `term_acc` | mention | terms found over terms expected | is each glossary target term present when its source term is? |
| `script_purity` | row | mean purity over scored rows | is the output in the script of its target? |
| `mong_source_retention` | corpus | chrF of the traditional-script arm over chrF of the Cyrillic arm | how much of the quality survives when the source is written in traditional script? |

### `numbers_acc`: numbers by value (spec T2 to T4)

A number is read by its VALUE, whatever convention wrote it: `1,250.50`, `1 250,50` and `1250.5` are one number; `7529` is not `7592` (D4's digit multiset cannot tell them apart; the
15 of the toy model's 67 wrong rows that it accepts are transpositions and lost decimal points, **spike**). **Measured** with `scratchpad/d12-t9/vectors.py` (the code at `ae4e816`):

| Text (language) | Numbers read |
|---|---|
| `1,250.50` (English), `1 250,50` (Khalkha) | 1250.50, both |
| `1,250` (English / Russian / Khalkha) | 1250 / 1.250 / either, flagged `ambiguous` |
| `1.250` (English / Russian) | 1.250 / 1250 |
| `12.03.2024` (Russian), `03/12/2024` (English), `2024-03-12` | the three components 12, 3 and 2024, in the order written (`:`, `/` and `-` always split, and so does a repeated `.` or `,` that is not a grouping of threes) |
| `10:30`; `10.30`; `5-6`; `-5` | 10 and 30; the decimal 10.30 (not a time); 5 and 6; minus 5 |
| `1,500₮`, `₮1500`, `1 500 ₮` (Khalkha) | 1500 with the unit `₮` (the first also reads as 1.500: `ambiguous`) |
| `25%`; `COVID-19`; `version 1.2.3` | 25 (`%` is not read); 19; 1, 2 and 3 |
| `1e5`; `.5`; `three`; `⟦3⟧ and 5` | 1 and 5; 5; none; 5 (a placeholder is not a number) |

The conventions table (T4) decides `1,250` and `1.250`: English, Chinese, Japanese and Korean write `.` for the decimal; Russian, German, Turkish and Kazakh write `,`; any other code
(Khalkha, traditional Mongolian, an unknown language) is mixed and the token carries both readings, flagged. The table is a prior (Mongolian text mixes both in practice and no corpus
statistic exists); the flag makes the exposure visible (`ambiguous_rows`), and the one false accept it allows is a reading the source itself allows.

A row of the `numbers` section is SCORED when the reference has at least one number and the source's numbers (read under `src_lang`) match the reference's one for one; otherwise it is
excluded (`no_numbers`, `source_reference_disagree`). The expected numbers are the REFERENCE's (what a human wrote in the target language, read under `tgt_lang`). The reference's and the
hypothesis's numbers are paired one to one by a maximum bipartite matching on shared readings (so no assignment is missed, and order never matters: `5 and 7` equals `7 and 5`, and a date
may change its layout). The row is `exact` when every reference number found a partner, no hypothesis number is left over and the units are compatible. A missing, extra or changed number,
a lost decimal point or a transposed digit fails the row. **The unit rule** (a fix of the review, A3): a currency symbol (`₮ $ € £ ¥ ₽ ₩`, with at most one space between) belongs to ONE
number, the one that took it after itself (so `5 $7` reads as `5 $` and `7`); a reference number that states a unit needs a hypothesis number of the same unit and one without a unit
accepts any; `units_ok` holds when SOME maximum matching gives every reference number a unit-compatible partner, so a text scored against itself is always exact even when one value
appears twice with different units (`5 and 5 ₮`: before the fix it failed; 39,906 of 300,000 random self-comparisons did). **Measured** on the finished code: `7529` against `7592`, `72,5`
against `725`: not exact; `5 and 7` against `7 and 5`: exact; `The price is 1,250.50 dollars.` against `Price 1 250,50`: exact (and not `surface_exact`); `It costs 1,500₮.` against
`It costs 1,500.`: the number is found but `units_ok` is false, so the row fails (`unit_failures`); `It costs 20%.` against `It costs 20.`: exact (`%` is not measured).

Beside the value, `details` holds `recall` (matched over expected tokens), `extra_tokens`, `surface_exact_rows`, `unit_failures`, `ambiguous_rows` and `capped_rows`. At most 65 numbers are
read per text (the scan stops there; a runaway hypothesis of 100,000 numbers was read in 0.011 s against 0.31 s for a full scan: the build's own measurement, from the ledger), the first 64 are matched and the rest count as extra. **Limits, on purpose:** number
words (`three`, `гурав`) and magnitude words (`сая`, `million`) are not read; `1e5` is 1 and 5; `.5` is 5; `10.30` is a decimal, not a time; digits glued to letters count (`COVID-19`,
`A4`); an ordinal (`3-р`) is its digit; `%` is not measured, so a dropped `%` is invisible (the toy's `percent` category scores 120 of 120 whatever the model writes for the sign);
`surface_exact_rows` compares the sorted multiset of the numbers as written, so it sees grouping spaces and decimal marks but NOT the layout of a date (`03/12/2024` and `12.03.2024` both
give the surfaces 12, 03 and 2024); and `numbers_acc` is value preservation, not format (the format question has no headline).

### `placeholder_acc`: `⟦n⟧` survive (spec T5)

A placeholder is the glyph pair of `spm.symbols` around one or two digits with no leading zero, index 0 to 99: exactly the hundred reserved symbols. A row of the `placeholders` section is
scored when its source holds at least one and its reference holds the same multiset (`no_placeholders`, `source_reference_disagree` otherwise). The hypothesis is judged as the engine wrote it:
it is `exact` when its multiset of indices equals the source's (each one present exactly as often as in the source, none foreign) and no glyph of the pair is damaged (`⟦ 3 ⟧`, `⟦3`,
`⟦03⟧`, `⟦100⟧`: **source-read**; **measured**: `⟦ 3 ⟧ ⟦3` damages 3 glyphs). Nothing is repaired: a policy that appended a lost placeholder would make every row pass and measure nothing. The ORDER is a detail
(`order_ok_rows`: exact rows whose order equals the reference's) and never part of the value, because order is legitimately free across languages. `details`: `missing`, `extra`, `damaged`,
`exact_rows`, `order_ok_rows`. What a placeholder stands for (a number, a URL, a tag, a name) is irrelevant: the metric only asks that the token survives once.

**What it shows about real models.** A model copies a placeholder only for the indices it was trained on: the toy trained on `⟦0⟧` to `⟦9⟧` copies those and 0 of 7 of `⟦10⟧` and up in each of the
four directions probed (**measured**: 0 of 28); the D11 teacher, trained with no placeholder, copies 0 of 600 rows. No training row of D6 to D10 holds a placeholder (`mixture.load_training_rows`,
**source-read**), so no real model copies one; a production mix that uses `⟦0⟧` to `⟦99⟧` has to train all 100 indices.

### `entity_acc`: names (spec T6)

There is no NER. An entity is an entry of the suite's ENTITY TABLE (`id -> language -> accepted spellings`), and a row of the `entities` section lists the ids it expects in the hypothesis,
written in the row's `tgt_lang`. A spelling is PRESENT in a text when, after NFC and casefolding of both, it occurs at a boundary (no letter, mark or digit directly before it; a run of spaces
inside matches any run of whitespace) followed by at most 4 letters or marks (a case suffix) and then a non-letter. The unit is a MENTION (a row with two names counts two); the row outcome
is `ok` when all its mentions are found (`details.rows_all_found`). Nothing is transliterated: a spelling is what the table lists. **Measured:** `Улаанбаатарт` matches `Улаанбаатар` (suffix
`т`); `Улаанбаатарынхаа` (5 letters) does not; `Сбат` does not match `Бат`. **Over-acceptance, recorded:** a short name is a prefix of a longer one and the hyphen is a boundary, so
`Батор` and `Сухэ-Батор` match `Бат` and `Ulan Bator` matches `Bat` (**measured**; at most 2 of the 480 mentions of the main suite, by the review's count).

### `term_acc`: glossary terms (spec T7), and what it does NOT measure

A row of the `terms` section lists glossary pairs `(src, tgt)`. A mention counts when the source term is present in the row's source (otherwise it is excluded as `absent_in_source`
and counted), and is found when the target term is present in the hypothesis (the same rule as the names). **With no mechanism in the system, `term_acc` is natural AGREEMENT with a glossary,
not terminology ADHERENCE** (review I-3): the glossary of the main suite is the toy's own noun table, so any model that translates the nouns passes, including the D11 teacher, which scores
1,041 of 1,044 on the main suite (**measured**) although it never saw a digit or a placeholder. The metric can fail (a glossary of words the model never writes scores 0 of 1,044, the control
suite) but no control shows a model failing because it translated a noun differently from the glossary. A real adherence metric needs entries whose required target DIFFERS from the model's
default, the unconstrained baseline scored beside the constrained run (the difference is the effect of the mechanism), entries whose target equals the source flagged (a source-echoing system
is credited for them), and real morphology instead of the 4-letter suffix. D12 builds none of that; the baseline is what a mechanism's builder will need.

### `script_purity`: the output is in the script of its target (spec T8)

The unit is a row of the `script` section `(source, src_lang or null, tgt_lang, script)`; `script` is `Mong`, `Cyrl` or `null` and is allowed only when `tgt_lang` is `khk_Cyrl` or
`mn_Mong`, as the engine's request is. The scripts the output must be in are those the registry gives the target, except that a row that names a script is judged against that script's
language (`Mong` against `mn_Mong`). The row's purity is `infer.script_mask.script_purity` (the share of its counted letters in them); a text with NO counted letter (empty, only digits,
punctuation or placeholders) has no purity and is EXCLUDED as `no_letters` (measured: `score_script("", "khk_Cyrl")` and a text of only digits and punctuation give `None`), unlike
`script_purity`, which gives it 1.0: a robustness figure must not count silence as pure. **So an empty or `no_output` hypothesis leaves the mean untouched and raises `excluded`: read
`coverage.script_purity.excluded` beside the mean.** The headline is the MEAN purity over the scored rows; `details`: `mean`, `min`, `pure_share` and `below_floor` (rows under the floor 0.98,
D11's prior, inclusive: a text exactly on it passes), `no_letters`, `floor`. The suite's rows are requests that provoke a leak: the trained directions, five pairs the model never trained
(`src=None`) and, in a separate suite, the converter's traditional-script output.

### `mong_source_retention`: reading the traditional script (spec T9)

A row of the `mong` section has the Khalkha Cyrillic `source`, the same sentence in traditional script `source_mong` and a `reference` in `tgt_lang`. The runner makes up to three requests
per row, the ARMS: `cyrl` (the control), `mong` (the traditional source as `mn_Mong`: the model reads the script itself) and, only when a converter is attached, `converted` (D5's
`ScriptConverter` turns the traditional source into Cyrillic first). The headline is `chrF(mong) / chrF(cyrl)`, both corpus chrF against the references (sacrebleu defaults), NOT clipped,
`null` when the Cyrillic chrF is 0. **It can exceed 1** (the direct arm beat the control: 1.0030 at beam 1, **measured**) **and it means nothing when the Cyrillic chrF is near zero**: two poor
scores give a ratio near 1 (the toy's hypotheses all rotated give 0.9978 over chrF 5.96 and 5.97, **measured**), which is why the report carries both chrF scores. `details`: `chrf_cyrl`,
`chrf_mong`, `chrf_signature` (sacrebleu's own, e.g. `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`), `agreement` (rows whose two arms wrote the same text), `chrf_converted`,
`retention_converted` (`null` without a converter), `rows`. The row outcome carries `ok` (the two arms agree), `value` (the sentence chrF of the `mong` arm) and `value_cyrl` (the sentence chrF
of the `cyrl` arm), so a resample of rows can recompute the ratio. A converted arm measures the converter as much as the model (the D5 tiny converter restores 5 of 37 toy nouns, **spike**), which is
why it is optional and reported beside the direct one. Whether a production model should read the script itself or be fed a conversion is a product decision the instrument informs and
does not make.

## The suite file

A JSON document, schema `rinchen.robustness.suite/1`, UTF-8, sorted keys, two-space indent, one final LF (`Suite.to_bytes`): `{schema, name, entities: {id: {language: [spellings]}}, sections:
{section: [row]}}` with up to six sections (`numbers`, `placeholders`, `entities`, `terms`, `script`, `mong`), at least one non-empty. Every row has exactly these fields (a missing one and an
unknown one are both refused, **measured** with `scratchpad/d12-t9/example/probe.py`):

| Field | Sections | Rule |
|---|---|---|
| `id` | all | `[A-Za-z0-9._-]{1,64}`, unique within its section |
| `src_lang`, `tgt_lang` | all | a code of the language registry; `src_lang` may be `null` only in `script` |
| `source` | all | not blank, at most 2,000 characters, no lone surrogate, no reserved symbol of the project other than a numbered placeholder |
| `reference` | all | a non-blank text in `numbers`, `placeholders`, `mong`; present and either a text or `null` in `entities`, `terms`, `script` |
| `category` | all | `[a-z0-9_]{1,24}`: the breakdown of `by_category` |
| `entities` | `entities` | a non-empty list of ids of the table, each with a spelling for the row's `tgt_lang` |
| `terms` | `terms` | a non-empty list of `{src, tgt}` (each at most 200 characters) |
| `script` | `script` | `"Mong"`, `"Cyrl"` or `null` |
| `source_mong` | `mong` | the `source` written in traditional script (with `src_lang` `khk_Cyrl`) |

The reader is as strict as `ModelRecord.from_bytes`: a repeated key, a duplicate id, a text of more than 2,000 characters, a reserved symbol (the engine would refuse the whole batch) and a
file of more than 8 MiB (checked before the parse) are refused with a `DataError` that names the section, the row index, the id and the field and NEVER quotes a text; the entity table's ids
are `[a-z0-9_]{1,32}` with at most 8 spellings per language. **A worked example** (all five sections but `mong`, whose rows hold `source_mong` in traditional script: see
`tests/fixtures/robustness/golden_suite.json`). This is the suite exactly as the scratch driver `scratchpad/d12-t9/example/make.py` wrote it, rows shown one per line; it parses and scores
(**measured**):

```
{"schema": "rinchen.robustness.suite/1", "name": "worked-example",
 "entities": {"bat": {"eng_Latn": ["Bat"], "khk_Cyrl": ["Бат"]},
              "ulaanbaatar": {"eng_Latn": ["Ulaanbaatar", "Ulan Bator"], "khk_Cyrl": ["Улаанбаатар"]}},
 "sections": {
  "numbers": [
   {"id": "n1", "category": "grouped", "src_lang": "eng_Latn", "tgt_lang": "khk_Cyrl", "source": "The price is 1,250.50 dollars.", "reference": "Үнэ 1 250,50 доллар."},
   {"id": "n2", "category": "date", "src_lang": "eng_Latn", "tgt_lang": "rus_Cyrl", "source": "Meeting on 03/12/2024 at 10:30.", "reference": "Встреча 12.03.2024 в 10:30."}],
  "placeholders": [
   {"id": "p1", "category": "noun", "src_lang": "eng_Latn", "tgt_lang": "khk_Cyrl", "source": "Where is ⟦3⟧?", "reference": "⟦3⟧ хаана байна вэ?"}],
  "entities": [
   {"id": "e1", "category": "person_city", "src_lang": "eng_Latn", "tgt_lang": "khk_Cyrl", "source": "Bat lives in Ulaanbaatar.", "reference": "Бат Улаанбаатарт амьдардаг.", "entities": ["bat", "ulaanbaatar"]}],
  "terms": [
   {"id": "t1", "category": "glossary", "src_lang": "eng_Latn", "tgt_lang": "khk_Cyrl", "source": "Where is the table?", "reference": "Ширээ хаана байна вэ?", "terms": [{"src": "table", "tgt": "ширээ"}]}],
  "script": [
   {"id": "s1", "category": "trained", "src_lang": "eng_Latn", "tgt_lang": "khk_Cyrl", "source": "This is a book.", "reference": null, "script": null}]}}
```

The hypotheses a system wrote for it are a JSON Lines file, one object per REQUEST with the keys `section`, `id`, `arm` (`main`; `cyrl`, `mong`, `converted` for `mong` rows), `status` (the engine's
`ok`, `empty`, `too_long`, `truncated`, `no_output`; it may be left out and then reads `ok`) and `text`, sorted keys, texts written as they are. Here one answer loses a digit group (`1 520,50`) and
one date changes layout (`03/12/2024`, the reference has `12.03.2024`):

```
{"arm": "main", "id": "n1", "section": "numbers", "status": "ok", "text": "Үнэ 1 520,50 доллар."}
{"arm": "main", "id": "n2", "section": "numbers", "status": "ok", "text": "Встреча 03/12/2024 в 10:30."}
{"arm": "main", "id": "p1", "section": "placeholders", "status": "ok", "text": "⟦3⟧ хаана байна вэ?"}
{"arm": "main", "id": "e1", "section": "entities", "status": "ok", "text": "Бат Улаанбаатарт амьдардаг."}
{"arm": "main", "id": "t1", "section": "terms", "status": "ok", "text": "Ширээ хаана байна вэ?"}
{"arm": "main", "id": "s1", "section": "script", "status": "ok", "text": "Энэ бол ном."}
```

The file must hold exactly the suite's requests, each once: a missing request, an unknown one and a repeated one are refused (`DataError`, exit 3) naming the section, the id and the arm. A file
holds at most 128 MiB (an operator cap; the spec named none). `tests/fixtures/robustness/` holds the only committed files: `golden_suite.json` (33 requests with known answers for every metric,
8,234 bytes, SHA-256 `2767112d...75b0`), `golden_hypotheses.jsonl` (3,345 bytes, `78bb56fc...07a8`) and `golden_robustness.json` (their scoring, 13,614 bytes, `80d08ddf...c87c`). The golden suite holds
two invisible characters as DATA: U+180B and U+180E in two traditional-script sources (real orthography). Every other suite is generated at test time from the toy.

## The commands

```
rinchen eval robustness --suite FILE (--model DIR | --hypotheses FILE) --out-dir DIR
    [--system NAME] [--beam-size 4] [--max-decoding-length 256] [--script-mask retry|always|off]
    [--device cpu|cuda] [--threads 1] [--script-converter-ct2 DIR --script-converter-spm FILE]
```

`--model` (or `RINCHEN_MODEL`) is an export `rinchen export convert` wrote; the engine translates every request in ONE call. `--hypotheses` reads the texts from a file (a model run writes
`hypotheses.jsonl`, which is such a file; a baseline system's output enters through the same file). Exactly one of the two; `RINCHEN_MODEL` set together with `--hypotheses` counts as both and is
refused. `--system` names the system (default: the export's name, or the stem of the hypotheses file). The converter pair attaches D5's converter (both options or neither, only with `--model`): the
engine writes the traditional script through it and every `mong` row gains the `converted` arm. `--device auto` is refused (typer's own message, exit 2), `--device cuda` is validated and was never run.

**The order of the steps** (the pattern of `rinchen export parity`): the options are checked; for `--model` the export is vouched for by `verify_export` (a directory our manifest does not vouch for
is exit 2 and nothing is written); the suite, the hypotheses file and the converter's files are HASHED before any of them is read, and the suite is parsed from the bytes that were hashed; an
output directory that is an input's directory or holds another stage's manifest is refused; the converter and the engine open, the engine's `model.bin` and manifest digests must be the ones the
command hashed (else exit 2); the engine is closed on every path; only then is anything written: the old manifest and results first, then `hypotheses.jsonl` (with `--model`), `robustness.json`,
`report.json` and, LAST, `manifest.json` (stage `eval_robustness`: every input with its SHA-256, the files written, the non-null values of the six keys, `rows.<section>`, `scored.<section>` and
`excluded.<section>`, `status` `robustness: ok`, `mode`, `script_mask`, the four tool versions, seed -1, `config_sha256` of the options). No text of a suite or a hypothesis is ever in a message.

**Measured** (the real binary, CPU, GPU hidden, one thread; `scratchpad/d12-t9/measure.py`). The golden files through `--hypotheses`, 0.88 s, and the output is the golden file byte for byte:

```
$ rinchen eval robustness --suite golden_suite.json --hypotheses golden_hypotheses.jsonl --system golden --out-dir <O>
suite: golden
system: golden
numbers: 0.5000 (3/6), excluded 2
placeholders: 0.4000 (2/5), excluded 1
entities: 0.5714 (4/7 mentions)
terms: 0.7500 (3/4 mentions), excluded 1
script: 0.7500 (mean of 5 rows), below floor 2, excluded 1
mong: 0.5018 (retention over 3 rows, chrF 50.18 over 100.00 Cyrillic)
result: <O>/robustness.json
report: <O>/report.json
manifest: <O>/manifest.json
```

The worked example above (0.79 s): `numbers: 0.5000 (1/2), excluded 0`, `placeholders: 1.0000 (1/1), excluded 0`, `entities: 1.0000 (2/2 mentions)`, `terms: 1.0000 (1/1 mentions), excluded 0`,
`script: 1.0000 (mean of 1 rows), below floor 0, excluded 0`; there is no `mong:` line because the suite has no such section, and `mong_source_retention` is `null` in the report. The real model, main
suite, defaults (beam 4, retry, float32; 2.10 s, 4,060 requests):

```
suite: robust-main
system: rinchen-e28d15664126
numbers: 0.9292 (892/960), excluded 0
placeholders: 0.9983 (599/600), excluded 0
entities: 0.9313 (447/480 mentions)
terms: 1.0000 (1044/1044 mentions), excluded 0
script: 1.0000 (mean of 900 rows), below floor 0, excluded 0
mong: 0.9976 (retention over 480 rows, chrF 99.49 over 99.73 Cyrillic)
```

A `null` value prints `n/a (no scored row)` in place of the figure and keeps the rest of the line (the exclusions stay visible). The engine's one `compute_type_effective` JSON line goes to stderr.

**The controls** (each is a command whose result must be far below the main run's; the gate runs all of them, "The STOP gate"). A hypotheses file rotated inside each (section, arm) group (every answer
is another row's): `PYTHONPATH=. uv run python -c "from pathlib import Path; from tests.support.robust_gate import rotate_hypotheses; rotate_hypotheses(Path('r1/hypotheses.jsonl'),
Path('rotated.jsonl'), 7)"` and then `--hypotheses rotated.jsonl` (add `groups={("mong", "mong")}` to move only the traditional-script arm). Any other model: `--model <its export>` over the same suite. A
glossary the model never writes and the placeholder indices it never trained: the two control suites of `tests/support/robust_suites.py`. The script mask: `--script-mask off`. A source echo as the
hypotheses: the hypotheses file whose `text` is the request's own text.

## Reading the results

`robustness.json` (schema `rinchen.robustness/1`; sorted keys, two-space indent, LF, no timestamp and no text of the suite or of a hypothesis: the same inputs give the same bytes) holds
`{schema, system, suite, robustness, sections, rows}`:

- `system`: `name`, `model_sha256` and `manifest_sha256` (the engine's; `null` for a hypotheses file), `hypotheses_sha256` (the digest of the hypotheses file that was scored; `null` for a model run,
  whose `hypotheses.jsonl` is an output) and `options` (`beam_size`, `converter`, `device`, `max_decoding_length`, `script_mask`, `threads`; `null` for a hypotheses file).
- `suite`: `name`, `sha256` (of the bytes that were parsed) and `rows` per section.
- `robustness`: the six values, each a float or `null`.
- `sections.<section>`: `unit` (`row`, or `mention` for `entities` and `terms`), `rows`, `scored`, `excluded` (`{reason: n}`; a reason that did not occur is absent), `ok`, `value`, `details`
  (as above), `by_category` and `by_direction` (`{scored, ok}`; `<src>-<tgt>`, `und` for a null source language) and `statuses` (the engine's status counts over EVERY hypothesis of the section, the
  excluded rows' included: a failure the engine flagged itself stays visible beside the metric).
- `rows.<section>`: one outcome per row in file order: `{id, category, direction, scored, reason, ok, expected, found, value, value_cyrl}` (the fields that do not apply are `null`; `value` is the purity of a script row
  and the sentence chrF of a `mong` row's direct arm, `value_cyrl` the sentence chrF of its control arm). These are what D13's paired bootstrap resamples.

`report.json` (schema `rinchen.report.robustness/1`, the file the STOP gate reads) holds `{schema, system: {name, model_sha256, manifest_sha256}, suite: {name, sha256}, robustness: {six values},
coverage, robustness_file, robustness_sha256}`. **`coverage` is the denominator of each value** (review I-2): for each of the six keys `{scored, excluded, total}` in the unit of its section (rows;
mentions for the names and the terms), and for `mong_source_retention` also `chrf_cyrl` and `chrf_mong`. A section the suite does not have has all three at 0. **Measured, the main run:**

```
"robustness": {"entity_acc": 0.93125, "mong_source_retention": 0.9976032236777934, "numbers_acc": 0.9291666666666667, "placeholder_acc": 0.9983333333333333,
               "script_purity": 1.0, "term_acc": 1.0}
"coverage":   {"entity_acc": {"scored": 480, "excluded": 0, "total": 480}, "numbers_acc": {"scored": 960, "excluded": 0, "total": 960},
               "placeholder_acc": {"scored": 600, ...}, "script_purity": {"scored": 900, ...}, "term_acc": {"scored": 1044, ...},
               "mong_source_retention": {"scored": 480, "excluded": 0, "total": 480, "chrf_cyrl": 99.7334611210574, "chrf_mong": 99.49442232291074}}
```

Why it matters: a `numbers_acc` of 1.0 over 10 scored rows of a suite that excluded 990 looks like a perfect run; `script_purity` rises when silence is excluded; and a retention near 1 over two near-zero chrF
scores is a useless control. Read `coverage` first. `manifest.json` repeats the non-null values and holds `scored.<section>` and `excluded.<section>` beside `rows.<section>`; the golden manifest
is 2,212 bytes, a model run's 4,118 (`inputs`: the suite, the five files of the export and its manifest, the converter's files when attached; `outputs`: the files written, with `rows` for
`hypotheses.jsonl`). Two runs give equal `robustness.json`, `report.json` and `hypotheses.jsonl` (**measured**) and manifests that differ only in `created` (and, here, in the output paths, because the two runs used two directories).

**Reading the main run** (**measured**). `numbers_acc` 892 of 960: the 68 failing rows are 63 of `two_numbers` (57 of 120 exact), 3 of `currency` (117 of 120: the model drops the `₮` in three rows, which the unit rule
counts), 2 of `grouped` (118 of 120); `int`, `decimal`, `percent`, `date` and `time` are 120 of 120. `placeholder_acc` 599 of 600: one `mn_Mong` row loses a placeholder at beam 4 (beam 1 loses none: D15's
"`placeholder_acc` 1.0" depends on the decode). `entity_acc` 447 of 480 mentions (208 of 240 rows with both names found): real errors of a small model (another person's name, `Bat` for `Bold`; a doubled letter, `Ulaaanbaatar`; **measured** on the first rows that fail). `script_purity` 1.0 over 900
rows. `mong_source_retention` 0.9976 (chrF 99.49 over 99.73; 467 of 480 texts equal).

## Adding a suite

1. **Write the rows** (a builder in code is the way: `tests/support/robust_suites.py` is the model, and `rinchen.eval.robust_suite.write_suite` writes the exact bytes). Take a `reference` that carries the numbers
   and the placeholders the source carries (a row whose reference disagrees with its source is excluded, not scored), list every name in the entity table with ALL the spellings a good output may use in
   each target language, and give a glossary term only when the target is meant to be a choice.
2. **Check self-consistency before anything else**: score the references AS the hypotheses (write a hypotheses file whose `text` is each request's reference; the `mong` arms and the `cyrl` arm both the
   reference). A sound suite scores 1.0 on the numbers, placeholders, names and terms with NO exclusion; `tests/unit/test_support_robust_suites.py` does exactly this for the toy's suites. A suite that
   excludes rows or scores below 1.0 against its own references measures the suite, not the system.
3. **Run it with a control**: the same suite with the rotated hypotheses must collapse (numbers, placeholders, terms near 0); a suite that does not collapse cannot fail.
4. **Keep the suite small if it is committed** (the repository's fixtures stay under 2 MB, a single fixture under 50 KB); generate large suites at test time.

D12 builds no generic builder from real parallel text (extract the numbers a source and its reference share, mask them with placeholders, list the capitalised tokens both sides keep): on any set
without an entity table or a glossary `entity_acc` and `term_acc` are `null` (spec section 6, item 4; D13 decides).

## What to do when a floor is missed

The command has no floor: it measures. The floors belong to the gate, and they are literals of `tests/integration/test_robustness_gate.py` that are never lowered. If a gate figure falls under its floor:
**raise `ROBUST_TEACHER_STEPS`** (`tests/support/robust_gate.py`; training is deterministic here, so the answer is exact), never lower the floor or the ceiling of a control. Before that, find out what moved:
read `coverage` (did rows leave the denominator?), `statuses` (did the engine flag rows?), `rows.<section>` (which rows failed, in which category and direction) and compare with the controls (a figure that
does not collapse under rotation is a suite problem). A change of the toy's generator, of the D7 nouns or of D5's lexicon moves every number: the digest test fails first ("The toy"). For a real system, the
thresholds belong to whoever decides what the system must do: put them in the report that embeds the block, not in this command.

## The toy

`tests/support/robust_toy.py` generates, deterministically from seed 7, a corpus of 29,352 rows over the D7 nouns and templates: base sentences; seven kinds of number (integer, decimal, grouped, percent,
currency `₮`, date in a different layout per language: English `03/12/2024`, Khalkha `2024.03.12`, Russian `12.03.2024`; time) and a two-number template; placeholder sentences (the base templates with
the nouns replaced by `⟦0⟧` to `⟦9⟧`, plus a genitive template that reverses two placeholders between English and Mongolian); ten persons and ten cities with one or two accepted English spellings; and
`mn_Mong` source rows (`origin: converted`, the Khalkha side written by D5's `rules.transliterate`). Number formats: English `1,250.50`, Khalkha and Russian `1 250,50` with a plain space. It is the
PORT of the spike's generator, exact: **measured** 29,352 rows, SHA-256 `4136d80ada4c416934080eff1ba32da5ae7398bdfe3c4115ecf56296c511bad1`, train 25,520 (base 9,000, entity 1,200, number 8,320, number2
2,800, placeholder 4,200), dev 1,432 and test 2,400 (base 600, entity 240, number 840, number2 120, placeholder 600), built in 0.144 s, two builds equal row for row.

**Why the digest is pinned.** The generator's whole contract is the order of its random draws (one `random.Random(7)`, in the order of `_draw_items`), the D7 nouns and templates and D5's lexicon (the
traditional-script strings are `rules.transliterate` of the Khalkha sentences). Change any of them and every gate figure moves with no sign of why; `tests/unit/test_support_robust_toy.py` pins the SHA-256
and the counts so that the change fails loudly in one place. If it fails: STOP and find the change; a new digest is a decision (re-measure the gate and record it), not an update. **To regenerate and
train** (nothing is written under the repository: give an empty directory outside it; this is the driver of the runbook's measurement, `scratchpad/d12-t9/train.py`):

```
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run python - <<'PY'
import sys; assert sys.version_info[:2] == (3, 14)
from pathlib import Path
from tests.support.robust_gate import export_robust, robust_workspace, train_robust_teacher, write_suites
root = Path("<an empty directory>")
workspace = robust_workspace(root / "ws")                       # the toy's corpus and SentencePiece model
teacher = train_robust_teacher(workspace)                       # about 190 s, 6,000 steps, from random initialisation
export = export_robust(workspace, teacher, root / "export").directory
suites = write_suites(root / "suites")                          # main, term_control, unseen_placeholders, mong_output
PY
```

or `uv run pytest tests/integration/test_robustness_gate.py -m subprocess` (the session fixture trains once). **Measured** in this run: workspace 0.37 s, training (`train ablate --prepare-only`, `train launch`
6,000 steps with the last two saves averaged, `train average`) 186.9 s, export 3.07 s, suites 0.19 s, 190.9 s of wall time for the script; `model.bin` 2,080,071 bytes, SHA-256
`e28d15664126d66e40a389ea256889f2965dd64920358cae84232bac7c91a6a6`, **byte-identical to the design spike's** (and to the gate's earlier retraining): training is deterministic on this machine, so the
gate's figures are exact, not statistical. This is not a claim that it is bit-for-bit on another CPU: the gate RECORDS the digest and does not assert it. The model is the tiny teacher of the D10 and D11
gates (2/2 layers, d64, `SinusoidalInterleaved`, the direction tag in the source) trained from random initialisation on the six directions `eng-khk`, `khk-eng`, `khk-rus`, `rus-khk`, `mn_Mong-eng`,
`mn_Mong-rus`; the SentencePiece model has 800 pieces (`train_spm` accepts 600 to 1,100 for this text and fails at 1,400); 3,000 steps are not enough for dates, two-number rows and entities, 6,000 are.
The vocabulary of the export (**measured**: 520 entries) holds the ten digits, `₮` and the 100 placeholders because the training data holds the first two (the placeholders are in every export: D9's invariant); the D11 teacher's vocabulary (344 entries) holds no digit and no `₮` (see `docs/VERIFY.md`, "D12 findings"). The main suite and the three
control suites are built from the toy by `tests/support/robust_suites.py` (main 1,034,294 bytes, SHA-256 `92702feb...a037`; term control 203,026; unseen placeholders 6,577; traditional-script output 25,155).

## Exit codes

| Code | Meaning | **Measured** here (or by the unit tests with fakes) |
|---:|---|---|
| 0 | it measured (also with every value `null`) | all the runs above |
| 2 | a bad option; both or neither of `--model` and `--hypotheses`; an export not vouched for; `--device auto`; the converter options without `--model` or only one of them; a bad `--system`; an `--out-dir` that is an input's directory; a target or a script the model cannot write; the engine loaded another file than the one hashed | a flipped byte in `model.bin`; `--device auto`; neither; both; `--hypotheses` with the converter pair; one converter option; `--system "a b"`; `--out-dir` the suite's directory; the traditional-script output suite without a converter (nothing written, the output directory absent) |
| 3 | a suite or a hypotheses file that is missing, unreadable or not one; another stage's manifest in `--out-dir` | an unknown key in a suite (`section 'entities', row index 0 (id 'ent-0001'): has a field that is not part of the format`, no text quoted); a missing suite; a hypotheses file lacking one request (`section 'mong', id 'mong-0480', arm 'mong'`); a foreign manifest (refused, not erased) |
| 4 | CTranslate2 failed | fakes only: UNMEASURED with a real failing CTranslate2 |
| 1 | the output cannot be written (no manifest is left) | fakes only |
| 130 | interrupted (a run that stops before its first write leaves the last run's files) | fakes only |

## Costs

**Measured**, one process, CPU, one intra thread, a quiet machine (load 1 to 5), `scratchpad/d12-t9/measure.py`; 3,580 suite rows, 4,060 requests (the `mong` rows make two requests each):

| What | Value |
|---|---|
| `compute_robustness` over the main suite (best of 5) | 0.1615 s: 0.040 s per 1,000 requests (0.045 s per 1,000 suite rows); with `to_bytes` 0.214 s; `read_suite` 0.032 s; `read_hypotheses` 0.024 s |
| the engine, 4,060 requests in one call, float32 | beam 1 0.448 s (0.110 s per 1,000 rows); beam 4 0.937 s (0.231 s per 1,000 rows); `Engine.open` of the 2 MB export 0.015 s |
| the command, wall time (Python and typer start-up included) | a refusal 0.77 to 0.81 s; the golden files 0.88 s; `--hypotheses` over the main suite 1.03 to 1.08 s; `--model` beam 1 1.51 s, beam 4 2.10 s, with the converter 2.26 s |
| `import rinchen.eval.robustness` | 0.164 s |
| the toy | built in 0.144 s; trained in 186.9 s (190.9 s with the export and the suites) |
| files of the main run | `robustness.json` 1,013,432 B, `hypotheses.jsonl` 469,819 B, `report.json` 1,406 B, `manifest.json` 4,118 B |
| the gate's two modules (**gate**) | 31 tests in 374.7 s (6 min 14 s) on a quiet machine, 442 s under load: the toy's teacher 186 s of it (3 min 6 s in the fixture), the two D11 teachers and their four exports 125 s and the D5 converter 45 s as fixtures; the full gate grows by that |

The metrics are cheaper than the engine (0.04 against 0.11 and 0.23 s per 1,000 rows, ARITHMETIC from the table) and both are negligible next to the toy's training; the spike's prototypes cost about 0.1 s per
1,000 rows (**spike**), the finished code 0.04 s. UNMEASURED: memory (the command reads the whole suite and all hypotheses; the caps bound the parse, not memory), a suite of tens of
thousands of rows (one `Engine.translate` call is one batch in memory: D13 chunks if it needs to), a real model, the GPU.

## What D13 takes over

The FLORES+ and NTREX suites and the Mongolia-sourced human sets; the runner over systems (Rinchen students and teacher, NLLB, MADLAD, Hy-MT2: their outputs enter through the hypotheses file); `result.json` of
`rinchen.eval/1` (it embeds `robustness_block` as it is); significance (it reads the per-row `ok`, `value` and `value_cyrl` of `robustness.json`); the Markdown and JSON report (`report.json` of D12 is the
narrow one: D13 removes `eval/robust_report.py` and the command's `report.json` line or keeps them as a per-suite artefact; `robustness_block`, `robustness.json` and the command stay); the model card and the hub
upload. D12 leaves room by keeping the metrics pure functions of rows, the runner over a protocol (`translate(rows, options)`), the hypotheses file as the system boundary and the result per row. Left out on
purpose: a generic builder of suites from real parallel text, a repair policy, a `--min-*` threshold, per-language conventions beyond the table, entity extraction without a table, number words, document-level
suites, and a `terminology` request field.

## The STOP gate

The master plan's gate for D12 is "robustness block populated in tiny report". It is met on the toy through the real `rinchen` binary, the real Eole, the real CTranslate2 and the real SentencePiece on the CPU
with the GPU hidden (`-m subprocess`, every subprocess bounded by 900 s). Two modules, 31 tests: `tests/integration/test_robustness_gate.py` (25: R1 to R8, the CPU checks of R10 and R11) and
`test_robustness_gate_refusals.py` (6: R9 and the scans of R10). Thresholds are literals of the module and are never lowered. The figures are **measured** in this run (they equal the gate's, the spike's
and the ledger's, except where the last column says so):

| Gate | Threshold | Measured |
|---|---|---|
| R1 main suite, beam 4, `retry`, float32 | numbers 0.90 (each single-number category 0.95), placeholders 0.99, entities 0.90, terms 0.99, purity 0.999, retention 0.95; every request `ok`; no row excluded | numbers **892/960 = 0.9292** (int, decimal, percent, date, time 120/120; currency 117; grouped 118; two numbers 57, no floor), placeholders 599/600 = 0.9983, entities 447/480 = 0.9313, terms 1044/1044, purity 1.0 over 900 rows, retention 0.9976 (chrF 99.49 / 99.73); statuses `ok` 960, 600, 240, 400, 900, 960; excluded `{}` |
| R2 the hypotheses rotated by 7 rows inside each group | numbers, placeholders, terms at most 0.02, entities at most 0.10; the `mong` arm alone: retention at most 0.2 | numbers 3/960, placeholders 2/600, entities 10/480 = 0.0208, terms 1/1044; the `mong` arm alone 0.0598 (the spike: 3, 2, 10, 1, 0.06) |
| R3 the D11 teacher (never saw a digit or a placeholder) over the same suite | placeholders and numbers at most 0.02, retention at most 0.5 | numbers 0/960, placeholders 0/600, entities 0/480, terms 1041/1044, retention 0.3172 (chrF 17.07 / 53.81); the spike over its own rows 0 of 960, 0 of 600, 0.317 |
| R4 a glossary the model never writes | `term_acc` at most 0.02 | 0 of 1,044 mentions |
| R5 placeholder indices 10 and up | at most 0.2 | 0 of 7 in each of the four directions (0 of 28) |
| R6 script masks | `off`: at least 50 rows under the floor and a mean under 0.97; `always` and `retry`: none and a mean of at least 0.999; the other values move by at most 0.02 | `off`: 101 of 900 under the floor (all among the 500 untrained: 99 of 200 English, 100 of 100 Khalkha, 200 of 200 Russian pure), mean 0.9389, min 0.0; **at beam 1 `off` leaves 110, mean 0.9499 (the spike's 110: ARITHMETIC, (0.9499 x 900 - 400) / 500 = 0.9098 over the 500 untrained, the spike's mean)**; `retry`, `always`: 0 under, mean 1.0; `always`: numbers 891/960, placeholders 598/600, entities 444/480, retention 0.9977 |
| R7 rescoring and determinism | `--hypotheses` over R1's `hypotheses.jsonl` equal; two `--model` runs equal | `robustness`, `sections`, `rows`, `suite` equal (`system` names no model; `hypotheses_sha256` equals the file's digest); `robustness.json`, `report.json`, `hypotheses.jsonl` byte-equal across two runs |
| R8 the D5 tiny converter | the five other sections equal R1's; the converted arm retains less than the direct arm; the traditional-script output suite pure | the five sections equal; `retention_converted` 0.5473 (chrF 54.58 against 99.49 and 99.73; the spike 0.562 at beam 1); output suite purity 1.0 over 100 rows, all `ok`; without a converter exit 2 with nothing written; the converter run's `hypotheses.jsonl` read back with `--hypotheses` gives the same sections |
| R9 refusals | exit codes 2, 3, 2 with nothing written | flipped byte 2, unknown key 3, `--device auto` 2 |
| R10, R11 no GPU; the exclusions | the AST scans; the golden suite through the real binary equals the golden result byte for byte | clean; equal (the main suite excludes no row, so R11 and the unit golden file are the only guards against a metric that counts exclusions as passes) |

The gate was run for real in the fix wave (**gate**: the two modules 31 passed in 442 s on a machine at load 58 to 130; 374.7 s, 6 min 14 s, on a quiet one at Task 8) and the whole gate gave
13,543 passed, 1 skipped, 10 deselected. **What the figures show:** that the instrument can fail (every control collapses) and passes on a model that reads and writes the toy's numbers, placeholders,
names and scripts. **What they do not show:** anything about quality, a real model, a real glossary, the traditional script as people write it, a GPU or a mechanism.

## Open gates

Not done by D12, because no real data, GPU, reader or mechanism exists: everything GPU-side (CTranslate2 on CUDA, `float16`, `int8_float16` and their effect on these metrics; the spike's int8 column, with
its prototype metrics, differed from float32 by a few rows (**spike**, not re-run) and ranks nothing); a MECHANISM for placeholders and terminology and the TRAINING DATA that would make a model copy
any of the 100 indices and use a term marker (nobody owns them: spec section 6, item 1); real data and real priors (the conventions table, the 4-letter suffix, the currency symbols, the 0.98 floor; real
entities have many spellings; number and magnitude words are not read); suites from real parallel text and real glossaries (D13); a real converter and real traditional-script text (the control-character
audit and the render check of D5 stay open); D15's own tiny suite and matching training rows (D12's generator is a test helper, not a package module; `placeholder_acc` 1.0 holds at beam 1 on the toy and not
at beam 4); and the gates assigned to D4 and the D5 to D11 gates that need real data or a reader.

## What was not verified

- **Anything on a GPU**, and any real model, corpus, glossary or reference. Every translation ran on the CPU with CTranslate2 on `device="cpu"`; every Eole run had `CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`.
- **That the metrics rank real systems.** The toy ranks nothing; no claim is made that int8 is lossless or that a mechanism helps.
- **A converter on real traditional-script text**: the D5 tiny converter is a scaffold (it restores 5 of 37 toy nouns).
- **The command's exit codes 4, 1 and 130 with real failures** (fakes only), `KeyboardInterrupt` with a real CTranslate2 call, memory at large suites.
- **CI**: no CI run covers D3 to D12. **Windows and macOS** were not run.
- **The scoped re-review of the fix wave** (`b2466ef..ae4e816`): it was running when this was written and its verdict is not recorded.

### Known limits, left open on purpose

Findings of the review that were parked, and limits of the build, one each:

1. **`term_acc` is natural agreement with a glossary, not adherence** (review I-3): the D11 teacher scores 1,041 of 1,044 on the main suite because the glossary is the toy's noun table. A real adherence metric needs entries
   whose required target differs from the model's default, the unconstrained baseline beside the constrained run, copy-through-equal entries flagged and real morphology instead of the 4-letter suffix.
2. **Over-acceptance of names**: the hyphen is a boundary and a short name is a prefix of a longer one (`Бат` in `Сухэ-Батор` and `Батор`, `Bat` in `Ulan Bator`); at most 2 of 480 mentions in the main suite (review minor 2);
   a suffix of 5 letters does not match.
3. **The converter's files are hashed but loaded without re-verification**, and `ScriptConverter` has no digest check and no `close` (review minor 3): a file swapped between the hash and the load is not caught.
4. **`read_suite` and `read_hypotheses` are used only by tests**; the command hashes with `hash_input`, reads with `verified_content` and parses with `from_bytes`, which reads the whole file, so the 8 MiB and 128 MiB
   caps bound the parse, not memory (review minor 4).
5. **Trivial solutions score high**: a source echo scores `numbers_acc` 1.0 (960 of 960) and `placeholder_acc` 1.0 (600 of 600), a digits-only output scores 1.0 on the numbers, and `%` is not measured
   (**measured** for the echo: entities 0.25, terms 0.0038, purity 0.5556, retention 0.40). Read `numbers_acc` and `placeholder_acc` next to chrF, never alone (review minor 7).
6. **No held-out `reorder` row**: the toy's training split takes every sentence of the one- and two-placeholder templates, genitive included, so the held-out split holds no genitive item and `order_ok` is exercised
   only by unit tests and the golden file (review minor 8).
7. **`5 $7` reads as `5 $` and `7`** ("the symbol after the number wins"): a reorder of such text cannot be faithful under any rule; pinned as a vector.
8. **Retention can exceed 1 and is meaningless near a zero Cyrillic chrF** (read `coverage.mong_source_retention.chrf_cyrl`); with a Cyrillic chrF of exactly 0 it is `null`.
9. **Script silence is excluded by design**: empty and `no_output` hypotheses leave the mean untouched and raise `excluded` (T8); read `coverage.script_purity` beside the mean.
10. **`surface_exact_rows` does not see the layout of a date** (the components are separate numbers: `03/12/2024` and `12.03.2024` have the same surfaces); a copy-through echo scores 720 of 960 (the spike's
    prototype, which compared digit strings with their separators, gave 568).
11. **Floors have margin and bit-identical training is not asserted**: numbers 892 against the floor 0.90 (864 rows) leaves 28 rows, entities 447 of 480 against 432 leaves 15 mentions; the gate records
    `model.bin`'s digest and does not assert it, so another CPU may move a count by a few rows (a miss is answered with more steps, never a lower floor).
12. **The sacrebleu signature pinned in the golden file fails on a sacrebleu bump on purpose** (`nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`): regenerate the golden result after checking the scores.
13. **Priors**: the conventions table, the 4-letter suffix, the currency symbols and the 0.98 floor; `10.30` is a decimal; digits glued to letters count; number words are not read; Khalkha `1,250` is ambiguous.
14. **One `Engine.translate` call per run** (a suite of tens of thousands of rows is one batch in memory), and an engine error names a row by its index in the call (`row N`), not by section, id and arm.
15. **`--help` of the command breaks a sentence over two lines** (an embedded newline of its docstring; cosmetic).
16. **A load race in a D8 test, fixed here as the ONE edit to a D8 test**: `tests/unit/test_train_launch_orphan.py::test_a_crashed_eole_leaves_no_rank_running_and_the_next_launch_is_accepted` polled `/proc/<pid>/stat` and
    caught only `FileNotFoundError`; a process that exits during the read raises `ProcessLookupError` (the reviewer's probe of 400 short-lived processes: 264 `FileNotFoundError`, 136 `ProcessLookupError`). The test failed
    twice under load (D11's Part A and D12's Task 6) and passed alone; the helper now catches both, as its sister `tests/unit/test_utils_logged_process.py` does, with a deterministic test (`338bb02`).
17. **Disclosed process slips of the D12 wave**, one line each: two accidental `git stash` runs (Task 3's `git stash -u`, dangling commits `6891dee` and `9284fc3`; Task 7's `git stash -q` followed at once by `git stash pop`,
    dangling commit `d538088`), both restored byte for byte (the reviewer compared the stash trees with the later commits; the stash list is empty and the tree clean); several launchers ignored signals (a first gate launch in
    Tasks 5 and 7 was killed by pid with its leftover test processes and relaunched; scratch trainings started with `&` ran with SIGINT ignored); one stray foreground `sleep 1` (Task 8); the fix wave's mutation harness
    gave false kills until a git repository was added to its copy; and in this runbook's own measurements two short foreground `sleep` calls (3 s and 20 s) while a launch and a training ran (nothing was killed or signalled; every
    driver ran in a session of its own). The design agent ran no GPU query of any kind, no docker command and no network access; the ledger records no `pkill -f`.

### The review, its fix wave and the re-review

**The combined review** (one independent reviewer, read-only, Tasks 1 to 8, `37634d1..b2466ef`): ready with fixes, 0 Critical, 4 Important and 9 Minor findings. It reproduced the real-binary figures
(892 of 960 and the rest) and found the rest sound: the maximum matching against brute force (20,000 random cases of identical sizes), `numbers_in` never raising on 300,000 random strings, runaway 1 to 3 MB
inputs in 0.05 to 0.3 s, `Decimal` exactness at a million digits, D4's behaviour unchanged (the old `digit_multiset` against the new: 0 mismatches on 300,000 random strings and every single code point), the
exclusions depending only on source and reference, the cap at 64, conservation (`scored + excluded` equals the total), placeholders embedded, adjacent, damaged and duplicated, no catastrophic backtracking in the span
matcher (2 million characters in 1 to 50 ms), the suite reader (cap before the parse, repeated keys, reserved symbols, a real `<2khk_Cyrl>` probe giving exit 3 with no text quoted), the hypotheses reader, the
interpolation scan (per-entry reasoned, exact both ways), the command (an output directory that is an input's, `RINCHEN_MODEL` with `--hypotheses`, a foreign manifest not erased, a stale `hypotheses.jsonl` deleted,
the converter round trip that a text cannot spoof, the engine's hashes re-checked and the engine closed in `finally`), the tool versions, NO leakage (0 of 2,400 test and 1,432 dev sentences and 0 rows of any suite
section appear in training, and no `mn_Mong` source is in training), the effectiveness of the gate controls and the hygiene (largest file 800 lines, largest source file 407).

| Finding | What | Fixed |
|---|---|---|
| **I-1** | the numbers metric FALSE-FAILED identical text when one value appeared twice with different units (`5 and 5 ₮` against itself: `exact=False`), because `units_ok` was computed on an arbitrary pairing; a symbol between two numbers was claimed by both (`859 571₮ 3 ширхэг`) | `64e45f0`: `units_ok` = a maximum matching over unit-compatible edges is as large as the unit-blind one; a currency symbol belongs to one number. 14 failing tests first; differential 300,000 random cases against brute force, 0 mismatches (11,659 legitimate unit failures; the old code mismatched 47,554); the gate's 892 of 960 unchanged |
| **I-2** | denominators invisible: `numbers_acc` over 10 scored rows of 1,000 looked perfect; silence raised the script mean; retention hid a useless control | `40dc2ee`: the `coverage` map of `report.json`, `scored.<section>` and `excluded.<section>` in the manifest, the retention line of the command with both chrF scores and `excluded N` on the script line |
| **I-3** | `term_acc` is not terminology adherence (the D11 teacher scores 1,041 of 1,044) | docs only: spec T7 amendment and Known limit 1 |
| **I-4** | the per-row outcomes could not reproduce the retention (a `mong` row stored only the direct arm's sentence chrF; corpus chrF is not a mean of sentence chrF) | `ce794ee`: `value_cyrl` per row; also minors 5 and 6: `chrf_signature` and `system.hypotheses_sha256`; the golden result regenerated (12,589 to 13,614 bytes), cross-checked: the new file minus the three additions equals the old |
| minors | stale 0.9323 and 895 in two CLI docstrings (`ae4e816`); the hyphen over-acceptance, the converter provenance gap, `read_suite` used only by tests, the trivial-solution controls, the missing `reorder` row, the floor margins (limits 2, 3, 4, 5, 6, 11); the D8 race test (`338bb02`); the stash slips | as stated |

**The fix wave** (five commits, `b2466ef..ae4e816`, failing test first, mutation-checked: 13, 18 and 12 mutants, no survivor that mattered) and its **full gate: 13,543 passed, 1 skipped, 10 deselected** (13,491 after
Task 8; the machine was at load 58 to 130, so the gate ran as five parallel pytest processes that cover every collected test exactly once, about 19 minutes), `ruff` (598 files) and `mypy` (211 files) clean, the floors
unchanged. Existing tests changed because the code forced it (the same list is in the commit messages): the pinned report keys, the stdout lines, the `system` dictionaries, the row fields and the mong details of the
tests of `eval`, the reviewed-entry wording of the message scan, and additions to the gate and to `tests/support/robust_golden.py`.

**The scoped re-review** of the fix wave (`b2466ef..ae4e816`) was RUNNING when this runbook was written. **Its verdict is NOT recorded here; a follow-up edit adds it**, as D9 to D11 did.

## Measured numbers

The machine: Intel Core Ultra 7 265K (20 cores as `nproc` reports), 46 GiB of RAM, Linux 7.0.0-42-generic, CPython 3.14.10, eole 0.6.4, ctranslate2 4.8.4, sentencepiece 0.2.1, sacrebleu 2.6.0, regex 2026.9.29; the GPU was
hidden from every process and every translator was built on `device="cpu"`. All on 2026-11-10, once each, code at `ae4e816` (the tree held only these documents), load average 1 to 5. The drivers were read-only
scripts (`scratchpad/d12-t9/`) that use the gate's own helpers (`tests/support/robust_gate.py`), so the numbers are the gate's; nothing was written under `data/` or `models/`. The D11 teacher and the D5 converter
were the cached artifacts of the D11 spike (`scratchpad/d11-t4/export_f32`, `d11-spike/work/converter`; the teacher verified by `rinchen export verify`, exit 0, `model.bin` `b7f33d88d5fa...`); the robustness model was
trained fresh.

| What | Value |
|---|---|
| the toy: build; digest; splits | 0.144 s; `4136d80a...bad1`; 25,520 / 1,432 / 2,400 |
| the toy's teacher: workspace, training, export, suites | 0.37 s, 186.94 s, 3.07 s, 0.19 s; `model.bin` 2,080,071 B, `e28d1566...a6` (the spike's, byte for byte) |
| the golden run (`--hypotheses`) | 0.88 s; `robustness.json` equal to the golden file; `report.json` 1,209 B; `manifest.json` 2,212 B |
| R1 main, beam 4 / beam 1 | 2.10 s / 1.51 s; numbers 892 / 893 of 960; placeholders 599 / 600 of 600; entities 447 / 450 of 480; terms 1044 / 1044; purity 1.0 / 1.0; retention 0.9976 (99.49 over 99.73) / 1.0030 (99.86 over 99.57) |
| the numbers by category (beam 4 / beam 1, of 120 each) | int, decimal, percent, date, time 120 / 120; currency 117 / 120; grouped 118 / 118; two numbers 57 / 55 |
| details of the numbers (beam 4) | recall 0.9483, extra tokens 81, surface-exact rows 895, unit failures 3, ambiguous rows 0, capped rows 0 |
| R2 rotated by 7 (all groups / the `mong` arm only) | numbers 3, placeholders 2, entities 10, terms 1, purity 0.7556 (220 under the floor), retention 0.9978 over chrF 5.96 / 5.97; the `mong` arm only: retention 0.0598, the others as R1 |
| R3 the D11 teacher | numbers 0/960, placeholders 0/600 (1,800 missing), entities 0/480, terms 1041/1044 (398 of 400 rows all found), purity 1.0, retention 0.3172 (17.07 over 53.81), 1.95 s |
| R4 term control, R5 unseen placeholders | terms 0/1044; placeholders 0/28 (0/7 in each of four directions) |
| R6 masks (beam 4): `off` / `always` | script 0.9389 with 101 under the floor / 1.0 with 0; `always`: numbers 891, placeholders 598, entities 444, retention 0.9977; `off` at beam 1: 110 under, mean 0.9499 |
| R7 rescoring, two runs | equal; `hypotheses_sha256` `e2bee59d...`; manifests differ only in `created` and the output paths (two directories) |
| R8 the converter | five sections equal R1's; `retention_converted` 0.5473 (54.58); mong-output suite purity 1.0 over 100 rows; without a converter exit 2, directory absent; round trip equal |
| R10 a source echo | numbers 960/960, placeholders 600/600, entities 120/480, terms 4/1044, purity 0.5556 (400 under), retention 0.4001 (7.30 over 18.26), surface-exact rows 720 |
| the vectors of `numbers_in`, spans, placeholders | the tables above |
| costs | the table of "Costs" |
| the tests | about 1,765 new; 13,543 passed, 1 skipped, 10 deselected after the fix wave (the gate's own run); `ruff format --check`, `ruff check`, `mypy src` clean on this tree |

**The numbers agree between the spike, the gates and this run** wherever they are not times: the six values of R1 at beam 1 (893, 600, 450, 1,044, 1.0, 1.0030) and the controls, the model's digest and the toy's digest.
Where they differ: (1) the numbers at beam 4 are 892 of 960 = 0.9292, not the spike's and the plan's 895 = 0.9323: the spike's rougher prototype ignored the currency unit and so counted three rows (`num-0547`, `num-0569`,
`num-0571`, where the model drops the `₮`) as exact; the spike's reference `numbers_ref.py` and the finished code agree on 892, and beam 1 gives 893 as the spike measured; the floor 0.90 stands with 0.029 of margin;
(2) mask `off` leaves 101 rows under the floor at beam 4, the command's default and the gate's setting, against the spike's 110 at beam 1 (reproduced exactly here at beam 1: the ledger's note that the 110 belonged to a
different request set was not right, the request set is the same); (3) the converted arm retains 0.5473 at beam 4 and the spike's 0.562 was at beam 1; (4) a copy-through source echo is surface-exact on 720 rows
of 960 where the spike's prototype said 568 (Known limit 10); (5) the machine's metric cost is 0.04 s per 1,000 requests where the plan assumed 0.1.

## Where the code lives

`src/rinchen/text/numerals.py`; `src/rinchen/eval/{robust_numbers,robust_placeholders,robust_spans,robust_script,robust_mong,robust_suite,robust_suite_rules,robust_hypotheses,robust_outcomes,robust_sections,robustness,robust_report}.py`;
`src/rinchen/cli/{eval,eval_robustness,eval_robustness_plan,eval_robustness_output}.py`; the edit of `src/rinchen/filter/numbers.py`; `tests/fixtures/robustness/{golden_suite.json,golden_hypotheses.jsonl,golden_robustness.json}`; unit tests
`tests/unit/test_{text_numerals,filter_numbers_fold,eval_robust_numbers,eval_robust_placeholders,eval_robust_spans,eval_robust_script,eval_robust_mong,eval_robust_suite,eval_robust_suite_refusals,eval_robust_suite_field_refusals,
eval_robust_hypotheses,eval_robustness,eval_robustness_document,eval_robustness_rows,eval_robust_report,eval_layering,eval_error_messages,cli_eval_robustness,cli_eval_robustness_refusals,cli_eval_robustness_failures,
cli_eval_robustness_manifest,cli_eval_robustness_messages,support_robust_toy,support_robust_suites}.py`, property tests `tests/property/test_eval_{robust_numbers,robust_spans,robust_suite,robustness}_property.py`; the gate
`tests/integration/test_robustness_gate.py` and `test_robustness_gate_refusals.py` with the fixtures `d12_robust_workspace`, `d12_robust_teacher`, `d12_robust_export` and `d12_suites` of `tests/integration/conftest.py`; and
the helpers `tests/support/{robust_toy,robust_toy_text,robust_suites,robust_gate,robust_golden,robust_cli,robust_suite_documents}.py` (and the extension of `error_messages.py`).
