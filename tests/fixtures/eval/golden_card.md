---
language:
- eng
- khk
- rus
library_name: ctranslate2
pipeline_tag: translation
tags:
- rinchen
- translation
- ctranslate2
- mongolian
- toy
---

# rinchen-07148803f29a

> These figures come from generated fixture suites. They test the tooling and say nothing about translation quality.

## What it is

rinchen-07148803f29a is a translation model of the Rinchen project, exported for CTranslate2 (float32). Its transformer has 2 encoder layers, 1 decoder layer, hidden size 64 and 124 vocabulary entries. It was trained from random initialisation: no pretrained model is its base. That is the project's rule; the export's record cannot prove it, and the checkpoint's manifest stage is named under Provenance.

Document level: no.

The tag of the target language goes in the source (`src`). Targets and their tags: eng_Latn `<2eng_Latn>`, khk_Cyrl `<2khk_Cyrl>`, rus_Cyrl `<2rus_Cyrl>`. The default script of a Mongolian target is Cyrillic (`Cyrl`). The traditional script (`Mong`) is not written by this export alone: it needs the script converter.

Repository: `rinchen-test/rinchen-tiny`.

## Directions

Trained for 6 directions: eng_Latn-khk_Cyrl, khk_Cyrl-eng_Latn, khk_Cyrl-rus_Cyrl, mn_Mong-eng_Latn, mn_Mong-rus_Cyrl, rus_Cyrl-khk_Cyrl.

## Files

| file | bytes | sha256 |
| --- | --- | --- |
| model.bin | 30 | 15d442008fc30bee1d28a81d2d926f17005fe428e465c2062627366bf2a68019 |
| shared_vocabulary.json | 2048 | 1ca5a3a5d4d83c97acbcdaa8aab41c3c8ae0b2646a343dad85ed7526508d74cb |
| config.json | 312 | 587cb980af76fdc7e52369fd0b9d926dff266976b6f8ac631e358fecc49ff8cf |
| spm.model | 128 | 26d706159878af9911c037ebf1fa8ee71a186d498a36583e4184910f3e8707e3 |
| rinchen.json | 1089 | 590688d0a34415f238ae0ad5ed45d88f68158515b77323eb7a851f3a5e4ad38c |
| report.json | 14894 | 9dbd9380216097555c9deac98aaf3b54257134bb78ccbae4d771b3935525d9ef |
| report.md | 6610 | a22350b28862150572f20bd8fc9de640d6fb5d9a0435c61423ad6d6ab8f96369 |

`README.md` is this card and `release.json` lists the digest of every other file, this card included.

## Usage

```python
import rinchen

with rinchen.Translator("path/to/this/directory") as translator:
    result = translator.translate("text to translate", tgt="khk_Cyrl")
print(result.text)
```

## Evaluation

The figures are those of the system `alpha` in `report.json`: every number below is a number of that report.

- Decoding settings of the run: beam_size=4, device=cpu, max_decoding_length=256, script_mask=retry, threads=1.
- A cell is a corpus score with its interval: `value [low, high]`.
- Intervals: 1,000 resamples, seed 20261110, percentile 95 %.

### Suite golden

- kind: fixture
- split: devtest
- rows: 10
- BLEU signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- chrF signature: `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`
- chrF++ signature: `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`

| direction | segments | resampling units | chrF++ | chrF | BLEU |
| --- | --- | --- | --- | --- | --- |
| khk_Cyrl-eng_Latn | 10 | 5 document units | 73.08 [54.91, 85.48] | 72.44 [53.76, 85.63] | 59.40 [36.34, 72.14] |
| eng_Latn-khk_Cyrl | 10 | 5 document units | 68.06 [50.67, 80.98] | 67.41 [49.84, 80.59] | 50.87 [25.54, 67.88] |

n/a (not in the suite): khk_Cyrl-rus_Cyrl, rus_Cyrl-khk_Cyrl, mn_Mong-eng_Latn, mn_Mong-rus_Cyrl.

n/a (not trained): khk_Cyrl-deu_Latn, deu_Latn-khk_Cyrl, khk_Cyrl-jpn_Jpan, jpn_Jpan-khk_Cyrl, khk_Cyrl-kaz_Cyrl, kaz_Cyrl-khk_Cyrl, khk_Cyrl-kor_Hang, kor_Hang-khk_Cyrl, khk_Cyrl-tur_Latn, tur_Latn-khk_Cyrl, khk_Cyrl-zho_Hans, zho_Hans-khk_Cyrl.

COMET-22, MetricX-24, MetricX-24-QE: n/a (no segment scores were given: no scorer is built (B1 and B2 are open)).

### Paired comparisons

n/a (no comparison of the report has this system as the candidate)

## Robustness

n/a (no robustness run was attached)

## Release targets

A target is a measurement: no command fails because one is not met.

| target | status | threshold | value |
| --- | --- | --- | --- |
| beats_baseline | unmeasured | 12 | n/a (0 of 16 directions measured) |
| off_target | unmeasured | 0.01 | n/a (no language identification is run) |
| numbers | unmeasured | 0.99 | n/a (no robustness run of a rinchen system was given) |
| placeholders | unmeasured | 1 | n/a (no robustness run of a rinchen system was given) |

## Limits

- The model is trained for the directions listed above and no others. Only the directions in the tables above are measured; a pair the model was not trained for (zero-shot) is never evaluated, and nothing here says that its output is in the language asked for.
- A dev split is used for model selection only: the report refuses a dev run, so every figure here is from a devtest or test split.
- Each interval resamples the unit its table names: a document when every row of a direction has a document id, a segment otherwise.
- 16 intervals and 4 paired comparisons are reported with no correction for multiple comparisons. Read a difference with its p-value.
- Not measured: COMET-22, MetricX-24 and MetricX-24-QE (no scorer is built), the share of outputs in the wrong language (no language identification is run), any run on a GPU (none was made) and real-world input beyond the suites above.

## Provenance

- rinchen_version: 0.0.0
- checkpoint kind: step
- checkpoint step: 200
- checkpoint manifest stage: train_launch
- checkpoint weights sha256: 1111111111111111111111111111111111111111111111111111111111111111
- checkpoint config sha256: 2222222222222222222222222222222222222222222222222222222222222222
- checkpoint vocabulary sha256: 3333333333333333333333333333333333333333333333333333333333333333
- checkpoint manifest sha256: 4444444444444444444444444444444444444444444444444444444444444444
- export manifest sha256: fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210
- report.json sha256: 00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff
