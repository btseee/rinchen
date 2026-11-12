# Tiny pipeline: pass (profile tiny)

## Stages

| stage | state | reason |
| --- | --- | --- |
| environment | ok |  |
| fetch | ok |  |
| corpus | ok |  |
| filter | ok |  |
| filter.script_ratio | ok |  |
| filter.numbers | ok |  |
| filter.length | ok | token_ratio_check skipped: the token-ratio check needs a SentencePiece model, which does not exist yet |
| filter.lid | skipped | no language-identification model: GlotLID is a download |
| filter.minhash | ok |  |
| filter.labse | ok |  |
| filter.qe | ok |  |
| spm | ok |  |
| converter | ok |  |
| synth | ok |  |
| prepare | ok |  |
| train | ok |  |
| average | ok |  |
| export | ok |  |
| parity | ok |  |
| serve | ok |  |
| robustness | ok |  |
| evaluate | ok |  |
| evaluate.memorised | ok |  |
| evaluate.heldout | ok |  |
| evaluate.neural_seam | ok |  |
| evaluate.report | ok |  |

## Thresholds

| threshold | bound | value | verdict | detail |
| --- | --- | --- | --- | --- |
| stages | <= 0 | 0 | pass | no stage failed and every skip has a reason of the closed list |
| train_chrf | >= 50.0 | 99.36 | pass | least of 5 directions, mn_Mong-eng_Latn, int8 export |
| parity_exact | >= 0.95 | 1 | pass | exact_share of the parity figures |
| parity_delta | <= 0.001 | 1.622e-06 | pass | mean_abs_delta of the parity figures |
| int8_chrf_drop | <= 1.0 | 0 | pass | largest float32 minus int8 chrF over 5 directions, eng_Latn-khk_Cyrl |
| script_purity | >= 1.0 | 1 | pass | script_purity of the robustness figures |
| placeholder_acc | >= 1.0 | 1 | pass | placeholder_acc of the robustness figures |
| wall_time | <= 1200 | - | pass | seconds from the first line of the run |

The wall-time figure is in manifest.json: this report holds no duration and no timestamp.

## Held-out

Held-out rows: generalisation on templated sentences, reported and never gated.

| direction | system | rows | chrF | chrF++ | BLEU | statuses |
| --- | --- | --- | --- | --- | --- | --- |
| eng_Latn-khk_Cyrl | int8 | 30 | 72.00 | 71.02 | 54.55 | ok 30 |
| khk_Cyrl-eng_Latn | int8 | 30 | 64.19 | 63.37 | 48.13 | ok 30 |
| khk_Cyrl-rus_Cyrl | int8 | 30 | 55.31 | 54.62 | 43.66 | ok 30 |
| rus_Cyrl-khk_Cyrl | int8 | 30 | 60.80 | 59.95 | 46.78 | ok 30 |

Neural score (metricx24_qe of tiny-int8-qe, planted scores): eng_Latn-khk_Cyrl 3.88 over 30 rows, khk_Cyrl-eng_Latn 4.31 over 30 rows.

## Robustness

The six values of the robustness measurement beside the reference floors of that measurement. The floors are references and not gates of this run.

| measure | value | reference floor | scored | total | excluded |
| --- | --- | --- | --- | --- | --- |
| entity_acc | 0.9792 | 0.90 | 240 | 240 | 0 |
| mong_source_retention | 0.9936 | 0.95 | 48 | 48 | 0 |
| numbers_acc | 1 | 0.90 | 240 | 240 | 0 |
| placeholder_acc | 1 | 0.99 | 238 | 238 | 0 |
| script_purity | 1 | 0.999 | 600 | 600 | 0 |
| term_acc | 0.9991 | 0.99 | 1134 | 1134 | 0 |

## Not measured

- any run on a GPU: no stage touched a device
- a real corpus: the fixtures are templated sentences
- a real generator behind the synthetic data stage
- neural metric scorers: only the seam of score files ran, with planted scores
- GitHub Actions: nothing ran there
- the time on a 4-core CPU runner
