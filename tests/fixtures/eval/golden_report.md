# Evaluation report

These figures come from generated fixture suites. They test the tooling and say nothing about translation quality.

## Systems

| system | kind | model | manifest | options |
| --- | --- | --- | --- | --- |
| alpha | hypotheses | n/a | n/a | none |
| beta | hypotheses | n/a | n/a | none |

## Suites

| suite | kind | benchmark | split | purpose | rows | directions | dataset | revision | sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| golden | fixture | golden | devtest | report | 10 | 2 | n/a (no source record) | n/a (no source record) | b707518d3913 |

## Scores

A cell is the corpus score with its interval, `value [low, high]`; the column beside it counts the outputs that were `ok` and names the others. A system that did not run a direction has `n/a` and the state; a direction the suite does not declare is `n/a (not in the suite)`.

### golden

Resampling unit: document (5 units in each direction).

#### chrF++

| direction | alpha | alpha ok | beta | beta ok |
| --- | --- | --- | --- | --- |
| khk_Cyrl-deu_Latn | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| deu_Latn-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-eng_Latn | 73.08 [54.91, 85.48] | 10 of 10 | 93.87 [85.24, 100.00] | 10 of 10 |
| eng_Latn-khk_Cyrl | 68.06 [50.67, 80.98] | 10 of 10 | 91.34 [85.07, 97.86] | 10 of 10 |
| khk_Cyrl-jpn_Jpan | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| jpn_Jpan-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-kaz_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| kaz_Cyrl-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-kor_Hang | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| kor_Hang-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-rus_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| rus_Cyrl-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-tur_Latn | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| tur_Latn-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-zho_Hans | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| zho_Hans-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |

#### chrF

| direction | alpha | alpha ok | beta | beta ok |
| --- | --- | --- | --- | --- |
| khk_Cyrl-deu_Latn | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| deu_Latn-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-eng_Latn | 72.44 [53.76, 85.63] | 10 of 10 | 93.46 [84.43, 100.00] | 10 of 10 |
| eng_Latn-khk_Cyrl | 67.41 [49.84, 80.59] | 10 of 10 | 91.07 [84.67, 97.84] | 10 of 10 |
| khk_Cyrl-jpn_Jpan | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| jpn_Jpan-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-kaz_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| kaz_Cyrl-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-kor_Hang | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| kor_Hang-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-rus_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| rus_Cyrl-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-tur_Latn | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| tur_Latn-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-zho_Hans | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| zho_Hans-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |

#### BLEU

| direction | alpha | alpha ok | beta | beta ok |
| --- | --- | --- | --- | --- |
| khk_Cyrl-deu_Latn | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| deu_Latn-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-eng_Latn | 59.40 [36.34, 72.14] | 10 of 10 | 94.42 [85.51, 100.00] | 10 of 10 |
| eng_Latn-khk_Cyrl | 50.87 [25.54, 67.88] | 10 of 10 | 91.07 [83.56, 97.80] | 10 of 10 |
| khk_Cyrl-jpn_Jpan | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| jpn_Jpan-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-kaz_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| kaz_Cyrl-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-kor_Hang | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| kor_Hang-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-rus_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| rus_Cyrl-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-tur_Latn | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| tur_Latn-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| khk_Cyrl-zho_Hans | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |
| zho_Hans-khk_Cyrl | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |

## Paired comparisons

16 intervals and 4 paired comparisons are reported with no correction for multiple comparisons.

### golden

Baseline: alpha. The verdict is the p-value's, not the interval's: `better` is a p-value below 0.05 with the difference in the system's favour, `worse` is a p-value below 0.05 with the difference against it, and `no significant difference` is anything else.

| direction | system | metric | delta | interval | p | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| khk_Cyrl-eng_Latn | beta | chrF++ | +20.79 | [+13.92, +33.41] | 0.002 | better |
| khk_Cyrl-eng_Latn | beta | BLEU | +35.01 | [+22.91, +54.78] | 0.002 | better |
| eng_Latn-khk_Cyrl | beta | chrF++ | +23.27 | [+12.65, +36.22] | 0.002 | better |
| eng_Latn-khk_Cyrl | beta | BLEU | +40.20 | [+29.20, +57.55] | 0.002 | better |

## Robustness

No robustness run was given.

## Not measured

- comet22: no segment scores were given: the COMET and MetricX scorers are not built (4 of 4 results)
- metricx24: no segment scores were given: the COMET and MetricX scorers are not built (4 of 4 results)
- metricx24_qe: no segment scores were given: the COMET and MetricX scorers are not built (4 of 4 results)
- golden: 14 expected directions are not in the suite.
- The share of outputs in the wrong language: no language identification is run.

## Release targets

A target is a measurement: no command fails because one is not met.

| target | status | value | threshold | basis |
| --- | --- | --- | --- | --- |
| beats_baseline | unmeasured | n/a | 12 | 0 of 16 directions measured |
| off_target | unmeasured | n/a | 0.01 | no language identification is run |
| numbers | unmeasured | n/a | 0.99 | no robustness run of a rinchen system was given |
| placeholders | unmeasured | n/a | 1 | no robustness run of a rinchen system was given |

## Provenance

- Bootstrap: 1000 resamples from the seed 20261110, percentile intervals at the level 0.95 (alpha 0.05), draws splitmix64-counter/1.
- BLEU signature: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
- chrF signature: nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0
- chrF++ signature: nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0

| suite | system | run.json |
| --- | --- | --- |
| golden | alpha | 1bc91f732e59 |
| golden | beta | b7e19a5d0144 |
