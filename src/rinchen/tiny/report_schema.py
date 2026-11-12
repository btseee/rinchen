"""The layout of `report.json` (D15, spec P10): the keys at every level, the kind of every value and
where a value may be null, written once as the closed shapes of `rinchen.tiny.report_shapes`.

The document, in sorted key order as it is written: `blocks`, `inputs` (the digests of the profile,
of the seven fixtures and of the lock), `profile`, `qe_fake`, `schema`, `stages` (the fifteen stages
in order, each with its state, closed reason, free `detail` and the state of each of its `parts`),
`thresholds` (the eight of the table, in order), `unmeasured` (the fixed list below) and `verdict`.

`blocks` holds the eleven measurements a run makes, each of them null when the stage that makes it
never got far enough to give the figures it is built from: `filter`, `synth`, `converter`, `train`
(with the digest of the averaged weights, null when the stage that makes it did not run), `parity`,
`serve`, `robustness` (D12's six values and the coverage each rests on, counts only), `memorised` and
`heldout` (one entry per direction and system), `neural_seam` and `int8`. A number is never a duration,
a text is never a path, a time or a line of another program's output, and nothing here names a
`manifest_sha256` or a `hypotheses_sha256`: what the report holds is the same on every machine.
"""

from __future__ import annotations

import re
from typing import Final

from rinchen.config.tiny import STAGE_IDS
from rinchen.tiny.fixtures import FIXTURE_FILES
from rinchen.tiny.report_shapes import (
    BOOL,
    COUNT,
    DIGEST,
    NAME,
    NONNEG,
    NUMBER,
    SCORE,
    SHARE,
    TEXT,
    Detail,
    Keys,
    Leaf,
    Nullable,
    Seq,
    Shape,
    Table,
    choice,
    equals,
)
from rinchen.tiny.states import STATES, is_closed_reason
from rinchen.tiny.thresholds import THRESHOLD_IDS

SCHEMA: Final = "rinchen.tiny/1"
VERDICTS: Final = ("pass", "fail")
UNMEASURED: Final[tuple[str, ...]] = (
    "any run on a GPU: no stage touched a device",
    "a real corpus: the fixtures are templated sentences",
    "a real generator behind the synthetic data stage",
    "neural metric scorers: only the seam of score files ran, with planted scores",
    "GitHub Actions: nothing ran there",
    "the time on a 4-core CPU runner",
)
# The statuses the engine gives a translation, in the order a summary lists them.
ENGINE_STATUSES: Final = ("ok", "empty", "too_long", "truncated", "no_output")
# D12's six values, in the order of the document.
ROBUSTNESS_KEYS: Final = (
    "entity_acc",
    "mong_source_retention",
    "numbers_acc",
    "placeholder_acc",
    "script_purity",
    "term_acc",
)
# The one value that is a ratio of two scores and may be above 1; the others are shares.
RATIO_KEY: Final = "mong_source_retention"
BLOCK_IDS: Final = (
    "filter",
    "synth",
    "converter",
    "train",
    "parity",
    "serve",
    "robustness",
    "memorised",
    "heldout",
    "neural_seam",
    "int8",
)

_PROFILE_NAME: Final = re.compile(r"[a-z][a-z0-9_]{0,15}")
_BOUND: Final = re.compile(r"(?:>=|<=) [0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?")

PROFILE: Final = Leaf(
    "a profile name: a lower-case letter and up to 15 lower-case letters, digits or underscores",
    lambda value: isinstance(value, str) and _PROFILE_NAME.fullmatch(value) is not None,
)
BOUND: Final = Leaf(
    "a bound: a comparison and a number",
    lambda value: isinstance(value, str) and _BOUND.fullmatch(value) is not None,
)
REASON: Final = Leaf(
    "one of the closed reasons of the stages",
    lambda value: isinstance(value, str) and is_closed_reason(value),
)


def _counts(*names: str) -> Keys:
    return Keys({name: COUNT for name in names})


_COUNTS_BY_NAME: Final = Table(NAME, COUNT)

_FILTER: Final = Keys(
    {
        "rows": COUNT,
        "kept": COUNT,
        "reasons": _COUNTS_BY_NAME,
        "authentic_rows": COUNT,
        "authentic_flagged": COUNT,
        "authentic_flagged_share": SHARE,
        "planted": Keys(
            {
                "total": COUNT,
                "detectable": COUNT,
                "matched": COUNT,
                "expected": _COUNTS_BY_NAME,
                "found": _COUNTS_BY_NAME,
            }
        ),
        "train_rows": COUNT,
    }
)
_SYNTH: Final = Keys(
    {
        "job": NAME,
        "generator": NAME,
        "qe_status": TEXT,
        "counts": _counts(
            "candidates",
            "dropped_copy",
            "dropped_empty",
            "dropped_purity",
            "dropped_qe",
            "no_candidates",
            "no_survivor",
            "selected",
            "sources",
        ),
        "shard_stats": _counts("generated", "planned", "regenerated_invalid", "reused"),
        "rows": COUNT,
        "chat_requests": COUNT,
    }
)
_CONVERTER: Final = Keys(
    {
        "cer": NONNEG,
        "word_accuracy": SHARE,
        "n": COUNT,
        "rule_fallbacks": COUNT,
        "invalid_outputs": COUNT,
    }
)
_TRAIN: Final = Keys(
    {
        "variant": NAME,
        "steps": COUNT,
        "rows_by_direction": _COUNTS_BY_NAME,
        "bt_rows": COUNT,
        "weights_sha256": Nullable(DIGEST),
    }
)
_PARITY: Final = Keys(
    {
        "exact": COUNT,
        "sentences": COUNT,
        "exact_share": SHARE,
        "min_exact_share": SHARE,
        "mean_abs_delta": NONNEG,
        "max_abs_delta": NONNEG,
        "max_mean_delta": NONNEG,
        "passes": BOOL,
        "gated": BOOL,
        "quantization": NAME,
        "placement": NAME,
        "model_sha256": Nullable(DIGEST),
        "checkpoint_weights_sha256": Nullable(DIGEST),
    }
)
_SERVE: Final = Keys(
    {
        "status_codes": Seq(COUNT, length=5),
        "statuses": Nullable(Seq(NAME)),
        "script_purity": Nullable(Seq(Nullable(SHARE))),
        "error_code": Nullable(NAME),
        "translate_200_total": Nullable(NONNEG),
    }
)
_ROBUSTNESS: Final = Keys(
    {
        "values": Keys(
            {
                key: Nullable(NONNEG if key == RATIO_KEY else SHARE)
                for key in ROBUSTNESS_KEYS
            }
        ),
        "coverage": Keys(
            {key: _counts("scored", "total", "excluded") for key in ROBUSTNESS_KEYS}
        ),
    }
)
_RUN: Final = Keys(
    {
        "direction": NAME,
        "system": NAME,
        "rows": COUNT,
        "chrf": SCORE,
        "chrf_pp": SCORE,
        "bleu": SCORE,
        "statuses": Table(choice(*ENGINE_STATUSES), COUNT, minimum=1),
    }
)
_NEURAL_SEAM: Final = Keys(
    {
        "metric": NAME,
        "system": NAME,
        "rows": Table(NAME, COUNT, minimum=1),
        "scores": Table(NAME, NUMBER, minimum=1),
    }
)
_INT8: Final = Keys(
    {
        "model_sha256": DIGEST,
        "chrf_drop": Keys({"least": NUMBER, "largest": NUMBER}),
        "agreement": Keys(
            {
                "by_direction": Table(NAME, _counts("rows", "equal"), minimum=1),
                "rows": COUNT,
                "equal": COUNT,
                "share": SHARE,
            }
        ),
    }
)
_BLOCK_SHAPES: Final[dict[str, Shape]] = {
    "filter": _FILTER,
    "synth": _SYNTH,
    "converter": _CONVERTER,
    "train": _TRAIN,
    "parity": _PARITY,
    "serve": _SERVE,
    "robustness": _ROBUSTNESS,
    "memorised": Seq(_RUN, minimum=1),
    "heldout": Seq(_RUN, minimum=1),
    "neural_seam": _NEURAL_SEAM,
    "int8": _INT8,
}
BLOCKS: Final = Keys({name: Nullable(_BLOCK_SHAPES[name]) for name in BLOCK_IDS})

_PART: Final = Keys({"id": NAME, "state": choice(*STATES), "reason": Nullable(REASON)})
_STAGE: Final = Keys(
    {
        "id": choice(*STAGE_IDS),
        "state": choice(*STATES),
        "reason": Nullable(REASON),
        "detail": Detail(),
        "parts": Seq(_PART),
    }
)
_THRESHOLD: Final = Keys(
    {
        "id": choice(*THRESHOLD_IDS),
        "bound": BOUND,
        "value": Nullable(NUMBER),
        "verdict": choice(*VERDICTS),
        "detail": TEXT,
    }
)
INPUTS: Final = Keys(
    {
        "config_sha256": DIGEST,
        "fixtures": Keys({name: DIGEST for name in FIXTURE_FILES}),
        "lock_sha256": DIGEST,
    }
)
DOCUMENT: Final = Keys(
    {
        "schema": equals(SCHEMA),
        "profile": PROFILE,
        "qe_fake": BOOL,
        "verdict": choice(*VERDICTS),
        "inputs": INPUTS,
        "stages": Seq(_STAGE, length=len(STAGE_IDS)),
        "thresholds": Seq(_THRESHOLD, length=len(THRESHOLD_IDS)),
        "blocks": BLOCKS,
        "unmeasured": equals(list(UNMEASURED)),
    }
)
