"""The observations of the golden run of the tiny pipeline, as the stages hand them back (D15 Task 6).

`golden_results()` is the fifteen `StageResult`s of the spike's run `p6` (the one measured end-to-end
run: its figures are copied here as literals so that no test reads the spike's files), each with the record and the facts the stage function would return, the details being what the
stage puts in its record. Figures that run did not record (the held-out figures of three directions,
which are the spike's finding F11; the rows by direction of the training corpus; the requests the chat
server answered; how often the two exports wrote the same text; the planted scores of the neural seam)
are ILLUSTRATIVE: they exercise the layout and say nothing about a model.

`tests/fixtures/tiny/golden_report.json` and `.md` were written from the same figures by a script that
imports nothing of `rinchen.tiny` (see the docstring of `tests/unit/test_tiny_report_golden.py`).
The helpers below change one stage of the run at a time.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Final

from rinchen.config.tiny import STAGE_IDS
from rinchen.tiny.context import StageResult
from rinchen.tiny.states import (
    PART_IDS,
    SKIP_EARLIER_FAILED,
    SKIP_NO_LID,
    SKIP_NO_NEURAL,
    SKIP_TOKEN_RATIO,
    PartRecord,
    StageRecord,
    fail_exit,
)

CONFIG_SHA256: Final = (
    "1ddb031f9f22480bbe7e07937ae72b97ae46fd386db65c61bf284ff2bf98ac79"
)
LOCK_SHA256: Final = "559b3954277eab053057fb0de73f92dca8ba11dffe193707424df5584e317b16"
FIXTURE_DIGESTS: Final = {
    "eng_Latn-khk_Cyrl.tsv": "7d1d15d7b7fc8e6ed5bc617990505519f37909ea0107f221728fdb38a962d229",
    "rus_Cyrl-khk_Cyrl.tsv": "dce119b354e907812361dce126b3d1266bebe874fc594c89a6247d3a0a8f6bbe",
    "mn_Mong.tsv": "16000ecdd3c099179c5e243cfa2a9c6d89bba565091cf680e7ac3231f0ad0f34",
    "mono_khk_Cyrl.tsv": "79f488e2ffe757dd34194b0ef056e41f29eb038c8b2d84dde57e23e7fa418952",
    "noise.tsv": "fa16dba1968d182ff50fc7f61c62b86b16fd7112481196c17ae70dfacafe3248",
    "nouns.tsv": "62fe9458a7d8592cb61ecedcaf2b1023b4a130fb357cb28b4e373d54fa2387be",
    "entities.tsv": "6cb6bb4ec68a042ff3a721d20798853d633fc4d36d1d2ed5e36e772f755cef63",
}
GOLDEN_ELAPSED: Final = 120.01
WEIGHTS_SHA256: Final = (
    "4e06ce45faef7c4b80b487a26606a5ac5ea2b2b6f740f82aaede088c1e33bb0e"
)
FLOAT32_SHA256: Final = (
    "f48811a45287337cfc4434b90f5e32e8869459d40ea90f3bd845debef2720dcc"
)
INT8_SHA256: Final = "8fa90a266950cfc3305a4e5f730f02f8ae04aaf9c60485834be591edbcff0aec"
VARIANT: Final = "t1_bt0.2_src_s1234"

MEMORISED_DIRECTIONS: Final = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "rus_Cyrl-khk_Cyrl",
    "khk_Cyrl-rus_Cyrl",
    "mn_Mong-eng_Latn",
)
MEMORISED_ROWS: Final = dict(
    zip(MEMORISED_DIRECTIONS, (299, 299, 300, 300, 48), strict=True)
)
# (chrF, BLEU, chrF++) of p6 by direction and system.
MEMORISED_FIGURES: Final = {
    "eng_Latn-khk_Cyrl": {
        "float32": (99.9593961645435, 99.75091642638047, 99.93605281692561),
        "int8": (99.9593961645435, 99.75091642638047, 99.93605281692561),
    },
    "khk_Cyrl-eng_Latn": {
        "float32": (99.64912545218539, 99.15375802916404, 99.59814399257519),
        "int8": (99.70392773428999, 99.23724783486942, 99.65512281302222),
    },
    "khk_Cyrl-rus_Cyrl": {
        "float32": (100.0, 100.0, 100.0),
        "int8": (100.0, 100.0, 100.0),
    },
    "mn_Mong-eng_Latn": {
        "float32": (99.29329142756212, 99.12979980302303, 99.3263860124633),
        "int8": (99.35849111913797, 99.18240523627664, 99.35140057662511),
    },
    "rus_Cyrl-khk_Cyrl": {
        "float32": (99.84338362894934, 99.54269493208409, 99.81879282325151),
        "int8": (99.84338362894934, 99.54269493208409, 99.81879282325151),
    },
}
# (chrF, BLEU, chrF++) of the held-out runs (int8): the first from p6, the rest from the spike's F11
# with illustrative chrF++ values.
HELDOUT_FIGURES: Final = {
    "eng_Latn-khk_Cyrl": (71.99500932252622, 54.54807542354906, 71.0192171727142),
    "khk_Cyrl-eng_Latn": (64.19, 48.13, 63.37),
    "khk_Cyrl-rus_Cyrl": (55.31, 43.66, 54.62),
    "rus_Cyrl-khk_Cyrl": (60.8, 46.78, 59.95),
}
ROBUSTNESS_VALUES: Final = {
    "entity_acc": 0.9791666666666666,
    "mong_source_retention": 0.9935849111913797,
    "numbers_acc": 1.0,
    "placeholder_acc": 1.0,
    "script_purity": 1.0,
    "term_acc": 0.9991181657848325,
}
ROBUSTNESS_TOTALS: Final = {
    "entity_acc": 240,
    "mong_source_retention": 48,
    "numbers_acc": 240,
    "placeholder_acc": 238,
    "script_purity": 600,
    "term_acc": 1134,
}


def surface(figures: tuple[float, float, float]) -> dict[str, float | None]:
    """The six metrics of a `result.json`: chrF, BLEU and chrF++ and the three neural ones, none."""
    chrf, bleu, chrf_pp = figures
    return {
        "bleu": bleu,
        "chrf": chrf,
        "chrf_pp": chrf_pp,
        "comet22": None,
        "metricx24": None,
        "metricx24_qe": None,
    }


def run_facts(rows: int, figures: tuple[float, float, float]) -> dict[str, Any]:
    return {"rows": rows, "metrics": surface(figures), "statuses": {"ok": rows}}


def robustness_facts() -> dict[str, Any]:
    coverage = {
        key: {"excluded": 0, "scored": total, "total": total}
        for key, total in ROBUSTNESS_TOTALS.items()
    }
    coverage["mong_source_retention"].update(
        chrf_cyrl=100.0, chrf_mong=99.35849111913797
    )
    return {
        "values": dict(ROBUSTNESS_VALUES),
        "coverage": coverage,
        "rows": {
            "entities": 120,
            "mong": 48,
            "numbers": 240,
            "placeholders": 238,
            "script": 600,
            "terms": 600,
        },
    }


def parity_facts() -> dict[str, Any]:
    """`parity.json` of the float32 export without its `lines` and `mismatches`."""
    return {
        "checkpoint_weights_sha256": WEIGHTS_SHA256,
        "exact": 70,
        "exact_share": 1.0,
        "export_name": "rinchen-f48811a45287",
        "gated": True,
        "max_abs_delta": 2.1517276763916016e-05,
        "max_lines": 50,
        "max_mean_delta": 0.001,
        "mean_abs_delta": 1.6221680949143363e-06,
        "min_exact_share": 0.95,
        "model_sha256": FLOAT32_SHA256,
        "passes": True,
        "placement": "src",
        "quantization": "float32",
        "sentences": 70,
    }


def evaluate_facts() -> dict[str, Any]:
    memorised: dict[str, dict[str, Any]] = {}
    for direction, by_system in MEMORISED_FIGURES.items():
        for system, figures in by_system.items():
            memorised.setdefault(direction, {})[system] = run_facts(
                MEMORISED_ROWS[direction], figures
            )
    equal = {
        "eng_Latn-khk_Cyrl": 299,
        "khk_Cyrl-eng_Latn": 296,
        "khk_Cyrl-rus_Cyrl": 300,
        "mn_Mong-eng_Latn": 47,
        "rus_Cyrl-khk_Cyrl": 300,
    }
    return {
        "memorised": memorised,
        "memorised_chrf": {
            direction: {system: figures[0] for system, figures in by_system.items()}
            for direction, by_system in MEMORISED_FIGURES.items()
        },
        "int8_chrf_drop": {"least": -0.06519969157584171, "largest": 0.0},
        "agreement": {
            "by_direction": {
                direction: {"rows": MEMORISED_ROWS[direction], "equal": count}
                for direction, count in equal.items()
            },
            "rows": 1246,
            "equal": 1242,
            "share": 0.9967897271268058,
        },
        "heldout": {
            direction: {"int8": run_facts(30, figures)}
            for direction, figures in HELDOUT_FIGURES.items()
        },
        "neural_seam": {
            "metric": "metricx24_qe",
            "system": "tiny-int8-qe",
            "rows": {"eng_Latn-khk_Cyrl": 30, "khk_Cyrl-eng_Latn": 30},
            "scores": {"eng_Latn-khk_Cyrl": 3.875, "khk_Cyrl-eng_Latn": 4.3125},
        },
        "report": {"runs": 2, "suites": 2},
    }


def synth_facts() -> dict[str, Any]:
    return {
        "job": "bt_eng_khk",
        "generator": "tiny_oracle",
        "qe_status": "skipped: --no-qe",
        "counts": {
            "candidates": 64,
            "dropped_copy": 0,
            "dropped_empty": 0,
            "dropped_purity": 0,
            "dropped_qe": 0,
            "no_candidates": 0,
            "no_survivor": 0,
            "selected": 64,
            "sources": 64,
        },
        "shard_stats": {
            "generated": 2,
            "planned": 2,
            "regenerated_invalid": 0,
            "reused": 0,
        },
        "rows": 64,
        "chat_requests": 64,
    }


def filter_facts() -> dict[str, Any]:
    return {
        "rows": 770,
        "kept": 756,
        "reasons": {
            "duplicate": 4,
            "labse": 3,
            "len_ratio_char": 3,
            "numbers": 2,
            "script_tgt": 2,
        },
        "authentic_rows": 760,
        "authentic_flagged": 4,
        "authentic_flagged_share": 0.005263157894736842,
        "planted": {
            "total": 10,
            "detectable": 10,
            "matched": 10,
            "expected": {
                "duplicate": 2,
                "labse": 2,
                "len_ratio_char": 2,
                "numbers": 2,
                "script_tgt": 2,
            },
            "found": {
                "duplicate": 2,
                "labse": 2,
                "len_ratio_char": 2,
                "numbers": 2,
                "script_tgt": 2,
            },
        },
        "train_rows": 1316,
    }


def _ok(
    stage_id: str, facts: Mapping[str, Any], parts: Sequence[PartRecord] = ()
) -> StageResult:
    record = StageRecord(stage_id, "ok", detail=copy.deepcopy(dict(facts)), parts=parts)
    return StageResult(record, dict(facts))


def _filter_result() -> StageResult:
    facts = filter_facts()
    detail = {
        "rows": 770,
        "kept": 756,
        "authentic_flagged": 4,
        "planted_detectable": 10,
        "planted_matched": 10,
        "train_rows": 1316,
        "token_ratio_check": SKIP_TOKEN_RATIO,
    }
    parts = (
        PartRecord("script_ratio", "ok"),
        PartRecord("numbers", "ok"),
        PartRecord("length", "ok"),
        PartRecord("lid", "skipped", SKIP_NO_LID),
        PartRecord("minhash", "ok"),
        PartRecord("labse", "ok"),
        PartRecord("qe", "ok"),
    )
    return StageResult(StageRecord("filter", "ok", detail=detail, parts=parts), facts)


def golden_results() -> list[StageResult]:
    """The fifteen stages of the golden run, in order, each `ok`."""
    environment = {
        "installed": 186,
        "locked": 186,
        "mismatched": 0,
        "extra": 0,
        "not_installed": 0,
    }
    corpus_facts = {
        "rows": 770,
        "dropped": 0,
        "dropped_by_corpus": {},
        "corpora": {},
        "splits": {},
    }
    train = {
        "variant": VARIANT,
        "steps": 1500,
        "rows_by_direction": {
            "eng_Latn-khk_Cyrl": 358,
            "khk_Cyrl-eng_Latn": 299,
            "khk_Cyrl-rus_Cyrl": 300,
            "mn_Mong-eng_Latn": 48,
            "rus_Cyrl-khk_Cyrl": 300,
        },
        "bt_rows": 59,
    }
    evaluate = evaluate_facts()
    evaluate_parts = tuple(
        PartRecord(part, "ok")
        for part in ("memorised", "heldout", "neural_seam", "report")
    )
    results = [
        _ok("environment", environment),
        _ok("fetch", {"files": 7, "bytes": 67104}),
        StageResult(
            StageRecord("corpus", "ok", detail={"rows": 770, "dropped": 0}),
            corpus_facts,
        ),
        _filter_result(),
        _ok(
            "spm",
            {
                "vocab_size": 640,
                "languages": ["eng_Latn", "khk_Cyrl", "mn_Mong", "rus_Cyrl"],
            },
        ),
        _ok(
            "converter",
            {
                "cer": 0.0285,
                "word_accuracy": 0.916,
                "n": 500,
                "rule_fallbacks": 0,
                "invalid_outputs": 0,
            },
        ),
        _ok("synth", synth_facts()),
        _ok("prepare", {"variant": VARIANT}),
        _ok("train", train),
        _ok("average", {"weights_sha256": WEIGHTS_SHA256}),
        _ok(
            "export", {"model_sha256": {"float32": FLOAT32_SHA256, "int8": INT8_SHA256}}
        ),
        _ok("parity", parity_facts()),
        _ok(
            "serve",
            {
                "status_codes": [200, 200, 200, 400, 200],
                "statuses": ["ok", "ok"],
                "script_purity": [1.0, 1.0],
                "error_code": "bad_request",
                "translate_200_total": 1.0,
            },
        ),
        _ok("robustness", robustness_facts()),
        _ok("evaluate", evaluate, evaluate_parts),
    ]
    assert [result.record.id for result in results] == list(STAGE_IDS)
    return results


def facts_of(results: Sequence[StageResult], stage_id: str) -> Mapping[str, Any]:
    return next(r.facts for r in results if r.record.id == stage_id)


def replaced(
    results: Sequence[StageResult], stage_id: str, new: StageResult
) -> list[StageResult]:
    """`results` with the result of `stage_id` replaced by `new`."""
    return [new if result.record.id == stage_id else result for result in results]


def failed(
    stage_id: str,
    *,
    reason: str | None = None,
    facts: Mapping[str, Any] | None = None,
    parts: Sequence[PartRecord] = (),
) -> StageResult:
    """A failed result of `stage_id` with the reason `exit code 4` unless another is given."""
    record = StageRecord(
        stage_id, "fail", reason or fail_exit(4), detail={}, parts=parts
    )
    return StageResult(record, dict(facts or {}))


def skipped_after(stage_id: str) -> StageResult:
    """The result the pipeline records for a stage that never ran."""
    return StageResult(StageRecord(stage_id, "skipped", SKIP_EARLIER_FAILED), {})


def without_neural_part() -> StageResult:
    """The evaluation of a run without `--qe-fake`: its neural part skipped for its closed reason and
    its `neural_seam` fact null."""
    facts = evaluate_facts()
    facts["neural_seam"] = None
    parts = [PartRecord(part, "ok") for part in PART_IDS["evaluate"]]
    parts[2] = PartRecord("neural_seam", "skipped", SKIP_NO_NEURAL)
    record = StageRecord("evaluate", "ok", detail=copy.deepcopy(facts), parts=parts)
    return StageResult(record, facts)


def stopped_after(count: int, *, failing: bool = False) -> list[StageResult]:
    """The golden run cut after `count` stages: the stages after it never ran (they are not in the
    list at all), and with `failing` the last one that ran failed, with no facts."""
    kept = golden_results()[:count]
    if failing:
        kept[-1] = failed(kept[-1].record.id)
    return kept
