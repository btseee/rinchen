"""The measurement blocks of `report.json`, built from the facts the stages observed (D15, spec P10).

`build_blocks` reads the facts of each stage by its id and nothing else: never a command's printed text,
a file or the clock. A block is built from the facts of its own stage (`train` also from the digest of
the averaged weights, `int8` from the digest of the int8 export and from the evaluation's drop and
agreement), and the facts of a record that failed are used all the same: a parity that missed its gate
has its figures and an evaluation that stopped after its first part has the figures of that part. What
the schema does not list is left out (the synthetic job's fingerprint names a directory and the port of a
listener; `parity.json` holds the name of the export), and a block whose facts are missing, or are not of
the kind the schema holds, is null and never an error: the report of a run that went wrong must still be
written, and its thresholds say what is missing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

from rinchen.tiny.report_schema import BLOCK_IDS, BLOCKS, ROBUSTNESS_KEYS
from rinchen.tiny.report_shapes import plain
from rinchen.tiny.thresholds import GATED_SYSTEM

Facts = Mapping[str, Mapping[str, Any]]

_FILTER_KEYS: Final = (
    "rows",
    "kept",
    "reasons",
    "authentic_rows",
    "authentic_flagged",
    "authentic_flagged_share",
    "train_rows",
)
_PLANTED_KEYS: Final = ("total", "detectable", "matched", "expected", "found")
_SYNTH_KEYS: Final = (
    "job",
    "generator",
    "qe_status",
    "counts",
    "shard_stats",
    "rows",
    "chat_requests",
)
_CONVERTER_KEYS: Final = (
    "cer",
    "word_accuracy",
    "n",
    "rule_fallbacks",
    "invalid_outputs",
)
_TRAIN_KEYS: Final = ("variant", "steps", "rows_by_direction", "bt_rows")
_PARITY_KEYS: Final = (
    "exact",
    "sentences",
    "exact_share",
    "min_exact_share",
    "mean_abs_delta",
    "max_abs_delta",
    "max_mean_delta",
    "passes",
    "gated",
    "quantization",
    "placement",
)
_PARITY_DIGESTS: Final = ("model_sha256", "checkpoint_weights_sha256")
_SERVE_KEYS: Final = (
    "statuses",
    "script_purity",
    "error_code",
    "translate_200_total",
)
_COUNT_KEYS: Final = ("scored", "total", "excluded")
_RUN_METRICS: Final = ("chrf", "chrf_pp", "bleu")

# What a malformed fact raises while a block is read out of it: a key that is not there, a value that
# cannot be indexed or iterated, a shape the schema refuses.
_UNREADABLE: Final = (KeyError, IndexError, TypeError, AttributeError, ValueError)


def _pick(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: source[key] for key in keys}


def _filter(facts: Facts) -> dict[str, Any]:
    source = facts["filter"]
    return {
        **_pick(source, _FILTER_KEYS),
        "planted": _pick(source["planted"], _PLANTED_KEYS),
    }


def _synth(facts: Facts) -> dict[str, Any]:
    return _pick(facts["synth"], _SYNTH_KEYS)


def _converter(facts: Facts) -> dict[str, Any]:
    return _pick(facts["converter"], _CONVERTER_KEYS)


def _train(facts: Facts) -> dict[str, Any]:
    average = facts.get("average")
    weights = average.get("weights_sha256") if isinstance(average, Mapping) else None
    return {**_pick(facts["train"], _TRAIN_KEYS), "weights_sha256": weights}


def _parity(facts: Facts) -> dict[str, Any]:
    source = facts["parity"]
    return {
        **_pick(source, _PARITY_KEYS),
        **{key: source.get(key) for key in _PARITY_DIGESTS},
    }


def _serve(facts: Facts) -> dict[str, Any]:
    source = facts["serve"]
    return {
        "status_codes": source["status_codes"],
        **{key: source.get(key) for key in _SERVE_KEYS},
    }


def _robustness(facts: Facts) -> dict[str, Any]:
    source = facts["robustness"]
    return {
        "values": {key: source["values"][key] for key in ROBUSTNESS_KEYS},
        "coverage": {
            key: _pick(source["coverage"][key], _COUNT_KEYS) for key in ROBUSTNESS_KEYS
        },
    }


def _runs(table: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One entry per direction and system, in the order of their names."""
    entries = []
    for direction in sorted(table):
        for system in sorted(table[direction]):
            run = table[direction][system]
            entries.append(
                {
                    "direction": direction,
                    "system": system,
                    "rows": run["rows"],
                    **_pick(run["metrics"], _RUN_METRICS),
                    "statuses": run["statuses"],
                }
            )
    return entries


def _memorised(facts: Facts) -> list[dict[str, Any]]:
    return _runs(facts["evaluate"]["memorised"])


def _heldout(facts: Facts) -> list[dict[str, Any]]:
    return _runs(facts["evaluate"]["heldout"])


def _neural_seam(facts: Facts) -> dict[str, Any]:
    return _pick(
        facts["evaluate"]["neural_seam"], ("metric", "system", "rows", "scores")
    )


def _int8(facts: Facts) -> dict[str, Any]:
    evaluation = facts["evaluate"]
    agreement = evaluation["agreement"]
    return {
        "model_sha256": facts["export"]["model_sha256"][GATED_SYSTEM],
        "chrf_drop": _pick(evaluation["int8_chrf_drop"], ("least", "largest")),
        "agreement": {
            "by_direction": {
                direction: _pick(counts, ("rows", "equal"))
                for direction, counts in agreement["by_direction"].items()
            },
            **_pick(agreement, ("rows", "equal", "share")),
        },
    }


_BUILDERS: Final[dict[str, Callable[[Facts], Any]]] = {
    "filter": _filter,
    "synth": _synth,
    "converter": _converter,
    "train": _train,
    "parity": _parity,
    "serve": _serve,
    "robustness": _robustness,
    "memorised": _memorised,
    "heldout": _heldout,
    "neural_seam": _neural_seam,
    "int8": _int8,
}


def _block(name: str, facts: Facts) -> Any:
    try:
        block = plain(_BUILDERS[name](facts))
        BLOCKS.shapes[name].check(block, name)
    except _UNREADABLE:
        return None
    return block


def build_blocks(facts: Facts) -> dict[str, Any]:
    """The eleven blocks of the report, in the order of `BLOCK_IDS`, from `facts` (the facts of
    each stage by its id): plain dicts and lists that share nothing with `facts`, each null when
    its facts are missing or are not what the schema holds."""
    return {name: _block(name, facts) for name in BLOCK_IDS}
