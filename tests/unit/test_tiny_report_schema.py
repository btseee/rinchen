"""The layout of `report.json` (D15 Task 6, spec P10): `rinchen.tiny.report_schema`.

The schema is the closed list of the keys at every level, the kind of every value and the places where
a value may be null. These tests pin the key sets as literals (a change of the layout is a decision made
here, with the golden files), check that the golden document fits, and refuse each way a document can be
wrong by the place the message names.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from rinchen.config.tiny import STAGE_IDS
from rinchen.tiny.report_schema import (
    BLOCK_IDS,
    BLOCKS,
    DOCUMENT,
    ENGINE_STATUSES,
    ROBUSTNESS_KEYS,
    SCHEMA,
    UNMEASURED,
)
from rinchen.tiny.report_shapes import Keys, Nullable, Seq, ShapeError
from rinchen.tiny.states import (
    FAIL_BUDGET,
    FAIL_INPUT,
    SKIP_REASONS,
    STATES,
    fail_check,
    fail_environment,
    fail_exit,
    fail_timeout,
)
from rinchen.tiny.thresholds import THRESHOLD_IDS
from tests.support.tiny_report_walk import (
    closed_places,
    golden_document,
    place_of,
    with_key,
    with_value,
    without_key,
)

TOP_KEYS = [
    "blocks",
    "inputs",
    "profile",
    "qe_fake",
    "schema",
    "stages",
    "thresholds",
    "unmeasured",
    "verdict",
]
BLOCK_KEYS = {
    "filter": [
        "authentic_flagged",
        "authentic_flagged_share",
        "authentic_rows",
        "kept",
        "planted",
        "reasons",
        "rows",
        "train_rows",
    ],
    "synth": [
        "chat_requests",
        "counts",
        "generator",
        "job",
        "qe_status",
        "rows",
        "shard_stats",
    ],
    "converter": ["cer", "invalid_outputs", "n", "rule_fallbacks", "word_accuracy"],
    "train": ["bt_rows", "rows_by_direction", "steps", "variant", "weights_sha256"],
    "parity": [
        "checkpoint_weights_sha256",
        "exact",
        "exact_share",
        "gated",
        "max_abs_delta",
        "max_mean_delta",
        "mean_abs_delta",
        "min_exact_share",
        "model_sha256",
        "passes",
        "placement",
        "quantization",
        "sentences",
    ],
    "serve": [
        "error_code",
        "script_purity",
        "status_codes",
        "statuses",
        "translate_200_total",
    ],
    "robustness": ["coverage", "values"],
    "neural_seam": ["metric", "rows", "scores", "system"],
    "int8": ["agreement", "chrf_drop", "model_sha256"],
}
RUN_KEYS = ["bleu", "chrf", "chrf_pp", "direction", "rows", "statuses", "system"]


def golden() -> dict[str, Any]:
    return golden_document()


def unwrapped(shape: Any) -> Any:
    return shape.shape if isinstance(shape, Nullable) else shape


def refusal(document: Any) -> str:
    with pytest.raises(ShapeError) as raised:
        DOCUMENT.check(document, "report")
    return str(raised.value)


def test_the_golden_document_fits_the_schema() -> None:
    DOCUMENT.check(golden(), "report")


def test_the_schema_name_and_the_closed_lists() -> None:
    assert SCHEMA == "rinchen.tiny/1"
    assert tuple(sorted(BLOCK_IDS)) == tuple(
        sorted(BLOCK_KEYS | {"memorised": 0, "heldout": 0})
    )
    assert BLOCK_IDS == (
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
    assert ROBUSTNESS_KEYS == (
        "entity_acc",
        "mong_source_retention",
        "numbers_acc",
        "placeholder_acc",
        "script_purity",
        "term_acc",
    )
    assert ENGINE_STATUSES == ("ok", "empty", "too_long", "truncated", "no_output")


def test_the_unmeasured_list_is_fixed() -> None:
    assert UNMEASURED == (
        "any run on a GPU: no stage touched a device",
        "a real corpus: the fixtures are templated sentences",
        "a real generator behind the synthetic data stage",
        "neural metric scorers: only the seam of score files ran, with planted scores",
        "GitHub Actions: nothing ran there",
        "the time on a 4-core CPU runner",
    )


def test_the_keys_of_the_document_are_these() -> None:
    assert sorted(DOCUMENT.shapes) == TOP_KEYS


def test_the_keys_of_every_block_are_these() -> None:
    assert sorted(BLOCKS.shapes) == sorted(BLOCK_IDS)
    for name, keys in BLOCK_KEYS.items():
        assert sorted(unwrapped(BLOCKS.shapes[name]).shapes) == keys, name
    for name in ("memorised", "heldout"):
        sequence = unwrapped(BLOCKS.shapes[name])
        assert isinstance(sequence, Seq)
        assert sorted(sequence.item.shapes) == RUN_KEYS


def test_the_keys_of_a_stage_a_part_a_threshold_and_the_inputs_are_these() -> None:
    stage = DOCUMENT.shapes["stages"].item
    assert sorted(stage.shapes) == ["detail", "id", "parts", "reason", "state"]
    assert sorted(stage.shapes["parts"].item.shapes) == ["id", "reason", "state"]
    threshold = DOCUMENT.shapes["thresholds"].item
    assert sorted(threshold.shapes) == ["bound", "detail", "id", "value", "verdict"]
    assert sorted(DOCUMENT.shapes["inputs"].shapes) == [
        "config_sha256",
        "fixtures",
        "lock_sha256",
    ]


def test_the_stage_ids_the_states_and_the_threshold_ids_are_those_of_the_pipeline() -> (
    None
):
    stage = DOCUMENT.shapes["stages"].item
    for shape, members in (
        (stage.shapes["id"], STAGE_IDS),
        (stage.shapes["state"], STATES),
        (DOCUMENT.shapes["thresholds"].item.shapes["id"], THRESHOLD_IDS),
    ):
        for member in members:
            shape.check(member, "x")
        with pytest.raises(ShapeError):
            shape.check("warn", "x")


def test_every_closed_mapping_of_the_golden_refuses_a_key_it_does_not_know() -> None:
    document = golden()
    places = list(closed_places(DOCUMENT, document))
    assert len(places) > 60
    for path in places:
        message = refusal(with_key(document, path, "unexpected_key", 1))
        assert message.startswith(f"{place_of(path)}: "), path
        assert "unexpected_key" not in message


def test_every_closed_mapping_of_the_golden_refuses_a_missing_key() -> None:
    document = golden()
    for path in closed_places(DOCUMENT, document):
        node = document
        for part in path:
            node = node[part]
        for key in list(node)[:1]:
            message = refusal(without_key(document, path, key))
            assert message.startswith(f"{place_of(path)}: "), path


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema",), "rinchen.tiny/2"),
        (("schema",), None),
        (("profile",), "Tiny"),
        (("profile",), "t" * 17),
        (("profile",), ""),
        (("profile",), 3),
        (("qe_fake",), "yes"),
        (("verdict",), "warn"),
        (("verdict",), True),
        (("inputs", "config_sha256"), "xyz"),
        (("inputs", "lock_sha256"), "A" * 64),
        (("inputs", "fixtures", "noise.tsv"), "short"),
        (("stages", 0, "id"), "bogus"),
        (("stages", 0, "state"), "warn"),
        (("stages", 0, "reason"), 5),
        (("stages", 0, "reason"), "an invented reason"),
        (("stages", 0, "reason"), "exit code four"),
        (("stages", 3, "parts", 3, "reason"), "no model, nothing to say"),
        (("stages", 3, "parts", 0, "id"), "x/y"),
        (("stages", 3, "parts", 0, "state"), "warn"),
        (("thresholds", 0, "verdict"), "warn"),
        (("thresholds", 1, "value"), math.nan),
        (("thresholds", 1, "value"), "50"),
        (("thresholds", 1, "bound"), 5),
        (("thresholds", 1, "detail"), ""),
        (("blocks", "filter", "authentic_flagged_share"), 1.5),
        (("blocks", "filter", "rows"), -1),
        (("blocks", "filter", "reasons"), {"not a name": 1}),
        (("blocks", "filter", "planted", "found"), {"duplicate": -1}),
        (("blocks", "synth", "qe_status"), "skipped: /tmp/x"),
        (("blocks", "converter", "n"), 1.5),
        (("blocks", "train", "steps"), "1500"),
        (("blocks", "train", "rows_by_direction", "eng_Latn-khk_Cyrl"), -1),
        (("blocks", "parity", "passes"), 1),
        (("blocks", "parity", "exact_share"), 1.1),
        (("blocks", "parity", "model_sha256"), "abc"),
        (("blocks", "serve", "status_codes"), [200, 200, 200, 400]),
        (("blocks", "serve", "statuses"), "ok"),
        (("blocks", "robustness", "values", "numbers_acc"), -0.1),
        (("blocks", "robustness", "values", "numbers_acc"), 1.1),
        (("blocks", "robustness", "values", "mong_source_retention"), -0.1),
        (("blocks", "robustness", "coverage", "numbers_acc", "scored"), -1),
        (("blocks", "memorised"), []),
        (("blocks", "memorised", 0, "chrf"), 100.01),
        (("blocks", "memorised", 0, "rows"), 0.5),
        (("blocks", "memorised", 0, "statuses"), {}),
        (("blocks", "memorised", 0, "statuses"), {"bogus_status": 1}),
        (("blocks", "heldout", 0, "bleu"), -1),
        (("blocks", "neural_seam", "scores"), {"eng_Latn-khk_Cyrl": math.inf}),
        (("blocks", "int8", "agreement", "share"), 2),
        (("blocks", "int8", "chrf_drop", "least"), "-0.06"),
        (("unmeasured",), []),
        (("unmeasured",), ["a real corpus: the fixtures are templated sentences"]),
        (("stages",), []),
        (("thresholds",), []),
    ],
    ids=lambda value: str(value)[:40],
)
def test_the_schema_refuses_a_wrong_value_and_names_where(
    path: tuple[str | int, ...], value: Any
) -> None:
    document = with_value(golden(), path, value)
    message = refusal(document)
    assert message.startswith(place_of(path)), (message, path)


@pytest.mark.parametrize(
    "reason",
    [
        *SKIP_REASONS,
        fail_exit(4),
        fail_exit(130),
        fail_timeout(600),
        FAIL_BUDGET,
        FAIL_INPUT,
        fail_environment(2, 1),
        fail_check("planted_noise"),
        fail_check("service_translate"),
        fail_check("neural_scores"),
    ],
)
def test_every_closed_reason_of_the_stages_fits_a_reason_of_the_report(
    reason: str,
) -> None:
    document = golden()
    document["stages"][0]["reason"] = reason
    DOCUMENT.check(document, "report")


@pytest.mark.parametrize(
    "bound", [">= 50.0", "<= 0.001", "<= 1e-05", "<= 1200", "<= 0", ">= 1.5e+16"]
)
def test_a_bound_is_a_comparison_and_a_plain_number(bound: str) -> None:
    document = golden()
    document["thresholds"][1]["bound"] = bound
    DOCUMENT.check(document, "report")


@pytest.mark.parametrize(
    "bound",
    [
        "<= 1.2.3",
        ">= -1",
        "== 3",
        ">= 5e",
        "= 5",
        ">=5",
        "<= ",
        "<= 1 2",
        "<= nan",
        ">= 1e",
        "",
    ],
)
def test_a_bound_that_is_not_one_is_refused(bound: str) -> None:
    document = golden()
    document["thresholds"][1]["bound"] = bound
    assert refusal(document).startswith("report.thresholds[1].bound: ")


def test_the_unmeasured_list_is_held_in_order() -> None:
    document = golden()
    document["unmeasured"] = list(reversed(document["unmeasured"]))
    assert refusal(document).startswith("report.unmeasured: ")


def test_the_fixtures_are_exactly_the_seven_files() -> None:
    document = golden()
    assert len(document["inputs"]["fixtures"]) == 7
    assert refusal(
        without_key(document, ("inputs", "fixtures"), "noise.tsv")
    ).startswith("report.inputs.fixtures: ")


@pytest.mark.parametrize(
    "path",
    [
        ("stages", 0, "reason"),
        ("blocks", "filter"),
        ("blocks", "synth"),
        ("blocks", "converter"),
        ("blocks", "train"),
        ("blocks", "parity"),
        ("blocks", "serve"),
        ("blocks", "robustness"),
        ("blocks", "memorised"),
        ("blocks", "heldout"),
        ("blocks", "neural_seam"),
        ("blocks", "int8"),
        ("blocks", "train", "weights_sha256"),
        ("blocks", "parity", "model_sha256"),
        ("blocks", "parity", "checkpoint_weights_sha256"),
        ("blocks", "serve", "statuses"),
        ("blocks", "serve", "script_purity"),
        ("blocks", "serve", "error_code"),
        ("blocks", "serve", "translate_200_total"),
        ("blocks", "robustness", "values", "term_acc"),
        ("thresholds", 7, "value"),
    ],
    ids=lambda path: ".".join(str(part) for part in path),
)
def test_a_block_may_be_null_and_so_may_the_values_that_may_be_missing(
    path: tuple[str | int, ...],
) -> None:
    DOCUMENT.check(with_value(golden(), path, None), "report")


@pytest.mark.parametrize(
    "path",
    [
        ("blocks", "filter", "rows"),
        ("blocks", "parity", "exact_share"),
        ("blocks", "memorised", 0, "chrf"),
        ("blocks", "robustness", "coverage"),
        ("blocks", "int8", "model_sha256"),
        ("inputs", "config_sha256"),
        ("profile",),
        ("verdict",),
        ("schema",),
    ],
    ids=lambda path: ".".join(str(part) for part in path),
)
def test_a_value_that_is_always_there_is_never_null(
    path: tuple[str | int, ...],
) -> None:
    assert refusal(with_value(golden(), path, None)).startswith(place_of(path))


def test_the_serve_purity_list_may_hold_a_null_for_a_text_without_letters() -> None:
    document = with_value(golden(), ("blocks", "serve", "script_purity"), [1.0, None])
    DOCUMENT.check(document, "report")


def test_the_robustness_values_hold_the_ratio_above_one_but_the_shares_stay_below() -> (
    None
):
    values = ("blocks", "robustness", "values")
    DOCUMENT.check(
        with_value(golden(), (*values, "mong_source_retention"), 1.3), "report"
    )
    DOCUMENT.check(with_value(golden(), (*values, "numbers_acc"), 1.0), "report")


def test_a_walk_of_the_schema_finds_a_closed_mapping_for_every_block() -> None:
    document = golden()
    paths = [
        path for path in closed_places(DOCUMENT, document) if path[:1] == ("blocks",)
    ]
    seen = {path[1] for path in paths if len(path) > 1}
    assert seen == set(BLOCK_IDS)
    assert isinstance(DOCUMENT, Keys)
