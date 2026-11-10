"""Every refusal of the strict reader of `result.json` (D13, spec T11): exactly the keys of the
format, each of its type, every metric either a number in `metrics` or a reason in `unmeasured`
(never both, never neither), `ci95` consistent with the metrics and the bootstrap, the six
robustness keys with their coverage and their source or none of the three, digests of 64
lowercase hexadecimal characters, and what the parts say of each other (the statuses add up to the
segments, the units fit the segments, the system of the provenance is the system of the result).

A refusal is a `DataError` that names the field and never quotes a value: each test plants a
marker, which is not a well-formed id, digest or code, in the value the refusal is about.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.eval.result import result_from_bytes
from tests.support.eval_result_documents import (
    DELETE,
    DIGEST,
    MARKER,
    NEURAL,
    edited,
    golden_result_document,
    serialised,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

PREFIX = "not an evaluation result: "


def refused(data: bytes | dict[str, Any]) -> str:
    raw = data if isinstance(data, bytes) else serialised(data)
    with pytest.raises(DataError) as caught:
        result_from_bytes(raw)
    message = str(caught.value)
    assert message.startswith(PREFIX)
    assert MARKER not in message
    return message


def robust_document() -> dict[str, Any]:
    """The golden result with a robustness block, its coverage and its source."""
    document = golden_result_document()
    document["robustness"] = {
        "numbers_acc": 0.5,
        "placeholder_acc": 0.4,
        "term_acc": 0.75,
        "script_purity": 0.75,
        "entity_acc": 0.5714285714285714,
        "mong_source_retention": None,
    }
    coverage: dict[str, Any] = {
        key: {"scored": 6, "excluded": 2, "total": 8} for key in document["robustness"]
    }
    coverage["mong_source_retention"] |= {"chrf_cyrl": None, "chrf_mong": 12.5}
    document["robustness_coverage"] = coverage
    document["robustness_source"] = {
        "suite": "golden",
        "suite_sha256": DIGEST,
        "robustness_sha256": DIGEST,
        "report_sha256": DIGEST,
        "binding": "name",
    }
    return document


def test_the_golden_document_and_the_robust_document_are_read() -> None:
    assert result_from_bytes(serialised(golden_result_document()))
    assert result_from_bytes(serialised(robust_document()))


# ---- the keys of the format --------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(golden_result_document()))
def test_every_key_is_required(key: str) -> None:
    message = refused(edited(golden_result_document(), key, DELETE))

    assert f"'{key}'" in message


def test_a_key_that_is_not_part_of_the_format_is_refused_and_never_named() -> None:
    for where in ("", "metrics.", "provenance.", "provenance.system.", "bootstrap."):
        document = edited(golden_result_document(), f"{where}{MARKER}", 1)

        assert "not part of the format" in refused(document)


def test_a_created_timestamp_is_not_part_of_the_format() -> None:
    document = edited(golden_result_document(), "created", "2026-11-10T21:00:00Z")

    assert "not part of the format" in refused(document)


def test_the_schema_must_be_the_first_version() -> None:
    for schema in ("rinchen.eval/2", "rinchen.eval/0", MARKER, 1, None):
        assert "'schema'" in refused(edited(golden_result_document(), "schema", schema))


# ---- scalars -----------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [10.0, True, 0, -1, "10", None, [10]])
def test_n_segments_is_an_integer_of_at_least_one(bad: object) -> None:
    assert "'n_segments'" in refused(
        edited(golden_result_document(), "n_segments", bad)
    )


@pytest.mark.parametrize("bad", [1, 0, "true", None, "yes"])
def test_trained_is_a_boolean(bad: object) -> None:
    assert "'trained'" in refused(edited(golden_result_document(), "trained", bad))


@pytest.mark.parametrize("field", ["system", "suite"])
@pytest.mark.parametrize("bad", ["", MARKER, "a b", "x" * 65, 3, None, "a/b"])
def test_a_system_and_a_suite_are_ids(field: str, bad: object) -> None:
    document = edited(golden_result_document(), field, bad)

    assert f"'{field}'" in refused(document)


@pytest.mark.parametrize(
    "bad",
    [
        "khk_Mong-eng_Latn",
        "eng_Latn-khk_Mong",
        "eng_Latn-eng_Latn",
        "eng_Latn",
        "eng_Latn-xx_Latn",
        "khk_Cyrl-eng_Latn-rus_Cyrl",
        "",
        MARKER,
        3,
        None,
    ],
)
def test_a_direction_is_two_different_languages_of_the_registry(bad: object) -> None:
    assert "'direction'" in refused(edited(golden_result_document(), "direction", bad))


def test_khk_mong_is_refused_by_name() -> None:
    message = refused(
        edited(golden_result_document(), "direction", "khk_Mong-eng_Latn")
    )

    assert "khk_Mong" in message


@pytest.mark.parametrize(
    "field", ["bleu_signature", "chrf_signature", "chrf_pp_signature"]
)
@pytest.mark.parametrize("bad", ["", 3, None, ["x"]])
def test_a_signature_is_a_text(field: str, bad: object) -> None:
    assert f"'{field}'" in refused(edited(golden_result_document(), field, bad))


# ---- statuses ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"weird": 10},
        {"ok": 9},
        {"ok": 11},
        {},
        {"ok": 10, "empty": 0},
        {"ok": 10.0},
        {"ok": True},
        {"ok": "10"},
        {"ok": -1, "empty": 11},
        [["ok", 10]],
        None,
    ],
)
def test_the_statuses_are_counts_of_the_engines_five_that_add_up_to_the_segments(
    bad: object,
) -> None:
    assert "'statuses'" in refused(edited(golden_result_document(), "statuses", bad))


def test_a_status_the_engine_does_not_have_is_not_named() -> None:
    document = edited(golden_result_document(), "statuses", {MARKER: 10})

    assert "'statuses'" in refused(document)


# ---- metrics and what is unmeasured ------------------------------------------------------------


def test_the_metrics_are_exactly_the_six() -> None:
    assert "'metrics'" in refused(
        edited(golden_result_document(), "metrics.bleu", DELETE)
    )
    assert "'metrics'" in refused(edited(golden_result_document(), "metrics.extra", 1))
    assert "'metrics'" in refused(edited(golden_result_document(), "metrics", None))


@pytest.mark.parametrize("bad", ["59.4", True, [59.4], {"a": 1}])
def test_a_metric_is_a_number_or_null(bad: object) -> None:
    assert "'metrics.bleu'" in refused(
        edited(golden_result_document(), "metrics.bleu", bad)
    )


@pytest.mark.parametrize("bare", ["NaN", "Infinity", "-Infinity"])
def test_a_metric_that_is_not_finite_is_refused(bare: str) -> None:
    data = serialised(golden_result_document()).replace(
        b'"bleu": 59.40481339822236', f'"bleu": {bare}'.encode(), 1
    )

    assert "'metrics.bleu'" in refused(data)


def test_a_metric_in_neither_metrics_nor_unmeasured_is_refused() -> None:
    document = edited(golden_result_document(), "metrics.bleu", None)

    message = refused(document)

    assert "'bleu'" in message
    assert "unmeasured" in message


def test_a_metric_in_both_is_refused() -> None:
    document = edited(golden_result_document(), "unmeasured.bleu", "some reason")

    message = refused(document)

    assert "'bleu'" in message


def test_a_null_metric_without_a_reason_is_refused() -> None:
    document = edited(golden_result_document(), "unmeasured.comet22", DELETE)

    assert "'comet22'" in refused(document)


@pytest.mark.parametrize("bad", ["", 3, None, ["x"], True])
def test_a_reason_is_a_text(bad: object) -> None:
    document = edited(golden_result_document(), "unmeasured.comet22", bad)

    assert "'unmeasured.comet22'" in refused(document)


def test_the_unmeasured_list_names_only_metrics_of_the_format() -> None:
    document = edited(golden_result_document(), f"unmeasured.{MARKER}", "reason")

    assert "'unmeasured'" in refused(document)
    assert "'unmeasured'" in refused(
        edited(golden_result_document(), "unmeasured", ["a"])
    )


def test_a_result_that_measured_a_neural_metric_has_no_reason_for_it() -> None:
    document = golden_result_document()
    document["metrics"]["comet22"] = 0.8
    document["ci95"]["comet22"] = [0.7, 0.9]
    document["unmeasured"].pop("comet22")

    assert result_from_bytes(serialised(document)).metrics["comet22"] == 0.8


# ---- the intervals -----------------------------------------------------------------------------


def test_the_intervals_are_exactly_the_six() -> None:
    assert "'ci95'" in refused(edited(golden_result_document(), "ci95.bleu", DELETE))
    assert "'ci95'" in refused(edited(golden_result_document(), "ci95.extra", None))


@pytest.mark.parametrize("name", NEURAL)
def test_a_null_metric_has_a_null_interval(name: str) -> None:
    document = edited(golden_result_document(), f"ci95.{name}", [0.0, 1.0])

    assert f"'ci95.{name}'" in refused(document)


@pytest.mark.parametrize(
    "bad",
    [
        [90.0, 80.0],
        [80.0],
        [1.0, 2.0, 3.0],
        "x",
        ["a", "b"],
        [True, False],
        {"lo": 1},
        5,
    ],
)
def test_an_interval_is_two_numbers_the_lower_not_above_the_upper(bad: object) -> None:
    assert "'ci95.bleu'" in refused(edited(golden_result_document(), "ci95.bleu", bad))


def test_an_interval_end_that_is_not_finite_is_refused() -> None:
    data = serialised(golden_result_document()).replace(
        b"72.13730354815289", b"Infinity", 1
    )

    assert "'ci95.bleu'" in refused(data)


def test_a_measured_metric_has_an_interval_exactly_when_there_are_two_units_or_more() -> (
    None
):
    no_interval = edited(golden_result_document(), "ci95.bleu", None)
    assert "'ci95.bleu'" in refused(no_interval)

    one_unit = golden_result_document()
    one_unit["bootstrap"]["units"] = 1
    for name in ("bleu", "chrf", "chrf_pp"):
        one_unit["ci95"][name] = None
    assert result_from_bytes(serialised(one_unit)).ci95["bleu"] is None
    one_unit["ci95"]["bleu"] = [1.0, 2.0]
    assert "'ci95.bleu'" in refused(one_unit)


# ---- the bootstrap -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("resamples", 99),
        ("resamples", True),
        ("resamples", 1000.0),
        ("resamples", "1000"),
        ("seed", -1),
        ("seed", 2**64),
        ("seed", True),
        ("seed", 1.5),
        ("alpha", 0.5),
        ("alpha", 0.0),
        ("alpha", -0.1),
        ("alpha", "0.05"),
        ("alpha", 1),
        ("interval", "basic"),
        ("interval", 3),
        ("rng", "numpy-pcg64"),
        ("rng", None),
        ("unit", "sentence"),
        ("unit", None),
        ("units", 0),
        ("units", 1.0),
        ("units", True),
        ("units", 11),
    ],
)
def test_the_bootstrap_record_holds_what_the_bootstrap_can_do(
    field: str, bad: object
) -> None:
    document = edited(golden_result_document(), f"bootstrap.{field}", bad)

    assert f"'bootstrap.{field}'" in refused(document)


def test_the_bootstrap_has_exactly_its_seven_fields() -> None:
    assert "'bootstrap'" in refused(
        edited(golden_result_document(), "bootstrap.rng", DELETE)
    )
    assert "'bootstrap'" in refused(edited(golden_result_document(), "bootstrap", None))


def test_a_segment_unit_means_one_unit_for_every_segment() -> None:
    document = golden_result_document()
    document["bootstrap"]["unit"] = "segment"

    assert "'bootstrap.units'" in refused(document)

    document["bootstrap"]["units"] = 10
    assert result_from_bytes(serialised(document)).bootstrap.unit == "segment"


# ---- the robustness block, its coverage and its source ---------------------------------------


@pytest.mark.parametrize(
    "name", ["robustness", "robustness_coverage", "robustness_source"]
)
def test_the_three_robustness_fields_are_all_null_or_all_there(name: str) -> None:
    for document in (
        edited(robust_document(), name, None),
        edited(golden_result_document(), name, robust_document()[name]),
    ):
        message = refused(document)

        assert "robustness" in message
        assert "comes with its coverage and its source" in message


def test_a_block_without_its_coverage_is_refused() -> None:
    message = refused(edited(robust_document(), "robustness_coverage", None))

    assert "'robustness_coverage'" in message


def test_a_block_without_exactly_the_six_keys_is_refused() -> None:
    assert "'robustness'" in refused(
        edited(robust_document(), "robustness.term_acc", DELETE)
    )
    assert "'robustness'" in refused(edited(robust_document(), "robustness.extra", 0.5))


@pytest.mark.parametrize("bad", ["0.5", True, [0.5], {"a": 1}])
def test_a_block_value_is_a_number_or_null(bad: object) -> None:
    document = edited(robust_document(), "robustness.numbers_acc", bad)

    assert "'robustness.numbers_acc'" in refused(document)


def test_five_robustness_keys_are_refused() -> None:
    document = edited(robust_document(), "robustness.entity_acc", DELETE)

    assert "'robustness'" in refused(document)


def test_a_coverage_has_the_six_keys_each_with_its_counts() -> None:
    for path in (
        "robustness_coverage.numbers_acc",
        "robustness_coverage.numbers_acc.scored",
        "robustness_coverage.numbers_acc.total",
        "robustness_coverage.mong_source_retention.chrf_mong",
        "robustness_coverage.mong_source_retention.chrf_cyrl",
    ):
        assert "'robustness_coverage" in refused(
            edited(robust_document(), path, DELETE)
        )
    for path in (
        "robustness_coverage.extra",
        "robustness_coverage.numbers_acc.extra",
    ):
        assert "'robustness_coverage" in refused(edited(robust_document(), path, 1))


@pytest.mark.parametrize("bad", [-1, 1.5, "3", True, None])
def test_a_coverage_count_is_a_non_negative_integer(bad: object) -> None:
    document = edited(robust_document(), "robustness_coverage.numbers_acc.scored", bad)

    assert "'robustness_coverage.numbers_acc.scored'" in refused(document)


def test_a_coverage_whose_counts_do_not_add_up_is_refused() -> None:
    document = edited(robust_document(), "robustness_coverage.numbers_acc.total", 99)

    assert "'robustness_coverage.numbers_acc'" in refused(document)


@pytest.mark.parametrize("bad", ["x", True, [1], float("inf")])
def test_the_two_chrf_scores_of_the_retention_are_numbers_or_null(bad: object) -> None:
    path = "robustness_coverage.mong_source_retention.chrf_mong"
    data = json.dumps(edited(robust_document(), path, bad), sort_keys=True).encode()

    assert f"'{path}'" in refused(data)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("binding", "id"),
        ("binding", None),
        ("suite", MARKER),
        ("suite_sha256", "ab" * 31 + "a"),
        ("suite_sha256", "AB" * 32),
        ("robustness_sha256", None),
        ("report_sha256", MARKER),
    ],
)
def test_the_source_of_the_block_is_exact(field: str, bad: object) -> None:
    document = edited(robust_document(), f"robustness_source.{field}", bad)

    assert f"'robustness_source.{field}'" in refused(document)


def test_the_source_has_exactly_its_five_fields() -> None:
    assert "'robustness_source'" in refused(
        edited(robust_document(), "robustness_source.binding", DELETE)
    )
    assert "'robustness_source'" in refused(
        edited(robust_document(), "robustness_source.extra", "x")
    )


# ---- digests and what the parts say of each other ---------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "provenance.suite_sha256",
        "provenance.outputs_sha256",
        "provenance.system.model_sha256",
        "provenance.system.manifest_sha256",
        "provenance.system.outputs_sha256",
        "manifest_sha",
    ],
)
@pytest.mark.parametrize("bad", [MARKER, "ab" * 31 + "a", "AB" * 32, 3, ""])
def test_a_digest_is_64_lowercase_hexadecimal_characters(
    path: str, bad: object
) -> None:
    assert f"'{path}'" in refused(edited(golden_result_document(), path, bad))


def test_the_digests_of_the_run_are_not_null() -> None:
    for path in ("provenance.suite_sha256", "provenance.outputs_sha256"):
        assert f"'{path}'" in refused(edited(golden_result_document(), path, None))


def test_the_manifest_digest_is_the_one_of_the_system() -> None:
    both = golden_result_document()
    both["manifest_sha"] = DIGEST
    assert "'manifest_sha'" in refused(both)

    other = golden_result_document()
    other["manifest_sha"] = DIGEST
    other["provenance"]["system"]["manifest_sha256"] = "cd" * 32
    assert "'manifest_sha'" in refused(other)

    agreeing = golden_result_document()
    agreeing["manifest_sha"] = DIGEST
    agreeing["provenance"]["system"]["manifest_sha256"] = DIGEST
    assert result_from_bytes(serialised(agreeing)).manifest_sha == DIGEST


def test_the_system_of_the_provenance_is_the_system_of_the_result() -> None:
    document = edited(golden_result_document(), "provenance.system.name", "beta")

    assert "'provenance.system.name'" in refused(document)


def test_the_system_record_has_its_six_fields_of_their_types() -> None:
    for path, bad in (
        ("provenance.system.kind", "weird"),
        ("provenance.system.kind", None),
        ("provenance.system.name", 3),
        ("provenance.system.options", [1]),
        ("provenance.system.options", {"a": [1]}),
        ("provenance.system.options", {"a": None}),
        ("provenance.system.options", {"a": {"b": 1}}),
    ):
        assert f"'{path}'" in refused(edited(golden_result_document(), path, bad))
    assert "'provenance.system'" in refused(
        edited(golden_result_document(), "provenance.system.options", DELETE)
    )
    assert "'provenance'" in refused(
        edited(golden_result_document(), "provenance.system", DELETE)
    )
    assert "'provenance'" in refused(
        edited(golden_result_document(), "provenance.extra", 1)
    )


def test_an_option_that_is_not_finite_is_refused() -> None:
    data = serialised(
        edited(golden_result_document(), "provenance.system.options", {"t": 0.0})
    ).replace(b'"t": 0.0', b'"t": NaN')

    assert "'provenance.system.options'" in refused(data)


def test_options_of_a_system_are_read_back_as_they_were_written() -> None:
    document = edited(
        golden_result_document(),
        "provenance.system.options",
        {"beam_size": 4, "device": "cpu", "temperature": 0.0, "sampling": False},
    )

    result = result_from_bytes(serialised(document))

    assert dict(result.provenance.system.options) == {
        "beam_size": 4,
        "device": "cpu",
        "sampling": False,
        "temperature": 0.0,
    }


# ---- bytes -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"{not json",
        b"[]",
        b"3",
        b"null",
        b'"text"',
        b"\xff\xfe{}",
        b"\xef\xbb\xbf{}",
    ],
)
def test_a_file_that_is_not_a_json_object_in_utf_8_is_refused(data: bytes) -> None:
    refused(data)


def test_a_repeated_key_is_refused_wherever_it_is() -> None:
    text = serialised(golden_result_document()).decode()
    top = text.replace('"suite": "golden",', '"suite": "golden", "suite": "golden",', 1)
    nested = text.replace('"units": 5', '"units": 5, "units": 5', 1)

    for data in (top, nested):
        assert "more than once" in refused(data.encode())


def test_a_planted_marker_never_comes_back_wherever_it_is_planted() -> None:
    for path in (
        "system",
        "suite",
        "direction",
        "schema",
        "bootstrap.interval",
        "bootstrap.unit",
        "provenance.system.kind",
        "provenance.system.name",
        "provenance.suite_sha256",
        "manifest_sha",
    ):
        message = refused(edited(golden_result_document(), path, MARKER))

        assert MARKER not in message, path


def test_a_truncated_document_is_refused_whatever_the_cut() -> None:
    data = serialised(golden_result_document())

    for cut in (1, 10, len(data) // 2, len(data) - 2):
        refused(data[:cut])
