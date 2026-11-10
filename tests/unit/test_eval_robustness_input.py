"""D12's files as D13 takes them in (spec T12): `read_robustness` reads the narrow `report.json`
(schema `rinchen.report.robustness/1`) and `robustness.json` (schema `rinchen.robustness/1`) from
the BYTES the command hashed, and returns the six values with their COVERAGE (a block without its
denominators hides how much each value left out) and the digests that say whose they are.

The report carries the block, the coverage and the digest of `robustness.json`; every one of them
is checked against the other file, so a report that was edited, or belongs to another run, is
refused naming the field. `bind` then says whether the block may sit in the result of a system:
by the model and manifest digests for a `rinchen` system (the block would describe another model
otherwise), by the weaker name for any other system, never by the name for a `rinchen` system.

The golden files are D12's (`tests/fixtures/robustness/golden_robustness.json`); a run bound to a
model is made from the same result with a system that has digests.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Callable
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.eval.robust_report import REPORT_SCHEMA, report_bytes, robustness_report
from rinchen.eval.robustness import (
    RESULT_SCHEMA,
    RobustnessResult,
    RobustnessSystem,
    robustness_block,
    robustness_coverage,
)
from rinchen.eval.robustness_input import (
    BINDINGS,
    BLOCK_KEYS,
    RobustnessInput,
    bind,
    read_robustness,
    source_record,
)
from rinchen.eval.systems import SystemRecord
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.robust_golden import (
    GOLDEN_COVERAGE,
    GOLDEN_HYPOTHESES_SHA256,
    GOLDEN_RESULT,
    GOLDEN_SUITE_SHA256,
    golden_result,
)

pytestmark = pytest.mark.usefixtures("no_network")

MARKER = "SECRET-MARKER-64"
MODEL = "a1" * 32
MANIFEST = "b2" * 32
OTHER = "c3" * 32
# The six values D12's golden result holds (D12's tests derived them independently).
GOLDEN_BLOCK = {
    "numbers_acc": 0.5,
    "placeholder_acc": 0.4,
    "term_acc": 0.75,
    "script_purity": 0.75,
    "entity_acc": 0.5714285714285714,
    "mong_source_retention": 0.5017570743156547,
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialised(document: Any) -> bytes:
    return f"{json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)}\n".encode()


def model_result(name: str = "toy", **digests: str | None) -> RobustnessResult:
    """The golden result as if a model had written the hypotheses."""
    system = RobustnessSystem(
        name=name,
        model_sha256=digests.get("model_sha256", MODEL),
        manifest_sha256=digests.get("manifest_sha256", MANIFEST),
        options={"beam_size": 4},
    )
    return dataclasses.replace(golden_result(), system=system)


def files_of(
    result: RobustnessResult,
    *,
    edit_robustness: Callable[[dict[str, Any]], None] | None = None,
    edit_report: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[bytes, bytes]:
    """`(report.json, robustness.json)` of `result`; the report names the digest of the
    robustness bytes it is given, after `edit_robustness` and before `edit_report`."""
    robustness = result.to_dict()
    if edit_robustness is not None:
        edit_robustness(robustness)
    robustness_bytes = serialised(robustness)
    report = robustness_report(result, sha(robustness_bytes))
    if edit_report is not None:
        edit_report(report)
    return report_bytes(report), robustness_bytes


def golden_files(**edits: Callable[[dict[str, Any]], None]) -> tuple[bytes, bytes]:
    return files_of(golden_result(), **edits)


def refused(report: bytes, robustness: bytes) -> str:
    with pytest.raises(DataError) as caught:
        read_robustness(report, robustness)
    message = str(caught.value)
    assert MARKER not in message
    return message


def system_record(
    kind: str = "rinchen",
    name: str = "toy",
    model: str | None = MODEL,
    manifest: str | None = MANIFEST,
) -> SystemRecord:
    return SystemRecord(
        kind=kind,  # type: ignore[arg-type]
        name=name,
        model_sha256=model,
        manifest_sha256=manifest,
    )


# ---- the golden files --------------------------------------------------------------------------


def test_the_golden_robustness_file_is_the_serialisation_of_the_golden_result() -> None:
    # The premise of the tests below: the files built here are D12's own bytes.
    assert golden_result().to_bytes() == GOLDEN_RESULT.read_bytes()


def test_the_golden_files_give_the_block_the_coverage_and_the_digests() -> None:
    report, robustness = golden_files()

    read = read_robustness(report, robustness)

    assert isinstance(read, RobustnessInput)
    assert dict(read.block) == GOLDEN_BLOCK
    assert list(read.block) == list(BLOCK_KEYS)
    assert {key: dict(value) for key, value in read.coverage.items()} == GOLDEN_COVERAGE
    assert read.system_name == "golden"
    assert read.model_sha256 is None
    assert read.manifest_sha256 is None
    assert read.hypotheses_sha256 == GOLDEN_HYPOTHESES_SHA256
    assert read.suite_name == "golden"
    assert read.suite_sha256 == GOLDEN_SUITE_SHA256
    assert read.robustness_sha256 == sha(robustness) == sha(GOLDEN_RESULT.read_bytes())
    assert read.report_sha256 == sha(report)


def test_the_block_and_the_coverage_are_what_d12_computes_from_the_result() -> None:
    result = golden_result()

    read = read_robustness(*golden_files())

    assert dict(read.block) == robustness_block(result)
    assert {key: dict(value) for key, value in read.coverage.items()} == (
        robustness_coverage(result)
    )


def test_the_keys_of_the_block_are_the_six_of_d12_in_its_order() -> None:
    assert BLOCK_KEYS == tuple(robustness_block(golden_result()))
    assert len(BLOCK_KEYS) == 6


def test_the_block_and_the_coverage_cannot_be_changed_through_the_input() -> None:
    read = read_robustness(*golden_files())

    with pytest.raises(TypeError):
        read.block["numbers_acc"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        read.coverage["numbers_acc"]["scored"] = 99  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        read.system_name = "other"  # type: ignore[misc]


def test_a_value_the_section_did_not_score_is_none_and_never_zero() -> None:
    result = golden_result()
    sections = dict(result.sections)
    del sections["entities"]
    without_entities = dataclasses.replace(result, sections=sections)

    read = read_robustness(*files_of(without_entities))

    assert read.block["entity_acc"] is None
    assert dict(read.coverage["entity_acc"]) == {"scored": 0, "excluded": 0, "total": 0}


def test_a_model_run_carries_its_digests() -> None:
    read = read_robustness(*files_of(model_result()))

    assert read.model_sha256 == MODEL
    assert read.manifest_sha256 == MANIFEST
    assert read.hypotheses_sha256 is None
    assert read.system_name == "toy"


# ---- the report and robustness.json must agree ----------------------------------------------


def test_a_report_naming_another_digest_of_robustness_json_is_refused() -> None:
    report, robustness = golden_files(
        edit_report=lambda r: r.__setitem__("robustness_sha256", OTHER)
    )

    message = refused(report, robustness)

    assert "robustness_sha256" in message
    assert OTHER not in message


def test_robustness_json_changed_after_the_report_was_written_is_refused() -> None:
    report, robustness = golden_files()
    changed = serialised({**json.loads(robustness), "rows": {}})

    assert "robustness_sha256" in refused(report, changed)


def test_a_block_with_a_value_changed_in_the_report_only_is_refused() -> None:
    report, robustness = golden_files(
        edit_report=lambda r: r["robustness"].__setitem__("numbers_acc", 0.9)
    )

    assert "robustness.numbers_acc" in refused(report, robustness)


def test_a_block_with_a_value_changed_in_robustness_json_only_is_refused() -> None:
    # The report names the digest of the CHANGED file, so only the block differs.
    report, robustness = golden_files(
        edit_robustness=lambda r: r["robustness"].__setitem__("script_purity", 0.1)
    )

    assert "robustness.script_purity" in refused(report, robustness)


def test_a_block_that_is_none_in_one_file_and_a_number_in_the_other_is_refused() -> (
    None
):
    report, robustness = golden_files(
        edit_report=lambda r: r["robustness"].__setitem__("entity_acc", None)
    )

    assert "robustness.entity_acc" in refused(report, robustness)


def test_a_system_that_differs_between_the_files_is_refused() -> None:
    for field, value in (
        ("name", "other"),
        ("model_sha256", MODEL),
        ("manifest_sha256", MANIFEST),
    ):
        report, robustness = golden_files(
            edit_report=lambda r, f=field, v=value: r["system"].__setitem__(f, v)
        )

        assert f"system.{field}" in refused(report, robustness)


def test_a_suite_that_differs_between_the_files_is_refused() -> None:
    for field, value in (("name", "other"), ("sha256", OTHER)):
        report, robustness = golden_files(
            edit_report=lambda r, f=field, v=value: r["suite"].__setitem__(f, v)
        )

        assert f"suite.{field}" in refused(report, robustness)


def test_a_coverage_that_the_sections_do_not_back_is_refused() -> None:
    for field in ("scored", "excluded", "total"):
        report, robustness = golden_files(
            edit_report=lambda r, f=field: r["coverage"]["numbers_acc"].__setitem__(
                f, r["coverage"]["numbers_acc"][f] + 1
            )
        )

        assert "coverage.numbers_acc" in refused(report, robustness)


def test_a_coverage_chrf_that_the_sections_do_not_back_is_refused() -> None:
    for field in ("chrf_cyrl", "chrf_mong"):
        report, robustness = golden_files(
            edit_report=lambda r, f=field: r["coverage"][
                "mong_source_retention"
            ].__setitem__(f, 1.5)
        )

        assert f"coverage.mong_source_retention.{field}" in refused(report, robustness)


# ---- the shape of the report -----------------------------------------------------------------


def test_every_key_of_the_report_is_required_and_no_other_is_allowed() -> None:
    keys = list(json.loads(golden_files()[0]))
    for key in keys:
        report, robustness = golden_files(edit_report=lambda r, k=key: r.pop(k))

        assert refused(report, robustness)
    report, robustness = golden_files(edit_report=lambda r: r.__setitem__("extra", 1))

    assert refused(report, robustness)


def test_a_schema_of_another_version_is_refused_in_either_file() -> None:
    other_report, robustness = golden_files(
        edit_report=lambda r: r.__setitem__("schema", "rinchen.report.robustness/2")
    )
    report, other_robustness = golden_files(
        edit_robustness=lambda r: r.__setitem__("schema", "rinchen.robustness/2")
    )

    assert "schema" in refused(other_report, robustness)
    assert "schema" in refused(report, other_robustness)
    assert REPORT_SCHEMA == "rinchen.report.robustness/1"
    assert RESULT_SCHEMA == "rinchen.robustness/1"


def test_the_two_files_must_not_be_swapped() -> None:
    report, robustness = golden_files()

    assert "schema" in refused(robustness, report)


def test_the_name_of_the_robustness_file_must_be_the_one_of_d12() -> None:
    report, robustness = golden_files(
        edit_report=lambda r: r.__setitem__("robustness_file", "other.json")
    )

    assert "robustness_file" in refused(report, robustness)


def test_a_block_without_exactly_the_six_keys_is_refused() -> None:
    for edit in (
        lambda r: r["robustness"].pop("term_acc"),
        lambda r: r["robustness"].__setitem__("extra_acc", 0.5),
    ):
        report, robustness = golden_files(edit_report=edit)

        assert "robustness" in refused(report, robustness)


def test_a_block_value_that_is_not_a_number_or_none_is_refused() -> None:
    for bad in ("0.5", True, [0.5], {"a": 1}):
        report, robustness = golden_files(
            edit_report=lambda r, b=bad: r["robustness"].__setitem__("numbers_acc", b),
            edit_robustness=lambda r, b=bad: r["robustness"].__setitem__(
                "numbers_acc", b
            ),
        )

        assert "robustness.numbers_acc" in refused(report, robustness)


def test_a_block_value_that_is_not_finite_is_refused() -> None:
    report, robustness = golden_files()
    for bare in (b"NaN", b"Infinity", b"-Infinity"):
        broken = report.replace(b'"numbers_acc": 0.5', b'"numbers_acc": ' + bare, 1)

        assert "robustness.numbers_acc" in refused(broken, robustness)


def test_a_coverage_without_the_six_keys_or_with_a_missing_count_is_refused() -> None:
    edits: list[Callable[[dict[str, Any]], None]] = [
        lambda r: r["coverage"].pop("entity_acc"),
        lambda r: r["coverage"].__setitem__("extra_acc", {}),
        lambda r: r["coverage"]["numbers_acc"].pop("scored"),
        lambda r: r["coverage"]["numbers_acc"].pop("total"),
        lambda r: r["coverage"]["numbers_acc"].__setitem__("extra", 1),
        lambda r: r["coverage"]["mong_source_retention"].pop("chrf_cyrl"),
        lambda r: r["coverage"]["mong_source_retention"].pop("chrf_mong"),
    ]
    for edit in edits:
        report, robustness = golden_files(edit_report=edit)

        assert "coverage" in refused(report, robustness)


def test_a_count_that_is_not_a_non_negative_integer_is_refused() -> None:
    for bad in (-1, 1.5, "3", True, None):
        report, robustness = golden_files(
            edit_report=lambda r, b=bad: r["coverage"]["numbers_acc"].__setitem__(
                "scored", b
            )
        )

        assert "coverage.numbers_acc.scored" in refused(report, robustness)


def test_a_count_that_does_not_add_up_is_refused() -> None:
    report, robustness = golden_files(
        edit_report=lambda r: r["coverage"]["numbers_acc"].__setitem__("total", 99)
    )

    assert "coverage.numbers_acc" in refused(report, robustness)


def test_a_digest_that_is_not_a_sha256_is_refused() -> None:
    for edit in (
        lambda r: r["suite"].__setitem__("sha256", MARKER),
        lambda r: r["system"].__setitem__("model_sha256", MARKER),
        lambda r: r.__setitem__("robustness_sha256", MARKER),
        lambda r: r["suite"].__setitem__("sha256", "AB" * 32),
    ):
        report, robustness = golden_files(edit_report=edit)

        refused(report, robustness)


def test_a_name_that_is_not_an_id_is_refused() -> None:
    report, robustness = golden_files(
        edit_report=lambda r: r["system"].__setitem__("name", f"{MARKER} x"),
        edit_robustness=lambda r: r["system"].__setitem__("name", f"{MARKER} x"),
    )

    assert "system.name" in refused(report, robustness)


def test_a_file_that_is_not_json_an_object_or_utf_8_is_refused() -> None:
    report, robustness = golden_files()
    for broken in (b"", b"{not json", b"[]", b"3", b"\xff\xfe", b'{"a": 1, "a": 2}'):
        assert refused(broken, robustness)
        assert refused(report, broken)


def test_a_repeated_key_in_either_file_is_refused() -> None:
    report, robustness = golden_files()
    repeated_report = report.replace(
        b'"robustness_file": "robustness.json"',
        b'"robustness_file": "robustness.json", "robustness_file": "robustness.json"',
        1,
    )

    assert refused(repeated_report, robustness)


def test_the_messages_never_hold_what_the_files_held() -> None:
    report, robustness = golden_files()
    document = json.loads(report)
    document["robustness"]["numbers_acc"] = MARKER
    document["system"]["name"] = MARKER
    planted = copy.deepcopy(document)
    planted["coverage"]["numbers_acc"]["scored"] = MARKER

    for bad in (document, planted):
        refused(report_bytes(bad), robustness)


# ---- the binding to a system -------------------------------------------------------------------


def input_of(result: RobustnessResult) -> RobustnessInput:
    return read_robustness(*files_of(result))


def test_a_rinchen_system_with_the_digests_of_the_robustness_run_is_bound_by_the_digests() -> (
    None
):
    assert bind(input_of(model_result()), system_record()) == "model_digest"


def test_a_rinchen_system_with_one_digest_different_is_refused_and_the_model_is_named() -> (
    None
):
    inp = input_of(model_result())
    for record in (
        system_record(model=OTHER),
        system_record(manifest=OTHER),
        system_record(model=OTHER, manifest=OTHER),
        system_record(model=None, manifest=None),
    ):
        with pytest.raises(ConfigError, match="another model") as caught:
            bind(inp, record)

        assert OTHER not in str(caught.value)
        assert MODEL not in str(caught.value)
        assert MANIFEST not in str(caught.value)


def test_the_name_of_a_rinchen_system_never_binds_it() -> None:
    # Same name, digests that differ: a name is not evidence about a model.
    with pytest.raises(ConfigError, match="another model"):
        bind(input_of(model_result(name="toy")), system_record(name="toy", model=OTHER))


def test_a_rinchen_system_cannot_be_bound_to_a_run_that_had_no_model() -> None:
    # The golden robustness run read hypotheses from a file: no digest to compare, and a
    # rinchen system compared by name alone is refused.
    inp = input_of(golden_result())

    with pytest.raises(ConfigError, match="without a model") as caught:
        bind(inp, system_record(name="golden"))

    assert "never by its name" in str(caught.value)


def test_another_kind_of_system_with_the_same_name_is_bound_by_the_name() -> None:
    inp = input_of(golden_result())

    assert (
        bind(
            inp,
            system_record(kind="baseline", name="golden", model=None, manifest=None),
        )
        == "name"
    )
    assert (
        bind(
            inp,
            system_record(kind="hypotheses", name="golden", model=None, manifest=None),
        )
        == "name"
    )


def test_another_kind_of_system_with_another_name_is_refused_naming_both_names() -> (
    None
):
    inp = input_of(golden_result())

    for kind in ("baseline", "hypotheses"):
        with pytest.raises(ConfigError, match="another system") as caught:
            bind(inp, system_record(kind=kind, name="nllb", model=None, manifest=None))

        assert "'golden'" in str(caught.value)
        assert "'nllb'" in str(caught.value)


def test_a_baseline_is_bound_by_the_name_even_to_a_run_that_had_a_model() -> None:
    inp = input_of(model_result(name="nllb"))

    assert (
        bind(
            inp, system_record(kind="baseline", name="nllb", model=None, manifest=None)
        )
        == "name"
    )


def test_the_two_bindings_are_the_two_words_a_result_records() -> None:
    assert BINDINGS == ("model_digest", "name")


def test_the_binding_message_never_holds_a_name_that_is_not_an_id() -> None:
    inp = input_of(golden_result())
    odd = f"{MARKER} x"

    with pytest.raises(ConfigError) as caught:
        bind(inp, system_record(kind="baseline", name=odd, model=None, manifest=None))

    assert MARKER not in str(caught.value)


# ---- what a result records of the source ---------------------------------------------------------


def test_the_source_record_names_the_suite_the_digests_and_the_binding() -> None:
    inp = read_robustness(*golden_files())

    record = source_record(inp, "name")

    assert record == {
        "suite": "golden",
        "suite_sha256": GOLDEN_SUITE_SHA256,
        "robustness_sha256": inp.robustness_sha256,
        "report_sha256": inp.report_sha256,
        "binding": "name",
    }
    assert list(record) == sorted(record)


def test_the_source_record_refuses_a_binding_that_is_none_of_the_two() -> None:
    inp = read_robustness(*golden_files())

    with pytest.raises(ValueError, match="binding"):
        source_record(inp, "id")
