"""The result of D13 (spec T11): `result.json`, schema `rinchen.eval/1`, one per (suite, system,
direction) that ran. Sorted keys, two-space indent, LF, NO timestamp (the bytes of a result are a
function of its inputs) and no path; a metric with no value is `null` with its reason in
`unmeasured` (never 0); the robustness block of D12 comes with its coverage and its source, bound
to the system by the model digest or the weaker name.

The golden result (`tests/fixtures/eval/golden_result_alpha_khk_eng.json`) is the result of the
system `alpha` over the golden suite, document unit. It holds a sacrebleu version string in its
signatures, so a sacrebleu bump fails it on purpose. To regenerate it after a bump or after any
change of the result: check each changed literal independently (`golden_result_document()` in
`tests/support/eval_result_documents.py` is built from the golden tables, not from the code),
write `golden_result().to_bytes()` to the file and copy its size and `sha256sum` into the
constants of this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any

import pytest

from rinchen.errors import ConfigError
from rinchen.eval.bootstrap import BootstrapSettings
from rinchen.eval.outputs import Output
from rinchen.eval.result import (
    OUTPUTS_FILE,
    RESULT_FILE,
    RESULT_SCHEMA,
    DirectionResult,
    Provenance,
    build_result,
    result_from_bytes,
)
from rinchen.eval.robust_report import report_bytes, robustness_report
from rinchen.eval.robustness_input import RobustnessInput, read_robustness
from rinchen.eval.scoring import DirectionScores, score_direction
from rinchen.eval.systems import SystemRecord
from tests.support.eval_fakes import golden_outputs, golden_suite
from tests.support.eval_golden import (
    GOLDEN_COUNT_METRICS,
    GOLDEN_NO_SCORER_REASON,
    MEAN_MICRO_A,
)
from tests.support.eval_result_documents import (
    DIRECTION,
    GOLDEN_ALPHA_OUTPUTS_SHA256,
    GOLDEN_RESULT_FILE,
    GOLDEN_SUITE_SHA256,
    NEURAL,
    golden_result_document,
    serialised,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.robust_golden import GOLDEN_COVERAGE, golden_result

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = BootstrapSettings()
# `sha256sum` and `wc -c` of the golden result file, taken after the file was checked against
# `golden_result_document()` (a document built from the golden tables with the standard library).
GOLDEN_RESULT_BYTES = 1966
GOLDEN_RESULT_SHA256 = (
    "f31880cb2147d82f7d6ecb0b2018639bb505a788720e07e926c217f69ec4b28b"
)
MODEL = "a1" * 32
MANIFEST = "b2" * 32
GOLDEN_BLOCK = {
    "numbers_acc": 0.5,
    "placeholder_acc": 0.4,
    "term_acc": 0.75,
    "script_purity": 0.75,
    "entity_acc": 0.5714285714285714,
    "mong_source_retention": 0.5017570743156547,
}


def alpha_scores(**changes: Any) -> DirectionScores:
    outputs = [
        output
        for key, output in golden_outputs("alpha").items()
        if key.direction == DIRECTION
    ]
    return score_direction(golden_suite().view(DIRECTION), outputs, SETTINGS, **changes)


def alpha_record(**changes: Any) -> SystemRecord:
    fields: dict[str, Any] = {
        "kind": "hypotheses",
        "name": "alpha",
        "outputs_sha256": GOLDEN_ALPHA_OUTPUTS_SHA256,
    }
    fields.update(changes)
    return SystemRecord(**fields)


def built(**changes: Any) -> DirectionResult:
    arguments: dict[str, Any] = {
        "scores": alpha_scores(),
        "system": alpha_record(),
        "suite_name": "golden",
        "suite_sha256": GOLDEN_SUITE_SHA256,
        "direction": DIRECTION,
        "outputs_sha256": GOLDEN_ALPHA_OUTPUTS_SHA256,
        "trained": True,
    }
    arguments.update(changes)
    return build_result(**arguments)


def robustness_input(name: str = "toy", *, model: bool = True) -> RobustnessInput:
    result = golden_result()
    system = dataclasses.replace(
        result.system,
        name=name,
        model_sha256=MODEL if model else None,
        manifest_sha256=MANIFEST if model else None,
    )
    result = dataclasses.replace(result, system=system)
    robustness = result.to_bytes()
    report = report_bytes(
        robustness_report(result, hashlib.sha256(robustness).hexdigest())
    )
    return read_robustness(report, robustness)


# ---- the golden file ---------------------------------------------------------------------------


def test_the_golden_result_is_the_committed_file_byte_for_byte() -> None:
    assert built().to_bytes() == GOLDEN_RESULT_FILE.read_bytes()


def test_the_committed_file_is_the_document_of_the_golden_tables_written_with_the_standard_library() -> (
    None
):
    # Independent of the code under test: the literals of `eval_golden` and `json`.
    assert GOLDEN_RESULT_FILE.read_bytes() == serialised(golden_result_document())


def test_the_committed_file_is_pinned_by_its_size_and_its_digest() -> None:
    data = GOLDEN_RESULT_FILE.read_bytes()

    assert len(data) == GOLDEN_RESULT_BYTES
    assert hashlib.sha256(data).hexdigest() == GOLDEN_RESULT_SHA256


def test_the_golden_result_holds_no_timestamp_no_path_and_no_created_field() -> None:
    text = GOLDEN_RESULT_FILE.read_text(encoding="utf-8")

    assert "created" not in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    assert not re.search(r'"/|[A-Za-z]:\\|\.\./', text)


def test_the_file_is_written_as_the_product_writes_every_document() -> None:
    data = GOLDEN_RESULT_FILE.read_bytes()
    text = data.decode("utf-8")

    assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    assert "\r" not in text
    assert text.startswith('{\n  "bleu_signature"')
    assert json.loads(text) == golden_result_document()


def test_the_keys_of_a_result_are_exactly_those_of_the_spec() -> None:
    assert sorted(built().to_dict()) == [
        "bleu_signature",
        "bootstrap",
        "chrf_pp_signature",
        "chrf_signature",
        "ci95",
        "direction",
        "manifest_sha",
        "metrics",
        "n_segments",
        "provenance",
        "robustness",
        "robustness_coverage",
        "robustness_source",
        "schema",
        "statuses",
        "suite",
        "system",
        "trained",
        "unmeasured",
    ]
    assert RESULT_SCHEMA == "rinchen.eval/1"
    assert (RESULT_FILE, OUTPUTS_FILE) == ("result.json", "hypotheses.jsonl")


# ---- what the builder takes from the scores ----------------------------------------------------


def test_the_result_holds_the_scores_the_signatures_and_the_bootstrap_as_they_are() -> (
    None
):
    scores = alpha_scores()

    result = built(scores=scores)

    assert dict(result.metrics) == dict(scores.metrics)
    assert {k: list(v) if v else v for k, v in result.ci95.items()} == dict(scores.ci95)
    assert result.bleu_signature == scores.signatures["bleu"]
    assert result.chrf_signature == scores.signatures["chrf"]
    assert result.chrf_pp_signature == scores.signatures["chrf_pp"]
    assert dict(result.statuses) == dict(scores.statuses)
    assert dict(result.unmeasured) == dict(scores.unmeasured)
    assert result.bootstrap == scores.bootstrap
    assert result.n_segments == scores.n_segments == 10


def test_a_metric_that_was_not_measured_is_null_with_its_reason_and_never_zero() -> (
    None
):
    document = built().to_dict()

    for name in NEURAL:
        assert document["metrics"][name] is None
        assert document["ci95"][name] is None
        assert document["unmeasured"][name] == GOLDEN_NO_SCORER_REASON
    for name in GOLDEN_COUNT_METRICS:
        assert document["metrics"][name] is not None
        assert name not in document["unmeasured"]
    assert '"comet22": null' in built().to_bytes().decode()


def test_a_metric_given_through_the_seam_is_a_number_and_leaves_the_unmeasured_list() -> (
    None
):
    scores = alpha_scores(
        segment_scores={"comet22": [m / 1_000_000 for m in MEAN_MICRO_A[:10]]}
    )

    document = built(scores=scores).to_dict()

    assert document["metrics"]["comet22"] is not None
    assert document["ci95"]["comet22"] is not None
    assert sorted(document["unmeasured"]) == ["metricx24", "metricx24_qe"]
    assert result_from_bytes(built(scores=scores).to_bytes()) == built(scores=scores)


def test_the_status_that_is_not_ok_is_in_the_result() -> None:
    outputs = [
        Output(output.text, "truncated" if index < 2 else "ok")
        for index, output in enumerate(
            o for k, o in golden_outputs("alpha").items() if k.direction == DIRECTION
        )
    ]
    scores = score_direction(golden_suite().view(DIRECTION), outputs, SETTINGS)

    assert built(scores=scores).to_dict()["statuses"] == {"ok": 8, "truncated": 2}


def test_the_provenance_names_the_suite_the_outputs_and_the_whole_system_record() -> (
    None
):
    result = built()

    assert result.provenance == Provenance(
        suite_sha256=GOLDEN_SUITE_SHA256,
        outputs_sha256=GOLDEN_ALPHA_OUTPUTS_SHA256,
        system=alpha_record(),
    )
    assert result.to_dict()["provenance"]["system"] == alpha_record().to_dict()


def test_the_outputs_digest_of_the_provenance_is_the_runs_own_file_not_the_systems() -> (
    None
):
    # A model run has no `outputs_sha256` of its own; the run's hypotheses file has one.
    record = SystemRecord(
        kind="rinchen", name="toy", model_sha256=MODEL, manifest_sha256=MANIFEST
    )

    result = built(system=record, outputs_sha256="ee" * 32)

    assert result.provenance.outputs_sha256 == "ee" * 32
    assert result.provenance.system.outputs_sha256 is None


def test_manifest_sha_is_the_manifest_that_vouches_for_the_model_and_null_otherwise() -> (
    None
):
    model = SystemRecord(
        kind="rinchen", name="toy", model_sha256=MODEL, manifest_sha256=MANIFEST
    )

    assert built(system=model).manifest_sha == MANIFEST
    assert built(system=model).to_dict()["manifest_sha"] == MANIFEST
    assert built().manifest_sha is None
    baseline = SystemRecord(kind="baseline", name="nllb", options={"n": 1})
    assert built(system=baseline).manifest_sha is None


def test_trained_is_what_the_caller_says() -> None:
    assert built(trained=True).trained is True
    assert built(trained=False).trained is False
    assert built(trained=False).to_dict()["trained"] is False


def test_the_system_and_the_suite_of_the_result_are_their_names() -> None:
    document = built().to_dict()

    assert (document["system"], document["suite"]) == ("alpha", "golden")
    assert document["direction"] == DIRECTION


# ---- the robustness block with its coverage ------------------------------------------------------


def test_without_a_robustness_run_the_three_robustness_fields_are_null() -> None:
    document = built().to_dict()

    assert document["robustness"] is None
    assert document["robustness_coverage"] is None
    assert document["robustness_source"] is None


def test_the_block_comes_with_its_coverage_and_its_source() -> None:
    system = SystemRecord(
        kind="rinchen", name="toy", model_sha256=MODEL, manifest_sha256=MANIFEST
    )
    inp = robustness_input()

    document = built(system=system, robustness=inp).to_dict()

    assert document["robustness"] == GOLDEN_BLOCK
    assert {k: dict(v) for k, v in document["robustness_coverage"].items()} == (
        GOLDEN_COVERAGE
    )
    assert document["robustness_source"] == {
        "suite": "golden",
        "suite_sha256": inp.suite_sha256,
        "robustness_sha256": inp.robustness_sha256,
        "report_sha256": inp.report_sha256,
        "binding": "model_digest",
    }


def test_the_block_is_the_same_in_every_direction_of_the_system() -> None:
    system = SystemRecord(
        kind="rinchen", name="toy", model_sha256=MODEL, manifest_sha256=MANIFEST
    )
    inp = robustness_input()

    first = built(system=system, robustness=inp, direction=DIRECTION)

    assert (
        first.robustness
        == built(
            system=system, robustness=inp, direction="eng_Latn-khk_Cyrl"
        ).robustness
    )


def test_a_system_that_is_not_a_model_is_bound_by_its_name_and_the_result_says_so() -> (
    None
):
    inp = robustness_input("alpha", model=False)

    document = built(robustness=inp).to_dict()

    assert document["robustness_source"]["binding"] == "name"
    assert document["robustness_coverage"] is not None


def test_a_robustness_run_of_another_model_is_not_embedded() -> None:
    system = SystemRecord(
        kind="rinchen", name="toy", model_sha256="ee" * 32, manifest_sha256=MANIFEST
    )

    with pytest.raises(ConfigError, match="another model"):
        built(system=system, robustness=robustness_input())


def test_a_rinchen_system_is_never_bound_to_a_robustness_run_by_name() -> None:
    system = SystemRecord(
        kind="rinchen", name="toy", model_sha256=MODEL, manifest_sha256=MANIFEST
    )

    with pytest.raises(ConfigError):
        built(system=system, robustness=robustness_input("toy", model=False))


# ---- the reader --------------------------------------------------------------------------------


def test_what_the_writer_writes_the_reader_reads() -> None:
    result = built()

    assert result_from_bytes(result.to_bytes()) == result
    assert result_from_bytes(GOLDEN_RESULT_FILE.read_bytes()) == result


def test_a_result_with_robustness_a_model_and_a_given_metric_round_trips() -> None:
    system = SystemRecord(
        kind="rinchen",
        name="toy",
        model_sha256=MODEL,
        manifest_sha256=MANIFEST,
        options={"beam_size": 4, "script_mask": "retry", "temperature": 0.0},
    )
    scores = alpha_scores(
        segment_scores={"metricx24": [m / 1_000_000 for m in MEAN_MICRO_A[:10]]}
    )
    result = built(system=system, scores=scores, robustness=robustness_input())

    assert result_from_bytes(result.to_bytes()) == result
    assert result_from_bytes(result.to_bytes()).to_bytes() == result.to_bytes()


def test_a_text_is_written_as_it_is_and_never_as_an_escape() -> None:
    mongolian = "\u0445\u044d\u043b"
    record = SystemRecord(
        kind="baseline", name="nllb", options={"label": mongolian, "n": 1}
    )

    data = built(system=record).to_bytes()

    assert mongolian.encode() in data
    assert b"\\u" not in data
    assert result_from_bytes(data).provenance.system == record


def test_two_builds_of_the_same_inputs_have_the_same_bytes() -> None:
    assert built().to_bytes() == built().to_bytes()


def test_a_read_result_cannot_be_changed_through_its_mappings() -> None:
    result = result_from_bytes(GOLDEN_RESULT_FILE.read_bytes())

    for mapping in (result.metrics, result.statuses, result.unmeasured, result.ci95):
        with pytest.raises(TypeError):
            mapping["bleu"] = 0  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.system = "beta"  # type: ignore[misc]
