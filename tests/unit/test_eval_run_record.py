"""The run record of D13 (spec T11): `run.json`, schema `rinchen.eval.run/1`, the table of contents
of a run. The suite and its purpose (`selection` for a `dev` suite, `report` for the others: the
report refuses the first), the system record, the settings of the scoring, one entry for EVERY
direction of the suite with its state and reason (a direction that did not run keeps its row
count and has no result file), and each file the run wrote with its SHA-256.

The reader is strict like the one of `result.json`: exactly the keys, types, every state one of
the five, a reason exactly when a direction did not run, a result path exactly when it did, no
path that leaves the run's directory, digests of 64 lowercase hexadecimal characters. A refusal
names the field and never quotes a value.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.eval.result import (
    OUTPUTS_FILE,
    RUN_FILE,
    RUN_SCHEMA,
    RunDirection,
    RunRecord,
    RunSettings,
    RunSuite,
    run_from_bytes,
)
from rinchen.eval.systems import SystemRecord
from tests.support.eval_result_documents import (
    DELETE,
    DIGEST,
    FIRST,
    GOLDEN_ALPHA_OUTPUTS_SHA256,
    GOLDEN_SUITE_SHA256,
    MARKER,
    OTHER_DIGEST,
    SECOND,
    THIRD,
    edited,
    run_document,
    serialised,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SOURCE = {
    "suite": "golden",
    "suite_sha256": DIGEST,
    "robustness_sha256": DIGEST,
    "report_sha256": OTHER_DIGEST,
    "binding": "model_digest",
}


def record() -> RunRecord:
    return run_from_bytes(serialised(run_document()))


def refused(document: dict[str, Any] | bytes) -> str:
    raw = document if isinstance(document, bytes) else serialised(document)
    with pytest.raises(DataError) as caught:
        run_from_bytes(raw)
    message = str(caught.value)
    assert message.startswith("not an evaluation run: ")
    assert MARKER not in message
    return message


# ---- the document ------------------------------------------------------------------------------


def test_the_file_names_and_the_schema() -> None:
    assert RUN_SCHEMA == "rinchen.eval.run/1"
    assert (RUN_FILE, OUTPUTS_FILE) == ("run.json", "hypotheses.jsonl")


def test_a_valid_document_is_read_into_the_record() -> None:
    read = record()

    assert read.suite == RunSuite("golden", GOLDEN_SUITE_SHA256)
    assert read.purpose == "report"
    assert read.system == SystemRecord(
        kind="hypotheses", name="alpha", outputs_sha256=GOLDEN_ALPHA_OUTPUTS_SHA256
    )
    assert read.settings == RunSettings(
        resamples=1000, seed=20261110, alpha=0.05, selection=None
    )
    assert [d.direction for d in read.directions] == [FIRST, SECOND, THIRD]
    assert read.directions[0] == RunDirection(
        FIRST, "run", None, 10, f"{FIRST}/result.json"
    )
    assert read.directions[2] == RunDirection(
        THIRD, "not_trained", "not one of the model's trained directions", 10, None
    )
    assert dict(read.files)[OUTPUTS_FILE] == GOLDEN_ALPHA_OUTPUTS_SHA256
    assert read.robustness_source is None
    assert dict(read.segment_scores) == {}


def test_what_the_writer_writes_is_the_document_and_the_reader_reads_it_back() -> None:
    read = record()

    assert read.to_bytes() == serialised(run_document())
    assert run_from_bytes(read.to_bytes()) == read


def test_the_keys_of_a_run_are_exactly_the_spec_and_there_is_no_timestamp() -> None:
    document = record().to_dict()

    assert sorted(document) == [
        "directions",
        "files",
        "purpose",
        "robustness_source",
        "schema",
        "segment_scores",
        "settings",
        "suite",
        "system",
    ]
    assert sorted(document["directions"][0]) == [
        "direction",
        "reason",
        "result",
        "rows",
        "state",
    ]
    assert "created" not in record().to_bytes().decode()


def test_a_run_with_a_selection_and_a_source_and_scores_round_trips() -> None:
    document = run_document()
    document["settings"]["selection"] = [FIRST]
    document["directions"][1].update(
        state="not_selected", reason="not in the selection", result=None
    )
    document["directions"][2].update(
        state="not_selected", reason="not in the selection"
    )
    del document["files"][f"{SECOND}/result.json"]
    document["robustness_source"] = dict(SOURCE)
    document["segment_scores"] = {"comet22": DIGEST, "metricx24": OTHER_DIGEST}

    read = run_from_bytes(serialised(document))

    assert read.settings.selection == (FIRST,)
    assert dict(read.robustness_source or {}) == SOURCE
    assert dict(read.segment_scores) == {"comet22": DIGEST, "metricx24": OTHER_DIGEST}
    assert run_from_bytes(read.to_bytes()) == read


def test_every_state_of_the_runner_is_a_state_of_the_record() -> None:
    for state in ("not_trained", "unsupported_language", "not_in_file"):
        document = edited(run_document(), "directions.2.state", state)

        assert run_from_bytes(serialised(document)).directions[2].state == state


def test_a_selection_dev_suite_purpose_is_a_purpose_too() -> None:
    document = edited(run_document(), "purpose", "selection")

    assert run_from_bytes(serialised(document)).purpose == "selection"


def test_the_record_is_read_only() -> None:
    read = record()

    with pytest.raises(TypeError):
        read.files["x"] = DIGEST  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        read.purpose = "selection"  # type: ignore[misc]
    assert isinstance(read.directions, tuple)


# ---- the keys and the scalars ------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(run_document()))
def test_every_key_is_required(key: str) -> None:
    assert f"'{key}'" in refused(edited(run_document(), key, DELETE))


def test_a_key_that_is_not_part_of_the_format_is_refused_and_never_named() -> None:
    for where in ("", "suite.", "settings.", "directions.0."):
        assert "not part of the format" in refused(
            edited(run_document(), f"{where}{MARKER}", 1)
        )


def test_the_schema_must_be_the_first_version() -> None:
    for schema in ("rinchen.eval.run/2", "rinchen.eval/1", MARKER, None):
        assert "'schema'" in refused(edited(run_document(), "schema", schema))


@pytest.mark.parametrize("bad", ["train", "", MARKER, None, "Report", 3])
def test_the_purpose_is_selection_or_report(bad: object) -> None:
    assert "'purpose'" in refused(edited(run_document(), "purpose", bad))


@pytest.mark.parametrize("bad", [MARKER, "", None, 3, "a b", "x" * 65])
def test_the_suite_has_an_id(bad: object) -> None:
    assert "'suite.name'" in refused(edited(run_document(), "suite.name", bad))


@pytest.mark.parametrize("bad", [MARKER, "", None, 3, "ab" * 31 + "a", "AB" * 32])
def test_the_suite_has_a_digest(bad: object) -> None:
    assert "'suite.sha256'" in refused(edited(run_document(), "suite.sha256", bad))


def test_the_system_is_a_record_of_its_six_fields() -> None:
    for path, bad in (
        ("system.kind", "weird"),
        ("system.kind", None),
        ("system.name", 3),
        ("system.model_sha256", MARKER),
        ("system.options", [1]),
        ("system.options", {"a": None}),
    ):
        assert f"'{path}'" in refused(edited(run_document(), path, bad))
    assert "'system'" in refused(edited(run_document(), "system.options", DELETE))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("resamples", 99),
        ("resamples", True),
        ("resamples", 1000.0),
        ("seed", -1),
        ("seed", 2**64),
        ("seed", False),
        ("alpha", 0.5),
        ("alpha", 0.0),
        ("alpha", "0.05"),
        ("selection", "khk_Cyrl-eng_Latn"),
        ("selection", [MARKER]),
        ("selection", [3]),
        ("selection", [FIRST, FIRST]),
        ("selection", [THIRD + "x"]),
        ("selection", []),
    ],
)
def test_the_settings_are_what_the_bootstrap_can_do(field: str, bad: object) -> None:
    assert f"'settings.{field}'" in refused(
        edited(run_document(), f"settings.{field}", bad)
    )


def test_the_settings_have_exactly_their_four_fields() -> None:
    assert "'settings'" in refused(edited(run_document(), "settings.seed", DELETE))
    assert "'settings'" in refused(edited(run_document(), "settings", None))


def test_a_selection_names_directions_of_the_run() -> None:
    document = run_document()
    document["settings"]["selection"] = ["deu_Latn-khk_Cyrl"]

    assert "'settings.selection'" in refused(document)


# ---- the directions ------------------------------------------------------------------------------


def test_a_run_lists_at_least_one_direction_each_once() -> None:
    assert "'directions'" in refused(edited(run_document(), "directions", []))
    assert "'directions'" in refused(edited(run_document(), "directions", {}))
    duplicate = run_document()
    duplicate["directions"][1]["direction"] = FIRST
    duplicate["directions"][1]["result"] = f"{FIRST}/result.json"

    assert "'directions.1.direction'" in refused(duplicate)


@pytest.mark.parametrize(
    "bad",
    ["khk_Mong-eng_Latn", "eng_Latn-eng_Latn", "eng_Latn", "", MARKER, None, 3],
)
def test_a_direction_is_two_different_languages_of_the_registry(bad: object) -> None:
    document = edited(run_document(), "directions.0.direction", bad)

    assert "'directions.0.direction'" in refused(document)


@pytest.mark.parametrize("bad", ["ran", "", MARKER, None, "Run", 3])
def test_a_state_is_one_of_the_five(bad: object) -> None:
    assert "'directions.0.state'" in refused(
        edited(run_document(), "directions.0.state", bad)
    )


def test_a_direction_that_ran_has_no_reason_and_one_that_did_not_has_one() -> None:
    assert "'directions.0.reason'" in refused(
        edited(run_document(), "directions.0.reason", "why")
    )
    for bad in (None, "", 3, ["x"]):
        assert "'directions.2.reason'" in refused(
            edited(run_document(), "directions.2.reason", bad)
        )


def test_a_direction_that_ran_has_its_result_file_and_one_that_did_not_has_none() -> (
    None
):
    assert "'directions.0.result'" in refused(
        edited(run_document(), "directions.0.result", None)
    )
    assert "'directions.2.result'" in refused(
        edited(run_document(), "directions.2.result", f"{THIRD}/result.json")
    )


@pytest.mark.parametrize(
    "bad",
    [
        "result.json",
        f"{SECOND}/result.json",
        "../result.json",
        MARKER,
        3,
        f"/{FIRST}/result.json",
    ],
)
def test_a_result_file_is_the_one_of_its_own_direction(bad: object) -> None:
    assert "'directions.0.result'" in refused(
        edited(run_document(), "directions.0.result", bad)
    )


@pytest.mark.parametrize("bad", [0, -1, 1.0, True, "10", None])
def test_the_rows_of_a_direction_are_a_positive_integer(bad: object) -> None:
    assert "'directions.0.rows'" in refused(
        edited(run_document(), "directions.0.rows", bad)
    )


def test_a_direction_has_exactly_its_five_fields() -> None:
    assert "'directions.0'" in refused(
        edited(run_document(), "directions.0.rows", DELETE)
    )
    assert "'directions.0'" in refused(edited(run_document(), "directions.0.extra", 1))
    assert "'directions.0'" in refused(edited(run_document(), "directions.0", "run"))


def test_not_selected_is_exactly_what_a_selection_leaves_out() -> None:
    selected_everything = run_document()
    selected_everything["settings"]["selection"] = [FIRST, SECOND, THIRD]
    selected_everything["directions"][2]["state"] = "not_selected"
    assert "'directions.2.state'" in refused(selected_everything)

    nothing_selected = run_document()
    nothing_selected["directions"][2]["state"] = "not_selected"
    assert "'directions.2.state'" in refused(nothing_selected)

    # The runner asks the selection first: a direction it leaves out is not_selected, never
    # not_trained.
    left_out = run_document()
    left_out["settings"]["selection"] = [FIRST, SECOND]
    assert "'directions.2.state'" in refused(left_out)
    left_out["directions"][2].update(
        state="not_selected", reason="not in the selection"
    )
    assert run_from_bytes(serialised(left_out))
    left_out["settings"]["selection"] = [FIRST]
    assert "'directions.1.state'" in refused(left_out)


# ---- files -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../x.json",
        "a/../x",
        "/abs/x",
        "a//b",
        "./x",
        "a/./b",
        "a\\b",
        "",
        "x" * 201,
        f"{MARKER}\n",
        "a\x00b",
    ],
)
def test_a_file_path_that_leaves_the_run_directory_is_refused(path: str) -> None:
    document = run_document()
    document["files"][path] = DIGEST

    assert "'files'" in refused(document)


@pytest.mark.parametrize("bad", ["ab" * 31 + "a", "AB" * 32, "", MARKER, None, 3])
def test_a_file_digest_is_64_lowercase_hexadecimal_characters(bad: object) -> None:
    assert "'files'" in refused(edited(run_document(), ("files", OUTPUTS_FILE), bad))


def test_the_hypotheses_file_and_every_result_file_are_listed() -> None:
    assert "'files'" in refused(edited(run_document(), ("files", OUTPUTS_FILE), DELETE))
    assert "'files'" in refused(
        edited(run_document(), ("files", f"{FIRST}/result.json"), DELETE)
    )


def test_a_run_cannot_list_its_own_file() -> None:
    document = run_document()
    document["files"][RUN_FILE] = DIGEST

    assert "'files'" in refused(document)


def test_the_files_are_a_mapping() -> None:
    for bad in ([], "x", None, [[OUTPUTS_FILE, DIGEST]]):
        assert "'files'" in refused(edited(run_document(), "files", bad))


# ---- the source of the block and the scores ------------------------------------------------------


def test_the_source_of_a_robustness_block_is_exact() -> None:
    for field, bad in (
        ("binding", "id"),
        ("suite", MARKER),
        ("suite_sha256", MARKER),
        ("robustness_sha256", "ab" * 31 + "a"),
        ("report_sha256", None),
    ):
        document = run_document()
        document["robustness_source"] = {**SOURCE, field: bad}

        assert f"'robustness_source.{field}'" in refused(document)
    document = run_document()
    document["robustness_source"] = {**SOURCE, "extra": "x"}
    assert "'robustness_source'" in refused(document)
    document["robustness_source"] = {"suite": "golden"}
    assert "'robustness_source'" in refused(document)


@pytest.mark.parametrize("name", ["bleu", "chrf_pp", MARKER, "", "comet"])
def test_scores_are_recorded_only_for_the_three_neural_metrics(name: str) -> None:
    document = edited(run_document(), f"segment_scores.{name}", DIGEST)

    assert "'segment_scores'" in refused(document)


@pytest.mark.parametrize("bad", ["ab" * 31 + "a", "AB" * 32, "", MARKER, None, 3])
def test_a_scores_file_digest_is_64_lowercase_hexadecimal_characters(
    bad: object,
) -> None:
    assert "'segment_scores.comet22'" in refused(
        edited(run_document(), "segment_scores.comet22", bad)
    )


def test_the_scores_are_a_mapping() -> None:
    for bad in ([], "x", None):
        assert "'segment_scores'" in refused(
            edited(run_document(), "segment_scores", bad)
        )


# ---- bytes ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [b"", b"{not json", b"[]", b"3", b"null", b"\xff\xfe{}", b'{"a": 1, "a": 2}'],
)
def test_a_file_that_is_not_a_json_object_in_utf_8_is_refused(data: bytes) -> None:
    refused(data)


def test_a_repeated_key_is_refused_wherever_it_is() -> None:
    text = serialised(run_document()).decode()
    nested = text.replace('"rows": 10,', '"rows": 10, "rows": 10,', 1)

    assert "more than once" in refused(nested.encode())


def test_a_planted_marker_never_comes_back() -> None:
    for path in (
        "schema",
        "purpose",
        "suite.name",
        "suite.sha256",
        "system.name",
        "directions.0.direction",
        "directions.0.state",
        "directions.0.result",
        "segment_scores.comet22",
    ):
        message = refused(edited(run_document(), path, MARKER))

        assert MARKER not in message, path


def test_a_truncated_document_is_refused_whatever_the_cut() -> None:
    data = serialised(run_document())

    for cut in (1, 10, len(data) // 2, len(data) - 2):
        refused(data[:cut])
