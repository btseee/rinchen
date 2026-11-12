"""The readers of what the evaluation commands of the tiny pipeline write (D15, spec P3 and P10).

A stage reads the results of a command from the files it wrote, never from what it printed. The
readers here check the shape of three of those files, `report.json` of `rinchen eval robustness`, the
`result.json` of a direction of `rinchen eval run` and its `hypotheses.jsonl`, over the recordings of
the spike, and refuse anything else with a `DataError` that names the file and the field and quotes no
text.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Final, get_args

import pytest

from rinchen.errors import DataError
from rinchen.infer.engine_types import Status
from rinchen.tiny.eval_readers import (
    STATUSES,
    Hypothesis,
    read_hypotheses,
    read_robustness_report,
    read_run_result,
)
from tests.support.tiny_run import RECORDINGS

REPORT: Final = RECORDINGS / "robustness_report.json"
RESULT: Final = RECORDINGS / "result_memorised_eng_khk_int8.json"
HELDOUT: Final = RECORDINGS / "result_heldout_eng_khk_int8.json"
HYPOTHESES: Final = RECORDINGS / "hypotheses_memorised_eng_int8.jsonl"
DIRECTION: Final = "eng_Latn-khk_Cyrl"
SIX: Final = (
    "entity_acc",
    "mong_source_retention",
    "numbers_acc",
    "placeholder_acc",
    "script_purity",
    "term_acc",
)
SECRET: Final = "SECRET-TEXT-OF-A-FILE"


def document(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def written(tmp_path: Path, content: object, name: str = "file.json") -> Path:
    path = tmp_path / name
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding="utf-8")
    return path


# ---- the robustness report -------------------------------------------------------------------------


def test_the_recorded_report_gives_the_six_values_and_the_coverage_beside_them() -> (
    None
):
    read = read_robustness_report(REPORT)

    assert sorted(read) == ["coverage", "values"]
    assert sorted(read["values"]) == list(SIX)
    assert read["values"] == document(REPORT)["robustness"]
    assert read["coverage"] == document(REPORT)["coverage"]
    assert read["coverage"]["placeholder_acc"] == {
        "excluded": 0,
        "scored": 238,
        "total": 238,
    }


def test_a_value_that_was_not_measured_is_null_and_is_accepted(tmp_path: Path) -> None:
    changed = document(REPORT)
    changed["robustness"]["term_acc"] = None

    assert (
        read_robustness_report(written(tmp_path, changed))["values"]["term_acc"] is None
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("robustness"),
        lambda d: d.pop("coverage"),
        lambda d: d.update(schema="rinchen.report/1"),
        lambda d: d.pop("schema"),
        lambda d: d["robustness"].pop("script_purity"),
        lambda d: d["robustness"].update(extra=1.0),
        lambda d: d["robustness"].update(script_purity="1.0"),
        lambda d: d["robustness"].update(script_purity=True),
        lambda d: d["robustness"].update(script_purity=-0.5),
        lambda d: d["robustness"].update(placeholder_acc=2.0),
        lambda d: d["robustness"].update(placeholder_acc=[1.0]),
        lambda d: d["coverage"].pop("numbers_acc"),
        lambda d: d["coverage"].update(extra={"scored": 1, "total": 1, "excluded": 0}),
        lambda d: d["coverage"].update(numbers_acc=5),
        lambda d: d["coverage"]["numbers_acc"].pop("scored"),
        lambda d: d["coverage"]["numbers_acc"].update(scored=-1),
        lambda d: d["coverage"]["numbers_acc"].update(scored=1.5),
        lambda d: d["coverage"]["numbers_acc"].update(total=True),
        lambda d: d["coverage"]["numbers_acc"].update(scored=300),
    ],
    ids=[
        "no-values",
        "no-coverage",
        "other-schema",
        "no-schema",
        "a-value-missing",
        "a-seventh-value",
        "text-value",
        "boolean-value",
        "negative-value",
        "value-above-one",
        "list-value",
        "a-coverage-missing",
        "a-seventh-coverage",
        "coverage-not-an-object",
        "no-scored",
        "negative-scored",
        "fractional-scored",
        "boolean-total",
        "more-scored-than-total",
    ],
)
def test_a_report_that_is_not_the_one_the_command_writes_is_refused(
    tmp_path: Path, change: Any
) -> None:
    changed = copy.deepcopy(document(REPORT))
    change(changed)

    with pytest.raises(DataError):
        read_robustness_report(written(tmp_path, changed))


def test_the_retention_may_exceed_one(tmp_path: Path) -> None:
    # `mong_source_retention` is a ratio of two chrF scores and is not a share.
    changed = document(REPORT)
    changed["robustness"]["mong_source_retention"] = 1.2

    read = read_robustness_report(written(tmp_path, changed))

    assert read["values"]["mong_source_retention"] == 1.2


def test_a_value_that_is_not_finite_is_refused(tmp_path: Path) -> None:
    text = json.dumps(document(REPORT)).replace(
        '"script_purity": 1.0', '"script_purity": NaN'
    )

    with pytest.raises(DataError):
        read_robustness_report(written(tmp_path, text))


@pytest.mark.parametrize("content", ["", "not json", "[]", '"text"', "{"])
def test_a_report_that_is_not_an_object_is_refused(
    tmp_path: Path, content: str
) -> None:
    with pytest.raises(DataError):
        read_robustness_report(written(tmp_path, content))


def test_a_report_that_is_not_there_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="report.json"):
        read_robustness_report(tmp_path / "report.json")


# ---- the result of a direction -----------------------------------------------------------------------


def test_a_result_gives_its_rows_the_six_metrics_and_the_statuses() -> None:
    read = read_run_result(
        RESULT, suite="tiny-memorised-eng", system="tiny-int8", direction=DIRECTION
    )

    recorded = document(RESULT)
    assert sorted(read) == ["metrics", "rows", "statuses"]
    assert read["rows"] == recorded["n_segments"] == 299
    assert read["metrics"] == recorded["metrics"]
    assert read["statuses"] == {"ok": 299}
    assert sorted(read["metrics"]) == [
        "bleu",
        "chrf",
        "chrf_pp",
        "comet22",
        "metricx24",
        "metricx24_qe",
    ]


def test_the_heldout_recording_is_read_too() -> None:
    read = read_run_result(
        HELDOUT, suite="tiny-heldout-eng", system="tiny-int8", direction=DIRECTION
    )

    assert read["rows"] == 30
    assert read["metrics"]["chrf"] == pytest.approx(71.99500932252622)


@pytest.mark.parametrize(
    "names",
    [
        {"suite": "tiny-memorised-rus"},
        {"system": "tiny-float32"},
        {"direction": "khk_Cyrl-eng_Latn"},
    ],
)
def test_a_result_of_another_suite_system_or_direction_is_refused(
    names: dict[str, str],
) -> None:
    asked = {
        "suite": "tiny-memorised-eng",
        "system": "tiny-int8",
        "direction": DIRECTION,
        **names,
    }

    with pytest.raises(DataError):
        read_run_result(RESULT, **asked)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("n_segments"),
        lambda d: d.update(n_segments=0),
        lambda d: d.update(n_segments=True),
        lambda d: d.update(n_segments="299"),
        lambda d: d.update(n_segments=300),
        lambda d: d.pop("statuses"),
        lambda d: d.update(statuses={}),
        lambda d: d.update(statuses={"ok": 298}),
        lambda d: d.update(statuses={"ok": 299, "other": 0}),
        lambda d: d.update(statuses={"ok": -1, "empty": 300}),
        lambda d: d.update(statuses={"ok": 299.0}),
        lambda d: d.update(statuses=[299]),
        lambda d: d.pop("metrics"),
        lambda d: d["metrics"].pop("chrf"),
        lambda d: d["metrics"].update(chrf=101.0),
        lambda d: d["metrics"].update(chrf=None),
        lambda d: d.pop("suite"),
        lambda d: d.pop("system"),
        lambda d: d.pop("direction"),
    ],
    ids=[
        "no-rows",
        "zero-rows",
        "boolean-rows",
        "text-rows",
        "rows-differ-from-the-statuses",
        "no-statuses",
        "empty-statuses",
        "statuses-sum-short",
        "unknown-status",
        "negative-count",
        "float-count",
        "statuses-a-list",
        "no-metrics",
        "no-chrf",
        "chrf-over-100",
        "null-chrf",
        "no-suite",
        "no-system",
        "no-direction",
    ],
)
def test_a_result_that_is_not_the_one_the_command_writes_is_refused(
    tmp_path: Path, change: Any
) -> None:
    changed = copy.deepcopy(document(RESULT))
    change(changed)

    with pytest.raises(DataError):
        read_run_result(
            written(tmp_path, changed),
            suite="tiny-memorised-eng",
            system="tiny-int8",
            direction=DIRECTION,
        )


def test_a_missing_result_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="result.json"):
        read_run_result(
            tmp_path / "result.json", suite="s", system="s", direction=DIRECTION
        )


def test_the_statuses_are_the_five_of_the_engine() -> None:
    assert STATUSES == get_args(Status)


# ---- the hypotheses -------------------------------------------------------------------------------------


def test_the_recorded_hypotheses_are_read_in_the_order_of_the_file() -> None:
    read = read_hypotheses(HYPOTHESES)

    lines = [
        json.loads(line) for line in HYPOTHESES.read_text(encoding="utf-8").splitlines()
    ]
    assert len(read) == len(lines) == 598
    assert read == [
        Hypothesis(line["direction"], line["id"], line["status"], line["text"])
        for line in lines
    ]
    assert {item.direction for item in read} == {DIRECTION, "khk_Cyrl-eng_Latn"}


def line(**changes: object) -> str:
    row = {"direction": DIRECTION, "id": "en-ek-0001", "status": "ok", "text": "x"}
    return json.dumps({**row, **changes}, ensure_ascii=False)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json\n",
        "[]\n",
        line() + "\n" + line() + "\n",
        line(status="fine") + "\n",
        line(text=5) + "\n",
        line(id=5) + "\n",
        line(direction=None) + "\n",
        json.dumps({"direction": DIRECTION, "id": "i", "status": "ok"}) + "\n",
        line() + "\n\n" + line(id="other") + "\n",
        line() + "\r\n",
        line() + "\n" + "{" + "\n",
        '{"direction": "d", "direction": "e", "id": "i", "status": "ok", "text": "t"}\n',
    ],
    ids=[
        "empty",
        "not-json",
        "not-an-object",
        "a-row-twice",
        "unknown-status",
        "text-not-a-string",
        "id-not-a-string",
        "direction-null",
        "no-text",
        "a-blank-line",
        "carriage-return",
        "a-broken-line",
        "a-repeated-key",
    ],
)
def test_hypotheses_that_are_not_the_ones_a_run_writes_are_refused(
    tmp_path: Path, content: str
) -> None:
    with pytest.raises(DataError):
        read_hypotheses(written(tmp_path, content, "hypotheses.jsonl"))


def test_a_message_names_the_line_and_quotes_no_text(tmp_path: Path) -> None:
    content = line() + "\n" + line(id="other", text=SECRET, status=SECRET) + "\n"

    with pytest.raises(DataError) as caught:
        read_hypotheses(written(tmp_path, content, "hypotheses.jsonl"))

    assert SECRET not in str(caught.value)
    assert "line 2" in str(caught.value)


def test_hypotheses_that_are_not_there_are_refused(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="hypotheses.jsonl"):
        read_hypotheses(tmp_path / "hypotheses.jsonl")
