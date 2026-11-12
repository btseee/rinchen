"""The helpers of the D15 gate's controls (`tests/support/tiny_gate_controls.py`), tested without starting
anything (D15 Task 9): where the files of a finished run are, the thresholds replayed over the figures of a
report, the rotation of a hypotheses file and the two suites of the controls. The report they replay is the
golden one of the pipeline's own tests, whose verdicts are known.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Final

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.tiny.thresholds import ThresholdInputs
from tests.support.tiny_gate_controls import (
    ROTATION,
    answer_with_the_source,
    averaged_directory,
    converter_options,
    export_directory,
    judged,
    replayed,
    report_inputs,
    rotate_texts,
    stage_path,
    verdicts,
    with_changes,
    write_control_suites,
)

GOLDEN: Final = REPO_ROOT / "tests" / "fixtures" / "tiny" / "golden_report.json"


def _run_with_the_golden_report(tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    shutil.copyfile(GOLDEN, out_dir / "report.json")
    return out_dir


def test_the_rotation_is_the_one_of_the_master_plans_control() -> None:
    assert ROTATION == 7


def test_a_stage_directory_is_the_numbered_one_of_the_output(tmp_path: Path) -> None:
    assert stage_path(tmp_path, "export", "int8") == tmp_path / "08_export" / "int8"
    assert stage_path(tmp_path, "robustness") == tmp_path / "10_robustness"
    assert stage_path(tmp_path, "train", "ablate") == tmp_path / "07_train" / "ablate"


def test_the_export_and_the_converter_are_where_the_stages_put_them(
    tmp_path: Path,
) -> None:
    assert export_directory(tmp_path, "float32") == tmp_path / "08_export" / "float32"
    assert converter_options(tmp_path) == [
        "--script-converter-ct2",
        tmp_path / "05_converter" / "model",
        "--script-converter-spm",
        tmp_path / "05_converter" / "run" / "converter_char.model",
    ]


def test_the_averaged_weights_are_the_one_variants(tmp_path: Path) -> None:
    averaged = tmp_path / "07_train" / "ablate" / "variants" / "a_variant" / "run"
    (averaged / "averaged").mkdir(parents=True)

    assert averaged_directory(tmp_path) == averaged / "averaged"


@pytest.mark.parametrize("names", [[], ["one", "two"]])
def test_a_run_without_exactly_one_variant_is_refused(
    tmp_path: Path, names: list[str]
) -> None:
    variants = tmp_path / "07_train" / "ablate" / "variants"
    variants.mkdir(parents=True)
    for name in names:
        (variants / name).mkdir()

    with pytest.raises(AssertionError, match="variant"):
        averaged_directory(tmp_path)


def test_the_thresholds_replayed_over_the_report_give_its_verdicts(
    tmp_path: Path,
) -> None:
    out_dir = _run_with_the_golden_report(tmp_path)
    document = json.loads(GOLDEN.read_text(encoding="utf-8"))

    results = replayed(out_dir)

    assert verdicts(results) == {
        item["id"]: item["verdict"] for item in document["thresholds"]
    }
    by_id = {result.id: result for result in results}
    for item in document["thresholds"]:
        if item["id"] != "wall_time":
            assert by_id[item["id"]].value == item["value"], item["id"]


def test_the_inputs_of_a_report_hold_the_five_directions_and_the_two_systems(
    tmp_path: Path,
) -> None:
    inputs = report_inputs(_run_with_the_golden_report(tmp_path))

    assert sorted(inputs.memorised_chrf) == [
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-eng_Latn",
        "khk_Cyrl-rus_Cyrl",
        "mn_Mong-eng_Latn",
        "rus_Cyrl-khk_Cyrl",
    ]
    assert all(
        sorted(systems) == ["float32", "int8"]
        for systems in inputs.memorised_chrf.values()
    )
    assert inputs.parity is not None
    assert sorted(inputs.parity) == ["exact_share", "mean_abs_delta"]
    assert inputs.robustness is not None
    assert inputs.robustness["script_purity"] == 1.0


def test_one_changed_figure_fails_only_its_own_threshold(tmp_path: Path) -> None:
    inputs = report_inputs(_run_with_the_golden_report(tmp_path))
    assert inputs.robustness is not None
    assert inputs.parity is not None

    placeholders = with_changes(
        inputs, robustness={**inputs.robustness, "placeholder_acc": 0.0}
    )
    delta = with_changes(inputs, parity={**inputs.parity, "mean_abs_delta": 2e-3})

    assert _failing(placeholders) == ["placeholder_acc"]
    assert _failing(delta) == ["parity_delta"]


def _failing(inputs: ThresholdInputs) -> list[str]:
    return sorted(result.id for result in judged(inputs) if result.verdict == "fail")


def test_the_rotation_moves_the_texts_of_each_direction_and_nothing_else(
    tmp_path: Path,
) -> None:
    rows = [
        {"direction": "a-b", "id": f"a-{n}", "status": "ok", "text": f"a text {n}"}
        for n in range(10)
    ] + [
        {"direction": "b-a", "id": f"b-{n}", "status": "empty", "text": f"b text {n}"}
        for n in range(8)
    ]
    source = tmp_path / "in.jsonl"
    source.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    target = tmp_path / "out.jsonl"

    rotate_texts(source, target, 7)

    moved = [json.loads(line) for line in target.read_text("utf-8").splitlines()]
    assert [(row["direction"], row["id"], row["status"]) for row in moved] == [
        (row["direction"], row["id"], row["status"]) for row in rows
    ]
    assert [row["text"] for row in moved[:10]] == [
        f"a text {(n + 7) % 10}" for n in range(10)
    ]
    assert [row["text"] for row in moved[10:]] == [
        f"b text {(n + 7) % 8}" for n in range(8)
    ]
    assert target.read_text("utf-8").endswith("\n")


def test_a_rotation_of_a_file_with_one_row_per_direction_changes_nothing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"direction": "a-b", "id": "x", "status": "ok", "text": "t"}) + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "out.jsonl"

    rotate_texts(source, target, 7)

    assert json.loads(target.read_text("utf-8"))["text"] == "t"


def test_the_answers_of_a_section_become_the_sources_of_its_rows(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "sections": {
                    "script": [
                        {"id": "s-1", "source": "first source"},
                        {"id": "s-2", "source": "second source"},
                    ],
                    "numbers": [{"id": "n-1", "source": "a number"}],
                }
            }
        ),
        encoding="utf-8",
    )
    lines = [
        {"arm": "main", "id": "s-1", "section": "script", "status": "ok", "text": "x"},
        {"arm": "main", "id": "n-1", "section": "numbers", "status": "ok", "text": "y"},
        {
            "arm": "main",
            "id": "s-2",
            "section": "script",
            "status": "empty",
            "text": "",
        },
    ]
    source = tmp_path / "in.jsonl"
    source.write_text(
        "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines),
        encoding="utf-8",
    )
    target = tmp_path / "out.jsonl"

    changed = answer_with_the_source(suite, source, target, section="script")

    written = [json.loads(line) for line in target.read_text("utf-8").splitlines()]
    assert changed == 2
    assert [(line["id"], line["text"]) for line in written] == [
        ("s-1", "first source"),
        ("n-1", "y"),
        ("s-2", "second source"),
    ]
    assert [(line["status"], line["arm"]) for line in written] == [
        (line["status"], line["arm"]) for line in lines
    ]


def test_an_answer_for_a_row_the_suite_does_not_have_is_refused(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"sections": {"script": [{"id": "s-1", "source": "x"}]}}),
        encoding="utf-8",
    )
    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"id": "s-9", "section": "script", "text": "x"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        answer_with_the_source(suite, source, tmp_path / "out.jsonl", section="script")


def test_the_two_control_suites_are_written_as_robustness_suites(
    tmp_path: Path,
) -> None:
    unseen, mong_output = write_control_suites(tmp_path / "suites")

    assert unseen.name == "unseen_placeholders.json"
    assert mong_output.name == "mong_output.json"
    for path in (unseen, mong_output):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["schema"] == "rinchen.robustness.suite/1"
    assert list(json.loads(unseen.read_text(encoding="utf-8"))["sections"]) == [
        "placeholders"
    ]
    assert list(json.loads(mong_output.read_text(encoding="utf-8"))["sections"]) == [
        "script"
    ]
