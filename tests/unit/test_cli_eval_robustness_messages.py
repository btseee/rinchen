"""`rinchen eval robustness`: no text of a suite or of a hypothesis is in any message of the
command (D12, Task 7). A marker is planted in every free-text field of a valid suite and of the
golden hypotheses, and in the values and keys of the ones the reader refuses; every failing path
is run and the marker must be in no standard output, no standard error and no log record. The
interpolation scan of `test_eval_error_messages.py` is the other half of the guarantee.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.infer.engine_types import Row, Translation
from tests.support.robust_cli import (
    RobustWorkspace,
    answers_from_golden,
    fail_replace,
    install,
    invoke,
    make_workspace,
)
from tests.support.robust_golden import GOLDEN_SUITE
from tests.support.robust_suite_documents import encode


@pytest.fixture
def workspace(tmp_path: Path) -> RobustWorkspace:
    return make_workspace(tmp_path)


# --- no text of a suite or of a hypothesis in a message ---------------------------------------

MARKER = "TEXT-OF-THE-FILE-91"


def golden_document() -> dict[str, Any]:
    return json.loads(GOLDEN_SUITE.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def marked_document() -> dict[str, Any]:
    """The golden suite with the marker in every free-text field of every section: sources,
    references, traditional sources, terms and the spellings of the entity table."""
    doc = golden_document()
    for rows in doc["sections"].values():
        for row in rows:
            row["source"] = f"{MARKER} {row['source']}"
            if row.get("reference") is not None:
                row["reference"] = f"{MARKER} {row['reference']}"
            if "source_mong" in row:
                row["source_mong"] += f" {MARKER}"
            for term in row.get("terms", ()):
                term["src"] += f" {MARKER}"
                term["tgt"] += f" {MARKER}"
    for languages in doc["entities"].values():
        for spellings in languages.values():
            spellings.append(f"{MARKER} spelling")
    return doc


def marked_suite() -> bytes:
    return encode(marked_document())


def marked_hypotheses(workspace: RobustWorkspace) -> list[dict[str, Any]]:
    """The golden hypotheses with the marker in every text."""
    lines = [
        json.loads(line)
        for line in workspace.hypotheses.read_text(encoding="utf-8").splitlines()
    ]
    for line in lines:
        line["text"] = f"{MARKER} {line['text']}"
    return lines


def marked_answers(rows: Sequence[Row]) -> list[Translation]:
    """The golden answers with the marker in every text."""
    return [
        dataclasses.replace(answer, text=f"{MARKER} {answer.text}")
        for answer in answers_from_golden(rows)
    ]


def dump(lines: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8") for line in lines
    )


def failing_runs(workspace: RobustWorkspace) -> list[tuple[str, list[str]]]:
    """Every way a run with planted texts can fail, each as the files it plants and the
    command line it runs. A run plants its files in `workspace` before it is invoked."""
    runs: list[tuple[str, list[str]]] = []

    def plant(name: str, suite: bytes, hypotheses: bytes) -> list[str]:
        directory = workspace.root / name
        directory.mkdir()
        (directory / "suite.json").write_bytes(suite)
        (directory / "hypotheses.jsonl").write_bytes(hypotheses)
        return workspace.hypotheses_args(
            suite=directory / "suite.json",
            hypotheses=directory / "hypotheses.jsonl",
            out_dir=workspace.root / f"out-{name}",
        )

    lines = marked_hypotheses(workspace)
    good_suite = marked_suite()
    runs.append(("missing", plant("missing", good_suite, dump(lines[1:]))))
    runs.append(("twice", plant("twice", good_suite, dump([*lines, lines[0]]))))
    stray = {"section": "numbers", "id": "zz9", "arm": "main", "text": MARKER}
    runs.append(("stray", plant("stray", good_suite, dump([*lines, stray]))))
    unknown_field = [{**lines[0], MARKER: MARKER}, *lines[1:]]
    runs.append(
        ("unknown_field", plant("unknown_field", good_suite, dump(unknown_field)))
    )
    runs.append(
        (
            "not_json",
            plant(
                "not_json", good_suite, (MARKER + " {not json\n").encode() + dump(lines)
            ),
        )
    )
    wrong_type = [{**lines[0], "text": [MARKER]}, *lines[1:]]
    runs.append(("wrong_type", plant("wrong_type", good_suite, dump(wrong_type))))
    bad_status = [{**lines[0], "status": MARKER}, *lines[1:]]
    runs.append(("bad_status", plant("bad_status", good_suite, dump(bad_status))))
    not_utf8 = MARKER.encode() + b"\xff\n"
    runs.append(("not_utf8", plant("not_utf8", good_suite, not_utf8)))
    doc = marked_document()
    doc["sections"]["numbers"][0]["source"] = f"{MARKER} <t> x"
    runs.append(("reserved", plant("reserved", encode(doc), dump(lines))))
    doc = marked_document()
    doc["sections"]["numbers"][0][MARKER] = MARKER
    runs.append(
        ("suite_unknown_key", plant("suite_unknown_key", encode(doc), dump(lines)))
    )
    doc = marked_document()
    doc["sections"]["numbers"][0]["source"] = [MARKER]
    runs.append(
        ("suite_wrong_type", plant("suite_wrong_type", encode(doc), dump(lines)))
    )
    doc = marked_document()
    doc[MARKER] = MARKER
    runs.append(
        ("suite_unknown_top", plant("suite_unknown_top", encode(doc), dump(lines)))
    )
    doc = marked_document()
    doc["entities"]["bat"][MARKER] = [MARKER]
    runs.append(
        (
            "suite_unknown_language",
            plant("suite_unknown_language", encode(doc), dump(lines)),
        )
    )
    runs.append(
        ("suite_not_json", plant("suite_not_json", MARKER.encode() + b"{", dump(lines)))
    )
    return runs


def test_no_text_of_a_suite_or_of_a_hypothesis_is_in_any_message_of_a_failing_run(
    workspace: RobustWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    install(monkeypatch, workspace.engine())
    caplog.set_level(logging.DEBUG)
    runs = failing_runs(workspace)

    for name, arguments in runs:
        result = invoke(arguments)

        assert result.exit_code == 3, (name, result.output)
        assert MARKER not in result.stdout, name
        assert MARKER not in result.stderr, name
        assert MARKER not in caplog.text, name
    assert len(runs) == 14


def test_no_text_of_a_suite_is_in_any_message_of_a_failing_model_run(
    workspace: RobustWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace.suite.write_bytes(marked_suite())
    caplog.set_level(logging.DEBUG)
    errors = (
        ConfigError("the target is not written"),
        DataError("row 3 holds a symbol"),
        UpstreamError("CTranslate2 failed"),
        KeyboardInterrupt(),
    )

    for error in errors:
        install(monkeypatch, workspace.engine(error=error))
        result = invoke(workspace.model_args())

        assert result.exit_code != 0
        assert MARKER not in result.stdout
        assert MARKER not in result.stderr
        assert MARKER not in caplog.text
    short = workspace.engine(answer=lambda rows: marked_answers(rows)[:-1])
    install(monkeypatch, short)
    result = invoke(workspace.model_args())
    assert result.exit_code == 4
    assert MARKER not in result.stdout + result.stderr + caplog.text
    fail_replace(monkeypatch, "report.json")
    install(monkeypatch, workspace.engine(answer=marked_answers))
    result = invoke(workspace.model_args())
    assert result.exit_code == 1
    assert MARKER not in result.stdout + result.stderr + caplog.text


def test_a_good_run_over_marked_files_prints_and_records_no_text(
    workspace: RobustWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace.suite.write_bytes(marked_suite())
    workspace.hypotheses.write_bytes(dump(marked_hypotheses(workspace)))
    caplog.set_level(logging.DEBUG)

    result = invoke(workspace.hypotheses_args("--system", "marked"))

    assert result.exit_code == 0, result.output
    assert MARKER not in result.stdout + result.stderr + caplog.text
    for name in ("robustness.json", "report.json", "manifest.json"):
        assert MARKER not in (workspace.out_dir / name).read_text(encoding="utf-8")
