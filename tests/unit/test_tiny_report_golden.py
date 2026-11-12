"""The golden report of D15 (Task 6): `tests/fixtures/tiny/golden_report.json` and `golden_report.md`
are the report of the golden run written byte for byte, what `build_report` and `to_bytes` make of that
run must equal the JSON, and the report holds nothing that varies from one run or one machine to the next.

The two files were written by a script of the task that imports nothing of `rinchen` (plain `json`, from
the spec's rules P4, P8, P10 and P12 and from the figures of the spike's run `p6`: the memorised chrF,
BLEU and chrF++ of five directions and two exports, the filter counts, the converter figures, the parity
figures, the robustness block and its coverage). The figures that run did not record are illustrative and
say nothing about a model: the held-out figures of three directions (the spike's finding F11 gives chrF
and BLEU, the chrF++ are made up), the rows by direction of the training corpus, the requests the chat
server answered, the agreement of the two exports and the planted scores of the neural seam. The golden
run passes every threshold: `wall_time` holds the verdict and a null value, because the seconds are the
manifest's and never the report's.

The script is not kept in the repository: after an intended change of the layout, write both files
again from the spec and the figures of `tests/support/tiny_report_inputs.py` with a script that, like
that one, shares no code with `rinchen.tiny`, check them against the spec and commit them with the
change. A new digest is a decision and not an update.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from typing import Any

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.tiny.context import StageResult
from rinchen.tiny.report import parse_report, report_document, to_bytes
from rinchen.tiny.states import StageRecord
from tests.support.tiny_report_build import build, variant_reports
from tests.support.tiny_report_inputs import golden_results
from tests.support.tiny_report_scans import fixture_sentences, violations
from tests.support.tiny_report_walk import (
    GOLDEN_JSON,
    GOLDEN_MD,
    dumps,
    golden_document,
)

# `sha256sum` of the two files and their sizes, taken from the files and not from the code.
GOLDEN_JSON_SHA256 = "98c2504bf16f5a4fcb6569f4399501f9023e8097c91cb26383c3c238ee0c0fc7"
GOLDEN_MD_SHA256 = "d617c28b40ee5078ad6802c6edddb708bf5f0212ea1859d44fad84c71c5f76bd"
GOLDEN_JSON_SIZE = 26309
GOLDEN_MD_SIZE = 3227


def golden_bytes() -> bytes:
    return GOLDEN_JSON.read_bytes()


# ---- G6.1: the golden files --------------------------------------------------------------------------


def test_the_golden_files_are_the_ones_that_were_written() -> None:
    assert len(golden_bytes()) == GOLDEN_JSON_SIZE
    assert GOLDEN_MD.stat().st_size == GOLDEN_MD_SIZE
    assert hashlib.sha256(golden_bytes()).hexdigest() == GOLDEN_JSON_SHA256
    assert hashlib.sha256(GOLDEN_MD.read_bytes()).hexdigest() == GOLDEN_MD_SHA256


def test_the_bytes_of_the_report_of_the_golden_run_are_the_golden_json() -> None:
    written = to_bytes(build())
    assert written == golden_bytes()
    assert len(written) == GOLDEN_JSON_SIZE
    assert hashlib.sha256(written).hexdigest() == GOLDEN_JSON_SHA256


def test_the_golden_json_is_the_sorted_two_space_form_with_one_final_line_feed() -> (
    None
):
    data = golden_bytes()
    assert data.endswith(b"}\n")
    assert not data.endswith(b"\n\n")
    assert dumps(golden_document()) == data
    assert b"\r" not in data
    assert b"\t" not in data


@pytest.mark.parametrize("path", [GOLDEN_JSON, GOLDEN_MD], ids=["json", "markdown"])
def test_the_golden_files_are_plain_ascii_with_line_feeds_only(path: Any) -> None:
    data = path.read_bytes()
    assert all(byte == 0x0A or 0x20 <= byte <= 0x7E for byte in data)
    assert data.endswith(b"\n")


def test_the_golden_json_parses_back_to_the_report_that_wrote_it_and_to_the_same_bytes() -> (
    None
):
    parsed = parse_report(golden_bytes())
    assert parsed == build()
    assert to_bytes(parsed) == golden_bytes()


def test_the_document_of_a_report_is_the_golden_document() -> None:
    assert report_document(build()) == golden_document()


def test_the_golden_run_has_every_threshold_passing_and_a_null_wall_time() -> None:
    document = golden_document()
    assert document["verdict"] == "pass"
    table = {item["id"]: item for item in document["thresholds"]}
    assert {item["verdict"] for item in table.values()} == {"pass"}
    assert table["wall_time"]["value"] is None
    assert table["wall_time"]["bound"] == "<= 1200"
    assert table["train_chrf"]["value"] == 99.35849111913797


def test_the_golden_json_keeps_the_note_of_the_token_ratio_check_in_the_filter() -> (
    None
):
    filtering = golden_document()["stages"][3]
    assert filtering["id"] == "filter"
    assert filtering["detail"]["token_ratio_check"] == (
        "the token-ratio check needs a SentencePiece model, which does not exist yet"
    )
    parts = {part["id"]: part for part in filtering["parts"]}
    assert parts["length"]["state"] == "ok"
    assert parts["lid"]["state"] == "skipped"


# ---- G6.2: what the report must not hold -------------------------------------------------------------


UNIQUE_ENVIRONMENT = {
    "HOME": "/zz-unique-home-4f1c",
    "USER": "zz-unique-user-9a2e",
    "PATH": "/zz-unique-path-77d0:/zz-unique-bin",
    "LOGNAME": "zz-unique-login-31bb",
    "TMPDIR": "/zz-unique-tmp-c05a",
    "SHELL": "/zz-unique-shell-e8d4",
}


def test_the_golden_report_holds_nothing_that_varies_from_run_to_run() -> None:
    assert violations(golden_bytes()) == []


def variants() -> dict[str, bytes]:
    return {name: to_bytes(report) for name, report in variant_reports().items()}


def test_no_report_of_any_ending_holds_a_time_a_path_a_duration_or_another_programs_output() -> (
    None
):
    for name, data in variants().items():
        assert violations(data) == [], name


def test_the_report_holds_no_value_of_the_environment_of_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in UNIQUE_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    data = to_bytes(build())
    for value in UNIQUE_ENVIRONMENT.values():
        assert value.encode() not in data
    assert violations(data, tuple(UNIQUE_ENVIRONMENT.values())) == []


def test_the_report_names_no_digest_of_a_file_that_embeds_a_path() -> None:
    for data in (golden_bytes(), *variants().values()):
        assert b"manifest_sha256" not in data
        assert b"hypotheses_sha256" not in data
        assert b"fingerprint" not in data


def test_the_report_holds_no_sentence_of_the_fixtures() -> None:
    sentences = fixture_sentences()
    assert len(sentences) > 500
    text = golden_bytes().decode()
    assert [s for s in sentences if s in text] == []


def test_the_scans_see_what_they_are_meant_to_see() -> None:
    cases = {
        "a time": b'{"a": "at 12:30"}',
        "a date": b'{"a": "2026-11-11"}',
        "a duration": b'{"seconds": 3}',
        "an elapsed key": b'{"x": {"elapsed_time": 3}}',
        "a path": b'{"a": "/home/someone/x"}',
        "a windows path": b'{"a": "C:\\\\Users"}',
        "a manifest digest": b'{"manifest_sha256": "x"}',
        "an output line": b'{"a": "Traceback (most recent call last)"}',
        "the environment": b'{"a": "zz-value"}',
    }
    for name, data in cases.items():
        assert violations(data, ("zz-value",)) != [], name
    assert violations(b'{"schema": "rinchen.tiny/1", "id": "wall_time"}') == []


# ---- G6.5: determinism and independence ---------------------------------------------------------------


def test_two_builds_over_the_same_observations_give_equal_bytes() -> None:
    assert to_bytes(build()) == to_bytes(build())


def reordered(value: Any) -> Any:
    """The same JSON value with the keys of every mapping inserted in the opposite order."""
    if isinstance(value, dict):
        return {key: reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list | tuple):
        return [reordered(item) for item in value]
    return value


def test_the_order_in_which_the_facts_were_written_changes_no_byte() -> None:
    shuffled = []
    for result in golden_results():
        record = result.record
        fresh = StageRecord(
            record.id,
            record.state,
            record.reason,
            reordered(dict(record.detail)),
            record.parts,
        )
        shuffled.append(StageResult(fresh, reordered(dict(result.facts))))
    assert to_bytes(build(shuffled)) == to_bytes(build())


CHILD = (
    "import hashlib\n"
    "from tests.support.tiny_report_build import build\n"
    "from rinchen.tiny.report import to_bytes\n"
    "first, second = to_bytes(build()), to_bytes(build())\n"
    "assert first == second\n"
    "print(hashlib.sha256(first).hexdigest())\n"
)
ENVIRONMENTS = [
    ("0", "C", "UTC"),
    ("3", "C.UTF-8", "Asia/Ulaanbaatar"),
    ("0", "C.UTF-8", "Asia/Ulaanbaatar"),
    ("3", "C", "UTC"),
]


@pytest.mark.parametrize(("hash_seed", "locale", "zone"), ENVIRONMENTS)
def test_the_bytes_are_the_same_under_another_hash_seed_locale_and_time_zone(
    hash_seed: str, locale: str, zone: str
) -> None:
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONHASHSEED": hash_seed,
        "LC_ALL": locale,
        "TZ": zone,
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "UV_OFFLINE": "1",
    }
    finished = subprocess.run(
        [sys.executable, "-c", CHILD],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr[-400:]
    assert finished.stdout.strip() == GOLDEN_JSON_SHA256
