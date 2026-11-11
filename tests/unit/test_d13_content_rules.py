"""The content rules of the project (CLAUDE.md) over everything D13 adds (Task 9, spec T16 and T22):
the rules exclude two subjects from every file, and the model card, the release record, the workflow,
the configurations, the documents and the tests of D13 carry no word of either. The Hub's own card
template is never used because it has a field for one of them.

The needles are the stems of the words of those subjects, spelled as code points (a tuple of integers
per needle) INSIDE this test, so that this file holds none of them and passes its own scan; a failure
names the file, the line and the number of the needle and never the word. A needle is looked for in
lower case. The rules about Inner Mongolia are not scanned: `khk_Mong` is named by the guards of D13
on purpose, to be refused.

What is scanned: the code D13 adds or edits (`rinchen.eval`, `rinchen.export`, `rinchen.metrics`, the
commands `eval*` and `release*`, the configurations `config/eval_*`, `acquire/hf.py`), `configs/eval/`,
the workflow `release.yml` and the runbook `13_release.md` when they exist, the fixtures of
`tests/fixtures/eval/`, the spec and the plan of D13, the tests and support modules of D13 (by name)
and the files the release stage generates (the card, the release record and the copies of the report).
The scan is itself tested on a text that holds a needle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_fixture import fabricated_release, invoke_stage

pytestmark = pytest.mark.usefixtures("no_network")

ROOT: Final = Path(__file__).resolve().parents[2]
# One tuple of code points per needle; the words are never written out.
NEEDLE_CODE_POINTS: Final[tuple[tuple[int, ...], ...]] = (
    (
        108,
        105,
        99,
        101,
        110,
    ),
    (
        99,
        111,
        112,
        121,
        114,
        105,
        103,
        104,
        116,
    ),
    (
        108,
        101,
        103,
        97,
        108,
    ),
    (
        108,
        97,
        119,
        115,
        117,
        105,
        116,
    ),
    (
        108,
        97,
        119,
        102,
        117,
        108,
    ),
    (
        108,
        97,
        119,
        121,
        101,
        114,
    ),
    (
        116,
        114,
        97,
        100,
        101,
        109,
        97,
        114,
        107,
    ),
    (
        112,
        97,
        116,
        101,
        110,
        116,
    ),
    (
        108,
        105,
        97,
        98,
        105,
        108,
        105,
        116,
        121,
    ),
    (
        119,
        97,
        114,
        114,
        97,
        110,
        116,
        121,
    ),
    (
        116,
        101,
        114,
        109,
        115,
        32,
        111,
        102,
        32,
        117,
        115,
        101,
    ),
    (
        116,
        101,
        114,
        109,
        115,
        32,
        111,
        102,
        32,
        115,
        101,
        114,
        118,
        105,
        99,
        101,
    ),
    (
        115,
        112,
        100,
        120,
    ),
    (
        97,
        116,
        116,
        111,
        114,
        110,
        101,
        121,
    ),
    (
        114,
        111,
        121,
        97,
        108,
        116,
    ),
    (
        101,
        117,
        108,
        97,
    ),
    (
        100,
        109,
        99,
        97,
    ),
    (
        106,
        117,
        114,
        105,
        115,
        100,
        105,
        99,
        116,
        105,
        111,
        110,
    ),
)
GLOBS: Final = (
    "src/rinchen/eval/*.py",
    "src/rinchen/export/*.py",
    "src/rinchen/metrics/*.py",
    "src/rinchen/cli/eval*.py",
    "src/rinchen/cli/release*.py",
    "src/rinchen/config/eval_*.py",
    "src/rinchen/acquire/hf.py",
    "configs/eval/*",
    ".github/workflows/release.yml",
    "docs/runbooks/13_release.md",
    "docs/superpowers/specs/2026-11-10-d13-release-design.md",
    "docs/superpowers/plans/2026-11-10-d13-release.md",
    "tests/fixtures/eval/*",
    "tests/support/eval_*.py",
    "tests/support/hub_*.py",
    "tests/support/release_*.py",
    "tests/support/no_network.py",
    "tests/unit/test_eval_*.py",
    "tests/unit/test_cli_eval_*.py",
    "tests/unit/test_cli_release_*.py",
    "tests/unit/test_export_release*.py",
    "tests/unit/test_export_model_card*.py",
    "tests/unit/test_export_hub*.py",
    "tests/unit/test_metrics_*.py",
    "tests/unit/test_config_eval_*.py",
    "tests/unit/test_release_*.py",
    "tests/unit/test_d13_*.py",
    "tests/property/test_eval_*.py",
    "tests/property/test_metrics_segment_stats_property.py",
    "tests/integration/test_eval_*.py",
    "tests/integration/test_release_*.py",
)
MINIMUM_FILES: Final = 80


def needles() -> tuple[str, ...]:
    return tuple("".join(map(chr, points)) for points in NEEDLE_CODE_POINTS)


def needles_in(text: str) -> list[int]:
    """The numbers of the needles that `text` holds, in lower case."""
    lowered = text.lower()
    return [number for number, needle in enumerate(needles()) if needle in lowered]


def lines_with_a_needle(text: str) -> list[tuple[int, int]]:
    """`(line number, needle number)` of every line of `text` that holds a needle."""
    found: list[tuple[int, int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        found += [(number, needle) for needle in needles_in(line)]
    return found


def scanned_files() -> list[Path]:
    paths = {
        path
        for pattern in GLOBS
        for path in ROOT.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    return sorted(paths)


# ---- the scan ----------------------------------------------------------------------------------


@pytest.mark.parametrize("number", range(len(NEEDLE_CODE_POINTS)))
def test_the_scan_finds_each_needle_in_any_case_inside_a_sentence(number: int) -> None:
    word = needles()[number]

    for planted in (word, word.upper(), word.capitalize()):
        assert needles_in(f"a line that says {planted} in the middle") == [number]


def test_the_scan_names_the_line_and_the_needle_and_never_the_word() -> None:
    text = "fine\n" + "x " + needles()[3] + "\nfine again\n"

    assert lines_with_a_needle(text) == [(2, 3)]


def test_the_scan_lets_the_sentences_of_the_card_through() -> None:
    clean = (
        "These figures come from generated fixture suites. They test the tooling and say "
        "nothing about translation quality. The model was trained from random initialisation. "
        "A comparison is significant when its p-value is below 0.05. n/a (not trained)."
    )

    assert needles_in(clean) == []


def test_no_needle_is_a_word_this_very_file_spells_out() -> None:
    assert needles_in(Path(__file__).read_text(encoding="utf-8")) == []


def test_the_needles_are_distinct_and_long_enough_to_mean_something() -> None:
    words = needles()

    assert len(set(words)) == len(words) >= 12
    assert all(len(word) >= 4 for word in words)


# ---- the files D13 adds -------------------------------------------------------------------------


def test_the_scan_reaches_what_d13_adds() -> None:
    names = {path.relative_to(ROOT).as_posix() for path in scanned_files()}

    assert len(names) >= MINIMUM_FILES
    for wanted in (
        "src/rinchen/eval/report_read.py",
        "src/rinchen/export/model_card.py",
        "src/rinchen/export/release.py",
        "src/rinchen/cli/release.py",
        "src/rinchen/cli/release_stage.py",
        "tests/fixtures/eval/golden_card.md",
        "tests/fixtures/eval/golden_report.md",
        "tests/support/release_fixture.py",
        "tests/unit/test_d13_content_rules.py",
        "docs/superpowers/specs/2026-11-10-d13-release-design.md",
        "docs/superpowers/plans/2026-11-10-d13-release.md",
    ):
        assert wanted in names, wanted


@pytest.mark.parametrize(
    "path", scanned_files(), ids=lambda path: path.relative_to(ROOT).as_posix()
)
def test_no_file_d13_adds_holds_a_word_of_the_excluded_subjects(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pytest.skip("not a text file")

    assert lines_with_a_needle(text) == [], (
        "needle (number, line) pairs, never the word"
    )


# ---- the files the stage generates ------------------------------------------------------------------


def test_the_card_the_release_record_and_the_copies_of_the_report_hold_none(
    tmp_path: Path,
) -> None:
    release = fabricated_release(tmp_path)
    assert invoke_stage(release).exit_code == 0

    for name in (
        "README.md",
        "release.json",
        "report.json",
        "report.md",
        "rinchen.json",
    ):
        text = (release.stage / name).read_text(encoding="utf-8")
        assert lines_with_a_needle(text) == [], name
