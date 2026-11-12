"""What a broken copy of the files of D15's Task 8 is made of (spec P16): the change (`Case`: a text
replaced in one file of a set, `Files`, and the sentence `findings` must say about the copy), the
sentences, written out here and not taken from the finder so that a finder that says the wrong thing
about a break is a case that fails and not a case that agrees with it, the anchors (a text that occurs
once in the valid file of its kind) and the small constructors of the table of
`tests/support/ci_workflow_breaks.py`.

`{tiny_minimum}` and `{nightly_minimum}` in a sentence are the least timeouts of the shipped
configuration, filled in by `sentence_of`.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from rinchen.config.maintain import MaintainConfig
from rinchen.config.tiny import TinyConfig
from tests.support.ci_workflow import (
    SUMMARY_RUN,
    TINY_RECIPE,
    Files,
    minimum_tiny_timeout,
    nightly_minimum_timeout,
)


class Case(NamedTuple):
    """`old` of the file `file` becomes `new`; `findings` must then say `expected`. An `old` that
    occurs more than once names the occurrence (from 0) that changes."""

    file: str
    old: str
    new: str
    expected: str
    occurrence: int | None = None


def replace_occurrence(text: str, old: str, new: str, occurrence: int | None) -> str:
    count = text.count(old)
    if occurrence is None:
        assert count == 1, f"{old!r} occurs {count} times, not once"
        return text.replace(old, new)
    assert count > occurrence, f"{old!r} occurs {count} times, not {occurrence + 1}"
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(old, start + 1)
    return text[:start] + new + text[start + len(old) :]


def broken(files: Files, case: Case) -> Files:
    """The set of files with the one change of `case`."""
    text = replace_occurrence(
        getattr(files, case.file), case.old, case.new, case.occurrence
    )
    return files.replacing(case.file, text)


def sentence_of(case: Case, tiny: TinyConfig, maintain: MaintainConfig) -> str:
    return case.expected.format(
        tiny_minimum=minimum_tiny_timeout(tiny),
        nightly_minimum=nightly_minimum_timeout(tiny, maintain),
    )


CI: Final = "ci.yml: "
NIGHTLY: Final = "nightly.yml: "
UPLOAD: Final = "the upload of job tiny "
SUMMARY: Final = "the summary step of job tiny "
TRIGGER: Final = CI + "the trigger is not a push to main and pull requests"
TOP_KEYS: Final = CI + "the workflow holds keys other than name, on and jobs"
JOBS: Final = CI + "the jobs are not gate and tiny, in order"
GATE_KEYS: Final = CI + "job gate holds keys other than runs-on and steps"
GATE_STEPS_SENTENCE: Final = (
    CI
    + "the steps of job gate are not the set-up, the sync, the gate and the doctor, in order"
)
TINY_KEYS: Final = (
    CI
    + "job tiny holds keys other than runs-on, timeout-minutes, permissions and steps"
)
TINY_STEPS_SENTENCE: Final = (
    CI + "the steps of job tiny are not the ones of the pipeline, in order"
)
NOT_ONCE: Final = CI + "job tiny does not run just tiny exactly once"
ONE_UPLOAD: Final = CI + "job tiny does not hold exactly one upload step"
NOT_45: Final = CI + "job tiny has a timeout other than 45 minutes"
NO_TIMEOUT: Final = CI + "job tiny has no timeout"
BELOW: Final = CI + "job tiny has a timeout below the bound of {tiny_minimum} minutes"
PERMISSIONS: Final = CI + "job tiny does not have the permissions contents: read"
NAMES_STUB: Final = " still names the stub of the pipeline"
ONE_VERSION: Final = "the workflows name more than one version of "
NOT_PINNED: Final = "an action is not pinned to a version tag"
RENAMED: Final = (
    NIGHTLY
    + "the names of the steps are not the old ones with the tiny pipeline renamed"
)
NIGHTLY_BELOW: Final = (
    NIGHTLY
    + "job maintain has a timeout below the check plus the pipeline, {nightly_minimum} minutes"
)
NOT_THE_RECIPE: Final = f"the tiny recipe is not {TINY_RECIPE}"
ONE_RECIPE: Final = "the justfile does not hold exactly one tiny recipe"
NOT_IGNORED: Final = (
    "the output directory of the tiny pipeline is not ignored at the root"
)

# The anchors: a text that occurs once in the valid file of its kind.
GATE_HEAD: Final = "  gate:\n    runs-on: ubuntu-24.04\n"
TINY_HEAD: Final = "  tiny:\n    runs-on: ubuntu-24.04\n"
TINY_TIMEOUT: Final = "    timeout-minutes: 45\n"
TINY_PERMISSIONS: Final = "    permissions:\n      contents: read\n"
PIPELINE_STEP: Final = "      - name: Tiny pipeline\n        run: just tiny\n\n"
RUN_TINY: Final = "        run: just tiny\n"
COVERAGE_STEP: Final = (
    "      - name: Upload coverage\n        if: always()\n        uses: actions/upload-artifact@v7\n"
    "        with:\n          name: coverage\n          path: coverage.xml\n          if-no-files-found: ignore\n\n"
)
DOCTOR_STEP: Final = (
    "      - name: Doctor (environment checks must all pass or skip, never fail)\n"
    "        run: just doctor\n\n"
)
UPLOAD_HEAD: Final = (
    "      - name: Upload the tiny report\n        if: always()\n"
    "        uses: actions/upload-artifact@v7\n"
)
UPLOAD_NAME_LINE: Final = "          name: tiny-report\n"
JSON_PATH: Final = "            artifacts/tiny/report.json\n"
MARKDOWN_PATH: Final = "            artifacts/tiny/report.md\n"
SUMMARY_LINE: Final = f"        run: {SUMMARY_RUN}\n"
RECIPE_LINE: Final = f"    {TINY_RECIPE}\n"
IGNORE_PAIR: Final = (
    "# The output directory of the tiny pipeline (just tiny)\n/artifacts/\n"
)
STUB_STEP: Final = (
    "      - name: Tiny pipeline (stub until D15)\n        run: just tiny\n\n"
)


def after_pipeline(name: str, body: str) -> str:
    """The pipeline step followed by one more step."""
    return RUN_TINY + f"\n      - name: {name}\n        {body}\n"


def ci(old: str, new: str, expected: str, occurrence: int | None = None) -> Case:
    return Case("ci", old, new, expected, occurrence)


def nightly(old: str, new: str, expected: str) -> Case:
    return Case("nightly", old, new, expected)


def release(old: str, new: str, expected: str) -> Case:
    return Case("release", old, new, expected)


def recipe(label: str, new: str, expected: str) -> tuple[str, Case]:
    """The line of the recipe becomes `new`."""
    return label, Case("justfile", RECIPE_LINE, new, expected)


def ignore(old: str, new: str, expected: str) -> Case:
    return Case("gitignore", old, new, expected)


def numbered(cases: list[tuple[str, Case]]) -> dict[str, Case]:
    table: dict[str, Case] = {}
    for name, case in cases:
        assert name not in table, name
        table[name] = case
    return table
