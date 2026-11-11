"""Property test of the report of a maintenance check (D14, spec M2, M3 and M11): over 200 generated
runs of `run_check` (a doctor of random outcomes, a lock check and a dry run that end in every way
a command can, an environment equal to the lock or not, any cadence, any selection, JUnit reports
of random counters, an export or none), the report and its Markdown hold nothing but observations.

For every run: no absolute path, timestamp, host name, line of the recorded uv output or value of
the environment is in the JSON or the Markdown, though the fakes' logs, the JUnit files and the
parent environment are full of them; every reason in the report is one of the closed strings of
`rinchen.maintain.states`; the bytes are the sorted two-space form with one final line feed and
the Markdown has line feeds only; a second run in another directory gives equal bytes; and the
overall state and the exit code are what an independent model computes from the six step states.
The model is another statement of M2 and M3: the worst state of the steps, `drift` only from the
drift step, the exit code 4 for a failure or for drift when asked.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.maintain.check import DoctorFinding
from rinchen.maintain.commands import CommandResult
from rinchen.maintain.report import (
    STEP_IDS,
    exit_code_of,
    report_document,
    report_to_bytes,
)
from rinchen.maintain.report_markdown import render_markdown
from rinchen.maintain.reruns import ExportInputs
from tests.support.maintain_check import (
    CPU_CHECKS,
    GPU_CHECK,
    SECRET_ENVIRONMENT,
    Doctor,
    execute,
    finding,
    make_options,
    make_seams,
    rerun_call,
    uv_call,
)
from tests.support.maintain_fakes import JUNIT_HOST, JUNIT_PATH, FakeRunner, junit_xml
from tests.support.maintain_scans import closed_reason, reasons_of, violations
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import (
    CHANGES_HAND_WRITTEN,
    NO_CHANGE_MEASURED,
    OFFLINE_FAILURE_MEASURED,
    SAMPLE_LOCKED,
)

pytestmark = pytest.mark.usefixtures("no_network")

RERUN_IDS = (
    "numerics",
    "parity_tiny",
    "parity_export",
    "robustness_tiny",
    "evaluation_tiny",
    "serving_tiny",
)
# What a command may leave in its log: a path, a host, a secret and the lines of uv.
NOISE = f"{JUNIT_PATH} {JUNIT_HOST} {SECRET_ENVIRONMENT['CI_TOKEN']} Using CPython at /opt/x\n"
DRY_TEXTS = (
    NO_CHANGE_MEASURED,
    CHANGES_HAND_WRITTEN,
    OFFLINE_FAILURE_MEASURED,
    "",
    "Update numpy v1.0 -> v2.0\n",
    f"Update numpy v2.5.4 -> v2.5.5\n{NOISE}",
    NOISE,
)
RESULTS = (
    CommandResult(0, False),
    CommandResult(1, False),
    CommandResult(5, False),
    CommandResult(None, True),
    CommandResult(None, False, False),
)
SECRETS = [*SECRET_ENVIRONMENT.values(), JUNIT_HOST, JUNIT_PATH, "/srv/example"]
PARITY_JSON = json.dumps(
    {"exact": 3, "sentences": 4, "mean_abs_delta": 1.5e-06}
).encode()

counts = st.tuples(*(st.integers(0, 4) for _ in range(3))).map(
    lambda three: (sum(three) + 2, *three)  # tests, failures, errors, skipped; 2 passed
)
statuses = st.sampled_from(["pass", "pass", "pass", "fail", "skip"])
scenarios = st.fixed_dictionaries(
    {
        "doctor": st.one_of(st.none(), st.lists(statuses, min_size=6, max_size=6)),
        "gpu": st.sampled_from(["pass", "fail", "skip", "absent"]),
        "uv": st.booleans(),
        "lock_check": st.sampled_from(RESULTS),
        "dry_result": st.sampled_from(RESULTS),
        "dry_text": st.sampled_from(DRY_TEXTS),
        "installed": st.sampled_from(["equal", "mismatch", "extra", "missing"]),
        "cadence": st.sampled_from(["nightly", "weekly"]),
        "drift": st.booleans(),
        "reruns": st.booleans(),
        "selected": st.lists(st.sampled_from(RERUN_IDS), max_size=3, unique=True),
        "export": st.booleans(),
        "results": st.lists(st.sampled_from(RESULTS), min_size=6, max_size=6),
        "junit": st.lists(st.one_of(st.none(), counts), min_size=6, max_size=6),
        "fail_on_drift": st.booleans(),
    }
)


def doctor_of(shape: Mapping[str, Any]) -> Callable[[], Sequence[DoctorFinding]]:
    if shape["doctor"] is None:

        def raising() -> None:
            raise RuntimeError(f"the doctor failed at {JUNIT_PATH}")

        return raising
    found = [
        finding(check, status)
        for check, status in zip(CPU_CHECKS, shape["doctor"], strict=True)
    ]
    if shape["gpu"] != "absent":
        found.append(finding(GPU_CHECK, shape["gpu"]))
    return Doctor(found)


def installed_of(shape: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    table = dict(SAMPLE_LOCKED)
    if shape["installed"] == "mismatch":
        table["numpy"] = ("0.0.1",)
    elif shape["installed"] == "extra":
        table["not-in-the-lock"] = ("1",)
    elif shape["installed"] == "missing":
        del table["rich"]
    return table


def runner_of(shape: Mapping[str, Any], out: Path) -> FakeRunner:
    runner = FakeRunner()
    runner.script(shape["lock_check"], log_text=NOISE, when=uv_call("--check"))
    runner.script(
        shape["dry_result"],
        log_text=shape["dry_text"],
        when=uv_call("--upgrade", "--dry-run"),
    )
    for rerun_id, result, junit in zip(
        RERUN_IDS, shape["results"], shape["junit"], strict=True
    ):
        data = None if junit is None else junit_xml(*junit)
        files = (
            {out / "parity_export" / "parity.json": PARITY_JSON}
            if rerun_id == "parity_export"
            else {}
        )
        runner.script(
            result,
            log_text=NOISE,
            junit=None if rerun_id == "parity_export" else data,
            files=files,
            when=rerun_call(rerun_id),
        )
    return runner


def run_scenario(shape: Mapping[str, Any], root: Path) -> Any:
    out = root / "out"
    export = (
        ExportInputs(root / "export", root / "ckpt", (root / "a.parquet",))
        if shape["export"]
        else None
    )
    runner = runner_of(shape, out)
    seams = make_seams(
        runner,
        doctor=doctor_of(shape),
        installed=lambda: installed_of(shape),
        uv="uv" if shape["uv"] else None,
    )
    options = make_options(
        cadence=shape["cadence"],
        drift=shape["drift"],
        reruns=shape["reruns"],
        selected=tuple(shape["selected"]),
        export=export,
    )
    return execute(root, runner, options=options, seams=seams)


def independent_state(document: Mapping[str, Any]) -> str:
    """The overall state from the six step states alone."""
    by_id = {step["id"]: step["state"] for step in document["steps"]}
    if "fail" in by_id.values():
        return "fail"
    return "drift" if by_id["drift"] == "drift" else "ok"


@settings(max_examples=200, deadline=None)
@given(scenarios)
def test_every_generated_report_holds_observations_only_and_decides_as_the_model_does(
    shape: dict[str, Any],
) -> None:
    with (
        tempfile.TemporaryDirectory() as first_dir,
        tempfile.TemporaryDirectory() as second_dir,
    ):
        first = run_scenario(shape, Path(first_dir))
        second = run_scenario(shape, Path(second_dir))
    report = first.result.report
    data = report_to_bytes(report)
    markdown = render_markdown(report)
    document = report_document(report)

    for text in (data.decode("utf-8"), markdown):
        assert violations(text, SECRETS) == []
    assert all(closed_reason(reason) for reason in reasons_of(document))
    assert [step["id"] for step in document["steps"]] == list(STEP_IDS)

    assert data.endswith(b"}\n") and not data.endswith(b"\n\n")
    assert (
        data.decode("utf-8")
        == json.dumps(json.loads(data), sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    )
    assert markdown.endswith("\n") and not markdown.endswith("\n\n")
    assert "\r" not in markdown and "\n\n\n" not in markdown

    assert data == report_to_bytes(second.result.report)
    assert markdown == render_markdown(second.result.report)

    assert document["state"] == independent_state(document)
    fail_on_drift = shape["fail_on_drift"]
    expected_exit = (
        4
        if document["state"] == "fail"
        or (document["state"] == "drift" and fail_on_drift)
        else 0
    )
    assert exit_code_of(report, fail_on_drift=fail_on_drift) == expected_exit
