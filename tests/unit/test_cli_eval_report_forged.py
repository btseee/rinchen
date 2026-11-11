"""`rinchen eval report` over a result file that was edited and re-digested (D13 fix wave 1, I1).

The edit is made consistent on purpose: the digest of the new `result.json` is written into
`run.json` and the digest of `run.json` into the manifest, so every hash of the chain holds. Only the
recomputation from the suite and the outputs can tell, and it must: exit 3, one line that names the
run, the suite, the direction and the field and quotes no value, and nothing written.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rinchen.eval.report_read import report_from_bytes
from tests.support.eval_report_cli import (
    FIRST,
    ReportWorkspace,
    forge_result,
    invoke_report,
    make_report_workspace,
)
from tests.support.launch_cli import assert_one_error
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

START = (
    f"the run of the system 'alpha' over the suite 'golden', direction {FIRST}: the "
)


PLANTED_FIELD = (
    "comet22 of the result has a value and the run lists no segment scores for it"
)
REASON_FIELD = (
    "unmeasured reason of the comet22 of the result is not the one the runner writes"
)


@pytest.fixture
def workspace(tmp_path: Path) -> ReportWorkspace:
    return make_report_workspace(tmp_path)


def forged_interval(document: dict[str, Any]) -> None:
    document["ci95"]["chrf_pp"] = [1.0, 2.0]


def forged_statuses(document: dict[str, Any]) -> None:
    document["statuses"] = {"empty": 4, "ok": 6}


def forged_signature(document: dict[str, Any]) -> None:
    document["chrf_pp_signature"] = document["chrf_pp_signature"].replace(
        "version:2.6.0", "version:9.9.9"
    )


def forged_trained(document: dict[str, Any]) -> None:
    document["trained"] = False


def planted_neural_value(document: dict[str, Any]) -> None:
    document["metrics"]["comet22"] = 0.77
    document["ci95"]["comet22"] = [0.7, 0.8]
    del document["unmeasured"]["comet22"]


def another_reason(document: dict[str, Any]) -> None:
    document["unmeasured"]["comet22"] = "the scorer crashed"


@pytest.mark.parametrize(
    ("edit", "field"),
    [
        (forged_interval, "ci95 of the chrf_pp of the result"),
        (forged_statuses, "statuses of the result"),
        (forged_signature, "chrf_pp_signature of the result"),
        (forged_trained, "trained flag of the result"),
        (planted_neural_value, PLANTED_FIELD),
        (another_reason, REASON_FIELD),
    ],
)
def test_a_consistently_re_digested_forgery_is_a_data_error_naming_the_field(
    workspace: ReportWorkspace, edit: Callable[[dict[str, Any]], None], field: str
) -> None:
    forge_result(workspace.alpha, FIRST, edit)

    result = invoke_report(workspace)

    assert_one_error(result, 3, START + field)
    assert result.stdout == ""
    assert not workspace.report_out.exists()


def test_the_signature_refusal_names_the_installed_sacrebleu_and_the_command_to_run(
    workspace: ReportWorkspace,
) -> None:
    forge_result(workspace.alpha, FIRST, forged_signature)

    result = invoke_report(workspace)

    assert_one_error(result, 3, START + "chrf_pp_signature of the result")
    assert f"({importlib.metadata.version('sacrebleu')})" in result.stderr
    assert "run `rinchen eval run` again" in result.stderr


def test_the_digest_chain_alone_does_not_see_the_forgery(
    workspace: ReportWorkspace,
) -> None:
    # The control of the tests above: the same edit made so that the chain holds is accepted by
    # every earlier check, which is why the recomputation exists. An edit that changes nothing
    # in the result's meaning (the same document) leaves a report that is made.
    forge_result(workspace.alpha, FIRST, lambda document: None)

    result = invoke_report(workspace)

    assert result.exit_code == 0, result.output
    assert report_from_bytes((workspace.report_out / "report.json").read_bytes())
