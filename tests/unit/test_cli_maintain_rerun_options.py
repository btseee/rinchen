"""The options that choose the reruns of `rinchen maintain check` (D14 fix wave 1, finding M6).

`--no-reruns` runs none and `--rerun X` runs exactly the ones it names: given together they ask
for opposite things, and the command used to let `--no-reruns` win in silence. The pair is now a
usage error (exit 2) that names both options and writes nothing, checked with the other options
and before any file is read. `--rerun` alone reports the reruns it left out as `not selected with
--rerun`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rinchen.maintain.states import SKIP_NOT_SELECTED
from tests.support.maintain_check import Doctor, rerun_id_of, step_of
from tests.support.maintain_cli import (
    UV,
    Workspace,
    install,
    invoke,
    make_workspace,
    snapshot,
)
from tests.support.maintain_fakes import FakeRunner
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return make_workspace(tmp_path)


def test_no_reruns_together_with_rerun_is_a_usage_error_that_names_both(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, doctor = FakeRunner(), Doctor([])
    install(monkeypatch, runner=runner, doctor=doctor)
    before = snapshot(ws.root)

    result = invoke(ws.args("--no-reruns", "--rerun", "numerics"))

    assert result.exit_code == 2, result.output
    assert "--no-reruns" in result.stderr and "--rerun" in result.stderr
    assert result.stdout == ""
    assert snapshot(ws.root) == before
    assert not ws.out.exists()
    assert runner.calls == [] and doctor.calls == 0


def test_the_order_of_the_two_options_does_not_matter(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, runner=FakeRunner(), doctor=Doctor([]))

    result = invoke(ws.args("--rerun", "numerics", "--no-reruns"))

    assert result.exit_code == 2, result.output
    assert "--no-reruns" in result.stderr


def test_the_pair_is_refused_before_any_file_is_read(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    (ws.project / "uv.lock").unlink()  # exit 3 if the files were read first
    install(monkeypatch, runner=FakeRunner(), doctor=Doctor([]))

    result = invoke(ws.args("--no-reruns", "--rerun", "numerics"))

    assert result.exit_code == 2, result.output


def test_no_reruns_alone_and_rerun_alone_are_accepted(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    assert invoke(ws.args("--no-reruns")).exit_code == 0

    install(monkeypatch)
    assert invoke(ws.args("--rerun", "numerics")).exit_code == 0


def test_a_named_rerun_leaves_the_others_not_selected_in_the_report_and_on_screen(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = install(monkeypatch)

    result = invoke(ws.args("--rerun", "numerics"))

    assert result.exit_code == 0, result.output
    document = json.loads((ws.out / "maintain_report.json").read_bytes())
    items = {
        item["id"]: (item["state"], item["reason"])
        for item in step_of(document, "reruns")["detail"]["items"]
    }
    assert items["numerics"] == ("ok", None)
    assert items["parity_tiny"] == ("skipped", SKIP_NOT_SELECTED)
    assert items["serving_tiny"] == ("skipped", SKIP_NOT_SELECTED)
    assert f"rerun parity_tiny: skipped ({SKIP_NOT_SELECTED})" in result.stdout
    assert [rerun_id_of(argv) for argv in runner.argvs if argv[0] != UV] == ["numerics"]
