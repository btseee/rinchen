"""What `--out-dir` of `rinchen maintain check` may already hold (D14 fix wave 1, finding M5).

The command removes the files the old manifest of its stage lists, writes its own documents and
logs, and says in its docstring that it writes nothing outside `--out-dir`. An output directory
that was any project subdirectory (`tests/`, the project root) would receive them. An EXISTING
`--out-dir` may now hold only the names this command writes: `maintain_report.json`,
`maintain_report.md`, `manifest.json` and the directories `logs`, `junit` and `parity_export`.
Anything else at its top level is a `ConfigError` (exit 2) that says how many other entries there
are and names none of them, and nothing is written. A directory that does not exist, an empty one
and the one an earlier run wrote are accepted. The manifest of another stage is refused first
(exit 3, as before).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.maintain_cli import (
    Workspace,
    install,
    invoke,
    make_workspace,
    snapshot,
)
from tests.support.maintain_fakes import FakeRunner
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

OWN_NAMES = (
    "maintain_report.json",
    "maintain_report.md",
    "manifest.json",
    "logs",
    "junit",
    "parity_export",
)
CANARY = "zqxcanary"


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return make_workspace(tmp_path)


def refused(ws: Workspace, monkeypatch: pytest.MonkeyPatch, out: Path) -> str:
    """Run into `out` with doubles; the run must end with exit 2 before anything runs or is
    written; return the error text."""
    runner = FakeRunner()
    install(monkeypatch, runner=runner)
    before = snapshot(ws.root)

    result = invoke(ws.args(out_dir=out))

    assert result.exit_code == 2, result.output
    assert snapshot(ws.root) == before
    assert runner.calls == []
    assert result.stdout == ""
    return result.stderr


def succeeds(ws: Workspace, monkeypatch: pytest.MonkeyPatch, out: Path) -> None:
    install(monkeypatch)
    result = invoke(ws.args(out_dir=out))
    assert result.exit_code == 0, result.output


# ---- what is refused -----------------------------------------------------------------------------


def test_a_directory_with_one_foreign_file_is_refused_and_the_message_counts_it(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "other"
    out.mkdir()
    (out / f"{CANARY}.txt").write_text("not ours")

    message = refused(ws, monkeypatch, out)

    assert "--out-dir" in message
    assert "1 entry" in message
    assert CANARY not in message


def test_a_part_file_left_by_a_killed_run_is_refused_and_the_message_says_what_it_is(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run killed while it wrote leaves `<name>.<hex>.part`: the directory an earlier run wrote
    is then refused, and the message must say why (it names no file)."""
    out = tmp_path / "other"
    out.mkdir()
    (out / "logs").mkdir()
    (out / f"manifest.json.{CANARY}.part").write_text("half")

    message = refused(ws, monkeypatch, out)

    assert "1 entry" in message
    assert ".part" in message
    assert "killed" in message
    assert CANARY not in message


def test_the_message_names_a_part_file_only_when_the_directory_holds_one(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "other"
    out.mkdir()
    (out / "a.txt").write_text("a")

    message = refused(ws, monkeypatch, out)

    assert ".part" not in message
    assert "killed" not in message


def test_a_foreign_directory_counts_as_an_entry_and_the_count_is_of_all_of_them(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "other"
    (out / f"{CANARY}_dir").mkdir(parents=True)
    (out / "a.txt").write_text("a")
    (out / "b.txt").write_text("b")
    (out / "logs").mkdir()  # one of ours: not counted

    message = refused(ws, monkeypatch, out)

    assert "3 entries" in message
    assert CANARY not in message


def test_a_tests_style_directory_is_refused(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = ws.project / "tests"

    message = refused(ws, monkeypatch, out)

    assert "entries" in message


def test_the_project_root_is_refused(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused(ws, monkeypatch, ws.project)


def test_a_directory_with_a_tracked_file_is_refused(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = ws.root / "tracked"
    out.mkdir()
    (out / "README.md").write_text("a tracked file")

    message = refused(ws, monkeypatch, out)

    assert "1 entry" in message


def test_a_directory_with_a_file_of_ours_and_one_of_another_is_refused(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws.out.mkdir()
    (ws.out / "maintain_report.json").write_text("old")
    (ws.out / "notes.txt").write_text("the operator's own note")

    message = refused(ws, monkeypatch, ws.out)

    assert "1 entry" in message


@pytest.mark.parametrize(
    "name", ["maintain_report.json.part", "Manifest.json", "log", "logs.txt", ".hidden"]
)
def test_a_name_that_is_not_one_of_ours_exactly_is_refused(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    ws.out.mkdir()
    (ws.out / name).write_text("x")

    refused(ws, monkeypatch, ws.out)


def test_a_foreign_manifest_is_refused_with_exit_3_before_the_foreign_entries(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws.out.mkdir()
    (ws.out / "notes.txt").write_text("not ours")
    store.write(
        ws.out,
        Manifest(
            stage="export_parity",
            created="2026-01-01T00:00:00+00:00",
            code_version="abc1234",
            config_sha256="0" * 64,
            seed=-1,
        ),
    )
    install(monkeypatch, runner=FakeRunner())

    result = invoke(ws.args())

    assert result.exit_code == 3, result.output
    assert "export_parity" in result.stderr


# ---- what is accepted ------------------------------------------------------------------------------


def test_a_directory_that_does_not_exist_is_accepted(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not ws.out.exists()

    succeeds(ws, monkeypatch, ws.out)

    assert (ws.out / "manifest.json").is_file()


def test_an_empty_directory_is_accepted(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws.out.mkdir()

    succeeds(ws, monkeypatch, ws.out)

    assert (ws.out / "manifest.json").is_file()


def test_the_directory_an_earlier_run_wrote_is_accepted(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    succeeds(ws, monkeypatch, ws.out)
    first = sorted(snapshot(ws.out))

    succeeds(ws, monkeypatch, ws.out)

    assert sorted(snapshot(ws.out)) == first


def test_a_directory_that_holds_every_name_of_ours_is_accepted(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws.out.mkdir()
    for name in OWN_NAMES:
        path = ws.out / name
        if "." in name:
            path.write_text("old")
        else:
            path.mkdir()

    succeeds(ws, monkeypatch, ws.out)


def test_a_foreign_file_inside_one_of_our_directories_is_left_alone_and_accepted(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    (ws.out / "logs").mkdir(parents=True)
    (ws.out / "logs" / "extra.log").write_text("not listed by any manifest")

    succeeds(ws, monkeypatch, ws.out)

    assert (ws.out / "logs" / "extra.log").read_text() == "not listed by any manifest"


def test_an_output_directory_that_is_a_file_is_not_this_rules_business(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    install(monkeypatch)

    result = invoke(ws.args(out_dir=blocker))

    assert result.exit_code == 1, result.output
