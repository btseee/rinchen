"""`rinchen release hub --upload` and the visibility of the repository (D13 fix wave 1, I5).

The default is a private repository, and `create_repo` cannot make an existing public one private.
So after the repository is created the command asks the Hub what it IS: a public repository without
`--public` is exit 2 ("the repository exists and is public: pass --public to upload into it") with no
commit and nothing written, and `hub_commit.json` and the manifest record the visibility the Hub
REPORTS (private or public), not the one that was asked for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.manifest import store
from tests.support.hub_fakes import RecordingHub, fake_token, install_open_client
from tests.support.hub_stage import (
    COMMIT_FILE,
    bytes_under,
    holds,
    invoke_hub,
    staged_release,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_fixture import Release

pytestmark = pytest.mark.usefixtures("no_network")

REFUSAL = "the repository exists and is public: pass --public to upload into it"


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return staged_release(tmp_path)


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "hub-out"


def uploading(
    release: Release,
    out: Path,
    monkeypatch: pytest.MonkeyPatch,
    hub: RecordingHub,
    *extra: str,
) -> Any:
    install_open_client(monkeypatch, hub)
    return invoke_hub(release, out, "--upload", *extra, env={"HF_TOKEN": fake_token()})


def recorded_visibility(out: Path) -> tuple[str, str]:
    """The visibility of `hub_commit.json` and of the manifest status."""
    commit = json.loads((out / COMMIT_FILE).read_text("utf-8"))
    manifest = store.read(out)
    assert manifest is not None
    return commit["visibility"], manifest.status["visibility"]


def test_an_existing_public_repository_is_exit_2_with_no_commit_and_nothing_written(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = RecordingHub(reports_private=False)
    before = snapshot(release.stage)

    made = uploading(release, out, monkeypatch, hub)

    assert_one_error(made, 2, REFUSAL)
    assert hub.commit_calls == []
    assert not out.exists()
    assert snapshot(release.stage) == before
    assert not holds(
        fake_token(), [made.stdout, made.stderr, *bytes_under(release.root)]
    )


def test_the_refusal_leaves_the_files_of_an_earlier_run_as_they_were(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke_hub(release, out).exit_code == 0
    before = snapshot(out)

    made = uploading(release, out, monkeypatch, RecordingHub(reports_private=False))

    assert_one_error(made, 2, REFUSAL)
    assert snapshot(out) == before


def test_public_uploads_into_a_public_repository_and_records_public(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = RecordingHub(reports_private=False)

    made = uploading(release, out, monkeypatch, hub, "--public")

    assert made.exit_code == 0, made.output
    assert len(hub.commit_calls) == 1
    assert recorded_visibility(out) == ("public", "public")


def test_a_private_repository_is_recorded_private(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = uploading(release, out, monkeypatch, RecordingHub(reports_private=True))

    assert made.exit_code == 0, made.output
    assert recorded_visibility(out) == ("private", "private")


def test_the_record_says_what_the_hub_reports_when_public_was_asked_for_a_private_repository(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = RecordingHub(reports_private=True)

    made = uploading(release, out, monkeypatch, hub, "--public")

    assert made.exit_code == 0, made.output
    assert hub.repo_calls[0].private is False
    assert recorded_visibility(out) == ("private", "private")


def test_a_hub_that_does_not_say_is_exit_4_with_no_commit_and_nothing_written(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = RecordingHub(reports_private=None)

    made = uploading(release, out, monkeypatch, hub)

    assert_one_error(made, 4, "the Hub did not say whether the repository is private")
    assert hub.commit_calls == []
    assert not out.exists()


# ---- the printed line (D13 fix wave 2, finding m5) -----------------------------------------------------------------


def commit_and_visibility_lines(stdout: str) -> list[str]:
    lines = stdout.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("commit: "))
    return lines[index : index + 2]


@pytest.mark.parametrize(
    ("reports_private", "extra", "expected"),
    [
        (True, (), "private"),
        (False, ("--public",), "public"),
        # the case of the review: --public over a repository that already exists and is private
        # exits 0 and used to say nothing; the line says what the Hub reports.
        (True, ("--public",), "private"),
    ],
    ids=["private", "public", "public-asked-private-reported"],
)
def test_the_visibility_the_hub_reported_is_printed_right_after_the_commit_line(
    release: Release,
    out: Path,
    monkeypatch: pytest.MonkeyPatch,
    reports_private: bool,
    extra: tuple[str, ...],
    expected: str,
) -> None:
    hub = RecordingHub(reports_private=reports_private)

    made = uploading(release, out, monkeypatch, hub, *extra)

    assert made.exit_code == 0, made.output
    assert commit_and_visibility_lines(made.stdout)[1] == f"visibility: {expected}"
    assert recorded_visibility(out) == (expected, expected)


def test_a_dry_run_prints_no_visibility_line(release: Release, out: Path) -> None:
    made = invoke_hub(release, out)

    assert made.exit_code == 0, made.output
    assert not any(line.startswith("visibility:") for line in made.stdout.splitlines())


def test_a_refused_public_repository_prints_no_visibility_line(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = uploading(release, out, monkeypatch, RecordingHub(reports_private=False))

    assert_one_error(made, 2, REFUSAL)
    assert "visibility:" not in made.stdout
