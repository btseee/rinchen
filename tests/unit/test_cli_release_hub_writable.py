"""`rinchen release hub --upload` finds out that its record cannot be written BEFORE the commit (D13
fix wave 1, M11).

The commit of an upload cannot be undone and the local record (`hub_commit.json`, the manifest) can
only be written after it, so a read-only `--out-dir` used to be found out after the irreversible
step. Now `_prepare` makes the directory (and the parents it has to), creates and removes a probe
file in it, and removes the directories it made, all before any client is opened: exit 1, the same
line as a write that fails at the end, nothing sent, nothing left behind. A dry run has nothing to
protect and does not probe.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tests.support.hub_fakes import RecordingHub, fake_token, install_open_client
from tests.support.hub_stage import (
    COMMIT_FILE,
    MANIFEST,
    PLAN_FILE,
    invoke_hub,
    staged_release,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_fixture import Release

pytestmark = [
    pytest.mark.usefixtures("no_network"),
    pytest.mark.skipif(
        os.geteuid() == 0,
        reason="running as root: a read-only directory is writable, so the case cannot arise",
    ),
]


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return staged_release(tmp_path)


@contextmanager
def read_only(directory: Path) -> Iterator[None]:
    before = stat.S_IMODE(directory.stat().st_mode)
    directory.chmod(0o500)
    try:
        yield
    finally:
        directory.chmod(before)


def uploading(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch, hub: RecordingHub
) -> tuple[Any, list[str]]:
    opened = install_open_client(monkeypatch, hub)
    made = invoke_hub(release, out, "--upload", env={"HF_TOKEN": fake_token()})
    return made, opened


def test_an_existing_read_only_out_directory_is_exit_1_before_any_client_is_opened(
    release: Release, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "hub-out"
    out.mkdir()
    hub = RecordingHub()

    with read_only(out):
        made, opened = uploading(release, out, monkeypatch, hub)

    assert_one_error(made, 1, f"cannot write the Hub plan under {out}")
    assert opened == [] and hub.calls == []
    assert made.stdout == ""


def test_a_missing_out_directory_under_a_read_only_parent_is_exit_1_and_makes_nothing(
    release: Release, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "locked"
    parent.mkdir()
    out = parent / "deeper" / "hub-out"
    hub = RecordingHub()

    with read_only(parent):
        made, opened = uploading(release, out, monkeypatch, hub)
        left = sorted(parent.iterdir())

    assert_one_error(made, 1, "cannot write the Hub plan under")
    assert opened == [] and hub.calls == []
    assert left == []


def test_an_earlier_record_in_a_read_only_directory_is_left_as_it_was(
    release: Release, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "hub-out"
    assert invoke_hub(release, out).exit_code == 0
    before = snapshot(out)
    hub = RecordingHub()

    with read_only(out):
        made, opened = uploading(release, out, monkeypatch, hub)

    assert_one_error(made, 1, "cannot write the Hub plan under")
    assert opened == [] and hub.calls == []
    assert snapshot(out) == before


def test_the_probe_leaves_no_file_and_no_directory_when_the_upload_then_fails(
    release: Release, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made_dir = tmp_path / "exists"
    made_dir.mkdir()
    missing = tmp_path / "a" / "b" / "missing"
    hub = RecordingHub(repo_error=RuntimeError("no"))

    first, _ = uploading(release, made_dir, monkeypatch, hub)
    second, _ = uploading(release, missing, monkeypatch, hub)

    assert first.exit_code == 4 and second.exit_code == 4
    assert list(made_dir.iterdir()) == []
    assert not (tmp_path / "a").exists()


def test_a_writable_out_directory_is_probed_and_the_upload_goes_through(
    release: Release, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "x" / "hub-out"
    hub = RecordingHub()

    made, opened = uploading(release, out, monkeypatch, hub)

    assert made.exit_code == 0, made.output
    assert len(opened) == 1 and len(hub.commit_calls) == 1
    assert sorted(path.name for path in out.iterdir()) == sorted(
        [COMMIT_FILE, MANIFEST, PLAN_FILE]
    )


def test_a_dry_run_does_not_probe_and_creates_nothing_when_it_refuses(
    release: Release, tmp_path: Path
) -> None:
    out = tmp_path / "dry" / "out"
    (release.stage / "stray.bin").write_bytes(b"x")

    made = invoke_hub(release, out)

    assert made.exit_code == 3, made.output
    assert not (tmp_path / "dry").exists()


def test_a_dry_run_into_a_read_only_directory_still_fails_at_the_end_with_1(
    release: Release, tmp_path: Path
) -> None:
    out = tmp_path / "hub-out"
    out.mkdir()

    with read_only(out):
        made = invoke_hub(release, out)

    assert made.exit_code == 1, made.output
    # The plan was printed before the write failed: a dry run has no probe in front of it.
    assert made.stdout.startswith("mode: dry-run\n")
