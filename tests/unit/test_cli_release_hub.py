"""`rinchen release hub` as a dry run (D13, Task 10, spec T15 and T18): the default mode prints the plan
of the upload and writes `hub_plan.json` and the manifest of the stage `release_hub`, and does nothing
else: it builds no operation, creates no client and reads no environment variable. What it refuses is
in `test_cli_release_hub_refusals.py` and what `--upload` does is in `test_cli_release_hub_upload.py`.

The stage is the one `rinchen release stage` wrote over a fabricated export; no model is loaded and
nothing reaches the Hub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.export.hub import plan_upload
from rinchen.export.release import UPLOAD_NAMES
from rinchen.manifest import store
from tests.support.hub_fakes import fake_token, install_open_client
from tests.support.hub_stage import (
    COMMIT_FILE,
    MANIFEST,
    PLAN_FILE,
    bytes_under,
    holds,
    invoke_hub,
    sha256_of,
    staged_release,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_fixture import Release
from tests.support.robust_cli import written

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return staged_release(tmp_path)


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "hub-out"


def file_lines(release: Release) -> list[str]:
    stage = release.stage
    return [
        f"{name} {(stage / name).stat().st_size} "
        f"{sha256_of((stage / name).read_bytes())[:12]}"
        for name in UPLOAD_NAMES
    ]


# ---- what it prints ----------------------------------------------------------------------------------------


def test_the_dry_run_prints_the_mode_the_repository_each_file_and_the_total(
    release: Release, out: Path
) -> None:
    made = invoke_hub(release, out)

    assert made.exit_code == 0, made.output
    assert made.stderr == ""
    total = sum((release.stage / name).stat().st_size for name in UPLOAD_NAMES)
    assert made.stdout.splitlines() == [
        "mode: dry-run",
        f"repo: {release.repo_id}",
        *file_lines(release),
        f"total: {total}",
        f"plan: {out / PLAN_FILE}",
        f"manifest: {out / MANIFEST}",
    ]


def test_the_lines_of_the_files_are_in_the_order_of_the_upload(
    release: Release, out: Path
) -> None:
    made = invoke_hub(release, out)

    names = [line.split()[0] for line in made.stdout.splitlines()[2:11]]
    assert names == list(UPLOAD_NAMES)


# ---- what it writes ----------------------------------------------------------------------------------------


def test_the_out_directory_holds_the_plan_and_the_manifest_and_nothing_else(
    release: Release, out: Path
) -> None:
    assert invoke_hub(release, out).exit_code == 0

    assert sorted(path.name for path in out.iterdir()) == sorted([MANIFEST, PLAN_FILE])


def test_the_plan_file_is_the_document_of_the_plan_as_the_disk_has_it(
    release: Release, out: Path
) -> None:
    assert invoke_hub(release, out).exit_code == 0

    plan = plan_upload(release.stage, release.repo_id)
    assert (out / PLAN_FILE).read_bytes() == plan.to_bytes()
    document = json.loads((out / PLAN_FILE).read_text(encoding="utf-8"))
    assert [file["name"] for file in document["files"]] == list(UPLOAD_NAMES)


def test_the_manifest_is_of_the_stage_release_hub_and_vouches_for_what_was_read(
    release: Release, out: Path
) -> None:
    assert invoke_hub(release, out).exit_code == 0

    manifest = store.read(out)
    assert manifest is not None
    assert manifest.stage == "release_hub"
    assert [Path(entry.path).name for entry in manifest.inputs] == [
        MANIFEST,
        *UPLOAD_NAMES,
    ]
    for entry in manifest.inputs:
        data = Path(entry.path).read_bytes()
        assert Path(entry.path).parent == release.stage
        assert entry.sha256 == sha256_of(data) and entry.bytes == len(data)
    assert [Path(entry.path).name for entry in manifest.outputs] == [PLAN_FILE]
    assert manifest.outputs[0].sha256 == sha256_of((out / PLAN_FILE).read_bytes())
    assert manifest.status == {
        "mode": "dry-run",
        "repo_id": release.repo_id,
        "visibility": "private",
    }
    assert manifest.seed == -1
    assert manifest.metrics["files"] == 9.0


def test_the_stage_manifest_is_hashed_before_it_is_read_and_read_as_hashed(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rinchen.cli.release_hub as command

    events: list[str] = []
    real_hash, real_read = command.hash_input, command.verified_content

    def hashing(path: Path, **options: Any) -> Any:
        events.append(f"hash {path.name}")
        return real_hash(path, **options)

    def reading(path: Path, digest: str) -> bytes:
        events.append(f"read {path.name}")
        return real_read(path, digest)

    monkeypatch.setattr(command, "hash_input", hashing)
    monkeypatch.setattr(command, "verified_content", reading)

    assert invoke_hub(release, out).exit_code == 0

    assert events == [f"hash {MANIFEST}", f"read {MANIFEST}"]


def test_two_runs_write_the_same_bytes_and_print_the_same_lines(
    release: Release, tmp_path: Path
) -> None:
    first, second = tmp_path / "one", tmp_path / "two"

    one = invoke_hub(release, first)
    two = invoke_hub(release, second)

    assert one.exit_code == two.exit_code == 0
    assert (first / PLAN_FILE).read_bytes() == (second / PLAN_FILE).read_bytes()
    assert one.stdout.replace(str(first), "O") == two.stdout.replace(str(second), "O")


def test_a_second_run_into_the_same_directory_replaces_the_files_by_the_same_bytes(
    release: Release, out: Path
) -> None:
    assert invoke_hub(release, out).exit_code == 0
    plan = (out / PLAN_FILE).read_bytes()

    assert invoke_hub(release, out).exit_code == 0

    assert (out / PLAN_FILE).read_bytes() == plan
    assert sorted(path.name for path in out.iterdir()) == sorted([MANIFEST, PLAN_FILE])


def test_a_file_the_old_manifest_does_not_list_is_left_alone(
    release: Release, out: Path
) -> None:
    assert invoke_hub(release, out).exit_code == 0
    (out / "notes.txt").write_bytes(b"mine")

    assert invoke_hub(release, out).exit_code == 0

    assert (out / "notes.txt").read_bytes() == b"mine"


def test_the_stage_is_not_touched(release: Release, out: Path) -> None:
    before = written(release.stage)

    assert invoke_hub(release, out).exit_code == 0

    assert written(release.stage) == before
    assert not (release.stage / PLAN_FILE).exists()


def test_the_dry_run_never_writes_a_commit_record(release: Release, out: Path) -> None:
    assert invoke_hub(release, out).exit_code == 0

    assert not (out / COMMIT_FILE).exists()


# ---- visibility ---------------------------------------------------------------------------------------------


def test_public_in_a_dry_run_is_recorded_as_what_an_upload_would_do_and_changes_no_file(
    release: Release, tmp_path: Path
) -> None:
    private, public = tmp_path / "private", tmp_path / "public"

    assert invoke_hub(release, private).exit_code == 0
    assert invoke_hub(release, public, "--public").exit_code == 0

    assert (private / PLAN_FILE).read_bytes() == (public / PLAN_FILE).read_bytes()
    manifest = store.read(public)
    assert manifest is not None and manifest.status["visibility"] == "public"
    manifest = store.read(private)
    assert manifest is not None and manifest.status["visibility"] == "private"


# ---- the dry run builds nothing and reads no environment variable ----------------------------------------------


def test_the_dry_run_creates_no_client_and_never_asks_for_the_token(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = install_open_client(monkeypatch, None)
    asked: list[str] = []
    monkeypatch.setattr(
        "rinchen.cli.release_hub.token_from_environment",
        lambda: asked.append("asked") or "unused",
    )
    token = fake_token()

    made = invoke_hub(release, out, env={"HF_TOKEN": token})

    assert made.exit_code == 0, made.output
    assert opened == [] and asked == []
    assert not holds(token, [made.stdout, made.stderr, *bytes_under(out)])


def test_the_dry_run_with_no_token_in_the_environment_is_the_same_run(
    release: Release, tmp_path: Path
) -> None:
    with_token, without = tmp_path / "with", tmp_path / "without"

    one = invoke_hub(release, with_token, env={"HF_TOKEN": fake_token()})
    two = invoke_hub(release, without, env={"HF_TOKEN": None})

    assert one.exit_code == two.exit_code == 0
    assert (with_token / PLAN_FILE).read_bytes() == (without / PLAN_FILE).read_bytes()


def test_the_help_names_the_options_and_says_the_dry_run_is_the_default() -> None:
    from typer.main import get_command

    from rinchen.cli.main import app
    from tests.support.eval_cli import invoke

    made = invoke(["release", "hub", "--help"])
    command = get_command(app).commands["release"].commands["hub"]  # type: ignore[attr-defined]
    helps = {param.name: param.help for param in command.params}

    assert made.exit_code == 0
    for option in ("--stage", "--repo-id", "--out-dir", "--upload", "--public"):
        assert option in made.stdout
    assert command.help is not None and "dry run" in command.help
    assert helps["upload_it"] == (
        "Upload for real. Needs the environment variable HF_TOKEN. "
        "Without it nothing is sent: the command is a dry run."
    )
    assert helps["public"] == (
        "Make the repository public when it is created. The default is private."
    )
