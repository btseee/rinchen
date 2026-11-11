"""`rinchen release hub --upload` over a recording client (D13, Task 10, spec T15 and T18): with the
token in `HF_TOKEN` the plan is printed, the repository is created private (public only with
`--public`), ONE commit holds the nine files, and `hub_commit.json`, `hub_plan.json` and, last, the
manifest are written. The token is read from the environment in this one path, is passed to the client
and is in no output: not a line, not a file, not a log record, not an error (an upstream error is named
by its class). The client is the seam `open_client`; no test builds a real one that could reach the Hub.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import release_hub
from rinchen.export.hub import HfHubClient, plan_upload
from rinchen.export.release import UPLOAD_NAMES
from rinchen.manifest import store
from tests.support.hub_fakes import (
    COMMIT_OID,
    FakeHfApi,
    RecordingHub,
    TokenInError,
    fake_token,
    install_open_client,
)
from tests.support.hub_stage import (
    COMMIT_FILE,
    MANIFEST,
    PLAN_FILE,
    bytes_under,
    holds,
    invoke_hub,
    release_document,
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


def uploading(
    release: Release,
    out: Path,
    monkeypatch: pytest.MonkeyPatch,
    hub: RecordingHub,
    *extra: str,
    token: str | None = None,
) -> tuple[Any, list[str], str]:
    """Run `--upload` with `hub` as the client; the result, the tokens `open_client` saw and the
    token the environment held."""
    opened = install_open_client(monkeypatch, hub)
    token = fake_token() if token is None else token
    made = invoke_hub(release, out, "--upload", *extra, env={"HF_TOKEN": token})
    return made, opened, token


# ---- the commit ---------------------------------------------------------------------------------------------------


def test_an_upload_creates_a_private_repository_and_makes_one_commit_of_the_nine_files(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = RecordingHub()

    made, opened, token = uploading(release, out, monkeypatch, hub)

    assert made.exit_code == 0, made.output
    assert opened == [token]
    assert [type(call).__name__ for call in hub.calls] == [
        "RepoCall",
        "VisibilityCall",
        "CommitCall",
    ]
    repo, commit = hub.repo_calls[0], hub.commit_calls[0]
    assert (repo.repo_id, repo.private, repo.exist_ok, repo.token) == (
        release.repo_id,
        True,
        True,
        token,
    )
    assert commit.token == token and commit.repo_id == release.repo_id
    assert [name for name, _ in commit.operations] == list(UPLOAD_NAMES)
    assert [path for _, path in commit.operations] == [
        release.stage / name for name in UPLOAD_NAMES
    ]
    assert commit.commit_message == f"Rinchen {release.export.record.name}"


def test_public_is_passed_through_and_is_the_only_way_to_a_public_repository(
    release: Release, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private, public = RecordingHub(), RecordingHub(reports_private=False)

    uploading(release, tmp_path / "a", monkeypatch, private)
    uploading(release, tmp_path / "b", monkeypatch, public, "--public")

    assert private.repo_calls[0].private is True
    assert public.repo_calls[0].private is False
    manifest = store.read(tmp_path / "b")
    assert manifest is not None and manifest.status["visibility"] == "public"


def test_the_token_of_the_environment_is_trimmed_and_passed_on(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = RecordingHub()
    token = fake_token()

    made, opened, _ = uploading(release, out, monkeypatch, hub, token=f"  {token}\n")

    assert made.exit_code == 0, made.output
    assert opened == [token] and hub.repo_calls[0].token == token


# ---- what it prints and writes ----------------------------------------------------------------------------------------


def test_the_plan_is_printed_before_the_upload_and_the_commit_after_it(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made, _, _ = uploading(release, out, monkeypatch, RecordingHub())

    lines = made.stdout.splitlines()
    total = sum((release.stage / name).stat().st_size for name in UPLOAD_NAMES)
    assert lines[:2] == ["mode: upload", f"repo: {release.repo_id}"]
    assert [line.split()[0] for line in lines[2:11]] == list(UPLOAD_NAMES)
    assert lines[11] == f"total: {total}"
    assert lines[12] == f"commit: {COMMIT_OID}"
    assert lines[13:] == [
        f"plan: {out / PLAN_FILE}",
        f"commit record: {out / COMMIT_FILE}",
        f"manifest: {out / MANIFEST}",
    ]


def test_the_commit_record_holds_the_repository_the_commit_and_the_release_digest(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploading(release, out, monkeypatch, RecordingHub())

    plan = plan_upload(release.stage, release.repo_id)
    data = (out / COMMIT_FILE).read_bytes()
    text = data.decode("utf-8")
    assert json.loads(text) == {
        "schema": "rinchen.hub_commit/1",
        "repo_id": release.repo_id,
        "commit_oid": COMMIT_OID,
        "release_sha256": plan.release_sha256,
        "visibility": "private",
    }
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"
    assert (out / PLAN_FILE).read_bytes() == plan.to_bytes()


def test_the_manifest_says_upload_and_lists_the_plan_and_the_commit_record(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploading(release, out, monkeypatch, RecordingHub())

    manifest = store.read(out)
    assert manifest is not None
    assert manifest.stage == "release_hub"
    assert manifest.status == {
        "mode": "upload",
        "repo_id": release.repo_id,
        "visibility": "private",
        "commit_oid": COMMIT_OID,
    }
    assert [Path(entry.path).name for entry in manifest.outputs] == [
        PLAN_FILE,
        COMMIT_FILE,
    ]
    for entry in manifest.outputs:
        assert entry.sha256 == sha256_of(Path(entry.path).read_bytes())
    assert [Path(entry.path).name for entry in manifest.inputs] == [
        MANIFEST,
        *UPLOAD_NAMES,
    ]
    assert manifest.tool_versions.get("huggingface_hub") not in (None, "not installed")


def test_the_manifest_is_written_last(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renamed: list[str] = []
    real = Path.replace

    def recording(self: Path, target: Any) -> Any:
        renamed.append(Path(target).name)
        return real(self, target)

    monkeypatch.setattr(Path, "replace", recording)

    uploading(release, out, monkeypatch, RecordingHub())

    assert renamed[-1] == MANIFEST
    assert set(renamed) == {PLAN_FILE, COMMIT_FILE, MANIFEST}


def test_the_out_directory_holds_the_plan_the_commit_record_and_the_manifest_only(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploading(release, out, monkeypatch, RecordingHub())

    assert sorted(path.name for path in out.iterdir()) == sorted(
        [COMMIT_FILE, MANIFEST, PLAN_FILE]
    )


def test_an_upload_into_the_directory_of_an_earlier_one_replaces_its_commit_record(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploading(release, out, monkeypatch, RecordingHub(oid="1" * 40))

    uploading(release, out, monkeypatch, RecordingHub(oid="2" * 40))

    assert json.loads((out / COMMIT_FILE).read_text("utf-8"))["commit_oid"] == "2" * 40


def test_the_stage_is_not_touched_by_an_upload(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = written(release.stage)

    uploading(release, out, monkeypatch, RecordingHub())

    assert written(release.stage) == before
    assert release_document(release.stage)["repo_id"] == release.repo_id


# ---- the token is in no output -------------------------------------------------------------------------------------------------


def test_the_token_is_in_no_line_no_file_and_no_log_record(
    release: Release,
    out: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    made, _, token = uploading(release, out, monkeypatch, RecordingHub())

    assert made.exit_code == 0, made.output
    assert not holds(token, [made.stdout, made.stderr, caplog.text, *bytes_under(out)])
    assert not holds(token, bytes_under(release.stage))


def test_an_error_whose_text_holds_the_token_is_exit_4_naming_the_class_only(
    release: Release,
    out: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    token = fake_token()
    hub = RecordingHub(commit_error=TokenInError(f"401 Unauthorized: {token}"))

    made, _, _ = uploading(release, out, monkeypatch, hub, token=token)

    assert made.exit_code == 4, made.output
    assert made.stderr == "error: the Hub refused the upload (TokenInError)\n"
    assert not holds(token, [made.stdout, made.stderr, caplog.text])
    assert not out.exists()
    assert not holds(token, bytes_under(release.root))


def test_a_repository_that_cannot_be_created_is_exit_4_and_no_commit_is_tried(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = fake_token()
    hub = RecordingHub(repo_error=TokenInError(f"{token} may not create"))

    made, _, _ = uploading(release, out, monkeypatch, hub, token=token)

    assert made.exit_code == 4, made.output
    assert made.stderr == "error: the Hub refused the repository (TokenInError)\n"
    assert hub.commit_calls == []
    assert not holds(token, [made.stdout, made.stderr])
    assert not out.exists()


def test_a_failed_upload_leaves_the_files_of_an_earlier_run_as_they_were(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke_hub(release, out).exit_code == 0
    before = written(out)
    hub = RecordingHub(commit_error=TokenInError("nope"))

    made, _, _ = uploading(release, out, monkeypatch, hub)

    assert made.exit_code == 4
    assert written(out) == before


def test_an_interrupt_during_the_upload_is_130_with_no_commit_record_and_no_manifest(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = RecordingHub(commit_error=KeyboardInterrupt())

    made, _, _ = uploading(release, out, monkeypatch, hub)

    assert made.exit_code == 130, made.output
    assert not out.exists()


def test_an_interrupt_leaves_an_earlier_dry_run_as_it_was(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke_hub(release, out).exit_code == 0
    before = written(out)

    made, _, _ = uploading(
        release, out, monkeypatch, RecordingHub(commit_error=KeyboardInterrupt())
    )

    assert made.exit_code == 130
    assert written(out) == before


def test_a_stage_that_changes_while_the_plan_is_printed_is_refused_before_the_client_is_asked(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = RecordingHub()
    real = release_hub.plan_upload

    def changing(stage: Path, repo_id: str) -> Any:
        plan = real(stage, repo_id)
        (stage / "model.bin").write_bytes(b"changed after the plan")
        return plan

    monkeypatch.setattr(release_hub, "plan_upload", changing)

    made, _, _ = uploading(release, out, monkeypatch, hub)

    assert made.exit_code == 2, made.output
    assert "model.bin" in made.stderr
    assert hub.calls == []
    assert not out.exists()


# ---- the seam -----------------------------------------------------------------------------------------------------------------------


def test_the_default_client_is_the_adapter_built_with_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    api = FakeHfApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", api)

    client = release_hub.open_client("tok")

    assert isinstance(client, HfHubClient)
    assert api.built_with == ["tok"] and api.calls == []


def test_an_upload_into_a_directory_of_another_stage_is_refused_before_the_client_is_asked(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    hub = RecordingHub()
    out.mkdir()
    shutil.copyfile(release.report_dir / MANIFEST, out / MANIFEST)
    before = written(out)

    made, opened, _ = uploading(release, out, monkeypatch, hub)

    assert made.exit_code == 3, made.output
    assert hub.calls == [] and opened == []
    assert written(out) == before


def test_an_upload_into_the_stage_itself_is_refused_before_the_client_is_asked(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = RecordingHub()

    made, opened, _ = uploading(release, release.stage, monkeypatch, hub)

    assert made.exit_code == 2, made.output
    assert hub.calls == [] and opened == []
