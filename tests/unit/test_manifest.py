"""Tests for the manifest model, store, and hash-based stage skip mechanism."""

import importlib.metadata
import re
import subprocess
from pathlib import Path

import pytest

from rinchen.manifest import stage, store
from rinchen.manifest.model import Manifest


def _write_manifest(stage_dir: Path, config_sha256: str, inputs: list[Path]) -> None:
    manifest = Manifest(
        stage="test_stage",
        created="2026-10-30T00:00:00Z",
        code_version="deadbeef",
        config_sha256=config_sha256,
        seed=0,
        inputs=stage.build_inputs(inputs),
    )
    store.write(stage_dir, manifest)


def test_store_round_trips(tmp_path):
    stage_dir = tmp_path / "stage"
    manifest = Manifest(
        stage="demo",
        created="2026-10-30T00:00:00Z",
        code_version="deadbeef",
        config_sha256="cfg-abc",
        seed=0,
    )

    path = store.write(stage_dir, manifest)
    assert path.is_file()

    loaded = store.read(stage_dir)
    assert loaded is not None
    assert loaded.stage == "demo"
    assert loaded.schema == "rinchen.manifest/1"


def test_store_read_returns_none_when_missing(tmp_path):
    assert store.read(tmp_path / "nonexistent") is None


def test_identical_inputs_and_config_are_skipped(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc") is True


def test_changed_config_reruns(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])

    assert stage.should_skip(stage_dir, [input_file], "cfg-different") is False


def test_changed_input_contents_reruns(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])
    input_file.write_text("changed", encoding="utf-8")

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc") is False


def test_no_prior_manifest_never_skips(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc") is False


def test_force_always_reruns(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc", force=True) is False


def test_status_round_trips_through_the_store(tmp_path):
    manifest = Manifest(
        stage="filter",
        created="2026-11-06T00:00:00Z",
        code_version="deadbeef",
        config_sha256="cfg-abc",
        seed=0,
        status={"lid": "ok", "qe": "skipped: no scorer configured"},
    )

    store.write(tmp_path, manifest)
    loaded = store.read(tmp_path)

    assert loaded is not None
    assert loaded.status == {"lid": "ok", "qe": "skipped: no scorer configured"}
    assert loaded.schema == "rinchen.manifest/1"


def test_manifest_without_status_loads_with_an_empty_one(tmp_path):
    (tmp_path / "manifest.json").write_text(
        """{"schema": "rinchen.manifest/1", "stage": "old",
        "created": "2026-10-30T00:00:00Z", "code_version": "deadbeef",
        "config_sha256": "cfg-abc", "seed": 0}""",
        encoding="utf-8",
    )

    loaded = store.read(tmp_path)

    assert loaded is not None
    assert loaded.status == {}


@pytest.mark.subprocess
def test_code_version_in_this_repository_is_a_short_sha_with_optional_dirty():
    assert re.fullmatch(r"[0-9a-f]{7}(-dirty)?", stage.code_version())


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def committed_repo(tmp_path, monkeypatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    (repo / "tracked.txt").write_text("one", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "--quiet", "-m", "first")
    monkeypatch.setattr(stage, "REPO_ROOT", repo)
    return repo


@pytest.mark.subprocess
def test_code_version_of_a_clean_checkout_is_the_first_seven_sha_characters(
    committed_repo,
):
    assert stage.code_version() == _git(committed_repo, "rev-parse", "HEAD")[:7]


@pytest.mark.subprocess
def test_code_version_marks_a_modified_tracked_file_dirty(committed_repo):
    (committed_repo / "tracked.txt").write_text("two", encoding="utf-8")

    assert (
        stage.code_version() == _git(committed_repo, "rev-parse", "HEAD")[:7] + "-dirty"
    )


@pytest.mark.subprocess
def test_code_version_marks_an_untracked_file_dirty(committed_repo):
    (committed_repo / "new.txt").write_text("new", encoding="utf-8")

    assert stage.code_version().endswith("-dirty")


@pytest.mark.subprocess
def test_code_version_is_unknown_outside_a_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "REPO_ROOT", tmp_path)

    assert stage.code_version() == "unknown"


@pytest.mark.subprocess
def test_code_version_is_unknown_for_a_repository_without_commits(
    tmp_path, monkeypatch
):
    _git(tmp_path, "init", "--quiet")
    monkeypatch.setattr(stage, "REPO_ROOT", tmp_path)

    assert stage.code_version() == "unknown"


def test_code_version_is_unknown_when_git_is_not_installed(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(stage.subprocess, "run", missing_git)

    assert stage.code_version() == "unknown"


def test_tool_versions_reports_installed_and_missing_packages():
    versions = stage.tool_versions(["polars", "no-such-package-xyz"])

    assert versions == {
        "polars": importlib.metadata.version("polars"),
        "no-such-package-xyz": "not installed",
    }


def test_tool_versions_of_nothing_is_empty():
    assert stage.tool_versions([]) == {}
