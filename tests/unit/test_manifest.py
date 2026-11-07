"""Tests for the manifest model, store, and hash-based stage skip mechanism."""

import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rinchen.errors import DataError
from rinchen.manifest import stage, store
from rinchen.manifest.model import Manifest, ManifestFile


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


def _demo_manifest(**fields):
    return Manifest(
        stage="demo",
        created="2026-11-06T00:00:00Z",
        code_version="deadbeef",
        config_sha256="cfg-abc",
        seed=0,
        **fields,
    )


def test_write_replaces_the_manifest_whole_and_leaves_no_part_file(tmp_path):
    store.write(tmp_path, _demo_manifest(metrics={"rows": 1.0}))

    store.write(tmp_path, _demo_manifest(metrics={"rows": 2.0}))

    loaded = store.read(tmp_path)
    assert loaded is not None
    assert loaded.metrics == {"rows": 2.0}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.json"]


def test_a_failed_write_leaves_the_old_manifest_and_no_part_file(tmp_path, monkeypatch):
    store.write(tmp_path, _demo_manifest(metrics={"rows": 1.0}))
    before = store.manifest_path(tmp_path).read_bytes()

    def refuse_to_rename(self, target):
        raise OSError("rename refused")

    monkeypatch.setattr(Path, "replace", refuse_to_rename)
    with pytest.raises(OSError, match="rename refused"):
        store.write(tmp_path, _demo_manifest(metrics={"rows": 2.0}))

    assert store.manifest_path(tmp_path).read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.json"]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"schema": "rinchen.manifest/1", "stage": "dem', id="truncated"),
        pytest.param('{"stage": "demo"}', id="not-a-manifest"),
        pytest.param("", id="empty"),
    ],
)
def test_reading_a_broken_manifest_is_a_data_error_naming_the_file(tmp_path, text):
    (tmp_path / "manifest.json").write_text(text, encoding="utf-8")

    with pytest.raises(DataError, match="manifest.json") as raised:
        store.read(tmp_path)

    assert raised.value.__cause__ is not None


def test_reading_a_manifest_that_is_not_utf8_is_a_data_error(tmp_path):
    (tmp_path / "manifest.json").write_bytes(b"\xff\xfe not utf-8")

    with pytest.raises(DataError, match="manifest.json"):
        store.read(tmp_path)


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
def test_importing_the_cli_raises_no_user_warning():
    # The CLI imports the manifest model; a pydantic warning at class creation
    # would print on every `rinchen` command.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-W",
            "error:Field name:UserWarning",
            "-c",
            "import rinchen.cli.main",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.subprocess
def test_code_version_in_this_repository_is_a_short_sha_with_optional_dirty():
    assert re.fullmatch(r"[0-9a-f]{7}(-dirty)?", stage.code_version())


def _git(repo: Path, *args: str) -> str:
    # A GIT_DIR or GIT_INDEX_FILE inherited from a git hook would point git
    # at the outer repository instead of `repo`.
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def no_inherited_git_variables(monkeypatch) -> None:
    """`stage.code_version` runs git with this process's environment."""
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)


@pytest.fixture
def committed_repo(tmp_path, monkeypatch, no_inherited_git_variables) -> Path:
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
def test_code_version_is_unknown_outside_a_repository(
    tmp_path, monkeypatch, no_inherited_git_variables
):
    # Without a ceiling, git would find a repository above the temporary directory.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setattr(stage, "REPO_ROOT", tmp_path)

    assert stage.code_version() == "unknown"


@pytest.mark.subprocess
def test_code_version_is_unknown_for_a_repository_without_commits(
    tmp_path, monkeypatch, no_inherited_git_variables
):
    _git(tmp_path, "init", "--quiet")
    monkeypatch.setattr(stage, "REPO_ROOT", tmp_path)

    assert stage.code_version() == "unknown"


class RecordingLogger:
    """Stands in for the structlog logger, which would bind the global
    logging setup to the stream of whichever test calls it first."""

    def __init__(self):
        self.warnings = []

    def warning(self, event, **fields):
        self.warnings.append((event, fields))


@pytest.mark.parametrize(
    ("failure", "error_type", "git_stderr"),
    [
        pytest.param(
            FileNotFoundError(2, "No such file or directory", "git"),
            "FileNotFoundError",
            None,
            id="git-not-installed",
        ),
        pytest.param(
            subprocess.TimeoutExpired("git", 30),
            "TimeoutExpired",
            None,
            id="git-timed-out",
        ),
        pytest.param(
            subprocess.CalledProcessError(
                128, "git", stderr="fatal: not a git repository\n"
            ),
            "CalledProcessError",
            "fatal: not a git repository",
            id="git-exited-with-an-error",
        ),
    ],
)
def test_code_version_is_unknown_and_warns_once_when_git_fails(
    monkeypatch, failure, error_type, git_stderr
):
    logger = RecordingLogger()
    monkeypatch.setattr(stage, "get_logger", lambda name: logger)

    def failing_git(*args, **kwargs):
        raise failure

    monkeypatch.setattr(stage.subprocess, "run", failing_git)

    assert stage.code_version() == "unknown"

    [(event, fields)] = logger.warnings
    assert event == "code_version_unknown"
    assert fields["error"].startswith(f"{error_type}: ")
    assert fields.get("git_stderr") == git_stderr


def test_tool_versions_reports_installed_and_missing_packages():
    versions = stage.tool_versions(["polars", "no-such-package-xyz"])

    assert versions == {
        "polars": importlib.metadata.version("polars"),
        "no-such-package-xyz": "not installed",
    }


def test_tool_versions_of_nothing_is_empty():
    assert stage.tool_versions([]) == {}


def test_output_file_records_path_hash_size_and_rows(tmp_path):
    written = tmp_path / "out.bin"
    written.write_bytes(b"abcdef")

    entry = stage.output_file(written, 3)

    assert entry == ManifestFile(
        path=str(written), sha256=stage.sha256_of(written), bytes=6, rows=3
    )


def test_output_file_without_a_row_count_leaves_rows_unset(tmp_path):
    written = tmp_path / "out.bin"
    written.write_bytes(b"")

    entry = stage.output_file(written, None)

    assert (entry.bytes, entry.rows) == (0, None)
