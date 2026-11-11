"""The code version of the manifest of `rinchen maintain check` (D14 fix wave 1, finding M4).

The nightly run writes its output directory (`maintain-out`) inside the checkout, and the code
version of a manifest is the sha of HEAD plus `-dirty` when the tree has changes, untracked files
included. The manifest said `<sha>-dirty` after every night because the command asked git AFTER it
had written its logs. The code version is now taken once, at the start of the command, before the
output directory is made or any log is written, and handed down to the manifest; and the output
directory of the workflow is git-ignored. A tree that is really dirty (a modified tracked file
before the start) is still `-dirty`.

The tests that ask git use a repository of their own in `tmp_path` (`stage.REPO_ROOT` is replaced).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from rinchen.cli import maintain_output
from rinchen.config.loader import REPO_ROOT
from rinchen.manifest import stage
from tests.support.maintain_cli import (
    CODE_VERSION,
    install,
    invoke,
    make_workspace,
    manifest_of,
)
from tests.support.nightly_workflow import MAINTAIN_RUN
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

OUT_DIRECTORY = "maintain-out"


def git(repo: Path, *args: str) -> str:
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    return subprocess.run(
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
    ).stdout.strip()


@pytest.fixture
def committed_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet")
    (repo / "tracked.txt").write_text("one", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "--quiet", "-m", "first")
    monkeypatch.setattr(stage, "REPO_ROOT", repo)
    return repo


def run_into(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    """A nightly run (doubles for the runner, the doctor and the installed versions) whose output
    directory is `<repo>/maintain-out`, as the workflow's is; the manifest it wrote."""
    workspace = make_workspace(tmp_path / "workspace")
    install(monkeypatch, constant_code_version=False)
    out = repo / OUT_DIRECTORY

    result = invoke(workspace.args(out_dir=out))

    assert result.exit_code == 0, result.output
    return manifest_of(out)


# ---- the version is the one before the run wrote anything ----------------------------------------------


@pytest.mark.subprocess
def test_a_clean_checkout_gets_its_plain_sha_although_the_run_writes_into_it(
    committed_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = run_into(committed_repo, tmp_path, monkeypatch)

    assert manifest["code_version"] == git(committed_repo, "rev-parse", "HEAD")[:7]
    assert (committed_repo / OUT_DIRECTORY / "logs").is_dir()


@pytest.mark.subprocess
def test_a_tree_with_a_modified_tracked_file_is_still_dirty(
    committed_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (committed_repo / "tracked.txt").write_text("two", encoding="utf-8")

    manifest = run_into(committed_repo, tmp_path, monkeypatch)

    head = git(committed_repo, "rev-parse", "HEAD")[:7]
    assert manifest["code_version"] == f"{head}-dirty"


@pytest.mark.subprocess
def test_a_tree_with_an_untracked_file_before_the_start_is_still_dirty(
    committed_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (committed_repo / "new.txt").write_text("new", encoding="utf-8")

    manifest = run_into(committed_repo, tmp_path, monkeypatch)

    assert str(manifest["code_version"]).endswith("-dirty")


def test_the_version_is_taken_once_before_the_output_directory_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_workspace(tmp_path)
    install(monkeypatch, constant_code_version=False)
    seen: list[bool] = []

    def spy() -> str:
        seen.append(workspace.out.exists())
        return CODE_VERSION

    monkeypatch.setattr(maintain_output, "code_version", spy)

    result = invoke(workspace.args())

    assert result.exit_code == 0, result.output
    assert seen == [False]
    assert manifest_of(workspace.out)["code_version"] == CODE_VERSION


# ---- the output directory of the workflow is ignored -----------------------------------------------------


def test_the_workflow_writes_into_the_directory_that_is_ignored() -> None:
    assert f"--out-dir {OUT_DIRECTORY} " in MAINTAIN_RUN


@pytest.mark.subprocess
def test_git_ignores_the_output_directory_of_the_workflow() -> None:
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "check-ignore",
            "--quiet",
            "--no-index",
            f"{OUT_DIRECTORY}/logs/lock_consistent.log",
        ],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )

    assert ignored.returncode == 0


def test_the_ignore_rule_is_anchored_to_the_root() -> None:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert f"/{OUT_DIRECTORY}/" in lines
