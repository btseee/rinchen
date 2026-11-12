"""No file of a directory that holds fixtures or configuration is on the disk and ignored by git.

A file that exists in the working tree and matches a rule of `.gitignore` passes every local run and is
missing from a clean checkout. The first CI run of the tiny pipeline failed at collection for that reason:
the recording `tests/fixtures/tiny_run/conv_eval.log` matched `*.log` and had never been committed, so the
tests that replay it could not read it on the runner. This module lists the ignored files under the
directories that tests and commands read data from and expects none, and shows on a throwaway repository
that the listing does see an ignored file and that a negation rule of `.gitignore` clears it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest

from rinchen.config.loader import REPO_ROOT

# The directories of the repository that hold data a test or a command reads. A cache or a build product
# is ignored on purpose and lives elsewhere (`__pycache__` is skipped below).
DATA_DIRECTORIES: Final = (
    "tests/fixtures",
    "tests/golden",
    "configs",
    "scripts",
    "docker",
    ".github",
)
_CACHE: Final = "__pycache__"
_GIT_TIMEOUT_SECONDS: Final = 60


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run `git` in `root` without the variables a hook may have set (`GIT_DIR`, `GIT_INDEX_FILE`)."""
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=_GIT_TIMEOUT_SECONDS,
    )


def ignored_files(root: Path, directories: Sequence[str]) -> list[str]:
    """The files under `directories` of the work tree `root` that are on the disk, not tracked and ignored."""
    listed = _git(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        *directories,
    )
    assert listed.returncode == 0, listed.stderr
    return sorted(
        line
        for line in listed.stdout.splitlines()
        if line and _CACHE not in Path(line).parts
    )


def _work_tree(root: Path) -> bool:
    if shutil.which("git") is None:
        return False
    return _git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() == "true"


@pytest.fixture
def scratch_repository(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    assert _git(tmp_path, "init", "-q").returncode == 0
    (tmp_path / "tests" / "fixtures" / "run").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "run" / "recorded.log").write_text(
        "one line\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "fixtures" / "run" / "kept.json").write_text(
        "{}\n", encoding="utf-8"
    )
    return tmp_path


def test_a_fixture_that_a_rule_of_gitignore_hides_is_listed(
    scratch_repository: Path,
) -> None:
    (scratch_repository / ".gitignore").write_text("*.log\n", encoding="utf-8")

    assert ignored_files(scratch_repository, DATA_DIRECTORIES) == [
        "tests/fixtures/run/recorded.log"
    ]


def test_a_negation_rule_of_gitignore_clears_the_listing(
    scratch_repository: Path,
) -> None:
    (scratch_repository / ".gitignore").write_text(
        "*.log\n!tests/fixtures/**/*.log\n", encoding="utf-8"
    )

    assert ignored_files(scratch_repository, DATA_DIRECTORIES) == []


def test_a_cache_directory_is_not_a_fixture(scratch_repository: Path) -> None:
    cache = scratch_repository / "configs" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "x.pyc").write_bytes(b"\x00")
    (scratch_repository / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")

    assert ignored_files(scratch_repository, DATA_DIRECTORIES) == []


def test_every_file_of_the_data_directories_that_is_on_the_disk_is_not_ignored() -> (
    None
):
    if not _work_tree(REPO_ROOT):
        pytest.skip("not a git work tree")

    hidden = ignored_files(REPO_ROOT, DATA_DIRECTORIES)

    assert hidden == [], (
        "ignored by .gitignore but on the disk, so absent from a clean checkout: "
        + ", ".join(hidden)
    )
