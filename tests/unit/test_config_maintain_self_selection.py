"""The guard against a rerun that selects a test of the maintenance check itself (D14 fix wave 1,
finding M3).

A rerun of kind `pytest` must not rerun the gate of the command that makes it. The guard is by
NAME: the loader refuses a target whose spelling, after NFKC normalisation and case folding, holds
`maintain` in any component of its path, and a short closed list of directories that hold
maintenance tests among others (a directory's name says nothing of the files under it). It does
not look inside a file, expand a directory, follow a link or see what a test imports: a test that
drives the maintenance check without `maintain` in its path is not seen. The last tests resolve
each shipped target against the repository and pin the two things the name cannot show: no file a
shipped target selects has `maintain` (or `fake_uv`) in its path, and every directory that holds a
maintenance test is one the guard refuses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.config.maintain import (
    DIRECTORIES_WITH_MAINTENANCE_TESTS,
    load_maintain_config,
)
from rinchen.errors import ConfigError
from tests.support.maintain_check import SHIPPED_CONFIG, shipped_config
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

FULLWIDTH_MAINTAIN = "".join(chr(0xFF00 + ord(letter) - 0x20) for letter in "maintain")
FULLWIDTH_CAPITALS = FULLWIDTH_MAINTAIN.upper()
TESTS = REPO_ROOT / "tests"


def document_with_targets(targets: list[str]) -> bytes:
    document: dict[str, Any] = yaml.safe_load(
        SHIPPED_CONFIG.read_text(encoding="utf-8")
    )
    document["reruns"][0]["targets"] = targets
    return yaml.safe_dump(document).encode("utf-8")


def refusal_of(target: str) -> str:
    with pytest.raises(ConfigError) as caught:
        load_maintain_config(document_with_targets([target]))
    return str(caught.value)


def accepted(target: str) -> bool:
    load_maintain_config(document_with_targets([target]))
    return True


def maintenance_test_files() -> list[Path]:
    """The collected test files of the repository (`test_*.py`) whose path names the maintenance
    check or its `uv` shim."""
    return sorted(
        path
        for path in TESTS.rglob("test_*.py")
        if any(
            word in path.relative_to(REPO_ROOT).as_posix().casefold()
            for word in ("maintain", "fake_uv")
        )
    )


# ---- the spelling ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "tests/unit/test_maintain_x.py",
        "tests/unit/test_MAINTAIN_x.py",
        "tests/unit/test_Maintain_x.py",
        f"tests/unit/test_{FULLWIDTH_MAINTAIN}_x.py",
        f"tests/unit/test_{FULLWIDTH_CAPITALS}_x.py",
        f"tests/unit/test_Main{FULLWIDTH_MAINTAIN[4:]}.py",
        "tests/maintain_stuff/test_x.py",
        f"tests/{FULLWIDTH_MAINTAIN}/test_x.py",
        "tests/golden/maintain/test_x.py",
        "tests/unit/maintain/test_x.py",
        "tests/unit/test_x.py::test_maintain_it",
        f"tests/unit/test_x.py::test_{FULLWIDTH_MAINTAIN}_it",
        "tests/unit/test_x.py::TestMaintain::test_it",
    ],
    ids=lambda target: target.encode("ascii", "backslashreplace").decode(),
)
def test_a_target_that_names_the_maintenance_check_in_any_spelling_is_refused(
    target: str,
) -> None:
    message = refusal_of(target)

    assert "targets" in message
    assert "maintenance check" in message


@pytest.mark.parametrize(
    "target",
    [
        "tests/unit/test_main.py",
        "tests/unit/test_maint.py",
        "tests/unit/test_mainta1n.py",
        "tests/unit/test_contain.py",
        "tests/golden",
        "tests/unit/test_x.py",
    ],
)
def test_a_target_without_the_name_is_not_refused_by_the_guard(target: str) -> None:
    assert accepted(target)


# ---- the directories that hold maintenance tests ------------------------------------------------------


def test_the_closed_list_of_directories_is_the_three_that_hold_maintenance_tests() -> (
    None
):
    assert DIRECTORIES_WITH_MAINTENANCE_TESTS == (
        "tests/unit",
        "tests/integration",
        "tests/property",
    )


@pytest.mark.parametrize("target", DIRECTORIES_WITH_MAINTENANCE_TESTS)
def test_a_directory_that_holds_maintenance_tests_is_refused_and_the_message_says_so(
    target: str,
) -> None:
    message = refusal_of(target)

    assert "targets" in message
    assert "holds tests of the maintenance check" in message


@pytest.mark.parametrize(
    "target",
    [
        "tests/unit/sub",
        "tests/integration/sub",
        "tests/property/sub",
        "tests/golden",
        "tests/unit/test_x.py",
        "tests/unitx",
        "tests/unit2",
    ],
)
def test_a_narrower_or_another_path_is_not_in_the_closed_list(target: str) -> None:
    assert accepted(target)


# ---- the repository ----------------------------------------------------------------------------------


def test_the_scan_of_maintenance_test_files_is_not_empty() -> None:
    names = {path.name for path in maintenance_test_files()}

    assert "test_maintain_gate.py" in names
    assert "test_fake_uv.py" in names
    assert "test_cli_maintain.py" in names
    assert "test_maintain_lock_property.py" in names


def test_every_directory_that_holds_a_maintenance_test_is_refused_as_a_target() -> None:
    directories = {
        parent.relative_to(REPO_ROOT).as_posix()
        for path in maintenance_test_files()
        for parent in path.parents
        if parent.is_relative_to(TESTS)
    }

    assert {"tests/unit", "tests/integration", "tests/property"} <= directories
    for directory in sorted(directories):
        with pytest.raises(ConfigError):
            load_maintain_config(document_with_targets([directory]))


def test_no_file_that_a_shipped_target_selects_names_the_maintenance_check() -> None:
    selected = 0
    for rerun in shipped_config().reruns:
        for target in rerun.targets:
            path = REPO_ROOT / target.split("::")[0]
            assert path.exists(), target
            files = [path] if path.is_file() else sorted(path.rglob("*.py"))
            for file in files:
                relative = file.relative_to(REPO_ROOT).as_posix().casefold()
                assert "maintain" not in relative, relative
                assert "fake_uv" not in relative, relative
                selected += 1
    assert selected >= 20
