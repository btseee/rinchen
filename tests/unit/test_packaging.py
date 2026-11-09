"""The package data and the built wheel (D11, ruling T14).

The language registry lives in `configs/`, which a wheel does not hold; `pyproject.toml` ships
it as package data (`rinchen/_data/languages.yaml`, a hatch `force-include`) and
`rinchen.config.languages` reads the repository copy when it exists and the packaged copy when
it does not. The wheel is built offline from a copy of the project made in a temporary
directory (the real tree is never written); when the build tool cannot run, the tests that
need the wheel are SKIPPED with the build's own message, never passed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.languages import PACKAGED_LANGUAGES_PATH
from rinchen.config.loader import REPO_ROOT

PYPROJECT = REPO_ROOT / "pyproject.toml"
FORCE_INCLUDE = {"configs/languages.yaml": "rinchen/_data/languages.yaml"}
CONSOLE_SCRIPT = "rinchen = rinchen.cli.main:app"
BUILD_TIMEOUT_SECONDS = 300
CHILD_TIMEOUT_SECONDS = 120
# The tail of the build's output that a skip says.
MESSAGE_LINES = 6


def pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


# --- pyproject.toml -------------------------------------------------------------------------


def test_the_wheel_ships_the_language_registry_as_package_data() -> None:
    wheel = pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert wheel["force-include"] == FORCE_INCLUDE


def test_every_force_included_source_is_a_file_of_the_repository() -> None:
    for source in FORCE_INCLUDE:
        assert (REPO_ROOT / source).is_file(), source


def test_the_published_path_of_the_registry_is_the_one_the_package_reads() -> None:
    (published,) = FORCE_INCLUDE.values()
    site = PACKAGED_LANGUAGES_PATH.parents[2]

    assert PACKAGED_LANGUAGES_PATH.relative_to(site).as_posix() == published


def test_the_package_and_the_console_script_are_declared_as_before() -> None:
    document = pyproject()

    assert document["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/rinchen"
    ]
    assert document["project"]["scripts"] == {"rinchen": "rinchen.cli.main:app"}


# --- the wheel ------------------------------------------------------------------------------


def _build_message(completed: subprocess.CompletedProcess[str]) -> str:
    tail = (completed.stderr or completed.stdout).strip().splitlines()[-MESSAGE_LINES:]
    return " | ".join(tail)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The wheel `uv build` makes from a copy of the project, offline (hatchling must be in
    uv's cache). Skipped, with the reason, when it cannot be built."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH: the wheel cannot be built")
    project = tmp_path_factory.mktemp("project")
    shutil.copy(PYPROJECT, project / "pyproject.toml")
    shutil.copytree(
        REPO_ROOT / "src",
        project / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for source in FORCE_INCLUDE:
        (project / source).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / source, project / source)
    out_dir = tmp_path_factory.mktemp("dist")
    # No network, ever: a build that needs it is skipped, not run online.
    environment = {**os.environ, "UV_OFFLINE": "1"}
    try:
        completed = subprocess.run(
            [
                uv,
                "build",
                "--wheel",
                "--no-progress",
                "--python",
                sys.executable,
                "--out-dir",
                str(out_dir),
                str(project),
            ],
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SECONDS,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.skip(f"uv build did not finish in {BUILD_TIMEOUT_SECONDS} s")
    if completed.returncode != 0:
        pytest.skip(f"uv build cannot run here: {_build_message(completed)}")
    (built,) = out_dir.glob("*.whl")
    return built


def test_the_wheel_holds_the_language_registry_byte_for_byte(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        packaged = archive.read("rinchen/_data/languages.yaml")

    assert packaged == (REPO_ROOT / "configs" / "languages.yaml").read_bytes()


@pytest.mark.parametrize(
    "member",
    [
        "rinchen/export/record.py",
        "rinchen/infer/engine.py",
        "rinchen/infer/translator.py",
        "rinchen/cli/translate.py",
        "rinchen/cli/translate_io.py",
        "rinchen/config/languages.py",
        "rinchen/py.typed",
    ],
)
def test_the_wheel_holds_the_modules_of_the_command_and_the_engine(
    wheel: Path, member: str
) -> None:
    with zipfile.ZipFile(wheel) as archive:
        assert member in archive.namelist()


def test_the_wheel_has_the_console_script(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        (entry_points,) = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/entry_points.txt")
        ]
        text = archive.read(entry_points).decode()

    assert text.splitlines() == ["[console_scripts]", CONSOLE_SCRIPT]


def test_the_wheel_holds_no_configs_directory(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    assert [name for name in names if "configs" in name] == []


def test_the_registry_of_the_wheel_loads_without_a_checkout(
    wheel: Path, tmp_path: Path
) -> None:
    """The wheel's own modules, ahead of everything on the path and with no `configs/`
    beside them, find the packaged registry and read it."""
    unpacked = tmp_path / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(unpacked)
    script = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from rinchen.config import languages\n"
        "assert languages.__file__.startswith(sys.argv[1]), languages.__file__\n"
        "assert languages.default_languages_path() == languages.PACKAGED_LANGUAGES_PATH\n"
        "print(' '.join(entry.code for entry in languages.load_languages()))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(unpacked)],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    codes = completed.stdout.split()
    assert codes[:3] == ["khk_Cyrl", "mn_Mong", "eng_Latn"]
    assert len(codes) == 10
