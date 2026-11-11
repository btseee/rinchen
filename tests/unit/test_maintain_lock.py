"""The lock reader of `rinchen.maintain.lock` (D14, spec M5): `uv.lock` read with `tomllib` into the
table of names and versions the environment check and the drift table stand on.

Three things are pinned: the table of a committed sample lock (a duplicate name, a name that is not
in normal form, a version with a local label, the editable project), every refusal (a `DataError`
that names the field and never quotes the bytes it was given), and the invariants of the REAL lock
of the repository, read against an independent line-based count (never a literal: the first
upgrade must not break a test).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import DataError
from rinchen.maintain.lock import LockFile, normalize_name, read_lock
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SAMPLE = REPO_ROOT / "tests" / "fixtures" / "maintain" / "uv.lock.sample"
REAL_LOCK = REPO_ROOT / "uv.lock"
CANARY = "SECRETCANARY"


def lock_text(*entries: str, head: str = "version = 1\nrevision = 5\n") -> bytes:
    """A lock with the given `[[package]]` bodies."""
    body = "".join(f"\n[[package]]\n{entry}\n" for entry in entries)
    return (head + body).encode()


def entry(name: str, version: str) -> str:
    return f'name = "{name}"\nversion = "{version}"'


# --- normalize_name ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "normal"),
    [
        ("alpha", "alpha"),
        ("Beta_Pkg", "beta-pkg"),
        ("Huggingface_Hub", "huggingface-hub"),
        ("zope.interface", "zope-interface"),
        ("a--b", "a-b"),
        ("a-_.b", "a-b"),
        ("A__B..C", "a-b-c"),
        ("NVIDIA_cudnn.cu13", "nvidia-cudnn-cu13"),
        ("x", "x"),
        ("", ""),
    ],
)
def test_names_are_put_in_the_normal_form_of_pep_503(name: str, normal: str) -> None:
    assert normalize_name(name) == normal
    assert normalize_name(normal) == normal


# --- the sample lock ----------------------------------------------------------------------------


def test_the_sample_lock_gives_the_table_of_its_six_entries() -> None:
    lock = read_lock(SAMPLE.read_bytes())

    assert isinstance(lock, LockFile)
    assert lock.format_version == 1
    assert lock.revision == 5
    assert lock.entries == 6
    assert sorted(lock.versions) == ["alpha", "beta-pkg", "delta", "gamma", "rinchen"]
    assert len(lock.versions) == 5
    assert lock.versions["alpha"] == ("1.2.3",)
    assert lock.versions["beta-pkg"] == ("2.0",)
    assert lock.versions["gamma"] == ("2.14.1+cu130",)
    assert lock.versions["delta"] == ("1.0",)
    assert lock.versions["rinchen"] == ("0.0.0",)


def test_a_dependency_line_of_an_entry_is_not_an_entry() -> None:
    # The sample's dependencies and `requires-dist` hold `{ name = "alpha" }` lines too.
    lock = read_lock(SAMPLE.read_bytes())
    assert "alpha" in lock.versions
    assert lock.entries == 6


def test_the_lock_is_frozen_and_its_table_read_only() -> None:
    lock = read_lock(SAMPLE.read_bytes())
    with pytest.raises(dataclasses.FrozenInstanceError):
        lock.entries = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        lock.versions["alpha"] = ("9.9",)  # type: ignore[index]
    assert not hasattr(lock, "__dict__")


def test_an_entry_of_one_name_with_two_versions_gives_both_sorted() -> None:
    lock = read_lock(lock_text(entry("pkg", "2.0"), entry("pkg", "1.0")))
    assert lock.entries == 2
    assert lock.versions == {"pkg": ("1.0", "2.0")}


def test_versions_are_sorted_as_strings_and_never_as_numbers() -> None:
    lock = read_lock(lock_text(entry("pkg", "9.0"), entry("pkg", "10.0")))
    assert lock.versions["pkg"] == ("10.0", "9.0")


def test_many_versions_of_one_name_come_out_in_one_sorted_order() -> None:
    versions = ["9.0", "1.5", "10.0", "2.0", "1.10", "3.1", "0.9", "1.0+cu130"]
    lock = read_lock(lock_text(*(entry("pkg", version) for version in versions)))
    assert lock.versions["pkg"] == tuple(sorted(versions))
    assert lock.versions["pkg"][0] == "0.9"


def test_a_local_label_is_kept_and_never_stripped() -> None:
    lock = read_lock(
        lock_text(entry("torch", "2.14.1+cu130"), entry("torch", "2.14.1"))
    )
    assert lock.versions["torch"] == ("2.14.1", "2.14.1+cu130")


def test_a_version_is_the_text_of_the_lock_and_is_not_parsed() -> None:
    odd = ("1.0.0", "01.0", "1.0.post1", "1!2.0", "2.0rc1", "1.0+local.7")
    lock = read_lock(lock_text(*(entry("pkg", version) for version in odd)))
    assert lock.versions["pkg"] == tuple(sorted(odd))


def test_names_are_normalised_and_their_versions_merged() -> None:
    lock = read_lock(lock_text(entry("Foo_Bar", "1.0"), entry("foo.bar", "2.0")))
    assert list(lock.versions) == ["foo-bar"]
    assert lock.versions["foo-bar"] == ("1.0", "2.0")
    assert lock.entries == 2


def test_the_names_of_the_table_are_in_sorted_order() -> None:
    lock = read_lock(
        lock_text(entry("zeta", "1"), entry("Alpha", "1"), entry("m", "1"))
    )
    assert list(lock.versions) == ["alpha", "m", "zeta"]


def test_other_keys_of_the_lock_and_of_its_entries_are_ignored() -> None:
    data = (
        b'version = 1\nrevision = 3\nrequires-python = ">=3.14"\nfuture-key = "x"\n'
        b'\n[[package]]\nname = "pkg"\nversion = "1.0"\nfuture = { a = 1 }\n'
        b'\n[package.metadata]\nrequires-dist = [{ name = "other" }]\n'
    )
    lock = read_lock(data)
    assert lock.revision == 3
    assert lock.entries == 1
    assert lock.versions == {"pkg": ("1.0",)}


def test_a_lock_without_a_revision_has_none() -> None:
    lock = read_lock(lock_text(entry("pkg", "1.0"), head="version = 1\n"))
    assert lock.revision is None


def test_a_lock_with_an_empty_package_list_has_no_entries() -> None:
    lock = read_lock(b"version = 1\npackage = []\n")
    assert lock.entries == 0
    assert lock.versions == {}


# --- the refusals -------------------------------------------------------------------------------

REFUSALS = [
    pytest.param(b"", "version", id="empty-file"),
    pytest.param(b"version = 1\n", "package", id="no-package-key"),
    pytest.param(
        b"version = 1\npackage = { a = 1 }\n", "package", id="package-a-table"
    ),
    pytest.param(b'version = 1\npackage = "x"\n', "package", id="package-a-string"),
    pytest.param(b"version = 1\npackage = [1]\n", "package[0]", id="entry-not-a-table"),
    pytest.param(
        lock_text(f'version = "1.0"\nother = "{CANARY}"'),
        "package[0].name",
        id="no-name",
    ),
    pytest.param(
        lock_text(f'name = "{CANARY}"'), "package[0].version", id="no-version"
    ),
    pytest.param(
        lock_text(entry("ok", "1"), 'name = 3\nversion = "1"'),
        "package[1].name",
        id="name-not-a-string",
    ),
    pytest.param(
        lock_text(f'name = "{CANARY}"\nversion = 3'),
        "package[0].version",
        id="version-not-a-string",
    ),
    pytest.param(
        lock_text(f'name = "{CANARY}"\nversion = true'),
        "package[0].version",
        id="version-a-bool",
    ),
    pytest.param(
        lock_text(f'name = ""\nversion = "{CANARY}"'),
        "package[0].name",
        id="empty-name",
    ),
    pytest.param(
        lock_text(f'name = "{CANARY}"\nversion = ""'),
        "package[0].version",
        id="empty-version",
    ),
    pytest.param(b"version = 2\npackage = []\n", "version", id="format-version-2"),
    pytest.param(
        b'version = "1"\npackage = []\n', "version", id="format-version-a-string"
    ),
    pytest.param(
        f'version = "{CANARY}"\npackage = []\n'.encode(),
        "version",
        id="format-version-a-canary",
    ),
    pytest.param(
        b"version = true\npackage = []\n", "version", id="format-version-a-bool"
    ),
    pytest.param(
        b"version = 1.0\npackage = []\n", "version", id="format-version-a-float"
    ),
    pytest.param(b"version = 0\npackage = []\n", "version", id="format-version-0"),
    pytest.param(b"package = []\n", "version", id="no-format-version"),
    pytest.param(
        b'version = 1\nrevision = "5"\npackage = []\n', "revision", id="revision-string"
    ),
    pytest.param(
        b"version = 1\nrevision = true\npackage = []\n", "revision", id="revision-bool"
    ),
    pytest.param(b"\xff", "UTF-8", id="not-utf-8"),
    pytest.param(b"version = 1\n\xff\xfe\n", "UTF-8", id="not-utf-8-late"),
    pytest.param(f"this is = = {CANARY}\n".encode(), "TOML", id="invalid-toml"),
    pytest.param(
        f'version = 1\npackage = [{{ name = "{CANARY}" ]\n'.encode(),
        "TOML",
        id="invalid-toml-inline-table",
    ),
]


@pytest.mark.parametrize(("data", "field"), REFUSALS)
def test_a_lock_that_cannot_be_read_is_a_data_error_naming_the_field(
    data: bytes, field: str
) -> None:
    with pytest.raises(DataError) as caught:
        read_lock(data)
    message = str(caught.value)
    assert field in message
    assert CANARY not in message


# --- the real lock ------------------------------------------------------------------------------


def independent_reading(text: str) -> list[tuple[str, str]]:
    """The (name, version) of each `[[package]]` entry, read from the lines of the text: the line
    after the header is the name and the one after it the version."""
    lines = text.splitlines()
    pairs: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        if line == "[[package]]":
            name = re.fullmatch(r'name = "([^"]+)"', lines[index + 1])
            version = re.fullmatch(r'version = "([^"]+)"', lines[index + 2])
            assert name is not None and version is not None, index
            pairs.append((name.group(1), version.group(1)))
    return pairs


def pep_503(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_the_real_lock_of_the_repository_is_read() -> None:
    data = REAL_LOCK.read_bytes()
    lock = read_lock(data)
    pairs = independent_reading(data.decode("utf-8"))

    assert lock.format_version == 1
    assert lock.entries == data.decode("utf-8").splitlines().count("[[package]]")
    assert lock.entries == len(pairs) > 100
    assert len(lock.versions) == len({pep_503(name) for name, _ in pairs})
    assert "rinchen" in lock.versions


def test_the_real_lock_gives_every_name_its_versions() -> None:
    lock = read_lock(REAL_LOCK.read_bytes())
    expected: dict[str, set[str]] = {}
    for name, version in independent_reading(REAL_LOCK.read_text(encoding="utf-8")):
        expected.setdefault(pep_503(name), set()).add(version)

    assert {name: set(found) for name, found in lock.versions.items()} == expected
    for name, found in lock.versions.items():
        assert list(found) == sorted(found), name
        assert all(isinstance(version, str) and version for version in found), name


def test_the_real_lock_names_are_in_normal_form() -> None:
    lock = read_lock(REAL_LOCK.read_bytes())
    assert all(normalize_name(name) == name for name in lock.versions)


def test_the_lock_is_read_from_the_bytes_and_not_from_a_path(tmp_path: Path) -> None:
    copy = tmp_path / "uv.lock"
    copy.write_bytes(SAMPLE.read_bytes())
    assert read_lock(copy.read_bytes()) == read_lock(SAMPLE.read_bytes())
