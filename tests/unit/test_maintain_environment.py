"""The environment check of `rinchen.maintain.environment` (D14, spec M5): the installed
distributions against the names and versions of the lock.

`fail` is an installed package the lock does not name (an exact sync removes it) or an installed
version the lock does not list for that name; a locked package that is not installed is only
COUNTED, because a lock holds packages for other platforms. Versions are strings compared for
equality, never numbers, and names are in PEP 503 form on both sides.
"""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import pytest

from rinchen.maintain.environment import (
    EnvironmentResult,
    Mismatch,
    check_environment,
    installed_versions,
)
from rinchen.maintain.lock import LockFile
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


def lock_of(table: Mapping[str, Sequence[str]]) -> LockFile:
    versions = {name: tuple(sorted(found)) for name, found in table.items()}
    return LockFile(
        format_version=1,
        revision=5,
        entries=sum(len(found) for found in versions.values()),
        versions=MappingProxyType(versions),
    )


# --- check_environment --------------------------------------------------------------------------


def test_equal_tables_are_ok() -> None:
    result = check_environment(
        lock_of({"alpha": ["1.0"], "beta": ["2.0"]}),
        {"alpha": ["1.0"], "beta": ["2.0"]},
    )
    assert result == EnvironmentResult(
        state="ok", installed=2, mismatched=(), extra=(), not_installed=0
    )


def test_an_empty_environment_against_an_empty_lock_is_ok() -> None:
    result = check_environment(lock_of({}), {})
    assert result == EnvironmentResult(
        state="ok", installed=0, mismatched=(), extra=(), not_installed=0
    )


def test_one_version_different_is_a_mismatch_with_both_versions() -> None:
    result = check_environment(
        lock_of({"alpha": ["1.0"], "beta": ["2.0"]}),
        {"alpha": ["1.0"], "beta": ["2.1"]},
    )
    assert result.state == "fail"
    assert result.mismatched == (Mismatch("beta", ("2.1",), ("2.0",)),)
    assert result.extra == ()
    assert result.not_installed == 0


def test_a_mismatch_holds_the_name_the_installed_and_the_locked_versions() -> None:
    mismatch = Mismatch(name="beta", installed=("2.1",), locked=("2.0",))
    assert (mismatch.name, mismatch.installed, mismatch.locked) == (
        "beta",
        ("2.1",),
        ("2.0",),
    )


def test_an_installed_name_the_lock_does_not_have_is_extra() -> None:
    result = check_environment(
        lock_of({"alpha": ["1.0"]}), {"alpha": ["1.0"], "zed": ["3"]}
    )
    assert result.state == "fail"
    assert result.extra == ("zed",)
    assert result.mismatched == ()
    assert result.installed == 2
    assert result.not_installed == 0


def test_an_extra_name_is_not_a_locked_name_that_was_found() -> None:
    result = check_environment(
        lock_of({"a": ["1"], "b": ["1"], "c": ["1"]}), {"a": ["1"], "z": ["1"]}
    )
    assert result.extra == ("z",)
    assert result.installed == 2
    assert result.not_installed == 2


def test_a_locked_name_that_is_not_installed_is_only_counted() -> None:
    result = check_environment(
        lock_of({"alpha": ["1.0"], "beta": ["2.0"], "gamma": ["3.0"]}),
        {"alpha": ["1.0"]},
    )
    assert result.state == "ok"
    assert result.not_installed == 2
    assert result.mismatched == ()
    assert result.extra == ()
    assert result.installed == 1


def test_two_installed_versions_where_the_lock_has_one_is_a_mismatch() -> None:
    result = check_environment(lock_of({"alpha": ["1.0"]}), {"alpha": ["1.0", "1.1"]})
    assert result.state == "fail"
    assert result.mismatched == (Mismatch("alpha", ("1.0", "1.1"), ("1.0",)),)


def test_one_installed_version_among_several_locked_ones_is_ok() -> None:
    # The lock holds one version per platform (a vendored wheel for each).
    result = check_environment(lock_of({"delta": ["1.0", "1.1"]}), {"delta": ["1.1"]})
    assert result.state == "ok"
    assert result.mismatched == ()


def test_versions_are_compared_as_text_and_never_as_numbers() -> None:
    locked = lock_of({"alpha": ["1.0"], "beta": ["2.14.1+cu130"]})
    for different in ("1.00", "01.0", "1.0.0", "1.0+local"):
        result = check_environment(
            locked, {"alpha": [different], "beta": ["2.14.1+cu130"]}
        )
        assert result.state == "fail", different
        assert [m.name for m in result.mismatched] == ["alpha"], different
    stripped = check_environment(locked, {"alpha": ["1.0"], "beta": ["2.14.1"]})
    assert stripped.state == "fail"
    assert [m.name for m in stripped.mismatched] == ["beta"]


def test_names_are_normalised_on_both_sides() -> None:
    result = check_environment(
        lock_of({"foo-bar": ["1.0"], "huggingface-hub": ["2.3.0"]}),
        {"Foo_Bar": ["1.0"], "Huggingface.Hub": ["2.3.0"]},
    )
    assert result == EnvironmentResult(
        state="ok", installed=2, mismatched=(), extra=(), not_installed=0
    )


def test_a_name_that_is_not_normal_in_the_lock_is_matched_all_the_same() -> None:
    result = check_environment(lock_of({"Foo_Bar": ["1.0"]}), {"foo-bar": ["1.0"]})
    assert result.state == "ok"
    assert result.not_installed == 0


def test_installed_names_that_normalise_to_one_are_one_name_with_both_versions() -> (
    None
):
    result = check_environment(
        lock_of({"foo-bar": ["1.0"]}), {"Foo_Bar": ["1.0"], "foo.bar": ["2.0"]}
    )
    assert result.installed == 1
    assert result.mismatched == (Mismatch("foo-bar", ("1.0", "2.0"), ("1.0",)),)


def test_a_name_with_no_installed_version_is_not_installed() -> None:
    result = check_environment(lock_of({"alpha": ["1.0"]}), {"alpha": []})
    assert result.state == "ok"
    assert result.installed == 0
    assert result.not_installed == 1


def test_the_outputs_are_sorted() -> None:
    many = ["10", "9", "1", "7", "3", "8", "2", "5"]
    result = check_environment(
        lock_of(
            {"a": ["9", "1", "5", "10", "3", "8"], "b": ["1"], "c": ["1"], "d": ["1"]}
        ),
        {"d": ["2"], "c": ["2"], "b": ["1"], "z": ["1"], "m": ["1"], "a": many},
    )
    assert [m.name for m in result.mismatched] == ["a", "c", "d"]
    assert result.mismatched[0].installed == tuple(sorted(many))
    assert result.mismatched[0].locked == tuple(sorted(["9", "1", "5", "10", "3", "8"]))
    assert result.extra == ("m", "z")
    assert result.installed == 6
    assert result.not_installed == 0


def test_a_result_is_ok_exactly_when_nothing_is_mismatched_or_extra() -> None:
    lock = lock_of({"a": ["1"], "b": ["1"]})
    cases: list[tuple[dict[str, list[str]], str]] = [
        ({"a": ["1"], "b": ["1"]}, "ok"),
        ({"a": ["1"]}, "ok"),
        ({}, "ok"),
        ({"a": ["2"], "b": ["1"]}, "fail"),
        ({"a": ["1"], "b": ["1"], "c": ["1"]}, "fail"),
        ({"c": ["1"]}, "fail"),
    ]
    for installed, state in cases:
        assert check_environment(lock, installed).state == state, installed


def test_the_result_is_frozen() -> None:
    result = check_environment(lock_of({}), {})
    with pytest.raises(AttributeError):
        result.state = "fail"  # type: ignore[misc]


# --- installed_versions -------------------------------------------------------------------------


def direct_listing() -> dict[str, set[str]]:
    """The installed distributions, listed straight from `importlib.metadata`."""
    listing: dict[str, set[str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name is None:
            continue
        listing.setdefault(re.sub(r"[-_.]+", "-", name).lower(), set()).add(
            distribution.version
        )
    return listing


def test_installed_versions_equals_a_direct_listing_of_the_distributions() -> None:
    found = installed_versions()
    expected = direct_listing()
    assert {name: set(versions) for name, versions in found.items()} == expected
    for versions in found.values():
        assert list(versions) == sorted(versions)
        assert isinstance(versions, tuple)


def test_installed_versions_holds_pytest_at_its_installed_version() -> None:
    found = installed_versions()
    assert importlib.metadata.version("pytest") in found["pytest"]
    assert "rinchen" in found


class FakeMetadata:
    """The metadata of a distribution as the next Python will have it: a missing field is a
    `KeyError` for `[]` and a `None` for `get`."""

    def __init__(self, fields: Mapping[str, str]) -> None:
        self._fields = dict(fields)

    def __getitem__(self, key: str) -> str:
        return self._fields[key]

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._fields.get(key, default)


class FakeDistribution:
    """What `importlib.metadata.distributions()` yields, reduced to the two fields read."""

    def __init__(self, name: str | None, version: str | None) -> None:
        fields = {}
        if name is not None:
            fields["Name"] = name
        if version is not None:
            fields["Version"] = version
        self.metadata = FakeMetadata(fields)

    @property
    def version(self) -> str:
        return self.metadata["Version"]


SPELLINGS = ["Foo_Bar", "foo.bar", "FOO-BAR"]


def fake_listing(
    monkeypatch: pytest.MonkeyPatch, *distributions: FakeDistribution
) -> dict[str, tuple[str, ...]]:
    monkeypatch.setattr(
        importlib.metadata, "distributions", lambda: iter(distributions)
    )
    return installed_versions()


def test_a_distribution_without_a_name_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = fake_listing(
        monkeypatch, FakeDistribution(None, "1.0"), FakeDistribution("alpha", "1.0")
    )
    assert found == {"alpha": ("1.0",)}


def test_installed_names_are_normalised_and_versions_sorted_and_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = ["9.0", "1.5", "10.0", "2.0", "1.10", "3.1", "0.9", "2.0"]
    found = fake_listing(
        monkeypatch,
        *(
            FakeDistribution(SPELLINGS[i % len(SPELLINGS)], v)
            for i, v in enumerate(versions)
        ),
        FakeDistribution("zed", "1"),
        FakeDistribution("Alpha", "1"),
    )
    assert list(found) == ["alpha", "foo-bar", "zed"]
    assert found["foo-bar"] == tuple(sorted(set(versions)))
    assert len(found["foo-bar"]) == len(versions) - 1


def test_a_distribution_without_a_version_is_not_a_listed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = fake_listing(
        monkeypatch, FakeDistribution("alpha", None), FakeDistribution("beta", "1")
    )
    assert "alpha" not in found
    assert found["beta"] == ("1",)
