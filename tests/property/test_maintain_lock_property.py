"""Property tests of the lock reader of D14 (`rinchen.maintain.lock.read_lock`): a generated lock,
written as TOML text, is read into the table that a small pure-Python model of the format
predicts (names in PEP 503 form, the distinct versions of a name sorted as strings, one entry per
`[[package]]` header), whatever the order of the keys of an entry, the extra keys and sub-tables
it holds (their own `name = ...` lines are not entries), the spelling of the names and how often
a name repeats; and a format version other than 1 is always refused. Under the `no_network`
guard like every D14 test module."""

from __future__ import annotations

import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.errors import DataError
from rinchen.maintain.lock import read_lock
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
WORDS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=6)
SEPARATORS = st.sampled_from(["-", "_", ".", "--", "-_", "__", ".."])
# The pool holds the versions that sort differently as text and as numbers, and local labels.
FIXED_VERSIONS = [
    "1.0",
    "10.0",
    "9.0",
    "1.2.3",
    "2.14.1+cu130",
    "2.14.1",
    "0.0.0",
    "1.0.post1",
    "1!2.0",
    "2.0rc1",
    "3.0+local.7",
]
EXTRA_LINES = [
    'source = { registry = "https://pypi.org/simple" }',
    'dependencies = [\n    { name = "zzz" },\n    { name = "yyy", version = "9" },\n]',
    'sdist = { url = "https://files.example.invalid/x.tar.gz", hash = "sha256:00", size = 1 }',
    "resolution-markers = [\n    \"sys_platform == 'win32'\",\n]",
    'wheels = [\n    { filename = "x.whl", hash = "sha256:00" },\n]',
]
SUB_TABLE = '\n[package.metadata]\nrequires-dist = [{ name = "qqq" }]\n'


@st.composite
def raw_names(draw: st.DrawFn) -> str:
    parts = draw(st.lists(WORDS, min_size=1, max_size=3))
    separators = [draw(SEPARATORS) for _ in parts[1:]]
    name = parts[0] + "".join(s + part for s, part in zip(separators, parts[1:]))
    return "".join(c.upper() if draw(st.booleans()) else c for c in name)


VERSIONS = st.one_of(
    st.sampled_from(FIXED_VERSIONS),
    st.builds(
        lambda numbers, label: ".".join(map(str, numbers)) + label,
        st.lists(st.integers(0, 120), min_size=1, max_size=4),
        st.sampled_from(["", "", "+cu130", "+local", ".post2"]),
    ),
)


@st.composite
def locks(draw: st.DrawFn) -> tuple[bytes, int | None, list[tuple[str, str]]]:
    """A lock as text, its revision and the (name, version) of each entry in file order."""
    pool = draw(st.lists(raw_names(), min_size=1, max_size=5))
    entries = draw(
        st.lists(st.tuples(st.sampled_from(pool), VERSIONS), min_size=0, max_size=10)
    )
    revision = draw(st.one_of(st.none(), st.integers(1, 9)))
    lines = ["version = 1"]
    if revision is not None:
        lines.append(f"revision = {revision}")
    lines.append('requires-python = "==3.14.*"')
    for name, version in entries:
        keys = [
            f'name = "{name}"',
            f'version = "{version}"',
            *draw(st.lists(st.sampled_from(EXTRA_LINES), unique=True, max_size=4)),
        ]
        keys = draw(st.permutations(keys))
        sub_table = SUB_TABLE if draw(st.booleans()) else ""
        lines.append("\n[[package]]\n" + "\n".join(keys) + sub_table)
    if not entries:
        lines.append("package = []")
    return "\n".join(lines).encode() + b"\n", revision, entries


def model(entries: list[tuple[str, str]]) -> dict[str, tuple[str, ...]]:
    """The table the format predicts, written out of the lines of the problem."""
    table: dict[str, set[str]] = {}
    for name, version in entries:
        normal = re.sub(r"[-_.]+", "-", name).lower()
        table.setdefault(normal, set()).add(version)
    return {normal: tuple(sorted(table[normal])) for normal in sorted(table)}


@SETTINGS
@given(locks())
def test_a_generated_lock_is_read_into_the_table_of_its_model(
    generated: tuple[bytes, int | None, list[tuple[str, str]]],
) -> None:
    data, revision, entries = generated
    lock = read_lock(data)

    assert lock.format_version == 1
    assert lock.revision == revision
    assert lock.entries == len(entries)
    assert dict(lock.versions) == model(entries)
    assert list(lock.versions) == sorted(lock.versions)


@SETTINGS
@given(st.integers().filter(lambda number: number != 1))
def test_a_format_version_other_than_one_is_always_refused(number: int) -> None:
    with pytest.raises(DataError) as caught:
        read_lock(f"version = {number}\npackage = []\n".encode())
    assert "version" in str(caught.value)


@SETTINGS
@given(locks())
def test_reading_a_lock_twice_gives_the_same_table(
    generated: tuple[bytes, int | None, list[tuple[str, str]]],
) -> None:
    assert read_lock(generated[0]) == read_lock(generated[0])
