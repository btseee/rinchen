"""The writer of the fixture suites (D15 added this one module to D13's eval package).

A pipeline that proves itself on its own fixtures needs evaluation suites made of them, and the devtest
guard (`tests/unit/test_eval_devtest_guard.py`) keeps every module outside `rinchen.eval` and the eval
commands from importing the readers of a suite file. This module is the write-only way in, inside the
tree the guard exempts: it takes plain data (`FixtureSuite`, `FixtureRow`), builds the real suite, has
the real reader's rules check it and writes it through the real writer, and it exposes no reader: the
suite types and the reading functions are imported under private names and never re-exported, and a test
pins the public names and lets one module of the pipeline import it.

What it adds to the rules of a suite is two refusals. The kind is always `fixture` (there is no
parameter for it), so a made suite never passes for a benchmark or for a human set; and the benchmark
label is not the name of a benchmark of `configs/eval/suites.yaml` (read through its loader, never a
second list) or of the human sets. Whatever else `suite_rules` refuses stays refused: the suite is
checked by the reader's own validator before a byte is written, so a bad suite leaves nothing behind.
A message names a rule and never quotes a sentence or a name of a suite's content.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from rinchen.config import eval_suites as _eval_suites
from rinchen.errors import ConfigError
from rinchen.eval import suite_rules as _rules
from rinchen.eval import suites as _suites

__all__ = ["FixtureRow", "FixtureSuite", "fixture_suite_path", "write_fixture_suite"]

_KIND: Final = "fixture"


@dataclass(frozen=True, slots=True)
class FixtureRow:
    """One sentence in one or more languages: its `id`, its `text` (language code to sentence)
    and the `doc` it belongs to, if any."""

    id: str
    text: Mapping[str, str]
    doc: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", MappingProxyType(dict(self.text)))


@dataclass(frozen=True, slots=True)
class FixtureSuite:
    """A fixture suite as plain data: its `name` (the file is `<name>.json`), the `benchmark`
    label it carries, its `split` (`dev`, `devtest` or `test`), the `directions` it declares and
    its `rows` in order."""

    name: str
    benchmark: str
    split: str
    directions: Sequence[str]
    rows: Sequence[FixtureRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "directions", tuple(self.directions))
        object.__setattr__(self, "rows", tuple(self.rows))


def fixture_suite_path(directory: Path, name: str) -> Path:
    """The file `write_fixture_suite` writes for the suite `name` in `directory`.

    Raises:
        DataError: `name` is not a name of a suite (a name never holds a separator).
    """
    return directory / f"{_rules.checked_name(name, 'suite', 'name')}.json"


def write_fixture_suite(suite: FixtureSuite, directory: Path) -> str:
    """Write `suite` as `<directory>/<name>.json` and return the SHA-256 of the bytes written.

    Raises:
        ConfigError: the benchmark label is the name of a configured benchmark or of the human
            sets, or the configuration of the benchmarks cannot be read.
        DataError: the suite breaks a rule of a suite (a name, a split, a language, a direction
            without rows, a sentence, a row id); nothing is written then, not even the directory.
    """
    _refuse_a_benchmark_name(suite.benchmark)
    built = _suites.Suite(
        name=suite.name,
        kind=_KIND,
        benchmark=suite.benchmark,
        split=suite.split,
        source=_rules.SuiteSource(None, None, False, {}, ()),
        directions=tuple(suite.directions),
        rows=tuple(_rules.SuiteRow(row.id, row.doc, row.text) for row in suite.rows),
    )
    # `to_bytes` writes what it is given: the reader's validator is what refuses a bad suite.
    _suites.Suite.from_bytes(built.to_bytes())
    path = fixture_suite_path(directory, suite.name)
    directory.mkdir(parents=True, exist_ok=True)
    return _suites.write_suite(built, path)


def _refuse_a_benchmark_name(benchmark: str) -> None:
    path = _eval_suites.DEFAULT_EVAL_SUITES_PATH
    try:
        config = _eval_suites.load_eval_suites_config(path.read_bytes())
    except OSError as error:
        raise ConfigError(
            "the configuration of the benchmarks cannot be read, so a fixture suite "
            "cannot be told from a benchmark"
        ) from error
    if benchmark in config.benchmarks or benchmark == _eval_suites.HUMAN_BENCHMARK:
        raise ConfigError(
            "the benchmark label of a fixture suite is the name of a configured benchmark "
            "or of the human sets"
        )
