"""What the options of `rinchen eval suite` ask for (D13): the revision of a fetch, the plan of a
build and the benchmark of the configuration they name, or a `ConfigError` before anything is
fetched, read or written.

The revision of a fetch is a commit hash from `--revision`, else the one the configuration pins,
and is None only when the operator said `--unpinned`: a fetch of a dataset as it is now is never
the default. A build is for a benchmark of the configuration (`--split` is required; the kind is
`benchmark` unless `fixture` is said) or for the benchmark `human` (the split is `test` and the
kind `human`, whatever is given must agree).
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final, NamedTuple

from rinchen.config.eval_suites import (
    HUMAN_BENCHMARK,
    BenchmarkConfig,
    EvalSuitesConfig,
)
from rinchen.errors import ConfigError
from rinchen.eval.robust_suite_rules import is_id

_REVISION: Final = re.compile(r"[0-9a-f]{40}")
_REVISION_RULE: Final = "40 lowercase hexadecimal characters (a commit hash)"
_NAME_RULE: Final = "1 to 64 characters: letters, digits, '.', '_' and '-'"


class SuiteKind(StrEnum):
    benchmark = "benchmark"
    human = "human"
    fixture = "fixture"


class Pin(NamedTuple):
    """The revision a fetch is asked for, from the options alone: a hash or None, and whether
    the operator said `--unpinned`."""

    revision: str | None
    unpinned: bool


class BuildPlan(NamedTuple):
    benchmark: str
    split: str
    kind: str
    name: str


def is_revision(value: str) -> bool:
    """Whether `value` is a commit hash: 40 lowercase hexadecimal characters."""
    return _REVISION.fullmatch(value) is not None


def pin_of(revision: str | None, unpinned: bool) -> Pin:
    """The `--revision` and `--unpinned` options as a pin.

    Raises:
        ConfigError: both are given, or the revision is not a commit hash.
    """
    if revision is not None and unpinned:
        raise ConfigError("give --revision or --unpinned, not both")
    if revision is not None and not is_revision(revision):
        raise ConfigError(f"--revision must be {_REVISION_RULE}")
    return Pin(revision, unpinned)


def revision_to_fetch(pin: Pin, bench: BenchmarkConfig) -> str | None:
    """The commit to fetch: the option, else the configuration's; None only for `--unpinned`.

    Raises:
        ConfigError: no revision is given or pinned and `--unpinned` was not said.
    """
    if pin.unpinned:
        return None
    chosen = pin.revision or bench.revision
    if chosen is None:
        raise ConfigError(
            "the revision of the dataset is not pinned: give --revision (a commit hash) "
            "or pin it in the configuration, or say --unpinned to fetch whatever the "
            "dataset is now"
        )
    return chosen


def plan_of(
    benchmark: str, split: str | None, kind: SuiteKind | None, name: str | None
) -> BuildPlan:
    """The options of a build as a plan; the name defaults to `<benchmark>-<split>`.

    Raises:
        ConfigError: the name is not an id, the split is missing or is not `test` for a human
            set, or the kind is not allowed for the benchmark.
    """
    if name is not None and not is_id(name):
        raise ConfigError(f"--name must be {_NAME_RULE}")
    if benchmark == HUMAN_BENCHMARK:
        if split not in (None, "test"):
            raise ConfigError("--split must be test for a human set (or left out)")
        if kind not in (None, SuiteKind.human):
            raise ConfigError(
                "--kind must be human for the benchmark human (or left out)"
            )
        return BuildPlan(benchmark, "test", "human", name or f"{benchmark}-test")
    if split is None:
        raise ConfigError("--split is required: dev or devtest")
    if kind is SuiteKind.human:
        raise ConfigError(
            "--kind human is for the benchmark human, which is built from an export"
        )
    return BuildPlan(
        benchmark,
        split,
        SuiteKind.benchmark.value if kind is None else kind.value,
        name or f"{benchmark}-{split}",
    )


def benchmark_of(suites: EvalSuitesConfig, name: str, split: str) -> BenchmarkConfig:
    """The benchmark `name` of the configuration, which must have the split `split`.

    Raises:
        ConfigError: the configuration has no such benchmark or the benchmark no such split.
    """
    bench = suites.benchmarks.get(name)
    if bench is None:
        raise ConfigError(
            f"the benchmark {name} is not in the configuration (it has "
            f"{', '.join(sorted(suites.benchmarks))})"
        )
    if split not in bench.splits:
        raise ConfigError(
            f"the split {split} is not a split of the benchmark {name} (it has "
            f"{', '.join(bench.splits)})"
        )
    return bench
