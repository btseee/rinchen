"""The writer of the fixture suites (D15 added `rinchen.eval.fixture_suites` to D13's eval package).

The devtest guard keeps every reader of a suite file out of the code that is not `rinchen.eval` or one of
the eval commands, and a pipeline that makes its own fixture suites needs to WRITE one. This module is the
write-only way in: it takes plain data, always writes a suite of kind `fixture`, refuses the name of a
configured benchmark, and exposes nothing that reads a suite. What a suite may hold stays the business of
`suite_rules`: a refusal there is a refusal here. These tests read what the writer wrote back through
`read_suite`, which only a test may do.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.config import eval_suites
from rinchen.errors import ConfigError, DataError
from rinchen.eval import fixture_suites, suite_rules, suite_sources, suites
from rinchen.eval.fixture_suites import (
    FixtureRow,
    FixtureSuite,
    fixture_suite_path,
    write_fixture_suite,
)
from tests.unit.test_eval_devtest_guard import PACKAGE_ROOT, module_of, reached

KHK: Final = "khk_Cyrl"
ENG: Final = "eng_Latn"
MONG: Final = "mn_Mong"
INNER: Final = "khk_" + "Mong"
# A Khalkha word of three letters: U+0413 U+043E U+043B.
KHK_WORD: Final = "".join(chr(code) for code in (0x413, 0x43E, 0x43B))
MODULE: Final = "rinchen.eval.fixture_suites"


def suite(**changes: Any) -> FixtureSuite:
    values: dict[str, Any] = {
        "name": "tiny-sample",
        "benchmark": "tiny",
        "split": "test",
        "directions": (f"{ENG}-{KHK}", f"{KHK}-{ENG}"),
        "rows": (
            FixtureRow("en-0001", {ENG: "The river.", KHK: KHK_WORD}),
            FixtureRow("en-0002", {ENG: "The window.", KHK: KHK_WORD + "."}, doc="d1"),
        ),
    }
    return FixtureSuite(**{**values, **changes})


def written(tmp_path: Path, spec: FixtureSuite | None = None) -> tuple[Path, str]:
    spec = spec or suite()
    digest = write_fixture_suite(spec, tmp_path)
    return fixture_suite_path(tmp_path, spec.name), digest


# ---- what is written ----------------------------------------------------------------------


def test_the_file_is_the_name_of_the_suite_in_the_directory(tmp_path: Path) -> None:
    path, _ = written(tmp_path)

    assert path == tmp_path / "tiny-sample.json"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["tiny-sample.json"]


def test_the_digest_is_the_sha256_of_the_bytes_that_were_written(
    tmp_path: Path,
) -> None:
    path, digest = written(tmp_path)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_reader_gets_back_what_was_written(tmp_path: Path) -> None:
    path, digest = written(tmp_path)

    read, read_digest = suites.read_suite(path)

    assert read_digest == digest
    assert (read.name, read.kind, read.benchmark, read.split) == (
        "tiny-sample",
        "fixture",
        "tiny",
        "test",
    )
    assert read.directions == (f"{ENG}-{KHK}", f"{KHK}-{ENG}")
    assert [(row.id, row.doc, dict(row.text)) for row in read.rows] == [
        ("en-0001", None, {ENG: "The river.", KHK: KHK_WORD}),
        ("en-0002", "d1", {ENG: "The window.", KHK: KHK_WORD + "."}),
    ]
    assert read.source == suite_rules.SuiteSource(None, None, False, {}, ())


def test_a_suite_of_the_traditional_script_is_written_too(tmp_path: Path) -> None:
    letters = "".join(chr(code) for code in (0x1820, 0x1821, 0x1822))
    spec = suite(
        name="tiny-mong",
        directions=(f"{MONG}-{ENG}",),
        rows=(FixtureRow("mm-0001", {MONG: letters, ENG: "The river."}),),
    )

    path, _ = written(tmp_path, spec)

    read, _ = suites.read_suite(path)
    assert read.rows[0].text[MONG] == letters


@pytest.mark.parametrize("split", ["dev", "devtest", "test"])
def test_each_split_of_a_fixture_suite_is_written(tmp_path: Path, split: str) -> None:
    path, _ = written(tmp_path, suite(split=split))

    assert suites.read_suite(path)[0].split == split


def test_the_directory_is_made_when_it_is_not_there(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"

    write_fixture_suite(suite(), target)

    assert (target / "tiny-sample.json").is_file()


# ---- the kind is always fixture ------------------------------------------------------------


def test_the_kind_is_not_a_choice() -> None:
    assert "kind" not in {field.name for field in dataclasses.fields(FixtureSuite)}
    assert "kind" not in {field.name for field in dataclasses.fields(FixtureRow)}
    assert list(inspect.signature(write_fixture_suite).parameters) == [
        "suite",
        "directory",
    ]
    with pytest.raises(TypeError):
        FixtureSuite(  # type: ignore[call-arg]
            name="x",
            benchmark="tiny",
            split="test",
            directions=(f"{ENG}-{KHK}",),
            rows=(),
            kind="human",
        )


@pytest.mark.parametrize("split", ["devtest", "test", "dev"])
def test_every_suite_that_is_written_has_the_kind_fixture(
    tmp_path: Path, split: str
) -> None:
    path, _ = written(tmp_path, suite(split=split))

    assert suites.read_suite(path)[0].kind == "fixture"


# ---- the names of real benchmarks -----------------------------------------------------------


def configured_names() -> set[str]:
    config = eval_suites.load_eval_suites_config(
        eval_suites.DEFAULT_EVAL_SUITES_PATH.read_bytes()
    )
    return set(config.benchmarks)


def test_the_shipped_configuration_names_the_benchmarks_the_writer_refuses() -> None:
    assert {"flores_plus", "ntrex"} <= configured_names()


@pytest.mark.parametrize(
    "benchmark",
    [*sorted(configured_names()), eval_suites.HUMAN_BENCHMARK],
)
def test_the_name_of_a_configured_benchmark_is_refused_and_nothing_is_written(
    tmp_path: Path, benchmark: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        write_fixture_suite(suite(benchmark=benchmark), tmp_path / "out")

    if benchmark != eval_suites.HUMAN_BENCHMARK:
        assert benchmark not in str(caught.value)
    assert not (tmp_path / "out").exists()


def test_the_refused_names_are_the_ones_of_the_configuration_not_a_list_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "suites.yaml"
    other.write_text(
        "benchmarks:\n"
        "  acme_set:\n"
        "    reader: lines_dir\n"
        "    splits: [devtest]\n"
        f"    languages: [{KHK}, {ENG}]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(eval_suites, "DEFAULT_EVAL_SUITES_PATH", other)

    with pytest.raises(ConfigError):
        write_fixture_suite(suite(benchmark="acme_set"), tmp_path / "out")
    # The shipped names mean nothing to a configuration that does not list them.
    write_fixture_suite(suite(benchmark="flores_plus"), tmp_path / "other")

    assert (tmp_path / "other" / "tiny-sample.json").is_file()


def test_a_configuration_that_cannot_be_read_is_a_refusal_not_a_free_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eval_suites, "DEFAULT_EVAL_SUITES_PATH", tmp_path / "missing")

    with pytest.raises(ConfigError):
        write_fixture_suite(suite(), tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_a_name_that_is_not_a_configured_benchmark_is_written(tmp_path: Path) -> None:
    for benchmark in ("tiny", "toy", "flores"):
        write_fixture_suite(suite(benchmark=benchmark, name=f"s-{benchmark}"), tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "s-flores.json",
        "s-tiny.json",
        "s-toy.json",
    ]
    for benchmark in ("tiny", "toy", "flores"):
        read, _ = suites.read_suite(tmp_path / f"s-{benchmark}.json")
        assert read.benchmark == benchmark


# ---- what the rules of a suite refuse stays refused ----------------------------------------


def test_inner_mongolian_is_refused_by_name_in_a_row_and_in_a_direction(
    tmp_path: Path,
) -> None:
    in_a_row = suite(rows=(FixtureRow("r-1", {ENG: "The river.", INNER: KHK_WORD}),))
    in_a_direction = suite(directions=(f"{ENG}-{INNER}",))

    for spec in (in_a_row, in_a_direction):
        with pytest.raises(DataError):
            write_fixture_suite(spec, tmp_path / "out")

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"split": "train"},
        {"name": "has space"},
        {"name": "../escape"},
        {"name": ""},
        {"directions": ()},
        {"directions": (f"{ENG}-{ENG}",)},
        {"directions": (f"{ENG}-deu_Latn",)},
        {"rows": ()},
        {"rows": (FixtureRow("r-1", {ENG: "   ", KHK: KHK_WORD}),)},
        {"rows": (FixtureRow("bad id", {ENG: "The river.", KHK: KHK_WORD}),)},
        {
            "rows": (
                FixtureRow("r-1", {ENG: "The river.", KHK: KHK_WORD}),
                FixtureRow("r-1", {ENG: "The window.", KHK: KHK_WORD}),
            )
        },
        {"rows": (FixtureRow("r-1", {ENG: "The\nriver.", KHK: KHK_WORD}),)},
        {"rows": (FixtureRow("r-1", {ENG: "The river."}),)},
    ],
    ids=[
        "no-train-split",
        "name-with-space",
        "name-with-slash",
        "empty-name",
        "no-direction",
        "a-language-to-itself",
        "a-language-of-no-registry",
        "no-rows",
        "blank-sentence",
        "bad-row-id",
        "repeated-row-id",
        "line-break",
        "a-direction-with-no-row",
    ],
)
def test_what_the_suite_rules_refuse_is_refused_and_nothing_is_written(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    with pytest.raises(DataError):
        write_fixture_suite(suite(**changes), tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_the_message_of_a_refusal_quotes_no_sentence(tmp_path: Path) -> None:
    secret = "SECRET-SENTENCE-OF-A-FIXTURE"
    spec = suite(rows=(FixtureRow("bad id", {ENG: secret, KHK: KHK_WORD}),))

    with pytest.raises(DataError) as caught:
        write_fixture_suite(spec, tmp_path / "out")

    assert secret not in str(caught.value)


def test_the_path_of_a_suite_refuses_a_name_that_is_not_one(tmp_path: Path) -> None:
    for name in ("", "a/b", "a\\b", "a b", "x" * 65):
        with pytest.raises(DataError):
            fixture_suite_path(tmp_path, name)

    assert fixture_suite_path(tmp_path, "tiny-x") == tmp_path / "tiny-x.json"


# ---- plain data -----------------------------------------------------------------------------


def test_the_specs_are_frozen_and_their_texts_cannot_be_changed() -> None:
    spec = suite()

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.rows[0].text[ENG] = "changed"  # type: ignore[index]
    assert isinstance(spec.rows, tuple)
    assert isinstance(spec.directions, tuple)


def test_a_text_mapping_that_is_changed_later_does_not_change_the_suite(
    tmp_path: Path,
) -> None:
    texts = {ENG: "The river.", KHK: KHK_WORD}
    spec = suite(rows=(FixtureRow("r-1", texts),))

    texts[ENG] = "Changed afterwards."
    path, _ = written(tmp_path, spec)

    assert suites.read_suite(path)[0].rows[0].text[ENG] == "The river."


def test_a_list_of_rows_is_taken_as_a_tuple(tmp_path: Path) -> None:
    spec = suite(
        rows=[FixtureRow("r-1", {ENG: "A.", KHK: KHK_WORD})],
        directions=[f"{ENG}-{KHK}"],
    )

    path, _ = written(tmp_path, spec)

    assert isinstance(spec.rows, tuple)
    assert isinstance(spec.directions, tuple)
    assert suites.read_suite(path)[0].directions == (f"{ENG}-{KHK}",)


# ---- determinism ------------------------------------------------------------------------------


def test_two_calls_write_equal_bytes(tmp_path: Path) -> None:
    first, _ = written(tmp_path / "one")
    second, _ = written(tmp_path / "two")

    assert first.read_bytes() == second.read_bytes()


SNIPPET: Final = (
    "import sys, tempfile, hashlib\n"
    "from pathlib import Path\n"
    "from rinchen.eval.fixture_suites import FixtureRow, FixtureSuite, write_fixture_suite\n"
    "rows = tuple(FixtureRow(f'r-{n}', {'eng_Latn': f'Row {n}.', 'khk_Cyrl': f'Khk {n}.'}) for n in range(30))\n"
    "spec = FixtureSuite('tiny-hash', 'tiny', 'test', ('eng_Latn-khk_Cyrl', 'khk_Cyrl-eng_Latn'), rows)\n"
    "with tempfile.TemporaryDirectory() as directory:\n"
    "    print(write_fixture_suite(spec, Path(directory)))\n"
)


@pytest.mark.subprocess
@pytest.mark.parametrize("hash_seed", ["0", "3"])
def test_the_bytes_do_not_depend_on_the_hash_seed(hash_seed: str) -> None:
    environment = {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.endswith(("_TOKEN", "_SECRET", "_PASSWORD", "_API_KEY"))
        },
        "PYTHONHASHSEED": hash_seed,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    rows = tuple(
        FixtureRow(f"r-{n}", {ENG: f"Row {n}.", KHK: f"Khk {n}."}) for n in range(30)
    )
    own = FixtureSuite(
        "tiny-hash", "tiny", "test", (f"{ENG}-{KHK}", f"{KHK}-{ENG}"), rows
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", SNIPPET],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with tempfile.TemporaryDirectory() as directory:
        assert result.stdout.strip() == write_fixture_suite(own, Path(directory))


# ---- write-only ---------------------------------------------------------------------------------

READERS: Final = (
    suites.Suite,
    suites.read_suite,
    suites.write_suite,
    suite_rules.SuiteRow,
    suite_rules.SuiteSource,
    suites,
    suite_rules,
    suite_sources,
)


def test_the_public_names_are_pinned() -> None:
    assert sorted(fixture_suites.__all__) == [
        "FixtureRow",
        "FixtureSuite",
        "fixture_suite_path",
        "write_fixture_suite",
    ]


def test_no_public_name_is_a_reader_or_a_re_export_of_the_suite_types() -> None:
    for name in vars(fixture_suites):
        if name.startswith("_"):
            continue
        value = getattr(fixture_suites, name)
        assert not any(value is reader for reader in READERS), name


def test_the_module_has_no_reader_under_any_public_name() -> None:
    for name in ("read_suite", "Suite", "SuiteRow", "SuiteSource", "suite_sources"):
        assert not hasattr(fixture_suites, name), name
    for name in vars(fixture_suites):
        assert not name.startswith("read"), name


# ---- who may import it --------------------------------------------------------------------------


def importers_of_the_module() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    root = PACKAGE_ROOT.parent
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "fixture_suites.py":
            continue
        name, is_package = module_of(path, root)
        hits = reached(
            path.read_text(encoding="utf-8"),
            name,
            is_package=is_package,
            guarded=(MODULE,),
        )
        if hits:
            found[path.relative_to(PACKAGE_ROOT).as_posix()] = hits
    return found


def test_the_pipeline_suites_are_the_only_importer_of_the_writer() -> None:
    assert sorted(importers_of_the_module()) == ["tiny/suites.py"]


@pytest.mark.parametrize(
    "source",
    [
        "import rinchen.eval.fixture_suites\n",
        "import rinchen.eval.fixture_suites as writer\n",
        "from rinchen.eval import fixture_suites\n",
        "from rinchen.eval.fixture_suites import write_fixture_suite\n",
        "def late():\n    from rinchen.eval import fixture_suites as w\n",
        "import importlib\nimportlib.import_module('rinchen.eval.fixture_suites')\n",
    ],
)
def test_the_scan_finds_every_way_to_import_the_writer(source: str) -> None:
    assert reached(source, "rinchen.train.x", guarded=(MODULE,)) == [MODULE]


@pytest.mark.parametrize(
    "source",
    [
        "from rinchen.eval import robust_suite\n",
        "from rinchen.eval.suites import Suite\n",
        "# rinchen.eval.fixture_suites\nx = 'rinchen.eval.fixture_suites'\n",
        "import rinchen.eval.fixture_suites_other\n",
    ],
)
def test_the_scan_does_not_flag_what_is_not_an_import_of_the_writer(
    source: str,
) -> None:
    assert reached(source, "rinchen.train.x", guarded=(MODULE,)) == []


def test_the_module_is_inside_the_exempt_tree_of_the_devtest_guard() -> None:
    assert (PACKAGE_ROOT / "eval" / "fixture_suites.py").is_file()
    assert Path(fixture_suites.__file__).parent == PACKAGE_ROOT / "eval"
