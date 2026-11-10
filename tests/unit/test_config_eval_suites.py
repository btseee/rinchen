"""The configuration of the benchmark suites (D13, T4): which benchmarks `rinchen eval suite` knows,
how each is read (parquet shards of a Hub dataset, or a directory of text files), its Hub dataset
and pinned revision, its splits, the columns of its shards and the languages to look for. The
shipped file `configs/eval/suites.yaml` loads and says what the spec says; every violation of the
schema is a `ConfigError` that names the field and never quotes a value of the file."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from rinchen.config.eval_suites import (
    BenchmarkConfig,
    EvalSuitesConfig,
    load_eval_suites_config,
)

from rinchen.config import eval_suites
from rinchen.config.languages import known_language_codes
from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError
from rinchen.eval import suite_rules
from tests.support.module_imports import (
    loaded_by_importing,
    project_imports,
    source_of,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SHIPPED = REPO_ROOT / "configs" / "eval" / "suites.yaml"
MARKER = "SECRET-MARKER-83"
NINE = sorted(known_language_codes() - {"mn_Mong"})


def document() -> dict[str, Any]:
    """A valid document, as a dict the test then breaks in one place."""
    return {
        "benchmarks": {
            "flores_plus": {
                "reader": "parquet_shards",
                "dataset": "openlanguagedata/flores_plus",
                "revision": None,
                "splits": ["dev", "devtest"],
                "columns": {"id": "id", "text": "text", "doc": "url"},
                "languages": ["khk_Cyrl", "eng_Latn", "rus_Cyrl"],
            },
            "ntrex": {
                "reader": "lines_dir",
                "splits": ["devtest"],
                "languages": ["khk_Cyrl", "eng_Latn", "rus_Cyrl"],
            },
        }
    }


def load(value: dict[str, Any] | str | bytes) -> EvalSuitesConfig:
    if isinstance(value, dict):
        value = yaml.safe_dump(value)
    data = value.encode("utf-8") if isinstance(value, str) else value
    return load_eval_suites_config(data)


def refused(value: dict[str, Any] | str | bytes) -> str:
    """The message of the ConfigError a document is refused with."""
    with pytest.raises(ConfigError) as caught:
        load(value)
    return str(caught.value)


def with_change(path: str, value: object) -> dict[str, Any]:
    """The valid document with the entry at `path` (dotted; digits index a list) replaced; a
    `...` value removes it."""
    data = document()
    node: Any = data
    *parents, last = path.split(".")
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    key: Any = int(last) if isinstance(node, list) else last
    if value is ...:
        del node[key]
    else:
        node[key] = value
    return data


# ---- the shipped file ---------------------------------------------------------------------


def test_the_shipped_file_is_the_one_of_the_spec() -> None:
    config = load(SHIPPED.read_bytes())

    assert sorted(config.benchmarks) == ["flores_plus", "ntrex"]
    flores = config.benchmarks["flores_plus"]
    assert flores.model_dump() == {
        "reader": "parquet_shards",
        "dataset": "openlanguagedata/flores_plus",
        "revision": None,
        "splits": ["dev", "devtest"],
        "columns": {"id": "id", "text": "text", "doc": "url"},
        "languages": flores.languages,
    }
    ntrex = config.benchmarks["ntrex"]
    assert ntrex.model_dump() == {
        "reader": "lines_dir",
        "dataset": None,
        "revision": None,
        "splits": ["devtest"],
        "columns": None,
        "languages": ntrex.languages,
    }


def test_the_shipped_file_looks_for_the_nine_registry_languages_other_than_traditional_mongolian() -> (
    None
):
    config = load(SHIPPED.read_bytes())

    assert len(NINE) == 9
    assert sorted(config.benchmarks["flores_plus"].languages) == NINE
    assert sorted(config.benchmarks["ntrex"].languages) == NINE
    assert "mn_Mong" not in config.benchmarks["flores_plus"].languages


def test_the_shipped_file_pins_no_revision_and_says_what_is_unverified() -> None:
    raw = SHIPPED.read_text(encoding="utf-8")
    prose = " ".join(line.lstrip("# ") for line in raw.splitlines()).lower()
    flores = load(raw.encode()).benchmarks["flores_plus"]

    assert flores.revision is None
    for sentence in ("from memory", "unverified", "pins", "nothing here was fetched"):
        assert sentence in prose, sentence


# ---- what a benchmark may say -------------------------------------------------------------


def test_a_lines_directory_benchmark_may_leave_out_its_dataset_columns_and_revision() -> (
    None
):
    ntrex = load(document()).benchmarks["ntrex"]

    assert (ntrex.dataset, ntrex.columns, ntrex.revision) == (None, None, None)


def test_a_pinned_revision_is_forty_lowercase_hexadecimal_characters() -> None:
    pinned = "0123456789abcdef0123456789abcdef01234567"

    config = load(with_change("benchmarks.flores_plus.revision", pinned))

    assert config.benchmarks["flores_plus"].revision == pinned


def test_the_document_column_is_optional_for_shards() -> None:
    data = with_change("benchmarks.flores_plus.columns.doc", ...)

    assert load(data).benchmarks["flores_plus"].columns is not None
    assert load(data).benchmarks["flores_plus"].columns.doc is None  # type: ignore[union-attr]


def test_traditional_script_mongolian_is_refused_only_for_flores_plus() -> None:
    allowed = with_change("benchmarks.ntrex.languages", ["khk_Cyrl", "mn_Mong"])
    refused_for_flores = with_change(
        "benchmarks.flores_plus.languages", ["khk_Cyrl", "mn_Mong"]
    )

    assert load(allowed).benchmarks["ntrex"].languages == ["khk_Cyrl", "mn_Mong"]
    assert "benchmarks.flores_plus.languages" in refused(refused_for_flores)
    assert "mn_Mong" in refused(refused_for_flores)


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        ("benchmarks.flores_plus.languages", ["khk_Cyrl", "khk_Mong"], "languages"),
        ("benchmarks.ntrex.languages", ["khk_Cyrl", "khk_Mong"], "languages"),
    ],
)
def test_inner_mongolian_is_refused_by_name(
    path: str, value: object, field: str
) -> None:
    message = refused(with_change(path, value))

    assert "khk_Mong is refused by name" in message
    assert path in message


@pytest.mark.parametrize("code", ["xx_Latn", "abc", "", "ENG_latn", "eng_Latn "])
def test_a_language_outside_the_registry_is_refused_without_quoting_it(
    code: str,
) -> None:
    message = refused(
        with_change("benchmarks.flores_plus.languages", ["khk_Cyrl", code])
    )

    assert "benchmarks.flores_plus.languages" in message
    assert "registry" in message


def test_a_language_that_is_not_a_string_is_refused() -> None:
    assert "languages" in refused(
        with_change("benchmarks.ntrex.languages", ["khk_Cyrl", 7])
    )


@pytest.mark.parametrize(
    "languages",
    [["eng_Latn", "rus_Cyrl"], [], ["khk_Cyrl", "khk_Cyrl", "eng_Latn"]],
    ids=["no-hub", "empty", "repeated"],
)
def test_the_languages_hold_the_hub_language_once_and_are_not_empty(
    languages: list[str],
) -> None:
    message = refused(with_change("benchmarks.ntrex.languages", languages))

    assert "benchmarks.ntrex.languages" in message


def test_the_message_for_a_missing_hub_language_names_it() -> None:
    assert "khk_Cyrl" in refused(
        with_change("benchmarks.ntrex.languages", ["eng_Latn", "rus_Cyrl"])
    )


@pytest.mark.parametrize("name", ["a b", "", "x" * 65, chr(0xE9), "a/b"])
def test_a_benchmark_name_must_be_an_id(name: str) -> None:
    data = document()
    data["benchmarks"][name] = data["benchmarks"].pop("ntrex")

    assert "benchmarks" in refused(data)


def test_the_name_human_is_taken_by_the_command() -> None:
    data = document()
    data["benchmarks"]["human"] = data["benchmarks"].pop("ntrex")

    assert "human" in refused(data)


@pytest.mark.parametrize(
    "splits",
    [[], ["test"], ["dev", "dev"], ["train"], "dev", [1]],
    ids=["empty", "test", "repeated", "train", "a-string", "a-number"],
)
def test_the_splits_are_dev_and_devtest_without_repeats_and_not_empty(
    splits: object,
) -> None:
    assert "benchmarks.ntrex.splits" in refused(
        with_change("benchmarks.ntrex.splits", splits)
    )


@pytest.mark.parametrize(
    "revision",
    ["a" * 39, "a" * 41, "A" * 40, "g" * 40, 12, ""],
    ids=["39", "41", "uppercase", "not-hex", "number", "empty"],
)
def test_a_pinned_revision_that_is_not_forty_hexadecimal_characters_is_refused(
    revision: object,
) -> None:
    message = refused(with_change("benchmarks.flores_plus.revision", revision))

    assert "benchmarks.flores_plus.revision" in message


@pytest.mark.parametrize("field", ["dataset", "columns", "revision"])
def test_a_lines_directory_benchmark_holds_no_hub_field(field: str) -> None:
    values: dict[str, object] = {
        "dataset": "openlanguagedata/flores_plus",
        "columns": {"id": "id", "text": "text"},
        "revision": "a" * 40,
    }

    message = refused(with_change(f"benchmarks.ntrex.{field}", values[field]))

    assert f"benchmarks.ntrex.{field}" in message


def test_a_shard_benchmark_needs_its_columns() -> None:
    assert "benchmarks.flores_plus.columns" in refused(
        with_change("benchmarks.flores_plus.columns", ...)
    )
    assert "benchmarks.flores_plus.columns" in refused(
        with_change("benchmarks.flores_plus.columns", None)
    )


@pytest.mark.parametrize("column", ["id", "text"])
def test_the_id_and_text_columns_are_required(column: str) -> None:
    message = refused(with_change(f"benchmarks.flores_plus.columns.{column}", ...))

    assert f"benchmarks.flores_plus.columns.{column}" in message


def test_the_columns_are_three_different_names() -> None:
    data = with_change("benchmarks.flores_plus.columns", {"id": "x", "text": "x"})

    assert "benchmarks.flores_plus.columns" in refused(data)
    assert "benchmarks.flores_plus.columns" in refused(
        with_change(
            "benchmarks.flores_plus.columns", {"id": "x", "text": "y", "doc": "x"}
        )
    )


@pytest.mark.parametrize("name", ["", "a b", "x" * 65, chr(0xE9)])
def test_a_column_name_is_a_plain_name(name: str) -> None:
    assert "benchmarks.flores_plus.columns.text" in refused(
        with_change("benchmarks.flores_plus.columns.text", name)
    )


@pytest.mark.parametrize("dataset", ["", "has space", "x" * 201, 5, "a:b"])
def test_a_dataset_is_an_id_of_the_hub(dataset: object) -> None:
    assert "benchmarks.flores_plus.dataset" in refused(
        with_change("benchmarks.flores_plus.dataset", dataset)
    )


def test_an_unknown_reader_is_refused() -> None:
    assert "benchmarks.flores_plus.reader" in refused(
        with_change("benchmarks.flores_plus.reader", "csv")
    )


def test_a_benchmark_may_not_hold_a_field_that_is_not_part_of_the_format() -> None:
    assert "benchmarks.ntrex.mirror" in refused(
        with_change("benchmarks.ntrex.mirror", "x")
    )
    assert "mirror" in refused({**document(), "mirror": 1})


def test_a_document_without_benchmarks_or_with_none_is_refused() -> None:
    assert "benchmarks" in refused({})
    assert "benchmarks" in refused({"benchmarks": {}})
    assert "benchmarks" in refused({"benchmarks": ["ntrex"]})


@pytest.mark.parametrize(
    "data",
    [b"\xff\xfe\x00", b"benchmarks: [", b"- a\n- b\n", b"just text", b""],
    ids=["not-utf8", "not-yaml", "a-list", "a-scalar", "empty"],
)
def test_text_that_is_not_a_mapping_is_a_config_error(data: bytes) -> None:
    with pytest.raises(ConfigError):
        load_eval_suites_config(data)


def test_no_message_quotes_a_value_of_the_file() -> None:
    planted = [
        with_change("benchmarks.flores_plus.languages", ["khk_Cyrl", MARKER]),
        with_change("benchmarks.flores_plus.dataset", MARKER + " x"),
        with_change("benchmarks.flores_plus.revision", MARKER),
        with_change("benchmarks.flores_plus.splits", [MARKER]),
        with_change("benchmarks.flores_plus.reader", MARKER),
        with_change("benchmarks.flores_plus.columns.text", MARKER + " x"),
    ]

    for data in planted:
        assert MARKER not in refused(data)


# ---- how the module sits in the project ----------------------------------------------------


def test_the_two_names_it_shares_with_the_suite_reader_are_the_readers() -> None:
    # The devtest guard keeps `rinchen.config` from importing the suite reader, so the
    # constants are written twice: this test keeps the two copies equal.
    assert eval_suites.FLORES_BENCHMARK == suite_rules.FLORES_BENCHMARK
    assert eval_suites.INNER_MONGOLIAN == suite_rules.FORBIDDEN_LANGUAGES[0]
    assert eval_suites.FLORES_EXCLUDED == suite_rules.FLORES_EXCLUDED
    assert eval_suites.HUB_LANGUAGE == suite_rules.HUB_LANGUAGE


def test_the_module_imports_nothing_of_the_suite_reader_or_the_commands() -> None:
    imported = project_imports(source_of("rinchen.config.eval_suites"))

    assert imported <= {
        "rinchen.config.languages",
        "rinchen.errors",
        "rinchen.config.loader",
    }


def test_importing_the_module_loads_no_heavy_library_and_no_other_layer() -> None:
    loaded = loaded_by_importing(
        ["rinchen.config.eval_suites"],
        [
            "polars",
            "datasets",
            "huggingface_hub",
            "sacrebleu",
            "ctranslate2",
            "rinchen.eval",
            "rinchen.cli",
            "rinchen.train",
        ],
    )

    assert loaded == []


def test_the_default_path_is_the_shipped_file() -> None:
    assert eval_suites.DEFAULT_EVAL_SUITES_PATH == SHIPPED
    assert BenchmarkConfig.model_fields.keys() >= {"reader", "splits", "languages"}
