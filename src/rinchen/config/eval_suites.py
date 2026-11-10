"""Configuration of the benchmark suites (`configs/eval/suites.yaml`, D13 T4): which benchmarks
`rinchen eval suite` knows, how each is read, where its dataset lives on the Hub and which
revision of it is pinned, its splits, the columns of its shards and the languages to look for.

A benchmark is read in one of two ways. `parquet_shards` is a Hub dataset as `rinchen eval suite
fetch` writes it: one directory per language holding Parquet shards, whose `columns` name the
sentence's `id`, its `text` and, optionally, the `doc` it belongs to. `lines_dir` is a directory of
text files, one per language, one sentence per line; it has no Hub dataset, no revision and no
columns. `languages` are the registry codes the benchmark is looked for in; the hub language
Khalkha is always among them, `khk_Mong` is refused by name and `mn_Mong` is not a language of
FLORES+. Nothing in the shipped file was checked against a real dataset.

The document is read from the bytes the caller hands in, so that the caller can hash the file
before it is read and parse the bytes it hashed. A message names the field at fault and never
quotes a value of the file.

The names shared with the suite reader (`rinchen.eval.suite_rules`) are written again here: the
devtest guard keeps `rinchen.config` from importing that module, and a test keeps the two copies
equal.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from rinchen.config.languages import known_language_codes
from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError

DEFAULT_EVAL_SUITES_PATH: Final = REPO_ROOT / "configs" / "eval" / "suites.yaml"
HUB_LANGUAGE: Final = "khk_Cyrl"
INNER_MONGOLIAN: Final = "khk_Mong"
FLORES_BENCHMARK: Final = "flores_plus"
FLORES_EXCLUDED: Final = ("mn_Mong",)
# The name the build command gives to a human set; no benchmark of the file may take it.
HUMAN_BENCHMARK: Final = "human"

Reader = Literal["parquet_shards", "lines_dir"]
SplitName = Literal["dev", "devtest"]

_NAME_PATTERN: Final = r"^[A-Za-z0-9._-]{1,64}$"
_COLUMN_PATTERN: Final = r"^[A-Za-z0-9_.-]{1,64}$"
_DATASET_PATTERN: Final = r"^[A-Za-z0-9._/-]{1,200}$"
_REVISION_PATTERN: Final = r"^[0-9a-f]{40}$"


def _distinct_splits(values: list[SplitName]) -> list[SplitName]:
    if len(set(values)) != len(values):
        raise ValueError("a split is listed twice")
    return values


def _registry_languages(values: list[str]) -> list[str]:
    for index, code in enumerate(values):
        if code == INNER_MONGOLIAN:
            raise ValueError(
                f"languages[{index}]: {INNER_MONGOLIAN} is refused by name: Inner "
                "Mongolian data is never used"
            )
        if code not in known_language_codes():
            raise ValueError(f"languages[{index}]: not a language code of the registry")
    if len(set(values)) != len(values):
        raise ValueError("a language is listed twice")
    if HUB_LANGUAGE not in values:
        raise ValueError(f"the hub language {HUB_LANGUAGE} must be listed")
    return values


class ColumnsConfig(BaseModel):
    """The columns of a shard: the sentence's `id` and `text` and, when the dataset has one, the
    `doc` (a document the sentence belongs to; the build turns it into an id)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(pattern=_COLUMN_PATTERN)]
    text: Annotated[str, Field(pattern=_COLUMN_PATTERN)]
    doc: Annotated[str, Field(pattern=_COLUMN_PATTERN)] | None = None

    @model_validator(mode="after")
    def _three_different_names(self) -> ColumnsConfig:
        names = [name for name in (self.id, self.text, self.doc) if name is not None]
        if len(set(names)) != len(names):
            raise ValueError("the columns must have different names")
        return self


class BenchmarkConfig(BaseModel):
    """One benchmark; see the module's text for what each field means. The fields of the Hub
    (`dataset`, `revision`, `columns`) belong to `parquet_shards` and are refused for `lines_dir`;
    `columns` is required for `parquet_shards`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reader: Reader
    dataset: Annotated[str, Field(pattern=_DATASET_PATTERN)] | None = None
    revision: Annotated[str, Field(pattern=_REVISION_PATTERN)] | None = None
    splits: Annotated[
        list[SplitName], Field(min_length=1), AfterValidator(_distinct_splits)
    ]
    columns: ColumnsConfig | None = Field(default=None, validate_default=True)
    languages: Annotated[
        list[str], Field(min_length=1), AfterValidator(_registry_languages)
    ]

    @field_validator("dataset", "revision")
    @classmethod
    def _only_shards_come_from_the_hub(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        if value is not None and info.data.get("reader") == "lines_dir":
            raise ValueError("a lines_dir benchmark reads local files and has none")
        return value

    @field_validator("columns")
    @classmethod
    def _shards_need_columns_and_files_have_none(
        cls, value: ColumnsConfig | None, info: ValidationInfo
    ) -> ColumnsConfig | None:
        reader = info.data.get("reader")
        if reader == "parquet_shards" and value is None:
            raise ValueError("a parquet_shards benchmark must name its columns")
        if reader == "lines_dir" and value is not None:
            raise ValueError("a lines_dir benchmark reads local files and has none")
        return value


class EvalSuitesConfig(BaseModel):
    """The whole file: the benchmarks by name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmarks: Annotated[
        dict[Annotated[str, Field(pattern=_NAME_PATTERN)], BenchmarkConfig],
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def _names_and_languages_of_the_benchmarks(self) -> EvalSuitesConfig:
        if HUMAN_BENCHMARK in self.benchmarks:
            raise ValueError(
                f"benchmarks.{HUMAN_BENCHMARK}: the name is taken by the build command for "
                "the human sets"
            )
        flores = self.benchmarks.get(FLORES_BENCHMARK)
        if flores is not None:
            for code in FLORES_EXCLUDED:
                if code in flores.languages:
                    raise ValueError(
                        f"benchmarks.{FLORES_BENCHMARK}.languages: {code} is not a "
                        f"language of {FLORES_BENCHMARK}"
                    )
        return self


def load_eval_suites_config(data: bytes) -> EvalSuitesConfig:
    """The benchmarks in the YAML `data`.

    Raises:
        ConfigError: `data` is not UTF-8 or not YAML, is not a mapping, or breaks the schema;
            the message names every field at fault and quotes no value.
    """
    try:
        raw: Any = yaml.safe_load(data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigError("invalid eval suites config: not UTF-8 text") from error
    except yaml.YAMLError as error:
        raise ConfigError(
            f"invalid eval suites config: cannot parse the YAML: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ConfigError(
            "invalid eval suites config: the top level must be a mapping of "
            f"field names to values, not {type(raw).__name__}"
        )
    try:
        return EvalSuitesConfig.model_validate(raw)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'config'}: {item['msg']}"
            for item in error.errors(include_input=False)
        )
        raise ConfigError(f"invalid eval suites config: {problems}") from error
