"""Shared setup of the tests of `rinchen eval suite` and of its fetch (D13, Task 4).

Nothing here touches the network: the fetch runs the REAL `rinchen.acquire.hf.fetch` against a
`datasets.load_dataset` that `FakeLoadDataset` replaces, which records every call and hands out
the rows of the language it was asked for. The shards `build` reads are written by the real
`rinchen.acquire.hf.write_shards`, which is what the fetch writes them with.

A `SuiteWorkspace` is the files of one run under a temporary directory: a tree of shards of three
languages (`eng_Latn`, `khk_Cyrl`, `rus_Cyrl`, four rows each; the rows of the three languages
share ids and documents, as a multi-way parallel benchmark does) and an output directory that
does not exist.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import polars as pl
import pytest
from click.testing import Result
from typer.testing import CliRunner

from rinchen.acquire import hf
from rinchen.cli.main import app

runner = CliRunner()

HUB: Final = "khk_Cyrl"
LANGUAGES: Final = ("eng_Latn", "khk_Cyrl", "rus_Cyrl")
SENTENCES: Final[Mapping[str, tuple[str, ...]]] = {
    "eng_Latn": (
        "the cat sat on the mat",
        "a dog runs in the park",
        "she reads a long book",
        "we eat bread and cheese",
    ),
    "khk_Cyrl": (
        "муур дэвсгэр дээр сууж байв",
        "нохой цэцэрлэгт хүрээлэнд гүйж байна",
        "тэр урт ном уншдаг",
        "бид талх бяслаг иддэг",
    ),
    "rus_Cyrl": (
        "кошка сидела на коврике",
        "собака бежит в парке",
        "она читает длинную книгу",
        "мы едим хлеб и сыр",
    ),
}
# What the `url` column holds in the fixture: addresses, which are not ids, so that the build
# has to turn a document into one.
DOCUMENT_URLS: Final = (
    "https://example.org/news/first",
    "https://example.org/news/first",
    "https://example.org/news/second",
    "https://example.org/news/second",
)
IDS: Final = ("s1", "s2", "s3", "s4")
REVISION: Final = "a" * 40
MANIFEST_FILE: Final = "manifest.json"
SUITE_FILE: Final = "flores_plus-devtest.json"
SHARD: Final = "shard-00000.parquet"


def sentences_of(language: str) -> tuple[str, ...]:
    """Four sentences of `language`: the invented ones above, or a numbered one for a language
    the fixture has none for."""
    return SENTENCES.get(language) or tuple(
        f"{language} sentence {number}" for number in range(1, 5)
    )


def shard_rows(
    language: str, *, ids: Sequence[object] = IDS, texts: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """The rows of `language` as the fetch writes them: `id`, `text` and `url`."""
    sentences = sentences_of(language) if texts is None else texts
    return [
        {"id": row_id, "text": text, "url": url}
        for row_id, text, url in zip(ids, sentences, DOCUMENT_URLS, strict=True)
    ]


def write_language_shards(
    root: Path,
    language: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    shard_size: int = 10_000,
) -> list[Path]:
    """Write `rows` under `root/language` with the shard writer of the fetch."""
    return hf.write_shards(rows, root / language, shard_size=shard_size)


@dataclass(frozen=True)
class LoadCall:
    """One call of `datasets.load_dataset`: its two positional arguments, the keywords it knows
    and the extra keywords."""

    dataset: str
    config: str | None
    split: str | None
    streaming: bool
    extra: Mapping[str, Any]

    def as_tuple(self) -> tuple[str, str | None, str | None, bool, dict[str, Any]]:
        return (self.dataset, self.config, self.split, self.streaming, dict(self.extra))


@dataclass
class FakeLoadDataset:
    """`datasets.load_dataset` as the fetch sees it: it records the call and answers with the
    rows of the language the call names (`config`), or raises `error` for the language
    `error_for` (every language when that is None)."""

    rows: Mapping[str, Sequence[Mapping[str, Any]]]
    error: BaseException | None = None
    error_for: str | None = None
    calls: list[LoadCall] = field(default_factory=list)

    def __call__(
        self,
        dataset: str,
        config: str | None = None,
        *,
        split: str | None = None,
        streaming: bool = False,
        **extra: Any,
    ) -> Iterator[dict[str, Any]]:
        self.calls.append(LoadCall(dataset, config, split, streaming, extra))
        if self.error is not None and self.error_for in (None, config):
            raise self.error
        assert config is not None
        return iter(dict(row) for row in self.rows[config])


def install_fake_load_dataset(
    monkeypatch: pytest.MonkeyPatch, fake: FakeLoadDataset
) -> FakeLoadDataset:
    """Replace `datasets.load_dataset`, the function the real `fetch` calls."""
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake)
    return fake


def fixture_rows(
    languages: Sequence[str] = LANGUAGES,
) -> dict[str, list[dict[str, Any]]]:
    return {language: shard_rows(language) for language in languages}


@dataclass(frozen=True)
class SuiteWorkspace:
    """The files of one `build` run: `shards` holds a directory per language, `out_dir` does not
    exist."""

    root: Path
    shards: Path
    out_dir: Path

    @property
    def manifest(self) -> Path:
        return self.out_dir / MANIFEST_FILE

    @property
    def suite(self) -> Path:
        return self.out_dir / SUITE_FILE

    def build_args(
        self,
        *extra: str,
        benchmark: str = "flores_plus",
        split: str = "devtest",
        source: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        """`eval suite build --benchmark B --split S --from P --out-dir O` and `extra`."""
        return [
            "eval",
            "suite",
            "build",
            "--benchmark",
            benchmark,
            "--split",
            split,
            "--from",
            str(self.shards if source is None else source),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def shard(self, language: str, name: str = SHARD) -> Path:
        return self.shards / language / name


def make_workspace(root: Path, languages: Sequence[str] = LANGUAGES) -> SuiteWorkspace:
    """Shards of `languages` under `root/shards` and an output directory that is not made."""
    shards = root / "shards"
    for language in languages:
        write_language_shards(shards, language, shard_rows(language))
    return SuiteWorkspace(root=root, shards=shards, out_dir=root / "out")


def invoke(args: Sequence[str], *, env: dict[str, str] | None = None) -> Result:
    """Run `rinchen` with `args` in this process; standard output and standard error stay
    apart (`result.stdout`, `result.stderr`)."""
    return runner.invoke(app, list(args), env=env)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(directory: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((directory / MANIFEST_FILE).read_text("utf-8"))
    return loaded


CONFIG_TEXT: Final = """\
benchmarks:
  flores_plus:
    reader: parquet_shards
    dataset: openlanguagedata/flores_plus
    revision: {revision}
    splits: [dev, devtest]
    columns: {{id: id, text: text, doc: url}}
    languages: {languages}
  flores_copy:
    reader: parquet_shards
    dataset: openlanguagedata/flores_plus
    revision: null
    splits: [dev, devtest]
    columns: {{id: id, text: text, doc: url}}
    languages: {languages}
  ntrex:
    reader: lines_dir
    splits: [devtest]
    languages: {languages}
"""


def write_config(
    path: Path, *, languages: Sequence[str] = LANGUAGES, revision: str | None = None
) -> Path:
    """A configuration of the two benchmarks over `languages`, the revision of `flores_plus`
    pinned to `revision` (or null)."""
    path.write_text(
        CONFIG_TEXT.format(
            revision="null" if revision is None else revision,
            languages="[" + ", ".join(languages) + "]",
        ),
        encoding="utf-8",
    )
    return path


# ---- D2's export of a human set -----------------------------------------------------------

EXPORT_COLUMNS: Final = (
    "src",
    "tgt",
    "src_lang",
    "tgt_lang",
    "doc_id",
    "seg_idx",
    "set",
    "origin",
    "split",
)
# The column types `rinchen.annotate.export.export_parquet` writes, so that an export with no
# document or no segment number still has them.
EXPORT_PARQUET_SCHEMA: Final[dict[str, Any]] = {
    "src": pl.Utf8,
    "tgt": pl.Utf8,
    "src_lang": pl.Utf8,
    "tgt_lang": pl.Utf8,
    "doc_id": pl.Utf8,
    "seg_idx": pl.Int64,
    "set": pl.Utf8,
    "origin": pl.Utf8,
    "split": pl.Utf8,
}


def export_row(
    doc_id: str | None,
    seg_idx: int | None,
    src: str,
    tgt: str,
    *,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "khk_Cyrl",
    set_name: str = "news",
    origin: str = "human",
    split: str = "test",
) -> dict[str, Any]:
    return {
        "src": src,
        "tgt": tgt,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "doc_id": doc_id,
        "seg_idx": seg_idx,
        "set": set_name,
        "origin": origin,
        "split": split,
    }


def golden_export() -> list[dict[str, Any]]:
    """Six rows of two documents, English to Khalkha."""
    return [
        export_row("d1", 1, "The cat sat.", "Муур сууж байв."),
        export_row("d1", 2, "A dog runs.", "Нохой гүйж байна."),
        export_row("d1", 3, "She reads.", "Тэр уншдаг."),
        export_row("d2", 1, "We eat bread.", "Бид талх иддэг."),
        export_row("d2", 2, "The sun is hot.", "Нар халуун байна."),
        export_row("d2", 3, "He drinks tea.", "Тэр цай уудаг."),
    ]


def write_export(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """`rows` as D2's export in the format the suffix of `path` names (`.jsonl`, `.tsv` or
    `.parquet`), written as `rinchen.annotate.export` writes it."""
    if path.suffix == ".jsonl":
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    elif path.suffix == ".tsv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    else:
        pl.DataFrame(list(rows), schema=EXPORT_PARQUET_SCHEMA).write_parquet(path)
    return path
