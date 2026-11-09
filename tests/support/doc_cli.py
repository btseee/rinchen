"""Shared setup of the `rinchen train docify` tests: a workspace holding a small corpus of
documents as Parquet, the toy SentencePiece model, a copy of the tiny teacher's base
config and the output directory.

`hand_built_rows()` is the corpus most tests use. Its windows can be counted by hand with
`--min-size 2 --max-size 2` (see the table below), and its piece counts are those of
`tests.support.doc_rows`:

    train  d1   eng->khk, sentences 0 to 3          2 windows (0 1) (2 3)
           d1   khk->eng, sentences 0 to 3          2 windows
           d2   eng->khk, sentences 4 to 6          1 window (4 5), the sentence 6 is a tail
           no document id (1 row), origin bt (1 row)
    dev    v1   both directions, sentences 6 and 7  1 window each
    test   t1   eng->khk, 3 rows of text no other row has

so 20 rows are read: 10 are in windows, 7 are not of the split (`not_train`), one has no
document id, one is of an origin that is not windowed and one is a tail.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli.main import app
from rinchen.config.loader import REPO_ROOT
from tests.support.doc_rows import MN_EN, doc_row, document
from tests.support.tiny_parallel import (
    build_toy_documents,
    train_toy_spm,
    write_corpus_parquet,
)

runner = CliRunner()

BASE_CONFIG = REPO_ROOT / "configs" / "eole" / "teacher_tiny.yaml"
# Windows of exactly two sentences, which the module's table counts by hand.
PAIRS = ["--min-size", "2", "--max-size", "2"]
TEST_ONLY_TEXT = ("Only the test split has this sentence.", "Зөвхөн туршилтын өгөгдөл.")
EN_MN_CORPUS = "eng_Latn-khk_Cyrl.doc"
MN_EN_CORPUS = "khk_Cyrl-eng_Latn.doc"
# The files a real run writes, relative to the output directory.
WRITTEN_FILES = (
    "config.yaml",
    "data/doc/eng_Latn-khk_Cyrl.doc.json",
    "data/doc/eng_Latn-khk_Cyrl.doc.src",
    "data/doc/eng_Latn-khk_Cyrl.doc.tgt",
    "data/doc/khk_Cyrl-eng_Latn.doc.json",
    "data/doc/khk_Cyrl-eng_Latn.doc.src",
    "data/doc/khk_Cyrl-eng_Latn.doc.tgt",
    "manifest.json",
    "valid.src",
    "valid.tgt",
)


def hand_built_rows() -> list[dict[str, Any]]:
    """The 20 rows of the module's table."""
    test_src, test_tgt = TEST_ONLY_TEXT
    return [
        *document("d1", range(4)),
        *document("d1", range(4), languages=MN_EN),
        *document("d2", range(4, 7)),
        doc_row(None, 0, 1),
        doc_row("d3", 0, 2, origin="bt"),
        *document("v1", range(6, 8), split="dev"),
        *document("v1", range(6, 8), split="dev", languages=MN_EN),
        *(
            doc_row("t1", position, index, split="test", src=test_src, tgt=test_tgt)
            for position, index in enumerate((0, 2, 4))
        ),
    ]


def edited_base_config(path: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    """A copy of the tiny teacher's base config at `path` after `edit` changed its
    mapping in place."""
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    edit(cfg)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


@dataclass(frozen=True, slots=True)
class DocifyWorkspace:
    """The files of one `rinchen train docify` test. `corpus`, `spm_model` and
    `base_config` exist; `out_dir` does not."""

    root: Path
    corpus: Path
    spm_model: Path
    base_config: Path
    out_dir: Path

    @property
    def manifest(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def config(self) -> Path:
        return self.out_dir / "config.yaml"

    @property
    def inputs(self) -> tuple[Path, Path, Path]:
        """The base config, the corpus and the SentencePiece model: the files a run
        hashes, in the order of its manifest."""
        return (self.base_config, self.corpus, self.spm_model)

    def args(
        self,
        *extra: str,
        corpora: Sequence[Path] | None = None,
        spm_model: Path | None = None,
        config: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        """The command line `rinchen train docify` with every required option, each
        with the workspace's path unless given, then `extra`."""
        chosen = (self.corpus,) if corpora is None else tuple(corpora)
        return [
            "train",
            "docify",
            *(part for path in chosen for part in ("--corpus", str(path))),
            "--spm-model",
            str(self.spm_model if spm_model is None else spm_model),
            "--config",
            str(self.base_config if config is None else config),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def written(self) -> dict[str, bytes]:
        """Every file under the output directory, by relative path."""
        return {
            path.relative_to(self.out_dir).as_posix(): path.read_bytes()
            for path in sorted(self.out_dir.rglob("*"))
            if path.is_file()
        }


def make_docify_workspace(
    root: Path,
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
    edit_base: Callable[[dict[str, Any]], None] | None = None,
    spm_model: Path | None = None,
) -> DocifyWorkspace:
    """The corpus of `rows` (`hand_built_rows()` by default), the toy SentencePiece model
    (the one given, or trained under `root`), the base config (the tiny teacher's, after
    `edit_base`) and a missing output directory, all under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    corpus = write_corpus_parquet(
        hand_built_rows() if rows is None else rows, root / "in" / "docs.parquet"
    )
    base_config = root / "base.yaml"
    if edit_base is None:
        shutil.copyfile(BASE_CONFIG, base_config)
    else:
        edited_base_config(base_config, edit_base)
    return DocifyWorkspace(
        root=root,
        corpus=corpus,
        spm_model=spm_model if spm_model is not None else train_toy_spm(root / "spm"),
        base_config=base_config,
        out_dir=root / "out",
    )


def toy_document_rows(documents_per_split: int = 6) -> list[dict[str, Any]]:
    """The first `documents_per_split` documents of every language pair of each split of
    `tests.support.tiny_parallel.build_toy_documents`, as rows."""
    toy = build_toy_documents()
    rows: list[dict[str, Any]] = []
    for split_rows in (toy.train, toy.dev, toy.test):
        kept: dict[tuple[str, str], list[str]] = {}
        for row in split_rows:
            pair = tuple(sorted((row["src_lang"], row["tgt_lang"])))
            seen = kept.setdefault((pair[0], pair[1]), [])
            if row["doc_id"] not in seen:
                seen.append(row["doc_id"])
        allowed = {
            doc_id for ids in kept.values() for doc_id in ids[:documents_per_split]
        }
        rows.extend(row for row in split_rows if row["doc_id"] in allowed)
    return rows


def run_ok(args: list[str]) -> Result:
    """Run `rinchen` with `args` and return the result, which must have exited 0."""
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def made_dir(directory: Path) -> Path:
    """`directory`, made (with its parents) if it was not there."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory
