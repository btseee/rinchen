"""Shared setup of the `rinchen train ablate` tests: a workspace holding the toy corpus
as Parquet, the toy SentencePiece model, a matrix file over a copy of the tiny base
config, and the output directory; and the CLI's two seams (`run_training` and
`evaluate_variant`) replaced by the fakes of `tests.support.fake_ablation`, so that no
Eole process starts.

The matrix is the one of `fake_ablation.make_matrix`: two variants, one per tag
placement, at temperature 0.5 and bt ratio 0.5 with seed 1234 (`ID_SRC`, `ID_TGT`).
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.cli import train as train_cli
from tests.support.fake_ablation import (
    BASE_CONFIG,
    ID_SRC,
    ID_TGT,
    FakeEvaluate,
    FakeTrain,
    make_matrix,
    write_toy_corpus,
)
from tests.support.tiny_parallel import train_toy_spm

VARIANT_IDS = (ID_SRC, ID_TGT)
RESULT_FILE_NAMES = ("results.tsv", "results.json", "results.md")
SEED = 1234


def write_matrix_file(path: Path, base: Path, **fields: Any) -> Path:
    """A matrix file at `path` over the base config `base`: the matrix of
    `make_matrix` with `fields` replacing whole fields."""
    document = make_matrix(**fields).model_dump(mode="json")
    document["base"] = str(base)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


@dataclass(frozen=True, slots=True)
class Workspace:
    """The files of one `rinchen train ablate` test. `corpus`, `spm_model`, `matrix`
    and `base_config` exist; `out_dir` does not. The base config is a copy of the
    tiny one, so that a test may change it."""

    root: Path
    corpus: Path
    spm_model: Path
    matrix: Path
    base_config: Path
    out_dir: Path

    @property
    def manifest(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def inputs(self) -> tuple[Path, Path, Path, Path]:
        """The matrix, the base config, the corpus and the SentencePiece model: the
        files a run hashes, in the order of its manifest."""
        return (self.matrix, self.base_config, self.corpus, self.spm_model)

    def result_files(self) -> list[Path]:
        return [self.out_dir / name for name in RESULT_FILE_NAMES]

    def variant_dir(self, variant_id: str) -> Path:
        return self.out_dir / "variants" / variant_id

    def config_file(self, variant_id: str) -> Path:
        return self.variant_dir(variant_id) / "config.yaml"

    def result_file(self, variant_id: str) -> Path:
        return self.variant_dir(variant_id) / "result.json"

    def train_record(self, variant_id: str) -> Path:
        return self.variant_dir(variant_id) / "train.json"

    def checkpoint(self, variant_id: str) -> Path:
        return self.variant_dir(variant_id) / "run" / "ckpt" / "model.00.safetensors"

    def args(
        self,
        *extra: str,
        matrix: Path | None = None,
        corpora: tuple[Path, ...] | None = None,
        spm_model: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        """The command line: every option, each with the workspace's path unless
        given."""
        chosen = (self.corpus,) if corpora is None else corpora
        return [
            "train",
            "ablate",
            "--matrix",
            str(self.matrix if matrix is None else matrix),
            *(part for path in chosen for part in ("--corpus", str(path))),
            "--spm-model",
            str(self.spm_model if spm_model is None else spm_model),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]


def make_workspace(root: Path, **matrix_fields: Any) -> Workspace:
    """The toy corpus, the toy SentencePiece model, the base config and the matrix
    file under `root` (made if needed), and an output directory that is not made."""
    root.mkdir(parents=True, exist_ok=True)
    corpus = write_toy_corpus(root / "in")
    base_config = root / "base.yaml"
    shutil.copyfile(BASE_CONFIG, base_config)
    return Workspace(
        root=root,
        corpus=corpus,
        spm_model=train_toy_spm(root / "spm"),
        matrix=write_matrix_file(root / "matrix.yaml", base_config, **matrix_fields),
        base_config=base_config,
        out_dir=root / "out",
    )


@dataclass(slots=True)
class Seams:
    """The fakes behind the CLI's two seams."""

    train: FakeTrain
    evaluate: FakeEvaluate


def install_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    train: FakeTrain | None = None,
    evaluate: FakeEvaluate | None = None,
) -> Seams:
    """Make `rinchen train ablate` train and evaluate through `train` and `evaluate`
    (fresh fakes unless given), so that no Eole process starts."""
    seams = Seams(
        train=train if train is not None else FakeTrain(),
        evaluate=evaluate if evaluate is not None else FakeEvaluate(),
    )
    monkeypatch.setattr(train_cli, "run_training", seams.train)
    monkeypatch.setattr(train_cli, "evaluate_variant", seams.evaluate)
    return seams


def spelling_of_one_place(
    workspace: Workspace, root: Path
) -> Mapping[str, tuple[Path, Path]]:
    """Ways to name the corpus file and an output directory that is its directory:
    (out_dir, corpus). `root` holds the helper directories and links."""
    in_dir = workspace.corpus.parent
    (root / "x").mkdir()
    alias = root / "alias"
    alias.symlink_to(in_dir, target_is_directory=True)
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "link.parquet").symlink_to(workspace.corpus)
    beside = root / "beside"
    beside.mkdir()
    (beside / "link.parquet").symlink_to(workspace.corpus)
    return {
        "plain": (in_dir, workspace.corpus),
        "dotdot": (root / "x" / ".." / "in", workspace.corpus),
        "symlinked-out-dir": (alias, workspace.corpus),
        "input-behind-a-link": (in_dir, elsewhere / "link.parquet"),
        "link-to-the-input-in-the-out-dir": (beside, beside / "link.parquet"),
    }
