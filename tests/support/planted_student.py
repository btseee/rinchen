"""The inputs and helpers of the tests of `rinchen.train.distill_prepare`: the toy corpora, a
KD file made of the authentic rows relabelled as forward translations by an oracle teacher,
three SentencePiece models (640 pieces, 500 pieces, and one without the tag `<ft>`) and the
expected numbers of the toy.

At temperature 0.5, a back-translation ratio of 0.5 and weight 100 D7 weighs the original
corpora 22 and 6 (en-mn, mn-en), 15 and 7 (mn-ru, ru-mn): the spike's numbers; the KD corpus
of a direction weighs the sum of those for `1:1`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece
import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.corpus.schema import corpus_frame
from rinchen.train.distill_mix import Mix
from rinchen.train.distill_prepare import PreparedStudent, prepare_student
from rinchen.train.eole_cfg import read_config
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN
from tests.support.planted_kd import write_kd_files
from tests.support.tiny_parallel import (
    build_toy,
    train_toy_spm,
    write_corpus_parquet,
)

BASE = REPO_ROOT / "configs" / "eole" / "student_tiny.yaml"
W = chr(0x2581)
DIRECTIONS = (EN_MN, MN_EN, MN_RU, RU_MN)
BT_DIRECTORY = "data/bt/r0.5_s1234"

# D7's weights of the toy at temperature 0.5, bt ratio 0.5, weight total 100.
ORIGINAL_WEIGHTS = {
    f"{EN_MN}.auth": 22,
    f"{EN_MN}.bt": 6,
    f"{MN_EN}.auth": 22,
    f"{MN_EN}.bt": 6,
    f"{MN_RU}.auth": 15,
    f"{MN_RU}.bt": 7,
    f"{RU_MN}.auth": 15,
    f"{RU_MN}.bt": 7,
}
KD_WEIGHTS = {
    f"{EN_MN}.kd": 28,
    f"{MN_EN}.kd": 28,
    f"{MN_RU}.kd": 22,
    f"{RU_MN}.kd": 22,
}
KD_ROWS = {
    f"{EN_MN}.kd": 1200,
    f"{MN_EN}.kd": 1200,
    f"{MN_RU}.kd": 600,
    f"{RU_MN}.kd": 600,
}


@dataclass(frozen=True)
class Inputs:
    corpus: Path
    kd: Path
    spm: Path
    spm500: Path
    no_ft_model: Path


def _model_without_ft(directory: Path) -> Path:
    """A SentencePiece model with the three direction tags and `<bt>` but not `<ft>`."""
    directory.mkdir(parents=True, exist_ok=True)
    text = directory / "text.txt"
    text.write_text(
        "".join(f"{row['src']}\n{row['tgt']}\n" for row in build_toy().train[:600]),
        encoding="utf-8",
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(text),
        model_prefix=str(directory / "noft"),
        vocab_size=200,
        user_defined_symbols=["<2eng_Latn>", "<2khk_Cyrl>", "<2rus_Cyrl>", "<bt>"],
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return directory / "noft.model"


def build_inputs(root: Path) -> Inputs:
    """The corpus file, the KD file and the three models, written under `root`."""
    toy = build_toy()
    corpus = write_corpus_parquet([*toy.train, *toy.bt, *toy.dev], root / "toy.parquet")
    oracle = [{**row, "origin": "ft", "corpus": "ft:toy"} for row in toy.train]
    kd = write_kd_files(root / "kd", corpus_frame(oracle))
    return Inputs(
        corpus=corpus,
        kd=kd,
        spm=train_toy_spm(root / "spm640"),
        spm500=train_toy_spm(root / "spm500", vocab_size=500),
        no_ft_model=_model_without_ft(root / "noft"),
    )


def prepare(inputs: Inputs, out_dir: Path, **overrides: Any) -> PreparedStudent:
    """`prepare_student` over the toy inputs with the standard options; the keyword arguments
    replace any of them."""
    arguments: dict[str, Any] = {
        "corpora": [inputs.corpus],
        "kd_files": [inputs.kd],
        "spm_model": inputs.spm,
        "base": BASE,
        "out_dir": out_dir,
        "mix": Mix(1, 1),
        "kd_tag": True,
        "placement": "src",
        "bt_ratio": 0.5,
        "temperature": 0.5,
        "seed": 1234,
        "dev_max_lines": 100,
        "weight_total": 100,
        "dry_run": False,
    } | overrides
    return prepare_student(**arguments)


def config_of(directory: Path) -> dict[str, Any]:
    return yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_under(directory: Path) -> list[str]:
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    )


def base_with(tmp_path: Path, **changes: Any) -> Path:
    """`tmp_path/base.yaml`: the tiny student base with `changes` made (a key of
    `training` for `train_from` and `reset_optim`, a top-level key otherwise)."""
    base = read_config(BASE, "base")
    for key, value in changes.items():
        if key in ("train_from", "reset_optim"):
            base["training"][key] = value
        else:
            base[key] = value
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path
