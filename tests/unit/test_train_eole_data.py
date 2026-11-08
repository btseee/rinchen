"""Tests for `rinchen.train.eole_data`: the Eole `data` section of a variant.

The sections are worked out by hand from the module docstring and the D7 spec (R6,
finding 4), not from the code.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError
from rinchen.train.corpus_files import (
    CorpusFiles,
    build_corpora,
    planned_corpus_files,
)
from rinchen.train.eole_cfg import merge, read_config, validate_train_config
from rinchen.train.eole_data import data_section
from rinchen.train.sampling import integer_weights, mixing_probabilities
from rinchen.train.validation_corpus import ValidPaths
from tests.support.corpus_rows import (
    EN_MN,
    MN_EN,
    MN_RU,
    RU_MN,
    sizes_of,
    toy_training_rows,
)

TINY_BASE = REPO_ROOT / "configs" / "eole" / "multilingual_tiny.yaml"


# --- data_section ------------------------------------------------------------------


def _files(direction: str, group: str, *, directory: str = "data/auth") -> CorpusFiles:
    source, target = direction.split("-")
    name = f"{direction}.{group}"
    return CorpusFiles(
        name=name,
        direction=direction,
        group=group,
        src_lang=source,
        tgt_lang=target,
        src_path=Path(f"{directory}/{name}.src"),
        tgt_path=Path(f"{directory}/{name}.tgt"),
        rows=10,
        src_sha256="0" * 64,
        tgt_sha256="1" * 64,
        reused=False,
    )


_VALID = ValidPaths(src=Path("variants/v/valid.src"), tgt=Path("variants/v/valid.tgt"))


def test_the_source_placement_puts_the_target_tag_in_the_source_prefix() -> None:
    section = data_section(
        [_files(EN_MN, "auth")],
        {f"{EN_MN}.auth": 7},
        "src",
        valid_paths=_VALID,
        filtertoolong=False,
    )

    assert section == {
        f"{EN_MN}.auth": {
            "path_src": f"data/auth/{EN_MN}.auth.src",
            "path_tgt": f"data/auth/{EN_MN}.auth.tgt",
            "transforms": ["sentencepiece", "prefix"],
            "weight": 7,
            "src_prefix": "<2khk_Cyrl>",
            "tgt_prefix": "",
        },
        "valid": {
            "path_src": "variants/v/valid.src",
            "path_tgt": "variants/v/valid.tgt",
        },
    }


def test_the_source_placement_puts_the_bt_tag_after_the_direction_tag() -> None:
    section = data_section(
        [_files(EN_MN, "auth"), _files(EN_MN, "bt", directory="data/bt/r1_s1")],
        {f"{EN_MN}.auth": 3, f"{EN_MN}.bt": 1},
        "src",
        valid_paths=_VALID,
        filtertoolong=False,
    )

    assert section[f"{EN_MN}.bt"] == {
        "path_src": f"data/bt/r1_s1/{EN_MN}.bt.src",
        "path_tgt": f"data/bt/r1_s1/{EN_MN}.bt.tgt",
        "transforms": ["sentencepiece", "prefix"],
        "weight": 1,
        "src_prefix": "<2khk_Cyrl> <bt>",
        "tgt_prefix": "",
    }
    assert list(section) == [f"{EN_MN}.auth", f"{EN_MN}.bt", "valid"]


def test_the_target_placement_puts_the_tag_in_the_target_prefix_and_bt_alone_in_the_source() -> (
    None
):
    section = data_section(
        [_files(EN_MN, "auth"), _files(EN_MN, "bt", directory="data/bt/r1_s1")],
        {f"{EN_MN}.auth": 3, f"{EN_MN}.bt": 1},
        "tgt",
        valid_paths=_VALID,
        filtertoolong=False,
    )

    assert section[f"{EN_MN}.auth"]["src_prefix"] == ""
    assert section[f"{EN_MN}.auth"]["tgt_prefix"] == "<2khk_Cyrl>"
    assert section[f"{EN_MN}.bt"]["src_prefix"] == "<bt>"
    assert section[f"{EN_MN}.bt"]["tgt_prefix"] == "<2khk_Cyrl>"


@pytest.mark.parametrize(
    ("direction", "tag"),
    [(EN_MN, "<2khk_Cyrl>"), (MN_EN, "<2eng_Latn>"), (MN_RU, "<2rus_Cyrl>")],
)
def test_the_tag_is_that_of_the_target_language_of_the_corpus(
    direction: str, tag: str
) -> None:
    section = data_section(
        [_files(direction, "auth")],
        {f"{direction}.auth": 1},
        "src",
        valid_paths=_VALID,
        filtertoolong=False,
    )

    assert section[f"{direction}.auth"]["src_prefix"] == tag


def test_filtertoolong_is_added_after_prefix_only_when_asked() -> None:
    corpora = [_files(EN_MN, "auth")]
    weights = {f"{EN_MN}.auth": 1}

    with_filter = data_section(
        corpora, weights, "src", valid_paths=_VALID, filtertoolong=True
    )
    without = data_section(
        corpora, weights, "src", valid_paths=_VALID, filtertoolong=False
    )

    assert with_filter[f"{EN_MN}.auth"]["transforms"] == [
        "sentencepiece",
        "prefix",
        "filtertoolong",
    ]
    assert without[f"{EN_MN}.auth"]["transforms"] == ["sentencepiece", "prefix"]


def test_the_valid_entry_carries_the_two_paths_and_nothing_else() -> None:
    section = data_section(
        [_files(EN_MN, "auth")],
        {f"{EN_MN}.auth": 1},
        "tgt",
        valid_paths=_VALID,
        filtertoolong=True,
    )

    assert set(section["valid"]) == {"path_src", "path_tgt"}


def test_the_weights_are_the_integers_given_whatever_the_corpus_sizes() -> None:
    corpora = [_files(EN_MN, "auth"), _files(RU_MN, "auth")]

    section = data_section(
        corpora,
        {f"{EN_MN}.auth": 61, f"{RU_MN}.auth": 39},
        "src",
        valid_paths=_VALID,
        filtertoolong=False,
    )

    assert [section[name]["weight"] for name in (f"{EN_MN}.auth", f"{RU_MN}.auth")] == [
        61,
        39,
    ]
    assert all(
        type(section[n]["weight"]) is int for n in (f"{EN_MN}.auth", f"{RU_MN}.auth")
    )


@pytest.mark.parametrize("weight", [0, -1, 1.0, True, "3", None])
def test_a_weight_that_is_not_a_positive_integer_is_a_value_error(weight: Any) -> None:
    with pytest.raises(ValueError, match=rf"corpus.*{EN_MN}\.auth.*positive integer"):
        data_section(
            [_files(EN_MN, "auth")],
            {f"{EN_MN}.auth": weight},
            "src",
            valid_paths=_VALID,
            filtertoolong=False,
        )


def test_a_corpus_without_a_weight_is_a_value_error() -> None:
    with pytest.raises(ValueError, match=rf"no weight for corpus.*{RU_MN}\.auth"):
        data_section(
            [_files(EN_MN, "auth"), _files(RU_MN, "auth")],
            {f"{EN_MN}.auth": 1},
            "src",
            valid_paths=_VALID,
            filtertoolong=False,
        )


def test_a_weight_for_a_corpus_that_is_not_there_is_a_value_error() -> None:
    with pytest.raises(ValueError, match=r"not in the mixture.*typo\.auth"):
        data_section(
            [_files(EN_MN, "auth")],
            {f"{EN_MN}.auth": 1, "typo.auth": 1},
            "src",
            valid_paths=_VALID,
            filtertoolong=False,
        )


@pytest.mark.parametrize("placement", ["suffix", "", "SRC", None])
def test_a_placement_other_than_src_or_tgt_is_a_value_error(placement: Any) -> None:
    with pytest.raises(ValueError, match="placement"):
        data_section(
            [_files(EN_MN, "auth")],
            {f"{EN_MN}.auth": 1},
            placement,
            valid_paths=_VALID,
            filtertoolong=False,
        )


def test_no_corpus_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="corpus"):
        data_section([], {}, "src", valid_paths=_VALID, filtertoolong=False)


@pytest.mark.parametrize(
    "where", ["corpus src", "corpus tgt", "valid src", "valid tgt"]
)
def test_a_dollar_sign_in_any_path_is_a_config_error(where: str) -> None:
    corpus = _files(EN_MN, "auth")
    if where == "corpus src":
        corpus = replace(corpus, src_path=Path("data/$HOME/a.src"))
    elif where == "corpus tgt":
        corpus = replace(corpus, tgt_path=Path("data/${HOME}/a.tgt"))
    valid = {
        "valid src": ValidPaths(Path("$V/valid.src"), Path("v/valid.tgt")),
        "valid tgt": ValidPaths(Path("v/valid.src"), Path("$V/valid.tgt")),
    }.get(where, _VALID)

    with pytest.raises(ConfigError, match=r"'\$'"):
        data_section(
            [corpus],
            {corpus.name: 1},
            "src",
            valid_paths=valid,
            filtertoolong=False,
        )


def test_two_corpora_of_one_name_are_a_value_error() -> None:
    corpus = _files(EN_MN, "auth")

    with pytest.raises(ValueError, match=rf"unique.*{EN_MN}\.auth"):
        data_section(
            [corpus, corpus],
            {corpus.name: 1},
            "src",
            valid_paths=_VALID,
            filtertoolong=False,
        )


def test_paths_are_written_as_posix_strings_as_given() -> None:
    corpus = _files(EN_MN, "auth", directory="/abs/data[1]/auth")

    section = data_section(
        [corpus], {corpus.name: 1}, "src", valid_paths=_VALID, filtertoolong=False
    )

    assert section[corpus.name]["path_src"] == f"/abs/data[1]/auth/{corpus.name}.src"


def _validated(section: dict[str, Any]) -> dict[str, Any]:
    base = read_config(TINY_BASE, "tiny base")
    config = merge(
        base,
        {
            "data": section,
            "transforms_configs": {
                "sentencepiece": {
                    "src_subword_model": "spm/toy.model",
                    "tgt_subword_model": "spm/toy.model",
                }
            },
            "src_vocab": "run/vocab.src",
            "training": {"model_path": "run/ckpt"},
        },
    )
    validate_train_config(config)
    return config


@pytest.mark.parametrize("placement", ["src", "tgt"])
@pytest.mark.parametrize("with_bt", [False, True])
def test_the_section_merged_into_the_tiny_base_passes_the_config_validator(
    placement: str, with_bt: bool
) -> None:
    corpora = [_files(EN_MN, "auth"), _files(MN_RU, "auth")]
    if with_bt:
        corpora += [_files(EN_MN, "bt", directory="data/bt/r1_s1")]
    weights = {corpus.name: index + 1 for index, corpus in enumerate(corpora)}

    section = data_section(
        corpora, weights, placement, valid_paths=_VALID, filtertoolong=True
    )

    config = _validated(section)
    assert config["data"]["valid"] == {
        "path_src": "variants/v/valid.src",
        "path_tgt": "variants/v/valid.tgt",
    }


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_the_files_a_build_would_write_give_the_section_the_written_files_give(
    tmp_path: Path, placement: Any
) -> None:
    training = toy_training_rows(tmp_path)
    data_dir = tmp_path / "data"
    planned = planned_corpus_files(training, 0.2, 1, data_dir)
    weights = {file.name: index + 1 for index, file in enumerate(planned)}

    built = build_corpora(training, bt_ratio=0.2, seed=1, data_dir=data_dir)

    assert data_section(
        planned, weights, placement, valid_paths=_VALID, filtertoolong=True
    ) == data_section(
        built.corpora, weights, placement, valid_paths=_VALID, filtertoolong=True
    )


def test_the_toy_corpora_weighted_by_temperature_make_a_valid_config(
    tmp_path: Path,
) -> None:
    training = toy_training_rows(tmp_path)
    built = build_corpora(training, bt_ratio=0.2, seed=1, data_dir=tmp_path / "data")
    probabilities = mixing_probabilities(sizes_of(built.corpora), 0.5)
    weights = integer_weights(probabilities, total=100)

    section = data_section(
        built.corpora, weights, "tgt", valid_paths=_VALID, filtertoolong=True
    )

    config = _validated(section)
    assert (
        sum(
            entry["weight"] for name, entry in config["data"].items() if name != "valid"
        )
        == 100
    )
    assert len(config["data"]) == 9
