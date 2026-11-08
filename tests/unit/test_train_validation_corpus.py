"""Tests for `rinchen.train.validation_corpus`: the pre-tokenised validation corpus
and the lines each direction occupies in it (D7 spec R7).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import sentencepiece

from rinchen.errors import ConfigError
from rinchen.train.mixture import DevSet
from rinchen.train.validation_corpus import (
    ValidPaths,
    dev_line_ranges,
    write_validation,
)
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN, file_lines
from tests.support.tiny_parallel import build_toy, train_toy_spm


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("mixture_spm"))


# --- write_validation --------------------------------------------------------------


def _dev_set(direction: str, pairs: Sequence[tuple[str, str]]) -> DevSet:
    source, target = direction.split("-")
    return DevSet(
        direction=direction,
        src_lang=source,
        tgt_lang=target,
        ids=tuple(f"id{index}" for index in range(len(pairs))),
        src=tuple(pair[0] for pair in pairs),
        tgt=tuple(pair[1] for pair in pairs),
    )


def _toy_dev() -> dict[str, DevSet]:
    toy = build_toy()
    by_direction: dict[str, list[tuple[str, str]]] = {}
    for row in toy.dev:
        by_direction.setdefault(f"{row['src_lang']}-{row['tgt_lang']}", []).append(
            (row["src"], row["tgt"])
        )
    # Two lines of each direction, entered in an order that is not the sorted one.
    return {
        direction: _dev_set(direction, by_direction[direction][:2])
        for direction in (MN_RU, EN_MN, RU_MN, MN_EN)
    }


def _pieces(spm_model: Path, text: str) -> list[str]:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))
    return list(processor.encode(text, out_type=str))


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_the_validation_lines_are_pre_tokenised_with_the_target_tag_on_one_side(
    tmp_path: Path, toy_spm: Path, placement: str
) -> None:
    dev = _toy_dev()

    paths = write_validation(dev, toy_spm, placement, tmp_path / "out")

    assert paths == ValidPaths(
        tmp_path / "out" / "valid.src", tmp_path / "out" / "valid.tgt"
    )
    expected_src: list[str] = []
    expected_tgt: list[str] = []
    for direction in sorted(dev):
        tag = {
            "khk_Cyrl": "<2khk_Cyrl>",
            "eng_Latn": "<2eng_Latn>",
            "rus_Cyrl": "<2rus_Cyrl>",
        }[dev[direction].tgt_lang]
        for source, target in zip(dev[direction].src, dev[direction].tgt, strict=True):
            source_pieces, target_pieces = (
                _pieces(toy_spm, source),
                _pieces(toy_spm, target),
            )
            if placement == "src":
                expected_src.append(" ".join([tag, *source_pieces]))
                expected_tgt.append(" ".join(target_pieces))
            else:
                expected_src.append(" ".join(source_pieces))
                expected_tgt.append(" ".join([tag, *target_pieces]))
    assert file_lines(paths.src) == expected_src
    assert file_lines(paths.tgt) == expected_tgt
    assert len(expected_src) == 8


def test_the_tag_is_the_first_piece_of_the_source_line_for_the_source_placement(
    tmp_path: Path, toy_spm: Path
) -> None:
    paths = write_validation(
        {EN_MN: _dev_set(EN_MN, [("This is the book.", "Энэ бол ном.")])},
        toy_spm,
        "src",
        tmp_path,
    )

    (source,) = file_lines(paths.src)
    (target,) = file_lines(paths.tgt)
    assert source.split(" ")[0] == "<2khk_Cyrl>"
    assert "<2khk_Cyrl>" not in target.split(" ")
    assert source.split(" ")[1:] == _pieces(toy_spm, "This is the book.")


def test_the_tag_is_the_first_piece_of_the_target_line_for_the_target_placement(
    tmp_path: Path, toy_spm: Path
) -> None:
    paths = write_validation(
        {MN_EN: _dev_set(MN_EN, [("Энэ бол ном.", "This is the book.")])},
        toy_spm,
        "tgt",
        tmp_path,
    )

    (source,) = file_lines(paths.src)
    (target,) = file_lines(paths.tgt)
    assert target.split(" ")[0] == "<2eng_Latn>"
    assert target.split(" ")[1:] == _pieces(toy_spm, "This is the book.")
    assert "<2eng_Latn>" not in source.split(" ")


def test_the_directions_are_concatenated_in_sorted_order_with_their_line_ranges(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = {
        MN_RU: _dev_set(MN_RU, [("Энэ бол ном.", "Это книга.")] * 3),
        EN_MN: _dev_set(EN_MN, [("This is the book.", "Энэ бол ном.")] * 2),
        RU_MN: _dev_set(RU_MN, [("Это книга.", "Энэ бол ном.")]),
    }

    paths = write_validation(dev, toy_spm, "src", tmp_path)

    assert dev_line_ranges(dev) == {
        EN_MN: range(2),
        MN_RU: range(2, 5),
        RU_MN: range(5, 6),
    }
    sources = file_lines(paths.src)
    assert len(sources) == len(file_lines(paths.tgt)) == 6
    assert [line.split(" ")[0] for line in sources] == [
        "<2khk_Cyrl>",
        "<2khk_Cyrl>",
        "<2rus_Cyrl>",
        "<2rus_Cyrl>",
        "<2rus_Cyrl>",
        "<2khk_Cyrl>",
    ]


def test_the_validation_files_are_lf_only_utf8_with_one_line_per_dev_row(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev()

    paths = write_validation(dev, toy_spm, "src", tmp_path)

    for path in (paths.src, paths.tgt):
        data = path.read_bytes()
        assert data.endswith(b"\n")
        assert b"\r" not in data
        assert data.count(b"\n") == sum(len(d.src) for d in dev.values())
        data.decode("utf-8")


def test_a_line_is_pieces_joined_by_single_spaces_only(
    tmp_path: Path, toy_spm: Path
) -> None:
    paths = write_validation(_toy_dev(), toy_spm, "tgt", tmp_path)

    for line in (*file_lines(paths.src), *file_lines(paths.tgt)):
        assert "  " not in line
        assert line == line.strip(" ")
        assert all(piece for piece in line.split(" "))


def test_the_output_directory_is_created_and_may_hold_glob_characters(
    tmp_path: Path, toy_spm: Path
) -> None:
    paths = write_validation(
        _toy_dev(), toy_spm, "src", tmp_path / "out[1]*?" / "valid"
    )

    assert paths.src.parent == tmp_path / "out[1]*?" / "valid"
    assert paths.src.is_file()


def test_a_direction_tag_that_is_not_a_piece_of_the_model_is_a_config_error(
    tmp_path: Path,
) -> None:
    text = tmp_path / "plain.txt"
    text.write_text(
        "".join(f"line {i} of some text\n" for i in range(200)), encoding="utf-8"
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(text),
        model_prefix=str(tmp_path / "plain"),
        vocab_size=40,
        minloglevel=2,
    )

    with pytest.raises(ConfigError, match=r"<2khk_Cyrl>"):
        write_validation(
            {EN_MN: _dev_set(EN_MN, [("a", "b")])},
            tmp_path / "plain.model",
            "src",
            tmp_path / "out",
        )


@pytest.mark.parametrize("what", ["missing", "directory", "not a model"])
def test_a_model_that_cannot_be_loaded_is_a_config_error(
    tmp_path: Path, what: str
) -> None:
    model = tmp_path / "model"
    if what == "directory":
        model.mkdir()
    elif what == "not a model":
        model.write_bytes(b"not a model")

    with pytest.raises(ConfigError, match="SentencePiece model"):
        write_validation(
            {EN_MN: _dev_set(EN_MN, [("a", "b")])}, model, "src", tmp_path / "out"
        )


def test_an_unknown_placement_or_no_dev_set_is_a_value_error(
    tmp_path: Path, toy_spm: Path
) -> None:
    placement: Any = "suffix"
    with pytest.raises(ValueError, match="placement"):
        write_validation(_toy_dev(), toy_spm, placement, tmp_path)
    with pytest.raises(ValueError, match="dev"):
        write_validation({}, toy_spm, "src", tmp_path)


def test_the_line_ranges_of_no_dev_sets_are_empty() -> None:
    assert dev_line_ranges({}) == {}
