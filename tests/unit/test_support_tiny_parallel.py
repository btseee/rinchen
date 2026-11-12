"""The toy parallel data describes itself truthfully: the sizes it documents, the
three languages being parallel, no sentence shared between training, the
back-translated group, dev and test, the Parquet files it writes and the
SentencePiece model it trains.

The three-way check parses every sentence back into its template and nouns with
regular expressions written here, not with the generator's own rendering, so a
template that drifts in one language is caught.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import pytest
import sentencepiece

from rinchen.config.loader import REPO_ROOT
from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.schema import CORPUS_SCHEMA, ORIGINS, SPLITS, make_row_id
from rinchen.errors import UpstreamError
from tests.support.tiny_parallel import (
    NOUNS,
    TOY_DIRECTIONS,
    TOY_SPM_VOCAB_RANGE,
    TOY_SPM_VOCAB_SIZE,
    ToyData,
    build_toy,
    train_toy_spm,
    write_corpus_parquet,
)

_TAGS = ("<2khk_Cyrl>", "<2eng_Latn>", "<2rus_Cyrl>", "<bt>")


@pytest.fixture(scope="module")
def toy() -> ToyData:
    return build_toy()


@pytest.fixture(scope="module")
def spm_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("toy_spm"))


def _by_direction(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['src_lang']}-{row['tgt_lang']}"].append(row)
    return dict(grouped)


def _sentences_by_language(rows: Sequence[dict[str, str]]) -> dict[str, set[str]]:
    sentences: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sentences[row["src_lang"]].add(row["src"])
        sentences[row["tgt_lang"]].add(row["tgt"])
    return dict(sentences)


# --- the table and the sizes ------------------------------------------------------


def test_the_noun_table_has_37_distinct_nouns_in_each_language() -> None:
    assert len(NOUNS) == 37
    for column in range(3):
        assert len({noun[column] for noun in NOUNS}) == 37


def test_the_four_directions_are_the_two_pairs_both_ways() -> None:
    assert TOY_DIRECTIONS == (
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-eng_Latn",
        "rus_Cyrl-khk_Cyrl",
        "khk_Cyrl-rus_Cyrl",
    )


def test_the_training_pairs_have_different_sizes_for_the_two_language_pairs(
    toy: ToyData,
) -> None:
    sizes = {d: len(rows) for d, rows in _by_direction(toy.train).items()}

    assert sizes == {
        "eng_Latn-khk_Cyrl": 1200,
        "khk_Cyrl-eng_Latn": 1200,
        "rus_Cyrl-khk_Cyrl": 600,
        "khk_Cyrl-rus_Cyrl": 600,
    }


def test_the_back_translated_group_has_300_rows_per_direction(toy: ToyData) -> None:
    sizes = {d: len(rows) for d, rows in _by_direction(toy.bt).items()}

    assert sizes == dict.fromkeys(TOY_DIRECTIONS, 300)


@pytest.mark.parametrize("split", ["dev", "test"])
def test_the_held_out_splits_have_100_rows_per_direction(
    toy: ToyData, split: str
) -> None:
    rows = getattr(toy, split)

    sizes = {d: len(group) for d, group in _by_direction(rows).items()}

    assert sizes == dict.fromkeys(TOY_DIRECTIONS, 100)


def test_every_row_has_the_keys_of_a_corpus_row_and_its_group_labels(
    toy: ToyData,
) -> None:
    expected_keys = {"src_lang", "tgt_lang", "src", "tgt", "origin", "corpus", "split"}
    groups = {
        "train": (toy.train, "human", "train"),
        "bt": (toy.bt, "bt", "train"),
        "dev": (toy.dev, "human", "dev"),
        "test": (toy.test, "human", "test"),
    }

    for rows, origin, split in groups.values():
        assert all(set(row) == expected_keys for row in rows)
        assert {row["origin"] for row in rows} == {origin}
        assert {row["split"] for row in rows} == {split}
    assert {origin for rows, origin, _ in groups.values()} <= set(ORIGINS)
    assert {split for _, _, split in groups.values()} <= set(SPLITS)


# --- determinism and the seed -----------------------------------------------------


def test_the_same_seed_gives_the_same_data() -> None:
    assert build_toy(seed=7) == build_toy(seed=7)


def test_another_seed_gives_other_training_and_held_out_sentences() -> None:
    first, second = build_toy(seed=7), build_toy(seed=8)

    assert first.train != second.train
    assert first.dev != second.dev
    assert first.bt != second.bt


# --- no overlap -------------------------------------------------------------------


def test_no_sentence_is_in_two_of_training_back_translation_dev_and_test(
    toy: ToyData,
) -> None:
    by_group = {
        "train": _sentences_by_language(toy.train),
        "bt": _sentences_by_language(toy.bt),
        "dev": _sentences_by_language(toy.dev),
        "test": _sentences_by_language(toy.test),
    }
    names = list(by_group)

    for language in ("khk_Cyrl", "eng_Latn", "rus_Cyrl"):
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                shared = by_group[first][language] & by_group[second][language]
                assert not shared, (language, first, second, sorted(shared)[:3])


def test_the_two_training_language_pairs_share_no_mongolian_sentence(
    toy: ToyData,
) -> None:
    pairs = _by_direction(toy.train)
    with_english = {row["tgt"] for row in pairs["eng_Latn-khk_Cyrl"]}
    with_russian = {row["tgt"] for row in pairs["rus_Cyrl-khk_Cyrl"]}

    assert len(with_english) == 1200
    assert len(with_russian) == 600
    assert not with_english & with_russian


def test_no_training_sentence_repeats_within_a_direction(toy: ToyData) -> None:
    for rows in (toy.train, toy.bt, toy.dev, toy.test):
        for direction, group in _by_direction(rows).items():
            assert len({row["src"] for row in group}) == len(group), direction
            assert len({row["tgt"] for row in group}) == len(group), direction


# --- the three languages are parallel ---------------------------------------------

_NOUN_BY_TEXT = [
    {noun[column]: index for index, noun in enumerate(NOUNS)} for column in range(3)
]
_WORD = r"(\w+)"
# One pattern per template and language, built by hand from the documented templates;
# the capital letter of a sentence's first noun is undone before the lookup.
_PATTERNS: list[list[tuple[int, re.Pattern[str]]]] = [
    [  # khk_Cyrl
        (1, re.compile(rf"Энэ бол {_WORD}\.")),
        (2, re.compile(rf"{_WORD} энд байна\.")),
        (3, re.compile(rf"{_WORD} хаана байна вэ\?")),
        (4, re.compile(rf"Тэнд {_WORD} байна\.")),
        (5, re.compile(rf"{_WORD} ба {_WORD}\.")),
        (6, re.compile(rf"{_WORD}, {_WORD} ба {_WORD}\.")),
    ],
    [  # eng_Latn
        (1, re.compile(rf"This is the {_WORD}\.")),
        (2, re.compile(rf"The {_WORD} is here\.")),
        (3, re.compile(rf"Where is the {_WORD}\?")),
        (4, re.compile(rf"The {_WORD} is there\.")),
        (5, re.compile(rf"The {_WORD} and the {_WORD}\.")),
        (6, re.compile(rf"The {_WORD}, the {_WORD} and the {_WORD}\.")),
    ],
    [  # rus_Cyrl
        (1, re.compile(rf"Это {_WORD}\.")),
        (2, re.compile(rf"{_WORD} здесь\.")),
        (3, re.compile(rf"Где {_WORD}\?")),
        (4, re.compile(rf"Там {_WORD}\.")),
        (5, re.compile(rf"{_WORD} и {_WORD}\.")),
        (6, re.compile(rf"{_WORD}, {_WORD} и {_WORD}\.")),
    ],
]
_LANGUAGE_INDEX = {"khk_Cyrl": 0, "eng_Latn": 1, "rus_Cyrl": 2}


def _parse(language: str, sentence: str) -> tuple[int, tuple[int, ...]]:
    """The (template, noun indexes) a sentence says, or a failed assertion."""
    column = _LANGUAGE_INDEX[language]
    for template, pattern in _PATTERNS[column]:
        match = pattern.fullmatch(sentence)
        if match:
            words = [word[0].lower() + word[1:] for word in match.groups()]
            return template, tuple(_NOUN_BY_TEXT[column][word] for word in words)
    raise AssertionError(f"{language} sentence matches no template: {sentence!r}")


def test_every_sentence_parses_into_one_of_the_six_templates(toy: ToyData) -> None:
    templates = Counter()
    for row in (*toy.train, *toy.bt, *toy.dev, *toy.test):
        templates[_parse(row["src_lang"], row["src"])[0]] += 1
        _parse(row["tgt_lang"], row["tgt"])

    assert set(templates) == {1, 2, 3, 4, 5, 6}


@pytest.mark.parametrize("split", ["dev", "test"])
def test_the_held_out_items_say_the_same_in_the_three_languages(
    toy: ToyData, split: str
) -> None:
    by_direction = _by_direction(getattr(toy, split))
    english = by_direction["eng_Latn-khk_Cyrl"]
    russian = by_direction["rus_Cyrl-khk_Cyrl"]
    from_mongolian_to_english = by_direction["khk_Cyrl-eng_Latn"]
    from_mongolian_to_russian = by_direction["khk_Cyrl-rus_Cyrl"]

    for position in range(100):
        meaning = _parse("khk_Cyrl", english[position]["tgt"])
        assert _parse("eng_Latn", english[position]["src"]) == meaning
        assert _parse("rus_Cyrl", russian[position]["src"]) == meaning
        assert _parse("khk_Cyrl", russian[position]["tgt"]) == meaning
        assert from_mongolian_to_english[position]["src"] == english[position]["tgt"]
        assert from_mongolian_to_english[position]["tgt"] == english[position]["src"]
        assert from_mongolian_to_russian[position]["src"] == russian[position]["tgt"]
        assert from_mongolian_to_russian[position]["tgt"] == russian[position]["src"]


@pytest.mark.parametrize("group", ["train", "bt"])
def test_each_pair_of_a_training_row_says_the_same_in_both_languages(
    toy: ToyData, group: str
) -> None:
    for row in getattr(toy, group):
        assert _parse(row["src_lang"], row["src"]) == _parse(
            row["tgt_lang"], row["tgt"]
        )


def test_each_training_sentence_is_in_the_reverse_direction_too(toy: ToyData) -> None:
    by_direction = _by_direction(toy.train)
    forward = by_direction["eng_Latn-khk_Cyrl"]
    backward = by_direction["khk_Cyrl-eng_Latn"]

    assert {(r["src"], r["tgt"]) for r in forward} == {
        (r["tgt"], r["src"]) for r in backward
    }


def test_the_held_out_items_cover_singles_pairs_and_triples(toy: ToyData) -> None:
    for rows in (toy.dev, toy.test):
        group = _by_direction(rows)["khk_Cyrl-eng_Latn"]
        nouns_per_sentence = Counter(
            len(_parse("khk_Cyrl", row["src"])[1]) for row in group
        )

        assert nouns_per_sentence == {1: 12, 2: 44, 3: 44}


# --- the Parquet writer -----------------------------------------------------------


def test_the_parquet_file_is_a_valid_corpus_table_of_the_rows(
    toy: ToyData, tmp_path: Path
) -> None:
    path = write_corpus_parquet(toy.dev, tmp_path / "dev.parquet")

    frame = scan_corpus(path).collect()

    assert path == tmp_path / "dev.parquet"
    assert frame.schema == CORPUS_SCHEMA
    assert frame.height == 400
    first = toy.dev[0]
    assert frame["id"][0] == make_row_id(
        first["src_lang"], first["tgt_lang"], first["src"], first["tgt"]
    )
    assert frame["src"].to_list() == [row["src"] for row in toy.dev]
    assert frame["split"].cast(pl.String).unique().to_list() == ["dev"]
    assert frame["origin"].cast(pl.String).unique().to_list() == ["human"]
    assert frame["filter_reason"].null_count() == frame.height


def test_the_parquet_file_is_the_same_on_every_write(
    toy: ToyData, tmp_path: Path
) -> None:
    first = write_corpus_parquet(toy.bt, tmp_path / "first.parquet")
    second = write_corpus_parquet(toy.bt, tmp_path / "second.parquet")

    assert first.read_bytes() == second.read_bytes()


def test_the_writer_makes_the_directory_and_accepts_any_row_list(
    toy: ToyData, tmp_path: Path
) -> None:
    path = write_corpus_parquet(toy.train[:3], tmp_path / "a[1]" / "tiny.parquet")

    assert scan_corpus(path).collect().height == 3


# --- the SentencePiece model ------------------------------------------------------


def test_every_direction_tag_and_the_back_translation_tag_is_one_piece(
    spm_model: Path,
) -> None:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))

    for tag in _TAGS:
        piece_id = processor.piece_to_id(tag)
        assert piece_id != processor.unk_id(), tag
        assert processor.id_to_piece(piece_id) == tag
        # Atomic in text too: SentencePiece never splits a user-defined symbol.
        assert tag in processor.encode(f"a{tag}b", out_type=str), tag


def test_the_model_has_the_documented_vocabulary_size(spm_model: Path) -> None:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))

    assert TOY_SPM_VOCAB_SIZE == processor.get_piece_size()


def test_the_model_covers_every_toy_sentence_without_unknown_pieces(
    toy: ToyData, spm_model: Path
) -> None:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))
    unknown = processor.unk_id()

    for row in (*toy.dev, *toy.test):
        assert unknown not in processor.encode(row["src"])
        assert unknown not in processor.encode(row["tgt"])


def test_the_model_is_trained_on_the_training_and_back_translated_text_only(
    toy: ToyData, tmp_path: Path
) -> None:
    train_toy_spm(tmp_path)

    texts = {
        language: set(
            (tmp_path / "corpus" / f"{language}.txt").read_text("utf-8").splitlines()
        )
        for language in ("khk_Cyrl", "eng_Latn", "rus_Cyrl")
    }

    for row in (*toy.train, *toy.bt):
        assert row["src"] in texts[row["src_lang"]]
        assert row["tgt"] in texts[row["tgt_lang"]]
    for row in (*toy.dev, *toy.test):
        assert row["src"] not in texts[row["src_lang"]]
        assert row["tgt"] not in texts[row["tgt_lang"]]


def test_the_documented_vocabulary_is_inside_the_feasible_range() -> None:
    low, high = TOY_SPM_VOCAB_RANGE

    assert (low, high) == (458, 669)
    assert low + 100 < TOY_SPM_VOCAB_SIZE < high - 10


@pytest.mark.parametrize("vocab_size", [458, 669])
def test_the_ends_of_the_feasible_range_train(tmp_path: Path, vocab_size: int) -> None:
    model = train_toy_spm(tmp_path, vocab_size=vocab_size)

    processor = sentencepiece.SentencePieceProcessor(model_file=str(model))
    assert processor.get_piece_size() == vocab_size


def test_a_vocabulary_below_the_feasible_range_is_refused(tmp_path: Path) -> None:
    # 118 reserved symbols, 256 byte pieces, 3 control symbols and 81 characters.
    with pytest.raises(UpstreamError, match=r"required_chars. 457 vs 458"):
        train_toy_spm(tmp_path, vocab_size=457)


def test_a_vocabulary_above_the_feasible_range_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UpstreamError, match=r"too high \(670\).*<= 669"):
        train_toy_spm(tmp_path, vocab_size=670)


def test_a_relative_model_directory_is_the_one_under_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `SpmConfig` anchors a relative path at the repository root, so the corpus
    # written under the working directory would never be found.
    monkeypatch.chdir(tmp_path)

    model = train_toy_spm(Path("relative_spm"), vocab_size=TOY_SPM_VOCAB_SIZE)

    assert model == (tmp_path / "relative_spm" / "toy.model").resolve()
    assert model.is_file()
    assert (tmp_path / "relative_spm" / "corpus" / "khk_Cyrl.txt").is_file()
    assert not (REPO_ROOT / "relative_spm").exists()
