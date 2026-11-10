"""The robustness toy of D12 (spec T17) is the toy of the design spike, port for port: the same
29,352 rows in the same order (the SHA-256 and the counts below were measured on the spike's
generator), the same text of every number, placeholder, entity and traditional-script sentence,
and nothing in it that a model or a metric could mistake for something else.

The literal sentences are taken from what the spike's generator wrote, not from this code's
renderers. The SentencePiece model is trained for real (a few seconds).
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
import pytest
import sentencepiece

from rinchen.infer.source_tokens import source_symbols
from rinchen.script_conv.rules import transliterate
from rinchen.spm.symbols import PLACEHOLDER_COUNT
from rinchen.train.mixture import load_training_rows
from tests.support.robust_toy import (
    ENTITIES,
    ROBUST_SEED,
    ROBUST_SPM_VOCAB_SIZE,
    RobustRow,
    RobustToy,
    _allocate,  # a private helper: its failure is a way to run out of supply, tested here
    build_robust_toy,
    robust_toy_digest,
    robust_toy_inputs,
    train_robust_spm,
)
from tests.support.tiny_parallel import NOUNS
from tests.support.tiny_variant import ToyInputs

DIGEST = "4136d80ada4c416934080eff1ba32da5ae7398bdfe3c4115ecf56296c511bad1"
SPLIT_COUNTS = {"train": 25_520, "dev": 1_432, "test": 2_400}
TEST_KINDS = {
    "base": 600,
    "entity": 240,
    "number": 840,
    "number2": 120,
    "placeholder": 600,
}
TRAIN_KINDS = {
    "base": 9_000,
    "entity": 1_200,
    "number": 8_320,
    "number2": 2_800,
    "placeholder": 4_200,
}
DIRECTIONS = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "khk_Cyrl-rus_Cyrl",
    "mn_Mong-eng_Latn",
    "mn_Mong-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
)
OPEN, CLOSE = chr(0x27E6), chr(0x27E7)
DASH = chr(0x2014)
TUGRIK = chr(0x20AE)
NO_BREAK_SPACES = (chr(0x00A0), chr(0x202F), chr(0x2009))
PLACEHOLDER = re.compile(f"{OPEN}([0-9]+){CLOSE}")


@pytest.fixture(scope="module")
def toy() -> RobustToy:
    return build_robust_toy()


@pytest.fixture(scope="module")
def spm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("robust_spm")


@pytest.fixture(scope="module")
def spm_model(toy: RobustToy, spm_dir: Path) -> Path:
    return train_robust_spm(spm_dir, toy)


def _count(toy: RobustToy, split: str) -> Counter[str]:
    return Counter(row.kind for row in toy.rows if row.split == split)


def _texts_by_language(rows: list[RobustRow]) -> dict[str, set[str]]:
    texts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        texts[row.src_lang].add(row.src)
        texts[row.tgt_lang].add(row.tgt)
    return texts


def _lines(toy: RobustToy) -> set[tuple[str, str, str, str]]:
    return {(r.src_lang, r.tgt_lang, r.src, r.tgt) for r in toy.rows}


# ---- the port is exact ---------------------------------------------------------------------


def test_the_toy_is_the_spikes_toy_row_for_row(toy: RobustToy) -> None:
    assert len(toy.rows) == 29_352
    assert robust_toy_digest(toy) == DIGEST


def test_the_seed_is_the_spikes() -> None:
    assert ROBUST_SEED == 7


def test_the_splits_and_kinds_have_the_spikes_sizes(toy: RobustToy) -> None:
    assert Counter(row.split for row in toy.rows) == SPLIT_COUNTS
    assert _count(toy, "test") == TEST_KINDS
    assert _count(toy, "train") == TRAIN_KINDS
    assert _count(toy, "dev") == {
        "base": 360,
        "entity": 72,
        "number": 560,
        "number2": 80,
        "placeholder": 360,
    }


def test_the_rows_come_train_then_dev_then_test(toy: RobustToy) -> None:
    splits = [row.split for row in toy.rows]

    assert [split for split, _ in _runs(splits)] == ["train", "dev", "test"]


def _runs(values: list[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


def test_the_first_row_is_the_first_train_sentence(toy: RobustToy) -> None:
    first = toy.rows[0]

    assert (first.split, first.src_lang, first.tgt_lang) == (
        "train",
        "eng_Latn",
        "khk_Cyrl",
    )
    assert (first.src, first.tgt) == ("Where is the tree?", "Мод хаана байна вэ?")
    assert (first.kind, first.template, first.nouns) == ("base", 3, (9,))
    assert NOUNS[9][:2] == ("мод", "tree")


def test_two_builds_are_equal_row_for_row(toy: RobustToy) -> None:
    again = build_robust_toy()

    assert again.rows == toy.rows
    assert again == toy


def test_another_seed_gives_other_rows_in_the_same_sizes(toy: RobustToy) -> None:
    other = build_robust_toy(ROBUST_SEED + 1)

    assert other.rows != toy.rows
    assert robust_toy_digest(other) != DIGEST
    assert Counter(row.split for row in other.rows) == SPLIT_COUNTS
    assert _count(other, "test") == TEST_KINDS
    assert _count(other, "train") == TRAIN_KINDS


def test_the_default_seed_is_the_spikes(toy: RobustToy) -> None:
    assert build_robust_toy(ROBUST_SEED) == toy


# ---- the digest ------------------------------------------------------------------------------


def _row(src: str, tgt: str, split: str = "test", **changes: object) -> RobustRow:
    fields: dict[str, object] = {
        "src_lang": "eng_Latn",
        "tgt_lang": "khk_Cyrl",
        "src": src,
        "tgt": tgt,
        "origin": "human",
        "corpus": "toy-base",
        "split": split,
        "kind": "base",
    }
    fields.update(changes)
    return RobustRow(**fields)  # type: ignore[arg-type]


def test_the_digest_is_the_sha256_of_tab_and_lf_lines_with_no_final_lf() -> None:
    toy = RobustToy(rows=(_row("A", "Б", "train"), _row("B", "В", "dev")))
    text = "train\teng_Latn\tkhk_Cyrl\tA\tБ\ndev\teng_Latn\tkhk_Cyrl\tB\tВ"

    assert robust_toy_digest(toy) == hashlib.sha256(text.encode()).hexdigest()


def test_the_digest_depends_on_the_order_of_the_rows() -> None:
    first, second = _row("A", "Б"), _row("B", "В")

    assert robust_toy_digest(RobustToy(rows=(first, second))) != robust_toy_digest(
        RobustToy(rows=(second, first))
    )


def test_the_digest_depends_on_every_column_it_names() -> None:
    base = robust_toy_digest(RobustToy(rows=(_row("A", "Б"),)))
    changed = [
        _row("A", "Б", "dev"),
        _row("A", "Б", src_lang="rus_Cyrl"),
        _row("A", "Б", tgt_lang="rus_Cyrl"),
        _row("C", "Б"),
        _row("A", "В"),
    ]

    digests = {robust_toy_digest(RobustToy(rows=(row,))) for row in changed}

    assert base not in digests
    assert len(digests) == len(changed)


def test_the_digest_ignores_what_it_does_not_name() -> None:
    plain = RobustToy(rows=(_row("A", "Б"),))
    annotated = RobustToy(
        rows=(_row("A", "Б", origin="converted", corpus="x", nouns=(1,)),)
    )

    assert robust_toy_digest(plain) == robust_toy_digest(annotated)


# ---- the rows --------------------------------------------------------------------------------


def test_a_row_is_a_corpus_row_with_the_seven_keys() -> None:
    row = _row("A", "Б", numbers=())

    assert row.corpus_row() == {
        "src_lang": "eng_Latn",
        "tgt_lang": "khk_Cyrl",
        "src": "A",
        "tgt": "Б",
        "origin": "human",
        "corpus": "toy-base",
        "split": "test",
    }
    assert list(row.corpus_row()) == [
        "src_lang",
        "tgt_lang",
        "src",
        "tgt",
        "origin",
        "corpus",
        "split",
    ]


def test_an_item_is_four_pair_rows_and_two_traditional_script_rows_unless_it_holds_a_number(
    toy: RobustToy,
) -> None:
    by_corpus = Counter((row.origin, row.corpus) for row in toy.rows)

    assert by_corpus == {
        ("human", "toy-base"): 1_660 * 4,
        ("human", "toy-entity"): 252 * 4,
        ("human", "toy-number"): 2_430 * 4,
        ("human", "toy-number2"): 750 * 4,
        ("human", "toy-placeholder"): 860 * 4,
        ("converted", "toy-base-mong"): 1_660 * 2,
        ("converted", "toy-entity-mong"): 252 * 2,
        ("converted", "toy-placeholder-mong"): 860 * 2,
    }


def test_the_traditional_script_rows_are_the_khalkha_side_written_by_the_converter(
    toy: RobustToy,
) -> None:
    khalkha_of = {
        ("eng_Latn", row.tgt): row.src
        for row in toy.rows
        if (row.src_lang, row.tgt_lang) == ("khk_Cyrl", "eng_Latn")
    }
    khalkha_of.update(
        {
            ("rus_Cyrl", row.tgt): row.src
            for row in toy.rows
            if (row.src_lang, row.tgt_lang) == ("khk_Cyrl", "rus_Cyrl")
        }
    )
    mong = [row for row in toy.rows if row.src_lang == "mn_Mong"]

    assert len(mong) == 4_800 + 264 + 480
    for row in mong:
        assert row.src == transliterate(khalkha_of[(row.tgt_lang, row.tgt)])
        assert row.tgt_lang in ("eng_Latn", "rus_Cyrl")
        assert row.kind in ("base", "entity", "placeholder")


def test_no_number_row_has_a_traditional_script_source(toy: RobustToy) -> None:
    kinds = {row.kind for row in toy.rows if row.src_lang == "mn_Mong"}

    assert kinds == {"base", "entity", "placeholder"}


def test_the_directions_of_the_toy_are_the_six_of_the_gate(toy: RobustToy) -> None:
    pairs = {f"{row.src_lang}-{row.tgt_lang}" for row in toy.rows}

    assert pairs == set(DIRECTIONS)


def test_no_sentence_of_one_split_is_in_another_in_the_same_language(
    toy: RobustToy,
) -> None:
    texts = {
        split: _texts_by_language([row for row in toy.rows if row.split == split])
        for split in SPLIT_COUNTS
    }

    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        for language in ("eng_Latn", "khk_Cyrl", "rus_Cyrl", "mn_Mong"):
            shared = texts[first][language] & texts[second][language]
            assert not shared, (language, first, second, sorted(shared)[:2])


def test_a_number_is_written_with_a_plain_space_between_its_groups(
    toy: RobustToy,
) -> None:
    for row in toy.rows:
        for text in (row.src, row.tgt):
            assert not any(space in text for space in NO_BREAK_SPACES)


# ---- the number formats, in sentences the spike wrote -------------------------------------------


@pytest.mark.parametrize(
    ("english", "khalkha", "russian"),
    [
        ("The fish is 1553.", "Загас 1553 байна.", f"Рыба {DASH} 1553."),
        ("The window is 4.25.", "Цонх 4,25 байна.", f"Окно {DASH} 4,25."),
        (
            "The rain is 470,163.50.",
            "Бороо 470 163,50 байна.",
            f"Дождь {DASH} 470 163,50.",
        ),
        ("The river is 4.5%.", "Гол 4,5% байна.", f"Река {DASH} 4,5%."),
        (
            f"The star is 440,326{TUGRIK}.",
            f"Од 440 326{TUGRIK} байна.",
            f"Звезда {DASH} 440 326{TUGRIK}.",
        ),
        (
            "The bird is 11/07/2028.",
            "Шувуу 2028.11.07 байна.",
            f"Птица {DASH} 07.11.2028.",
        ),
        ("The teacher is 23:35.", "Багш 23:35 байна.", f"Учитель {DASH} 23:35."),
        (
            "The apple: 80.05, the city: 7529.",
            "Алим: 80,05, хот: 7529.",
            "Яблоко: 80,05, город: 7529.",
        ),
    ],
    ids=[
        "int",
        "decimal",
        "grouped",
        "percent",
        "currency",
        "date",
        "time",
        "two_numbers",
    ],
)
def test_one_item_reads_in_the_conventions_of_each_language(
    toy: RobustToy, english: str, khalkha: str, russian: str
) -> None:
    lines = _lines(toy)

    assert ("eng_Latn", "khk_Cyrl", english, khalkha) in lines
    assert ("khk_Cyrl", "eng_Latn", khalkha, english) in lines
    assert ("rus_Cyrl", "khk_Cyrl", russian, khalkha) in lines
    assert ("khk_Cyrl", "rus_Cyrl", khalkha, russian) in lines


def test_a_grouped_number_has_a_comma_in_english_and_a_space_elsewhere(
    toy: RobustToy,
) -> None:
    grouped = [
        row
        for row in toy.rows
        if row.kind == "number" and row.numbers[0].kind == "grouped"
    ]
    english = [row for row in grouped if row.src_lang == "eng_Latn"]
    others = [row for row in grouped if row.src_lang != "eng_Latn"]

    assert len(english) == 260 + 20 + 30
    assert len(others) == 3 * (260 + 20 + 30)
    assert all(re.search(r"[0-9],[0-9]{3}", row.src) for row in english)
    assert all(re.search(r"[0-9] [0-9]{3}", row.src) for row in others)
    assert all(re.search(r"[0-9] [0-9]{3}", row.tgt) for row in english)


# ---- the annotations ---------------------------------------------------------------------------


def test_a_number_row_names_the_kind_and_the_values_of_its_numbers(
    toy: RobustToy,
) -> None:
    kinds = Counter(
        tuple(number.kind for number in row.numbers)
        for row in toy.rows
        if row.split == "test" and row.src_lang == "eng_Latn" and row.numbers
    )

    for kind in ("int", "decimal", "grouped", "percent", "currency", "date", "time"):
        assert kinds[(kind,)] == 30
    assert sum(count for pair, count in kinds.items() if len(pair) == 2) == 30
    values = {
        number.kind: dict(number.values)
        for row in toy.rows
        if row.src == "The star is 440,326" + TUGRIK + "."
        for number in row.numbers
    }
    assert values == {"currency": {"n": 440_326}}


def test_the_placeholder_indices_of_a_row_are_those_of_its_text(
    toy: RobustToy,
) -> None:
    rows = [row for row in toy.rows if row.kind == "placeholder"]

    assert rows
    for row in rows:
        found = [int(index) for index in PLACEHOLDER.findall(row.tgt)]
        assert sorted(row.placeholders) == sorted(found), row
        assert sorted(row.placeholders) == sorted(
            int(index) for index in PLACEHOLDER.findall(row.src)
        )


def test_the_placeholders_are_the_ten_indices_zero_to_nine(toy: RobustToy) -> None:
    for split in SPLIT_COUNTS:
        used = {
            int(index)
            for row in toy.rows
            if row.split == split
            for text in (row.src, row.tgt)
            for index in PLACEHOLDER.findall(text)
        }
        assert used <= set(range(10)), split
    train = {
        index for row in toy.rows if row.split == "train" for index in row.placeholders
    }
    assert train == set(range(10))


def test_only_the_training_split_holds_the_genitive_placeholder_sentence(
    toy: RobustToy,
) -> None:
    genitive = Counter(
        row.split for row in toy.rows if row.kind == "placeholder" and row.template == 7
    )
    triples = Counter(
        row.split
        for row in toy.rows
        if row.kind == "placeholder" and len(row.placeholders) == 3
    )

    assert genitive == {"train": 90 * 6}
    assert triples == {"train": 480 * 6, "dev": 60 * 6, "test": 100 * 6}


def test_the_toy_holds_no_placeholder_outside_its_placeholder_rows(
    toy: RobustToy,
) -> None:
    for row in toy.rows:
        if row.kind != "placeholder":
            assert OPEN not in row.src + row.tgt
            assert row.placeholders == ()


def test_an_entity_row_names_its_person_and_its_city(toy: RobustToy) -> None:
    rows = [row for row in toy.rows if row.kind == "entity"]

    assert rows
    for row in rows:
        person, city = row.entities
        assert person in PERSON_IDS
        assert city in CITY_IDS


PERSON_IDS = (
    "bat",
    "saraa",
    "dorj",
    "tuya",
    "naran",
    "bold",
    "oyun",
    "ganaa",
    "munkh",
    "enkh",
)
CITY_IDS = (
    "ulaanbaatar",
    "darkhan",
    "erdenet",
    "khovd",
    "choibalsan",
    "murun",
    "bayankhongor",
    "sukhbaatar",
    "zuunmod",
    "uliastai",
)


def test_the_entity_table_holds_ten_persons_and_ten_cities_in_three_languages() -> None:
    assert tuple(ENTITIES) == (*PERSON_IDS, *CITY_IDS)
    for spellings in ENTITIES.values():
        assert set(spellings) == {"eng_Latn", "khk_Cyrl", "rus_Cyrl"}
        assert all(names for names in spellings.values())


def test_the_entity_table_holds_the_spellings_of_the_spike() -> None:
    assert ENTITIES["tuya"] == {
        "eng_Latn": ("Tuya",),
        "khk_Cyrl": ("Туяа",),
        "rus_Cyrl": ("Туя",),
    }
    assert ENTITIES["ulaanbaatar"] == {
        "eng_Latn": ("Ulaanbaatar", "Ulan Bator"),
        "khk_Cyrl": ("Улаанбаатар",),
        "rus_Cyrl": ("Улан-Батор",),
    }
    assert ENTITIES["sukhbaatar"]["rus_Cyrl"] == ("Сухэ-Батор",)


def test_every_name_the_toy_writes_is_a_spelling_of_the_table(
    toy: RobustToy,
) -> None:
    rows = [row for row in toy.rows if row.kind == "entity"]

    assert rows
    for row in rows:
        assert _entities_written(row.tgt, row.tgt_lang) == set(row.entities), row
        if row.src_lang != "mn_Mong":
            assert _entities_written(row.src, row.src_lang) == set(row.entities), row


def _entities_written(text: str, language: str) -> set[str]:
    """The ids of the table whose first spelling starts a word of `text` (a case suffix
    may follow it)."""
    return {
        entity
        for entity, by_language in ENTITIES.items()
        if re.search(rf"(?<![\w-]){re.escape(by_language[language][0])}", text)
    }


def test_a_base_row_names_the_nouns_of_its_sentence(toy: RobustToy) -> None:
    rows = [
        row
        for row in toy.rows
        if row.kind == "base" and row.src_lang in ("eng_Latn", "rus_Cyrl", "khk_Cyrl")
    ]
    columns = {"khk_Cyrl": 0, "eng_Latn": 1, "rus_Cyrl": 2}

    assert rows
    for row in rows:
        for noun in row.nouns:
            assert NOUNS[noun][columns[row.src_lang]] in row.src.lower()
            assert NOUNS[noun][columns[row.tgt_lang]] in row.tgt.lower()


def test_only_a_base_row_names_nouns(toy: RobustToy) -> None:
    for row in toy.rows:
        assert bool(row.nouns) == (row.kind == "base")


def test_the_toy_holds_no_reserved_symbol_but_the_placeholders(
    toy: RobustToy,
) -> None:
    reserved = source_symbols()

    for row in toy.rows:
        for text in (row.src, row.tgt):
            assert not [symbol for symbol in reserved if symbol in text], text


# ---- running out of supply ----------------------------------------------------------------------


def test_asking_for_more_new_items_than_a_draw_can_give_is_an_error() -> None:
    with pytest.raises(ValueError, match="test"):
        _allocate(
            random.Random(0),
            {"test": []},
            set(),
            {"test": 3},
            lambda rng: ("item", ("the same key",)),
        )


def test_new_items_go_to_the_splits_in_the_order_of_the_counts_and_repeats_are_skipped() -> (
    None
):
    keys = iter([0, 0, 1, 1, 2, 3, 3, 4])
    items: dict[str, list[int]] = {"train": [], "test": []}
    seen: set[int] = set()

    def draw(rng: random.Random) -> tuple[int, int]:
        key = next(keys)
        return key * 10, key

    _allocate(random.Random(0), items, seen, {"train": 2, "test": 3}, draw)

    assert items == {"train": [0, 10], "test": [20, 30, 40]}
    assert seen == {0, 1, 2, 3, 4}


def test_the_draw_gets_the_generator_it_is_to_use() -> None:
    given: list[random.Random] = []
    rng = random.Random(0)

    _allocate(rng, {"a": []}, set(), {"a": 1}, lambda r: (given.append(r) or 1, 1))  # type: ignore[arg-type]

    assert given == [rng]


# ---- the SentencePiece model -------------------------------------------------------------------


def test_the_model_is_trained_on_the_four_languages_of_the_training_rows_only(
    toy: RobustToy, spm_model: Path, spm_dir: Path
) -> None:
    corpus = spm_dir / "corpus"
    expected: dict[str, set[str]] = defaultdict(set)
    for row in toy.rows:
        if row.split == "train":
            expected[row.src_lang].add(row.src)
            expected[row.tgt_lang].add(row.tgt)

    assert spm_model == spm_dir / "rtoy.model"
    assert sorted(path.name for path in corpus.iterdir()) == [
        "eng_Latn.txt",
        "khk_Cyrl.txt",
        "mn_Mong.txt",
        "rus_Cyrl.txt",
    ]
    for language, texts in expected.items():
        lines = (corpus / f"{language}.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(set(lines))
        assert set(lines) == texts, language


def test_a_relative_directory_is_made_absolute(
    toy: RobustToy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    model = train_robust_spm(Path("relative"), toy)

    assert model == tmp_path / "relative" / "rtoy.model"
    assert model.is_file()


def test_the_model_has_800_pieces(spm_model: Path) -> None:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))

    assert ROBUST_SPM_VOCAB_SIZE == 800
    assert processor.get_piece_size() == 800


def test_every_placeholder_is_one_piece(spm_model: Path) -> None:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))

    for index in range(PLACEHOLDER_COUNT):
        piece = f"{OPEN}{index}{CLOSE}"
        assert processor.id_to_piece(processor.piece_to_id(piece)) == piece
        assert processor.piece_to_id(piece) != processor.unk_id()
        assert processor.encode(f"Это {piece}.", out_type=str).count(piece) == 1


def test_the_ten_digits_and_the_tugrik_are_pieces(spm_model: Path) -> None:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))

    for piece in (*"0123456789", TUGRIK):
        assert processor.piece_to_id(piece) != processor.unk_id(), piece
    pieces = processor.encode(f"1 250,50{TUGRIK} 12:30", out_type=str)
    assert not [piece for piece in pieces if re.fullmatch("<0x[0-9A-F]{2}>", piece)]


# ---- the inputs of a training ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> ToyInputs:
    return robust_toy_inputs(tmp_path_factory.mktemp("robust_inputs"))


def test_the_corpus_has_the_training_and_dev_rows_and_no_test_row(
    inputs: ToyInputs,
) -> None:
    frame = pl.read_parquet(inputs.corpus)

    assert frame.height == 25_520 + 1_432
    assert dict(Counter(frame["split"])) == {"train": 25_520, "dev": 1_432}


def test_the_corpus_has_the_six_directions_of_the_gate(inputs: ToyInputs) -> None:
    training = load_training_rows([inputs.corpus])

    assert training.directions == DIRECTIONS
    assert training.stats.rows_kept == 25_520


def test_every_direction_has_a_dev_set_of_one_hundred_lines(inputs: ToyInputs) -> None:
    assert sorted(inputs.dev) == list(DIRECTIONS)
    for direction, dev_set in inputs.dev.items():
        assert len(dev_set.ids) == 100, direction
        assert len(dev_set.src) == len(dev_set.tgt) == 100


def test_the_inputs_name_a_model_of_800_pieces_beside_the_corpus(
    inputs: ToyInputs,
) -> None:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(inputs.spm_model))

    assert inputs.spm_model.name == "rtoy.model"
    assert inputs.corpus.parent == inputs.spm_model.parent.parent
    assert processor.get_piece_size() == 800
