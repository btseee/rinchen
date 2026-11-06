"""The B7 `GlotLID` wrapper and the `lid` stage: label normalisation, the
traditional-script pre-detector, and the `lid_src` / `lid_tgt` verdicts. The
model is the tiny fastText identifier of `tests.support.tiny_lid`; fakes stand
in for it where a test needs to control or count its answers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from rinchen.config.filters import LidConfig
from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.errors import ConfigError, DataError
from rinchen.filter import lid
from rinchen.filter.lid import GlotLID
from tests.support import pseudo_text as pt
from tests.support.planted_corpus import PlantedCorpus, build_planted_corpus
from tests.support.tiny_lid import train_tiny_lid_model

CONFIG = LidConfig()
# "mongol" in traditional Mongolian script, one code point at a time.
TRADITIONAL = "".join(
    chr(code) for code in (0x182E, 0x1823, 0x1829, 0x182D, 0x1823, 0x182F)
)
KHK_TEXT = pt.fixture_lines("khk_Cyrl")[0]
RUS_TEXT = pt.fixture_lines("rus_Cyrl")[0]
ENG_TEXT = pt.fixture_lines("eng_Latn")[0]
FIRST_TRADITIONAL = chr(0x1800)
LAST_TRADITIONAL = chr(0x18AF)
BEFORE_TRADITIONAL = chr(0x17FF)
AFTER_TRADITIONAL = chr(0x18B0)


class FakeModel:
    """Answers every text with the same labels and probabilities and records
    the batches it was asked about, in the shape of fastText's list API."""

    def __init__(self, answer: Sequence[tuple[str, float]]) -> None:
        self._answer = answer
        self.calls: list[tuple[list[str], int]] = []

    def predict(self, texts: list[str], k: int) -> tuple[Any, Any]:
        self.calls.append((list(texts), k))
        chosen = self._answer[:k]
        labels = [[f"__label__{label}" for label, _ in chosen] for _ in texts]
        probs = [np.array([prob for _, prob in chosen]) for _ in texts]
        return labels, probs


def _use_model(monkeypatch: pytest.MonkeyPatch, model: object) -> None:
    monkeypatch.setattr(lid, "_load_model", lambda path: model)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_tiny_lid_model(tmp_path_factory.mktemp("lid") / "lid.bin")


@pytest.fixture(scope="module")
def glotlid(model_path: Path) -> GlotLID:
    return GlotLID(model_path)


@pytest.fixture(scope="module")
def planted() -> PlantedCorpus:
    return build_planted_corpus()


def _row(
    src: str, tgt: str, *, src_lang: str = "khk_Cyrl", tgt_lang: str = "eng_Latn"
) -> dict[str, str]:
    return {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src": src,
        "tgt": tgt,
        "origin": "human",
        "corpus": "unit",
        "split": "train",
    }


def _frame(*rows: dict[str, str]) -> pl.DataFrame:
    return corpus_frame(list(rows))


def _scores(result: pl.DataFrame, side: str) -> list[float | None]:
    return result["scores"].struct.field(f"lid_{side}").to_list()


def _reasons(result: pl.DataFrame) -> list[str | None]:
    return result["filter_reason"].to_list()


def _config(model_path: Path, **options: Any) -> LidConfig:
    return LidConfig(model_path=model_path, **options)


# --- GlotLID.predict: labels and probabilities --------------------------------------


def test_labels_come_back_without_the_fasttext_prefix(glotlid: GlotLID) -> None:
    predictions = glotlid.predict([KHK_TEXT])

    assert predictions[0][0][0] == "khk_Cyrl"
    for label, _ in predictions[0]:
        assert not label.startswith("__label__")


def test_each_text_gets_its_top_k_labels_with_descending_probabilities(
    glotlid: GlotLID,
) -> None:
    predictions = glotlid.predict([KHK_TEXT, RUS_TEXT, ENG_TEXT], k=3)

    assert len(predictions) == 3
    assert [text_predictions[0][0] for text_predictions in predictions] == [
        "khk_Cyrl",
        "rus_Cyrl",
        "eng_Latn",
    ]
    for text_predictions in predictions:
        assert len(text_predictions) == 3
        probs = [prob for _, prob in text_predictions]
        assert probs == sorted(probs, reverse=True)
        assert all(0.0 <= prob <= 1.0 for prob in probs)
        assert len({label for label, _ in text_predictions}) == 3


def test_k_defaults_to_three(glotlid: GlotLID) -> None:
    assert len(glotlid.predict([KHK_TEXT])[0]) == 3


def test_fewer_labels_than_k_give_every_label_the_model_has(
    glotlid: GlotLID,
) -> None:
    labels = [label for label, _ in glotlid.predict([KHK_TEXT], k=50)[0]]

    assert sorted(labels) == ["deu_Latn", "eng_Latn", "khk_Cyrl", "rus_Cyrl"]


def test_a_batch_is_answered_text_by_text(glotlid: GlotLID) -> None:
    texts = [KHK_TEXT, ENG_TEXT, RUS_TEXT]

    batched = glotlid.predict(texts)

    for text, predictions in zip(texts, batched, strict=True):
        assert predictions == glotlid.predict([text])[0]


def test_no_texts_give_no_predictions(glotlid: GlotLID) -> None:
    assert glotlid.predict([]) == []


def test_k_below_one_is_rejected(glotlid: GlotLID) -> None:
    with pytest.raises(ValueError, match="k"):
        glotlid.predict([KHK_TEXT], k=0)


def test_a_probability_above_one_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    # fastText reports 1.00001 for a certain answer.
    _use_model(monkeypatch, FakeModel([("khk_Cyrl", 1.00001), ("rus_Cyrl", -0.0)]))

    predictions = GlotLID(Path("unused.bin")).predict([KHK_TEXT], k=2)

    assert predictions == [[("khk_Cyrl", 1.0), ("rus_Cyrl", 0.0)]]


def test_the_model_is_called_once_for_a_whole_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel([("khk_Cyrl", 0.9)])
    _use_model(monkeypatch, model)

    GlotLID(Path("unused.bin")).predict(["a", "b", "c"], k=1)

    assert model.calls == [(["a", "b", "c"], 1)]


# --- GlotLID.predict: the traditional-script pre-detector ---------------------------


def test_traditional_script_text_is_mn_mong_without_running_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel([("khk_Cyrl", 0.9)])
    _use_model(monkeypatch, model)

    predictions = GlotLID(Path("unused.bin")).predict([TRADITIONAL], k=3)

    assert predictions == [[("mn_Mong", 1.0)]]
    assert model.calls == []


def test_the_model_only_sees_the_texts_the_pre_detector_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel([("khk_Cyrl", 0.9)])
    _use_model(monkeypatch, model)

    predictions = GlotLID(Path("unused.bin")).predict(["one", TRADITIONAL, "two"], k=1)

    assert predictions == [
        [("khk_Cyrl", 0.9)],
        [("mn_Mong", 1.0)],
        [("khk_Cyrl", 0.9)],
    ]
    assert model.calls == [(["one", "two"], 1)]


def test_traditional_script_text_is_mn_mong_with_the_real_model(
    glotlid: GlotLID,
) -> None:
    assert glotlid.predict([TRADITIONAL]) == [[("mn_Mong", 1.0)]]


@pytest.mark.parametrize(
    ("traditional", "other", "detected"),
    [
        (6, 4, True),
        (5, 5, True),
        (4, 6, False),
        (1, 0, True),
        (0, 5, False),
        (3, 7, False),
        (7, 3, True),
    ],
)
def test_the_pre_detector_needs_half_of_the_non_whitespace_characters(
    glotlid: GlotLID, traditional: int, other: int, detected: bool
) -> None:
    text = TRADITIONAL[0] * traditional + "a" * other

    result = glotlid.predict([text])[0]

    assert (result == [("mn_Mong", 1.0)]) is detected


def test_whitespace_does_not_count_in_the_share(glotlid: GlotLID) -> None:
    mostly_space = TRADITIONAL[0] * 4 + " \t  " * 30 + "a" * 6
    half_with_space = TRADITIONAL[0] * 5 + " \t  " * 30 + "a" * 5

    assert glotlid.predict([mostly_space])[0] != [("mn_Mong", 1.0)]
    assert glotlid.predict([half_with_space])[0] == [("mn_Mong", 1.0)]


@pytest.mark.parametrize("edge", [FIRST_TRADITIONAL, LAST_TRADITIONAL])
def test_the_block_edges_count_as_traditional_script(
    glotlid: GlotLID, edge: str
) -> None:
    assert glotlid.predict([edge * 6])[0] == [("mn_Mong", 1.0)]


@pytest.mark.parametrize("outside", [BEFORE_TRADITIONAL, AFTER_TRADITIONAL])
def test_characters_just_outside_the_block_do_not_count(
    glotlid: GlotLID, outside: str
) -> None:
    assert glotlid.predict([outside * 6])[0] != [("mn_Mong", 1.0)]


def test_cyrillic_mongolian_is_not_pre_detected(glotlid: GlotLID) -> None:
    assert glotlid.predict([KHK_TEXT])[0][0][0] == "khk_Cyrl"


# --- GlotLID.predict: odd text ------------------------------------------------------


@pytest.mark.parametrize("text", ["", " ", "\t", "\n", " \n \r\n\t "])
def test_blank_text_gets_no_predictions_and_does_not_raise(
    glotlid: GlotLID, text: str
) -> None:
    assert glotlid.predict([text]) == [[]]


def test_blank_texts_do_not_disturb_their_neighbours(glotlid: GlotLID) -> None:
    predictions = glotlid.predict([KHK_TEXT, "", ENG_TEXT, "  ", TRADITIONAL])

    assert [bool(entry) for entry in predictions] == [True, False, True, False, True]
    assert predictions[0][0][0] == "khk_Cyrl"
    assert predictions[2][0][0] == "eng_Latn"
    assert predictions[4] == [("mn_Mong", 1.0)]


def test_only_blank_texts_never_reach_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel([("khk_Cyrl", 0.9)])
    _use_model(monkeypatch, model)

    GlotLID(Path("unused.bin")).predict(["", " "])

    assert model.calls == []


def test_a_newline_inside_a_text_is_read_as_a_space(glotlid: GlotLID) -> None:
    words = KHK_TEXT.split()
    with_newline = " ".join(words[:2]) + "\n" + " ".join(words[2:])

    assert glotlid.predict([with_newline]) == glotlid.predict([KHK_TEXT])


def test_a_newline_is_replaced_before_the_model_sees_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel([("khk_Cyrl", 0.9)])
    _use_model(monkeypatch, model)

    GlotLID(Path("unused.bin")).predict(["one\ntwo\n", "three"], k=1)

    assert model.calls == [(["one two ", "three"], 1)]


# --- GlotLID: loading ---------------------------------------------------------------


def test_a_missing_model_file_raises_config_error_naming_the_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "models" / "nowhere.bin"

    with pytest.raises(ConfigError, match=str(missing)):
        GlotLID(missing)


def test_a_file_that_is_not_a_model_raises_config_error_naming_the_path(
    tmp_path: Path,
) -> None:
    garbage = tmp_path / "garbage.bin"
    garbage.write_bytes(b"this is not a fastText model")

    with pytest.raises(ConfigError, match=str(garbage)):
        GlotLID(garbage)


def test_a_directory_instead_of_a_model_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=str(tmp_path)):
        GlotLID(tmp_path)


# --- apply: the planted corpus ------------------------------------------------------


def test_planted_corpus_flags_exactly_the_planted_lid_rows(
    planted: PlantedCorpus, model_path: Path
) -> None:
    result = lid.apply(planted.frame.lazy(), _config(model_path)).collect()

    flagged = {
        row["id"]: row["filter_reason"]
        for row in result.iter_rows(named=True)
        if row["filter_reason"] is not None
    }
    assert set(flagged) == planted.isolated_failures["lid"]
    assert len(flagged) == 4
    assert flagged == {
        row_id: planted.expected_reason[row_id]
        for row_id in planted.isolated_failures["lid"]
    }
    assert sorted(flagged.values()) == ["lid_src", "lid_src", "lid_tgt", "lid_tgt"]
    assert result.schema == CORPUS_SCHEMA


def test_planted_corpus_scores_every_row_and_only_the_lid_fields(
    planted: PlantedCorpus, model_path: Path
) -> None:
    result = lid.apply(planted.frame.lazy(), _config(model_path)).collect()

    for side in ("src", "tgt"):
        scores = _scores(result, side)
        assert None not in scores
        assert all(0.0 <= score <= 1.0 for score in scores if score is not None)
    assert result["scores"].struct.field("lid_src").dtype == pl.Float32
    others = [
        name
        for name in result["scores"].struct.fields
        if name not in ("lid_src", "lid_tgt")
    ]
    assert others
    for name in others:
        assert result["scores"].struct.field(name).null_count() == result.height
    assert result.drop("scores", "filter_reason").equals(
        planted.frame.drop("scores", "filter_reason")
    )


def test_a_flagged_side_scores_below_the_threshold_and_a_clean_one_above(
    planted: PlantedCorpus, model_path: Path
) -> None:
    result = lid.apply(planted.frame.lazy(), _config(model_path)).collect()

    for row in result.iter_rows(named=True):
        reason = planted.expected_reason[row["id"]]
        scores = row["scores"]
        assert (scores["lid_src"] < CONFIG.min_prob) == (reason == "lid_src")
        assert (scores["lid_tgt"] < CONFIG.min_prob) == (reason == "lid_tgt")


# --- apply: scores ------------------------------------------------------------------


def test_a_row_in_its_languages_scores_high_and_a_swapped_row_scores_low(
    model_path: Path,
) -> None:
    frame = _frame(_row(KHK_TEXT, ENG_TEXT), _row(ENG_TEXT, KHK_TEXT))

    result = lid.apply(frame.lazy(), _config(model_path)).collect()

    src, tgt = _scores(result, "src"), _scores(result, "tgt")
    assert src[0] is not None and src[0] > 0.9
    assert tgt[0] is not None and tgt[0] > 0.9
    assert src[1] is not None and src[1] < 0.1
    assert tgt[1] is not None and tgt[1] < 0.1
    assert _reasons(result) == [None, "lid_src"]


def test_traditional_script_for_mn_mong_scores_one_and_cyrillic_scores_low(
    model_path: Path,
) -> None:
    frame = _frame(
        _row(TRADITIONAL, ENG_TEXT, src_lang="mn_Mong"),
        _row(KHK_TEXT, ENG_TEXT, src_lang="mn_Mong"),
    )

    result = lid.apply(frame.lazy(), _config(model_path)).collect()

    src = _scores(result, "src")
    assert src[0] == 1.0
    assert src[1] is not None and src[1] < 0.5
    assert _reasons(result) == [None, "lid_src"]


def test_traditional_script_in_the_wrong_slot_is_rejected(model_path: Path) -> None:
    frame = _frame(_row(TRADITIONAL, ENG_TEXT))

    result = lid.apply(frame.lazy(), _config(model_path)).collect()

    assert _scores(result, "src") == [0.0]
    assert _reasons(result) == ["lid_src"]


def test_the_score_sums_the_accepted_labels_among_the_top_k(
    monkeypatch: pytest.MonkeyPatch, model_path: Path
) -> None:
    _use_model(
        monkeypatch,
        FakeModel([("eng_Latn", 0.5), ("khk_Cyrl", 0.25), ("rus_Cyrl", 0.125)]),
    )
    accept = {"khk_Cyrl": ["khk_Cyrl", "rus_Cyrl"]}
    frame = _frame(_row("a", "b")).lazy()

    def src_score(top_k: int) -> list[float | None]:
        cfg = _config(model_path, accept=accept, top_k=top_k)
        return _scores(lid.apply(frame, cfg).collect(), "src")

    assert src_score(1) == [0.0]
    assert src_score(2) == [0.25]
    assert src_score(3) == [0.375]


def test_accept_widens_the_labels_that_count_for_a_language(model_path: Path) -> None:
    frame = _frame(_row(RUS_TEXT, ENG_TEXT)).lazy()

    narrow = lid.apply(frame, _config(model_path)).collect()
    widened = lid.apply(
        frame, _config(model_path, accept={"khk_Cyrl": ["khk_Cyrl", "rus_Cyrl"]})
    ).collect()

    assert _reasons(narrow) == ["lid_src"]
    assert _reasons(widened) == [None]
    narrow_score = _scores(narrow, "src")[0]
    assert narrow_score is not None and narrow_score < 0.1
    widened_score = _scores(widened, "src")[0]
    assert widened_score is not None and widened_score > 0.9


def test_accept_for_a_language_replaces_its_own_label(model_path: Path) -> None:
    frame = _frame(_row(KHK_TEXT, ENG_TEXT)).lazy()

    result = lid.apply(
        frame, _config(model_path, accept={"khk_Cyrl": ["rus_Cyrl"]})
    ).collect()

    assert _reasons(result) == ["lid_src"]


def test_accept_for_another_language_changes_nothing(model_path: Path) -> None:
    frame = _frame(_row(KHK_TEXT, ENG_TEXT)).lazy()

    plain = lid.apply(frame, _config(model_path)).collect()
    other = lid.apply(
        frame, _config(model_path, accept={"zho_Hans": ["zho_Hans", "cmn_Hani"]})
    ).collect()

    assert other.equals(plain)


def test_mn_mong_is_accepted_through_the_accept_mapping(model_path: Path) -> None:
    frame = _frame(_row(KHK_TEXT, ENG_TEXT, src_lang="mn_Mong")).lazy()

    result = lid.apply(
        frame, _config(model_path, accept={"mn_Mong": ["khk_Cyrl"]})
    ).collect()

    assert _reasons(result) == [None]


def test_a_summed_score_never_exceeds_one(
    monkeypatch: pytest.MonkeyPatch, model_path: Path
) -> None:
    _use_model(monkeypatch, FakeModel([("khk_Cyrl", 0.7), ("rus_Cyrl", 0.7)]))
    cfg = _config(model_path, accept={"khk_Cyrl": ["khk_Cyrl", "rus_Cyrl"]})

    result = lid.apply(_frame(_row("a", "b")).lazy(), cfg).collect()

    assert _scores(result, "src") == [1.0]


# --- apply: the threshold -----------------------------------------------------------


def test_a_score_exactly_at_the_threshold_is_not_flagged(
    monkeypatch: pytest.MonkeyPatch, model_path: Path
) -> None:
    # 0.3 has no exact Float32 form: the score and the threshold must be
    # compared after the same rounding.
    _use_model(monkeypatch, FakeModel([("khk_Cyrl", 0.3)]))
    frame = _frame(_row("a", "b", tgt_lang="khk_Cyrl")).lazy()

    at = lid.apply(frame, _config(model_path, min_prob=0.3)).collect()
    above = lid.apply(
        frame, _config(model_path, min_prob=float(np.nextafter(np.float32(0.3), 1)))
    ).collect()

    assert _scores(at, "src") == [pytest.approx(0.3, abs=1e-7)]
    assert _reasons(at) == [None]
    assert _reasons(above) == ["lid_src"]


def test_the_threshold_comes_from_the_config(model_path: Path) -> None:
    frame = _frame(_row(KHK_TEXT, ENG_TEXT)).lazy()
    score = _scores(lid.apply(frame, _config(model_path)).collect(), "src")[0]
    assert score is not None and 0.5 < score < 1.0

    strict = lid.apply(frame, _config(model_path, min_prob=1.0)).collect()
    lenient = lid.apply(frame, _config(model_path, min_prob=0.001)).collect()

    assert _reasons(strict) == ["lid_src"]
    assert _reasons(lenient) == [None]


def test_src_is_flagged_before_tgt(model_path: Path) -> None:
    frame = _frame(_row(ENG_TEXT, KHK_TEXT)).lazy()

    result = lid.apply(frame, _config(model_path)).collect()

    assert _reasons(result) == ["lid_src"]
    tgt = _scores(result, "tgt")[0]
    assert tgt is not None and tgt < CONFIG.min_prob


def test_a_tgt_failure_alone_is_flagged_as_lid_tgt(model_path: Path) -> None:
    frame = _frame(_row(KHK_TEXT, KHK_TEXT)).lazy()

    result = lid.apply(frame, _config(model_path)).collect()

    assert _reasons(result) == ["lid_tgt"]


def test_a_row_that_already_has_a_reason_keeps_it(model_path: Path) -> None:
    df = _frame(_row(ENG_TEXT, KHK_TEXT), _row(ENG_TEXT, KHK_TEXT)).with_columns(
        pl.Series("filter_reason", ["numbers", None], dtype=pl.String)
    )

    result = lid.apply(df.lazy(), _config(model_path)).collect()

    assert _reasons(result) == ["numbers", "lid_src"]
    assert None not in _scores(result, "src")
    assert None not in _scores(result, "tgt")


# --- apply: empty, missing and odd text ---------------------------------------------


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
def test_blank_sides_score_zero_and_are_flagged_without_crashing(
    model_path: Path, blank: str
) -> None:
    frame = _frame(_row(blank, ENG_TEXT), _row(KHK_TEXT, blank))

    result = lid.apply(frame.lazy(), _config(model_path)).collect()

    assert _scores(result, "src")[0] == 0.0
    assert _scores(result, "tgt")[1] == 0.0
    assert _reasons(result) == ["lid_src", "lid_tgt"]


def test_a_null_side_is_scored_like_an_empty_one(model_path: Path) -> None:
    empty = _frame(_row("", ENG_TEXT), _row(KHK_TEXT, ""))
    nulled = empty.with_columns(
        pl.when(pl.col("src") == "")
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("src"))
        .alias("src"),
        pl.when(pl.col("tgt") == "")
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("tgt"))
        .alias("tgt"),
    )
    assert nulled["src"].null_count() == 1
    assert nulled["tgt"].null_count() == 1

    from_empty = lid.apply(empty.lazy(), _config(model_path)).collect()
    from_null = lid.apply(nulled.lazy(), _config(model_path)).collect()

    assert _scores(from_null, "src") == _scores(from_empty, "src")
    assert _scores(from_null, "tgt") == _scores(from_empty, "tgt")
    assert _reasons(from_null) == _reasons(from_empty)


def test_a_newline_inside_a_row_does_not_crash_the_stage(model_path: Path) -> None:
    words = KHK_TEXT.split()
    split_text = " ".join(words[:2]) + "\n" + " ".join(words[2:])
    frame = _frame(_row(split_text, ENG_TEXT), _row(KHK_TEXT, ENG_TEXT))

    result = lid.apply(frame.lazy(), _config(model_path)).collect()

    assert _scores(result, "src")[0] == _scores(result, "src")[1]
    assert _reasons(result) == [None, None]


def test_apply_on_an_empty_frame_returns_an_empty_frame_of_the_same_schema(
    model_path: Path,
) -> None:
    result = lid.apply(corpus_frame([]).lazy(), _config(model_path)).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA


# --- apply: language codes ----------------------------------------------------------


@pytest.mark.parametrize("side", ["src_lang", "tgt_lang"])
def test_an_unknown_language_code_raises_data_error_naming_it(
    model_path: Path, side: str
) -> None:
    frame = (
        _frame(_row(KHK_TEXT, ENG_TEXT))
        .lazy()
        .with_columns(pl.lit("xx_Latn").alias(side))
    )

    with pytest.raises(DataError, match="xx_Latn"):
        lid.apply(frame, _config(model_path)).collect()


def test_a_null_language_code_raises_data_error(model_path: Path) -> None:
    frame = (
        _frame(_row(KHK_TEXT, ENG_TEXT))
        .lazy()
        .with_columns(pl.lit(None, dtype=pl.String).alias("src_lang"))
    )

    with pytest.raises(DataError, match="None"):
        lid.apply(frame, _config(model_path)).collect()


def test_an_unknown_language_code_is_found_before_the_model_runs(
    monkeypatch: pytest.MonkeyPatch, model_path: Path
) -> None:
    model = FakeModel([("khk_Cyrl", 0.9)])
    _use_model(monkeypatch, model)
    frame = _frame(
        _row(KHK_TEXT, ENG_TEXT, src_lang="xx_Latn", tgt_lang="yy_Latn")
    ).lazy()

    with pytest.raises(DataError):
        lid.apply(frame, _config(model_path)).collect()

    assert model.calls == []


# --- apply: the model ---------------------------------------------------------------


def test_a_missing_model_raises_config_error_when_apply_is_called(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.bin"

    with pytest.raises(ConfigError, match=str(missing)):
        lid.apply(
            _frame(_row(KHK_TEXT, ENG_TEXT)).lazy(), LidConfig(model_path=missing)
        )


def test_the_model_is_loaded_once_per_apply_and_asked_once_per_side(
    monkeypatch: pytest.MonkeyPatch, model_path: Path
) -> None:
    loads: list[Path] = []
    model = FakeModel([("khk_Cyrl", 0.9)])

    def load(path: Path) -> FakeModel:
        loads.append(path)
        return model

    monkeypatch.setattr(lid, "_load_model", load)
    rows = [_row(f"src {index}", f"tgt {index}") for index in range(9)]

    lid.apply(_frame(*rows).lazy(), _config(model_path)).collect()

    assert loads == [model_path]
    assert [len(texts) for texts, _ in model.calls] == [9, 9]
    assert [k for _, k in model.calls] == [CONFIG.top_k, CONFIG.top_k]


def test_top_k_comes_from_the_config(
    monkeypatch: pytest.MonkeyPatch, model_path: Path
) -> None:
    model = FakeModel([("khk_Cyrl", 0.9)])
    _use_model(monkeypatch, model)

    lid.apply(_frame(_row("a", "b")).lazy(), _config(model_path, top_k=5)).collect()

    assert {k for _, k in model.calls} == {5}


def test_a_column_in_several_chunks_scores_like_one_chunk(
    planted: PlantedCorpus, model_path: Path
) -> None:
    parts = [planted.frame.slice(0, 10), planted.frame.slice(10, 25)]
    parts.append(planted.frame.slice(35, 15))
    chunked = pl.concat(parts, rechunk=False)
    assert chunked["src"].n_chunks() == 3

    result = lid.apply(chunked.lazy(), _config(model_path)).collect()

    expected = lid.apply(planted.frame.lazy(), _config(model_path)).collect()
    assert result.equals(expected)


def test_scoring_under_the_streaming_engine_gives_the_same_result(
    planted: PlantedCorpus, model_path: Path
) -> None:
    lazy = lid.apply(planted.frame.lazy(), _config(model_path))

    streamed = lazy.collect(engine="streaming")

    assert streamed.sort("id").equals(
        lid.apply(planted.frame.lazy(), _config(model_path)).collect().sort("id")
    )
