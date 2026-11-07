"""Selection of synthetic candidates: the empty, purity, copy and QE filters with
exact counts per reason, the minimum-Bayes-risk pick among the survivors, and
the corpus rows built from the winners. The QE scorer is a fake, so nothing
here needs a GPU or a model."""

from __future__ import annotations

import copy
import dataclasses
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, get_args

import numpy as np
import polars as pl
import pytest

from rinchen.config.filters import QeConfig
from rinchen.config.synth import SelectionConfig
from rinchen.corpus.schema import CORPUS_SCHEMA, SCORES_DTYPE, make_row_id
from rinchen.errors import ConfigError, UpstreamError
from rinchen.mbr import mbr
from rinchen.metrics.bleu_chrf import chrf
from rinchen.synth.select import Selection, SourceRows, build_rows, select_candidates

ZERO_WIDTH_SPACE = chr(0x200B)

CAT_SITS = "муур сууж байна"
CAT_SITS_PARTICLE = "муур сууж байна шүү"
CAT_SITS_SHORT = "муур сууна"
CAT_SLEEPS = "муур унтаж байна"
DOG_RUNS = "нохой гүйж байна"
DOG_RUNS_PARTICLE = "нохой гүйж байна шүү"
DOG_RUNS_TWICE = "нохой гүйж байна шүү шүү"

COUNT_KEYS = {
    "sources",
    "candidates",
    "dropped_empty",
    "dropped_purity",
    "dropped_qe",
    "dropped_copy",
    "no_candidates",
    "no_survivor",
    "selected",
}

METRICX = QeConfig(scorer="metricx_qe", max_metricx_qe=8.0, batch_size=3)
COMETKIWI = QeConfig(scorer="cometkiwi", min_cometkiwi=0.55, batch_size=3)


class FakeScorer:
    """Scores a pair by whichever of its two texts is listed in `by_text`,
    else `default`. Records every call."""

    requires_cuda = False

    def __init__(
        self, by_text: Mapping[str, float] | None = None, *, default: float = 0.0
    ) -> None:
        self._by_text = dict(by_text or {})
        self._default = default
        self.calls: list[tuple[list[str], list[str]]] = []

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        self.calls.append((list(sources), list(translations)))
        return [
            next(
                (self._by_text[text] for text in pair if text in self._by_text),
                self._default,
            )
            for pair in zip(sources, translations, strict=True)
        ]


class ScriptedScorer:
    """Answers every call with `answer(sources, translations)`."""

    requires_cuda = False

    def __init__(
        self, answer: Callable[[Sequence[str], Sequence[str]], list[float]]
    ) -> None:
        self._answer = answer

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        return self._answer(sources, translations)


# One source per planted situation, for a Mongolian-writing job (`ft`, the
# synthetic side is `khk_Cyrl`). Source 0 holds one candidate for every way to
# be dropped plus three survivors; source 1 already is Mongolian text, so a
# candidate can be its copy; source 2 got a refusal; source 3 loses everything.
PLANTED_SOURCES = SourceRows(
    ids=("s0", "s1", "s2", "s3"),
    texts=("The cat is sitting.", CAT_SLEEPS, "A dog ran.", "Hello."),
)
PLANTED_CANDIDATES: list[list[str]] = [
    [
        "",  # 0: empty
        "The cat sits.",  # 1: Latin, not Mongolian
        DOG_RUNS_PARTICLE,  # 2: QE 9.0, above the MetricX threshold
        DOG_RUNS_TWICE,  # 3: QE NaN
        DOG_RUNS,  # 4: QE exactly 8.0, at the threshold, passes
        CAT_SITS,  # 5: survivor
        CAT_SITS_PARTICLE,  # 6: survivor, the consensus of the survivors
    ],
    [CAT_SLEEPS, CAT_SITS_SHORT],  # 0: copy of the real text; 1: the only survivor
    [],
    ["", "Hello"],
]
PLANTED_QE = {
    DOG_RUNS_PARTICLE: 9.0,
    DOG_RUNS_TWICE: math.nan,
    DOG_RUNS: 8.0,
    CAT_SITS: 1.0,
    CAT_SITS_PARTICLE: 3.5,
    CAT_SITS_SHORT: 2.0,
}


def _sources(*texts: str) -> SourceRows:
    return SourceRows(
        ids=tuple(f"id{index}" for index in range(len(texts))), texts=texts
    )


def _cfg(qe: QeConfig = METRICX, min_script_share: float = 0.9) -> SelectionConfig:
    return SelectionConfig(qe=qe, min_script_share=min_script_share)


def _select(
    sources: SourceRows,
    candidates: Sequence[Sequence[str]],
    *,
    kind: str = "ft",
    synthetic_lang: str = "khk_Cyrl",
    cfg: SelectionConfig | None = None,
    scorer: FakeScorer | ScriptedScorer | None = None,
) -> Selection:
    return select_candidates(
        sources,
        candidates,
        kind=kind,  # type: ignore[arg-type]
        synthetic_lang=synthetic_lang,
        cfg=cfg if cfg is not None else _cfg(),
        scorer=scorer,
    )


def _planted(scorer: FakeScorer | None = None) -> Selection:
    return _select(
        PLANTED_SOURCES,
        PLANTED_CANDIDATES,
        scorer=FakeScorer(PLANTED_QE) if scorer is None else scorer,
    )


# --- the planted set ----------------------------------------------------------


def test_the_planted_set_counts_every_reason_exactly() -> None:
    selection = _planted()

    assert selection.counts == {
        "sources": 4,
        "candidates": 11,
        "dropped_empty": 2,
        "dropped_purity": 2,
        "dropped_qe": 2,
        "dropped_copy": 1,
        "no_candidates": 1,
        "no_survivor": 1,
        "selected": 2,
    }


def test_the_counts_always_hold_every_key_and_are_read_only() -> None:
    selection = _select(_sources(), [])

    assert dict(selection.counts) == dict.fromkeys(COUNT_KEYS, 0)
    with pytest.raises(TypeError):
        selection.counts["sources"] = 1  # type: ignore[index]


def test_the_counts_add_up_over_sources_and_over_candidates() -> None:
    counts = _planted().counts
    survivors = 4  # source 0: three, source 1: one

    assert counts["sources"] == (
        counts["no_candidates"] + counts["no_survivor"] + counts["selected"]
    )
    dropped = sum(value for key, value in counts.items() if key.startswith("dropped_"))
    assert counts["candidates"] == dropped + survivors


def test_the_winner_is_reported_as_an_index_into_the_original_candidates() -> None:
    selection = _planted()

    # Source 0: survivors are candidates 4, 5 and 6 and the consensus is the
    # third of them, candidate 6; source 1: candidate 1 survives alone.
    assert selection.chosen == (6, 1, None, None)


def test_the_planted_winner_is_the_survivor_the_others_agree_with_most() -> None:
    survivors = [DOG_RUNS, CAT_SITS, CAT_SITS_PARTICLE]
    expected = [
        (chrf(survivors[0], survivors[1]) + chrf(survivors[0], survivors[2])) / 2,
        (chrf(survivors[1], survivors[0]) + chrf(survivors[1], survivors[2])) / 2,
        (chrf(survivors[2], survivors[0]) + chrf(survivors[2], survivors[1])) / 2,
    ]
    assert expected[2] > expected[1] > expected[0]  # about 61.8, 55.2, 28.7

    assert _planted().chosen[0] == 6


def test_a_candidate_dropped_by_qe_never_competes_in_mbr() -> None:
    # Candidates 2 and 3 look like candidate 4 and, with all five Mongolian
    # candidates in the vote, a QE reject (candidate 3) is the consensus.
    without_scorer = _select(PLANTED_SOURCES, PLANTED_CANDIDATES)

    assert without_scorer.chosen[0] == 3
    assert _planted().chosen[0] == 6


def test_the_winners_carry_their_qe_score_and_purity() -> None:
    selection = _planted()

    assert selection.qe_scores == (3.5, 2.0, None, None)
    assert selection.purity == (1.0, 1.0, None, None)


def test_the_purity_of_a_winner_is_its_share_of_letters_in_the_language() -> None:
    candidate = CAT_SITS_SHORT + " k"  # nine Cyrillic letters and one Latin

    selection = _select(_sources("a cat"), [[candidate]], scorer=FakeScorer())

    assert selection.purity == (pytest.approx(0.9),)


# --- no candidate, no survivor, one survivor -----------------------------------


def test_a_refusal_counts_as_no_candidates_not_as_no_survivor() -> None:
    selection = _select(_sources("A dog ran."), [[]])

    assert selection.chosen == (None,)
    assert selection.counts["no_candidates"] == 1
    assert selection.counts["no_survivor"] == 0
    assert selection.counts["candidates"] == 0
    assert selection.counts["selected"] == 0


def test_a_source_whose_candidates_all_drop_counts_as_no_survivor() -> None:
    selection = _select(_sources("Hello."), [["", "Hello", ZERO_WIDTH_SPACE]])

    assert selection.chosen == (None,)
    assert selection.counts["no_survivor"] == 1
    assert selection.counts["no_candidates"] == 0
    assert selection.counts["dropped_empty"] == 2
    assert selection.counts["dropped_purity"] == 1


def test_a_single_survivor_wins_without_the_utility_being_consulted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrf_calls: list[tuple[str, str]] = []

    def counting_chrf(hyp: str, ref: str) -> float:
        chrf_calls.append((hyp, ref))
        return 0.0

    monkeypatch.setattr("rinchen.mbr.utility.chrf", counting_chrf)
    mbr_calls = _spy_on_mbr(monkeypatch)

    selection = _select(
        _sources("a cat", "a dog"),
        [["", CAT_SITS_SHORT], [DOG_RUNS]],
    )

    assert selection.chosen == (1, 0)
    assert chrf_calls == []
    assert mbr_calls == []


def test_several_survivors_of_every_source_go_through_one_mbr_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mbr_calls = _spy_on_mbr(monkeypatch)
    candidates = [
        [DOG_RUNS, CAT_SITS, CAT_SITS_PARTICLE],
        [CAT_SITS_SHORT],
        ["", CAT_SITS_PARTICLE, DOG_RUNS_PARTICLE, CAT_SITS],
    ]

    selection = _select(_sources("a", "b", "c"), candidates)

    assert mbr_calls == [
        (
            [
                [DOG_RUNS, CAT_SITS, CAT_SITS_PARTICLE],
                [CAT_SITS_PARTICLE, DOG_RUNS_PARTICLE, CAT_SITS],
            ],
            None,
        )
    ]
    assert selection.chosen == (2, 0, 1)


def test_each_survivor_is_scored_against_the_other_survivors_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrf_calls: list[tuple[str, str]] = []
    real_chrf = chrf

    def counting_chrf(hyp: str, ref: str) -> float:
        chrf_calls.append((hyp, ref))
        return real_chrf(hyp, ref)

    monkeypatch.setattr("rinchen.mbr.utility.chrf", counting_chrf)
    survivors = [DOG_RUNS, CAT_SITS, CAT_SITS_PARTICLE]

    _select(_sources("a"), [["", *survivors]])

    assert len(chrf_calls) == 6
    assert all(hyp != ref for hyp, ref in chrf_calls)
    assert {(hyp, ref) for hyp, ref in chrf_calls} == {
        (hyp, ref) for hyp in survivors for ref in survivors if hyp != ref
    }


def _spy_on_mbr(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[list[list[str]], Any]]:
    calls: list[tuple[list[list[str]], Any]] = []
    real_select = mbr.select

    def spy(cands: Any, utility: Any, pseudo_refs: Any = None) -> list[int]:
        calls.append(([list(segment) for segment in cands], pseudo_refs))
        return real_select(cands, utility, pseudo_refs)

    monkeypatch.setattr(mbr, "select", spy)
    return calls


# --- the empty filter ----------------------------------------------------------


@pytest.mark.parametrize(
    "candidate",
    ["", "   ", "\n\t", "...", "- !", ZERO_WIDTH_SPACE, chr(0xFEFF)],
    ids=[
        "empty",
        "spaces",
        "controls",
        "punctuation",
        "dash",
        "zero-width-space",
        "byte-order-mark",
    ],
)
def test_a_candidate_without_a_letter_or_digit_is_empty(candidate: str) -> None:
    counts = _select(_sources("a cat"), [[candidate]]).counts

    assert counts["dropped_empty"] == 1
    assert counts["dropped_purity"] == 0
    assert counts["no_survivor"] == 1


def test_a_candidate_of_digits_only_is_not_empty_but_has_no_script_share() -> None:
    counts = _select(_sources("one two"), [["12"]]).counts

    assert counts["dropped_empty"] == 0
    assert counts["dropped_purity"] == 1


# --- the purity filter -----------------------------------------------------------


@pytest.mark.parametrize(
    ("language", "candidate", "kept"),
    [
        ("khk_Cyrl", CAT_SITS, True),
        ("khk_Cyrl", "the cat sits", False),
        ("khk_Cyrl", "ελληνικά", False),
        ("khk_Cyrl", CAT_SITS_SHORT + " k", True),  # 9 of 10 letters: 0.9
        ("khk_Cyrl", "муур суун ab", False),  # 8 of 10 letters: 0.8
        ("eng_Latn", "the cat sits", True),
        ("eng_Latn", CAT_SITS, False),
        ("rus_Cyrl", CAT_SITS, True),
        ("jpn_Jpan", "日本語です", True),  # Han and Hiragana both count
        ("zho_Hans", "日本語です", False),  # only 3 of 5 letters are Han
    ],
)
def test_purity_is_the_share_of_letters_in_the_scripts_of_the_synthetic_language(
    language: str, candidate: str, kept: bool
) -> None:
    selection = _select(_sources("real text"), [[candidate]], synthetic_lang=language)

    assert selection.chosen == ((0,) if kept else (None,))
    assert selection.counts["dropped_purity"] == (0 if kept else 1)


def test_a_share_exactly_at_min_script_share_passes() -> None:
    mixed = "муур суун ab"  # 8 of 10 letters are Cyrillic

    at_the_floor = _select(_sources("a"), [[mixed]], cfg=_cfg(min_script_share=0.8))
    above_the_floor = _select(_sources("a"), [[mixed]], cfg=_cfg(min_script_share=0.81))

    assert at_the_floor.chosen == (0,)
    assert above_the_floor.chosen == (None,)


# Three scripts whose shares add up to 0.8999999999999999 and 0.9999999999999999
# in float64: the comparison with the floor has to be done in float32, as the
# D4 script_ratio stage does, or a candidate exactly at the floor fails.
JAPANESE_NINE_TENTHS = "日本語" + "ですま" + "テスト" + "a"  # 3 + 3 + 3 + 1 letters
PURE_JAPANESE_SPLITS = [(1, 6, 15), (6, 1, 15), (3, 8, 17)]


def _japanese(han: int, hiragana: int, katakana: int) -> str:
    return "日" * han + "あ" * hiragana + "ア" * katakana


def test_a_three_script_candidate_exactly_at_the_floor_passes() -> None:
    selection = _select(
        _sources("real text"),
        [[JAPANESE_NINE_TENTHS]],
        synthetic_lang="jpn_Jpan",
        cfg=_cfg(min_script_share=0.9),
    )

    assert selection.chosen == (0,)
    assert selection.counts["dropped_purity"] == 0


def test_a_three_script_candidate_just_below_the_floor_fails() -> None:
    selection = _select(
        _sources("real text"),
        [[JAPANESE_NINE_TENTHS]],
        synthetic_lang="jpn_Jpan",
        cfg=_cfg(min_script_share=0.91),
    )

    assert selection.chosen == (None,)
    assert selection.counts["dropped_purity"] == 1


@pytest.mark.parametrize(("han", "hiragana", "katakana"), PURE_JAPANESE_SPLITS)
def test_a_pure_japanese_candidate_passes_a_floor_of_one(
    han: int, hiragana: int, katakana: int
) -> None:
    selection = _select(
        _sources("real text"),
        [[_japanese(han, hiragana, katakana)]],
        synthetic_lang="jpn_Jpan",
        cfg=_cfg(min_script_share=1.0),
    )

    assert selection.chosen == (0,)


def test_a_single_script_candidate_at_nine_tenths_passes_and_eight_tenths_fails() -> (
    None
):
    nine = _select(_sources("a"), [[CAT_SITS_SHORT + " k"]])
    eight = _select(_sources("a"), [["муур суун ab"]])

    assert nine.chosen == (0,)
    assert eight.chosen == (None,)


def test_a_mixed_candidate_at_eight_tenths_fails_the_default_floor() -> None:
    selection = _select(_sources("a"), [["муур суун ab"]])

    assert selection.chosen == (None,)
    assert selection.counts["dropped_purity"] == 1


def test_an_unknown_synthetic_language_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="xx_Xxxx"):
        _select(_sources("a"), [["b"]], synthetic_lang="xx_Xxxx")


# --- the QE filter ---------------------------------------------------------------


def _kept_by_qe(cfg: SelectionConfig, score: float) -> bool:
    # A score beyond the float32 range must not leak numpy's overflow warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        selection = _select(
            _sources("The cat."),
            [[CAT_SITS]],
            cfg=cfg,
            scorer=FakeScorer({CAT_SITS: score}),
        )
    return selection.chosen == (0,)


@pytest.mark.parametrize(
    ("threshold", "score", "kept"),
    [
        (8.0, 7.99, True),
        (8.0, 8.0, True),
        (8.0, 8.01, False),
        (8.0, 9.0, False),
        (8.0, math.inf, False),
        (8.0, -math.inf, True),
        (8.0, math.nan, False),
        (8.0, 0.0, True),
        (8.0, -3.0, True),
        (5.0, 5.0, True),
        (5.0, 6.0, False),
        (5.0, 4.0, True),
        # Compared in float32, as the D4 qe stage does: these round to the
        # threshold and pass, a float64 comparison would reject them.
        (8.0, 8.0000004, True),
        (7.3, 7.30000019, True),
        (7.3, 7.3000015, False),
        # Beyond the float32 range a score is the infinity of its sign.
        (8.0, 1e300, False),
        (8.0, -1e300, True),
        (1e300, 1e299, True),
    ],
)
def test_metricx_qe_rejects_a_score_above_the_threshold(
    threshold: float, score: float, kept: bool
) -> None:
    cfg = _cfg(QeConfig(scorer="metricx_qe", max_metricx_qe=threshold, batch_size=3))

    assert _kept_by_qe(cfg, score) is kept


@pytest.mark.parametrize(
    ("threshold", "score", "kept"),
    [
        (0.55, 0.9, True),
        (0.55, 0.55, True),
        (0.55, 0.5499, False),
        (0.55, 0.0, False),
        (0.55, -math.inf, False),
        (0.55, math.inf, True),
        (0.55, math.nan, False),
        (0.55, 9.0, True),
        (0.8, 0.8, True),
        (0.8, 0.7, False),
        (0.8, 0.9, True),
        # Compared in float32, as the D4 qe stage does.
        (0.55, 0.54999999, True),
        (0.7, 0.699999988, True),
        (0.7, 0.6999998, False),
        # Beyond the float32 range a score is the infinity of its sign.
        (0.55, 1e300, True),
        (0.55, -1e300, False),
        (-1e300, 0.5, True),
    ],
)
def test_cometkiwi_rejects_a_score_below_the_threshold(
    threshold: float, score: float, kept: bool
) -> None:
    cfg = _cfg(QeConfig(scorer="cometkiwi", min_cometkiwi=threshold, batch_size=3))

    assert _kept_by_qe(cfg, score) is kept


def test_the_qe_drops_are_counted_per_candidate_under_qe() -> None:
    counts = _planted().counts

    assert counts["dropped_qe"] == 2


def test_without_a_scorer_the_qe_step_is_skipped_and_says_so() -> None:
    selection = _select(PLANTED_SOURCES, PLANTED_CANDIDATES)

    assert selection.qe_status == "skipped: no scorer"
    assert selection.counts["dropped_qe"] == 0
    assert selection.qe_scores == (None, None, None, None)
    assert selection.chosen == (3, 1, None, None)


def test_with_a_scorer_the_status_is_ok() -> None:
    assert _planted().qe_status == "ok"


def test_a_scorer_that_is_never_needed_still_reports_ok_and_is_not_called() -> None:
    scorer = FakeScorer()

    selection = _select(_sources("Hello."), [["", "Hello"]], scorer=scorer)

    assert selection.qe_status == "ok"
    assert scorer.calls == []


def test_the_scorer_sees_only_candidates_that_passed_empty_purity_and_copy() -> None:
    scorer = FakeScorer(PLANTED_QE)

    _select(
        PLANTED_SOURCES,
        PLANTED_CANDIDATES,
        scorer=scorer,
        cfg=_cfg(QeConfig(scorer="metricx_qe", batch_size=100)),
    )

    assert scorer.calls == [
        (
            ["The cat is sitting."] * 5 + [CAT_SLEEPS],
            [
                DOG_RUNS_PARTICLE,
                DOG_RUNS_TWICE,
                DOG_RUNS,
                CAT_SITS,
                CAT_SITS_PARTICLE,
                CAT_SITS_SHORT,
            ],
        )
    ]


def test_the_scorer_is_called_in_chunks_of_batch_size() -> None:
    scorer = FakeScorer(PLANTED_QE)

    _select(
        PLANTED_SOURCES,
        PLANTED_CANDIDATES,
        scorer=scorer,
        cfg=_cfg(QeConfig(scorer="metricx_qe", batch_size=4)),
    )

    assert [len(sources) for sources, _ in scorer.calls] == [4, 2]
    assert all(
        len(sources) == len(translations) for sources, translations in scorer.calls
    )
    assert [text for _, translations in scorer.calls for text in translations] == [
        DOG_RUNS_PARTICLE,
        DOG_RUNS_TWICE,
        DOG_RUNS,
        CAT_SITS,
        CAT_SITS_PARTICLE,
        CAT_SITS_SHORT,
    ]


def test_a_batch_size_of_one_calls_the_scorer_once_per_pair() -> None:
    scorer = FakeScorer()

    _select(
        _sources("a"),
        [[CAT_SITS, DOG_RUNS]],
        scorer=scorer,
        cfg=_cfg(QeConfig(scorer="metricx_qe", batch_size=1)),
    )

    assert [len(sources) for sources, _ in scorer.calls] == [1, 1]


def test_forward_pairs_are_scored_as_real_text_then_candidate() -> None:
    scorer = FakeScorer()

    _select(
        _sources("The cat is sitting."),
        [[CAT_SITS, CAT_SITS_PARTICLE]],
        kind="ft",
        synthetic_lang="khk_Cyrl",
        scorer=scorer,
    )

    assert scorer.calls == [
        (["The cat is sitting."] * 2, [CAT_SITS, CAT_SITS_PARTICLE])
    ]


def test_backward_pairs_are_scored_as_candidate_then_real_text() -> None:
    scorer = FakeScorer()

    _select(
        _sources(CAT_SITS),
        [["the cat is sitting", "the cat sits"]],
        kind="bt",
        synthetic_lang="eng_Latn",
        scorer=scorer,
    )

    assert scorer.calls == [
        (["the cat is sitting", "the cat sits"], [CAT_SITS, CAT_SITS])
    ]


@pytest.mark.parametrize("scores_returned", [0, 2, 4])
def test_a_wrong_number_of_scores_is_an_upstream_error(scores_returned: int) -> None:
    scorer = ScriptedScorer(lambda sources, translations: [0.0] * scores_returned)
    candidates = [[CAT_SITS, CAT_SITS_PARTICLE, DOG_RUNS]]

    with pytest.raises(
        UpstreamError, match=f"returned {scores_returned} scores for 3 pairs"
    ):
        _select(_sources("a"), candidates, scorer=scorer)


def test_a_wrong_number_of_scores_in_a_later_chunk_is_an_upstream_error() -> None:
    answers = iter([[0.0, 0.0], [0.0, 0.0]])
    scorer = ScriptedScorer(lambda sources, translations: next(answers))
    candidates = [[CAT_SITS, CAT_SITS_PARTICLE, DOG_RUNS]]

    with pytest.raises(UpstreamError, match="returned 2 scores for 1 pairs"):
        _select(
            _sources("a"),
            candidates,
            scorer=scorer,
            cfg=_cfg(QeConfig(scorer="metricx_qe", batch_size=2)),
        )


@pytest.mark.parametrize(
    ("bad", "type_name"),
    [
        ("high", "str"),
        ("0.5", "str"),  # a number written as text is not a number
        (None, "NoneType"),
        ([0.5], "list"),
        (b"1", "bytes"),
    ],
    ids=["text", "numeric text", "None", "nested list", "bytes"],
)
def test_a_score_that_is_not_a_number_is_an_upstream_error_naming_the_pair(
    bad: object, type_name: str
) -> None:
    scorer = ScriptedScorer(lambda sources, translations: [0.5, 0.5, bad])
    candidates = [[CAT_SITS, CAT_SITS_PARTICLE, DOG_RUNS]]

    with pytest.raises(UpstreamError) as raised:
        _select(_sources("a"), candidates, scorer=scorer)

    assert str(raised.value) == (
        f"QE scorer returned a {type_name} for pair 2, not a number"
    )


def test_the_pair_of_a_score_that_is_not_a_number_counts_from_the_first_chunk() -> None:
    answers = iter([[0.5, 0.5], [0.5, None]])
    scorer = ScriptedScorer(lambda sources, translations: next(answers))
    candidates = [[CAT_SITS, CAT_SITS_PARTICLE, DOG_RUNS, CAT_SITS_SHORT]]

    with pytest.raises(UpstreamError, match="returned a NoneType for pair 3,"):
        _select(
            _sources("a"),
            candidates,
            scorer=scorer,
            cfg=_cfg(QeConfig(scorer="metricx_qe", batch_size=2)),
        )


@pytest.mark.parametrize(
    ("reply", "type_name"),
    [(None, "NoneType"), (0.5, "float"), (iter([0.5]), "list_iterator")],
    ids=["None", "a bare number", "an iterator"],
)
def test_a_reply_that_is_not_a_list_of_scores_is_an_upstream_error(
    reply: object, type_name: str
) -> None:
    scorer = ScriptedScorer(lambda sources, translations: reply)

    with pytest.raises(UpstreamError) as raised:
        _select(_sources("a"), [[CAT_SITS]], scorer=scorer)

    assert str(raised.value) == (
        f"QE scorer returned a {type_name}, not a list of scores"
    )


def test_an_error_about_the_scores_carries_no_text() -> None:
    scorer = ScriptedScorer(lambda sources, translations: [None])

    with pytest.raises(UpstreamError) as raised:
        _select(_sources("a"), [[CAT_SITS]], scorer=scorer)

    assert CAT_SITS not in str(raised.value)


def test_non_finite_scores_are_numbers_and_the_threshold_decides() -> None:
    # NaN and +inf fail a "lower is better" threshold, -inf passes it.
    scorer = ScriptedScorer(
        lambda sources, translations: [math.nan, math.inf, -math.inf]
    )
    candidates = [[CAT_SITS, CAT_SITS_PARTICLE, DOG_RUNS]]

    selection = _select(_sources("a"), candidates, scorer=scorer)

    assert selection.counts["dropped_qe"] == 2
    assert selection.chosen == (2,)


def test_scores_from_a_numpy_array_are_numbers() -> None:
    scorer = ScriptedScorer(
        lambda sources, translations: np.array([1.0, 9.0, 2.0], dtype=np.float32)
    )
    candidates = [[CAT_SITS, CAT_SITS_PARTICLE, DOG_RUNS]]

    selection = _select(_sources("a"), candidates, scorer=scorer)

    assert selection.counts["dropped_qe"] == 1  # 9.0 is above the 8.0 threshold


def test_a_scorer_that_cannot_reach_its_files_is_an_upstream_error() -> None:
    failure = FileNotFoundError("no such model: metricx-24")

    def fail(sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        raise failure

    with pytest.raises(UpstreamError) as raised:
        _select(_sources("a"), [[CAT_SITS]], scorer=ScriptedScorer(fail))

    assert str(raised.value) == "the QE scorer failed: no such model: metricx-24"
    assert raised.value.__cause__ is failure


def test_a_scorer_that_raises_propagates_unchanged() -> None:
    class ScorerDown(Exception):
        pass

    failure = ScorerDown("out of memory")

    def fail(sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        raise failure

    with pytest.raises(ScorerDown) as raised:
        _select(_sources("a"), [[CAT_SITS]], scorer=ScriptedScorer(fail))

    assert raised.value is failure


# --- the copy rule -----------------------------------------------------------------


def test_a_candidate_equal_to_the_real_text_is_dropped_as_a_copy() -> None:
    selection = _select(_sources(CAT_SITS), [[CAT_SITS, CAT_SITS_SHORT]])

    assert selection.chosen == (1,)
    assert selection.counts["dropped_copy"] == 1


def test_a_copy_is_counted_per_candidate_and_leaves_no_survivor_alone() -> None:
    selection = _select(_sources(CAT_SITS), [[CAT_SITS, CAT_SITS]])

    assert selection.chosen == (None,)
    assert selection.counts["dropped_copy"] == 2
    assert selection.counts["no_survivor"] == 1


def test_a_near_copy_is_not_a_copy() -> None:
    selection = _select(_sources(CAT_SITS), [[CAT_SITS + "."]])

    assert selection.chosen == (0,)
    assert selection.counts["dropped_copy"] == 0


def test_a_copy_in_another_script_than_the_synthetic_side_fails_purity_first() -> None:
    selection = _select(_sources("The cat."), [["The cat."]])

    assert selection.counts["dropped_purity"] == 1
    assert selection.counts["dropped_copy"] == 0


def test_a_copy_that_would_also_fail_qe_counts_as_a_copy() -> None:
    scorer = FakeScorer({CAT_SITS: 9.0})

    selection = _select(_sources(CAT_SITS), [[CAT_SITS]], scorer=scorer)

    assert selection.counts["dropped_copy"] == 1
    assert selection.counts["dropped_qe"] == 0


def test_a_copy_is_dropped_before_it_costs_a_qe_call() -> None:
    scorer = FakeScorer()

    _select(_sources(CAT_SITS), [[CAT_SITS, DOG_RUNS]], scorer=scorer)

    assert scorer.calls == [([CAT_SITS], [DOG_RUNS])]


def test_a_copy_of_a_real_text_with_trailing_whitespace_is_still_a_copy() -> None:
    # The translator clients strip their candidates, so the real text has to be
    # stripped for a copy to match it.
    selection = _select(
        _sources("Hello world\n"),
        [["Hello world"]],
        synthetic_lang="eng_Latn",
    )

    assert selection.chosen == (None,)
    assert selection.counts["dropped_copy"] == 1


def test_a_candidate_with_surrounding_whitespace_is_still_a_copy() -> None:
    selection = _select(
        _sources("Hello world"),
        [["  Hello world\n"]],
        synthetic_lang="eng_Latn",
    )

    assert selection.counts["dropped_copy"] == 1


def test_a_candidate_that_differs_in_case_is_not_a_copy() -> None:
    selection = _select(
        _sources("Hello world"),
        [["hello world"]],
        synthetic_lang="eng_Latn",
    )

    assert selection.chosen == (0,)
    assert selection.counts["dropped_copy"] == 0


def test_whitespace_inside_the_text_makes_it_a_different_text() -> None:
    selection = _select(
        _sources("Hello world"),
        [["Hello  world"]],
        synthetic_lang="eng_Latn",
    )

    assert selection.chosen == (0,)
    assert selection.counts["dropped_copy"] == 0


# --- inputs ----------------------------------------------------------------------------


def test_a_candidate_list_per_source_is_required() -> None:
    with pytest.raises(ValueError, match="2 candidate lists for 1 sources"):
        _select(_sources("a"), [[CAT_SITS], [DOG_RUNS]])
    with pytest.raises(ValueError, match="0 candidate lists for 2 sources"):
        _select(_sources("a", "b"), [])


def test_a_kind_other_than_bt_or_ft_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="'rt'"):
        _select(_sources("a"), [[CAT_SITS]], kind="rt")


def test_a_flat_list_of_strings_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="source 0"):
        _select(_sources("a"), [CAT_SITS])  # type: ignore[list-item]


def test_source_rows_need_one_text_per_id() -> None:
    with pytest.raises(ValueError, match="2 ids but 1 texts"):
        SourceRows(ids=("a", "b"), texts=("text",))


def test_selecting_does_not_change_its_inputs() -> None:
    sources = SourceRows(ids=PLANTED_SOURCES.ids, texts=PLANTED_SOURCES.texts)
    candidates = copy.deepcopy(PLANTED_CANDIDATES)

    _select(sources, candidates, scorer=FakeScorer(PLANTED_QE))

    assert candidates == PLANTED_CANDIDATES
    assert sources == PLANTED_SOURCES


def test_tuples_of_candidates_select_like_lists() -> None:
    as_tuples = tuple(tuple(texts) for texts in PLANTED_CANDIDATES)

    from_tuples = _select(PLANTED_SOURCES, as_tuples, scorer=FakeScorer(PLANTED_QE))

    assert from_tuples == _planted()


def test_equal_survivors_are_all_kept_and_mbr_takes_the_first_of_a_tie() -> None:
    # Removing duplicates is the translator side's job: here both copies of
    # CAT_SITS survive, agree perfectly with each other and tie.
    selection = _select(_sources("a"), [[DOG_RUNS, CAT_SITS, CAT_SITS]])

    assert selection.chosen == (1,)
    assert selection.counts["candidates"] == 3
    assert selection.counts["selected"] == 1
    assert selection.counts["dropped_empty"] == 0
    assert selection.counts["dropped_purity"] == 0
    assert selection.counts["dropped_copy"] == 0
    assert selection.counts["dropped_qe"] == 0


def test_a_utility_other_than_chrf_is_a_config_error() -> None:
    cfg = SelectionConfig.model_construct(mbr_utility="comet")

    with pytest.raises(ConfigError, match="'comet'"):
        _select(_sources("a"), [[CAT_SITS]], cfg=cfg)


def test_selection_is_deterministic() -> None:
    first = _planted(FakeScorer(PLANTED_QE))
    second = _planted(FakeScorer(PLANTED_QE))

    assert first == second


# --- build_rows ------------------------------------------------------------------------

STAMP = datetime(2026, 11, 8, 12, 30, tzinfo=UTC)
NULL_SCORE_FIELDS = ("lid_src", "lid_tgt", "len_ratio", "num_ok", "labse")


def _score(frame: pl.DataFrame, field: str) -> list[Any]:
    return frame["scores"].struct.field(field).to_list()


def _rows(
    sources: SourceRows,
    candidates: Sequence[Sequence[str]],
    selection: Selection,
    *,
    kind: str = "bt",
    generator_kind: str = "vllm",
    generator_name: str = "vllm_default",
    mono_lang: str = "khk_Cyrl",
    other_lang: str = "eng_Latn",
    qe_scorer: str | None = "metricx_qe",
    created: datetime | None = STAMP,
) -> pl.DataFrame:
    return build_rows(
        sources,
        candidates,
        selection,
        kind=kind,  # type: ignore[arg-type]
        generator_kind=generator_kind,
        generator_name=generator_name,
        mono_lang=mono_lang,
        other_lang=other_lang,
        qe_scorer=qe_scorer,
        created=created,
    )


# Backward job: the real side is Mongolian, the generator wrote English.
BT_SOURCES = _sources(CAT_SITS, "муур", DOG_RUNS, "Мэнд")
DOG_RUN_MIXED = "the dog run" + "ж"  # 9 of 10 letters are Latin: purity 0.9
BT_CANDIDATES = [
    ["", "the cat is sitting"],
    [CAT_SITS_SHORT],  # Cyrillic, wrong for an English synthetic side
    [DOG_RUN_MIXED],
    [],
]
BT_QE = {"the cat is sitting": 2.5, DOG_RUN_MIXED: 4.0}


def _bt() -> tuple[Selection, pl.DataFrame]:
    selection = _select(
        BT_SOURCES,
        BT_CANDIDATES,
        kind="bt",
        synthetic_lang="eng_Latn",
        scorer=FakeScorer(BT_QE),
    )
    return selection, _rows(BT_SOURCES, BT_CANDIDATES, selection)


def test_the_planted_backward_job_chooses_what_the_rows_below_assume() -> None:
    selection, _ = _bt()

    assert selection.chosen == (1, None, 0, None)


def test_a_backward_row_has_the_synthetic_text_as_source() -> None:
    _, frame = _bt()

    assert frame["src"].to_list() == ["the cat is sitting", DOG_RUN_MIXED]
    assert frame["tgt"].to_list() == [CAT_SITS, DOG_RUNS]
    assert frame["src_lang"].to_list() == ["eng_Latn", "eng_Latn"]
    assert frame["tgt_lang"].to_list() == ["khk_Cyrl", "khk_Cyrl"]


def test_a_backward_row_scores_the_source_side_only() -> None:
    _, frame = _bt()

    assert _score(frame, "script_src") == [pytest.approx(1.0), pytest.approx(0.9)]
    assert _score(frame, "script_tgt") == [None, None]


def test_the_qe_score_goes_to_the_column_of_the_scorer_that_ran() -> None:
    selection, frame = _bt()

    assert _score(frame, "metricx_qe") == [pytest.approx(2.5), pytest.approx(4.0)]
    assert _score(frame, "cometkiwi") == [None, None]
    assert selection.qe_scores == (2.5, None, 4.0, None)


def test_a_cometkiwi_score_goes_to_the_cometkiwi_column() -> None:
    selection = _select(
        BT_SOURCES,
        BT_CANDIDATES,
        kind="bt",
        synthetic_lang="eng_Latn",
        cfg=_cfg(COMETKIWI),
        scorer=FakeScorer({"the cat is sitting": 0.75, DOG_RUN_MIXED: 0.625}),
    )

    frame = _rows(BT_SOURCES, BT_CANDIDATES, selection, qe_scorer="cometkiwi")

    assert _score(frame, "cometkiwi") == [pytest.approx(0.75), pytest.approx(0.625)]
    assert _score(frame, "metricx_qe") == [None, None]


def test_without_a_scorer_every_qe_column_is_null() -> None:
    selection = _select(BT_SOURCES, BT_CANDIDATES, kind="bt", synthetic_lang="eng_Latn")

    frame = _rows(BT_SOURCES, BT_CANDIDATES, selection, qe_scorer=None)

    assert frame.height == 2
    assert _score(frame, "metricx_qe") == [None, None]
    assert _score(frame, "cometkiwi") == [None, None]
    assert _score(frame, "script_src") == [pytest.approx(1.0), pytest.approx(0.9)]


def test_every_other_score_field_is_null() -> None:
    _, frame = _bt()

    for field in NULL_SCORE_FIELDS:
        assert _score(frame, field) == [None, None], field


def test_the_other_columns_of_a_backward_row_are_the_plain_job_values() -> None:
    _, frame = _bt()

    assert frame["origin"].to_list() == ["bt", "bt"]
    assert frame["corpus"].to_list() == ["bt:vllm_default", "bt:vllm_default"]
    assert frame["split"].to_list() == ["train", "train"]
    assert frame["doc_id"].to_list() == [None, None]
    assert frame["seg_idx"].to_list() == [None, None]
    assert frame["filter_reason"].to_list() == [None, None]
    assert frame["created"].to_list() == [STAMP, STAMP]


def test_a_row_id_is_the_corpus_row_id_of_the_final_pair() -> None:
    _, frame = _bt()

    assert frame["id"].to_list() == [
        make_row_id("eng_Latn", "khk_Cyrl", "the cat is sitting", CAT_SITS),
        make_row_id("eng_Latn", "khk_Cyrl", DOG_RUN_MIXED, DOG_RUNS),
    ]


# Forward job: the real side is English, the generator wrote Mongolian.
FT_SOURCES = _sources("The cat is sitting.", "Hello.", "A dog ran.")
FT_CANDIDATES = [[CAT_SITS], ["", "Hello"], ["", DOG_RUNS]]


def _ft(**overrides: Any) -> pl.DataFrame:
    selection = _select(
        FT_SOURCES,
        FT_CANDIDATES,
        kind="ft",
        synthetic_lang="khk_Cyrl",
        cfg=_cfg(COMETKIWI),
        scorer=FakeScorer({CAT_SITS: 0.75, DOG_RUNS: 0.625}),
    )
    arguments: dict[str, Any] = {
        "kind": "ft",
        "mono_lang": "eng_Latn",
        "other_lang": "khk_Cyrl",
        "generator_kind": "ct2",
        "generator_name": "nllb_600m",
        "qe_scorer": "cometkiwi",
    }
    return _rows(FT_SOURCES, FT_CANDIDATES, selection, **{**arguments, **overrides})


def test_a_forward_row_has_the_real_text_as_source() -> None:
    frame = _ft()

    assert frame["src"].to_list() == ["The cat is sitting.", "A dog ran."]
    assert frame["tgt"].to_list() == [CAT_SITS, DOG_RUNS]
    assert frame["src_lang"].to_list() == ["eng_Latn", "eng_Latn"]
    assert frame["tgt_lang"].to_list() == ["khk_Cyrl", "khk_Cyrl"]


def test_a_forward_row_scores_the_target_side_only() -> None:
    frame = _ft()

    assert _score(frame, "script_tgt") == [pytest.approx(1.0), pytest.approx(1.0)]
    assert _score(frame, "script_src") == [None, None]
    assert _score(frame, "cometkiwi") == [pytest.approx(0.75), pytest.approx(0.625)]
    assert _score(frame, "metricx_qe") == [None, None]


def test_a_forward_row_has_the_origin_of_its_generator_family() -> None:
    frame = _ft()

    assert frame["origin"].to_list() == ["synth_nmt", "synth_nmt"]
    assert frame["corpus"].to_list() == ["ft:nllb_600m", "ft:nllb_600m"]
    assert frame["split"].to_list() == ["train", "train"]


@pytest.mark.parametrize(
    ("kind", "generator_kind", "origin"),
    [
        ("bt", "vllm", "bt"),
        ("bt", "ct2", "bt"),
        ("ft", "vllm", "synth_llm"),
        ("ft", "llamacpp", "synth_llm"),
        ("ft", "ct2", "synth_nmt"),
    ],
)
def test_the_origin_comes_from_the_job_kind_and_generator_kind(
    kind: str, generator_kind: str, origin: str
) -> None:
    selection = Selection(
        chosen=(0,),
        qe_scores=(None,),
        purity=(1.0,),
        counts={},
        qe_status="skipped: no scorer",
    )

    frame = _rows(
        _sources("a"),
        [["b"]],
        selection,
        kind=kind,
        generator_kind=generator_kind,
        qe_scorer=None,
    )

    assert frame["origin"].to_list() == [origin]
    assert frame["corpus"].to_list() == [f"{kind}:vllm_default"]


def test_the_frame_has_the_corpus_schema_and_only_the_chosen_sources() -> None:
    frame = _ft()

    assert frame.schema == CORPUS_SCHEMA
    assert frame.height == 2


def test_rows_follow_the_order_of_the_sources() -> None:
    sources = _sources("one", "two", "three")
    candidates = [[CAT_SITS], [DOG_RUNS], [CAT_SITS_SHORT]]
    selection = _select(sources, candidates)

    frame = _rows(
        sources,
        candidates,
        selection,
        kind="ft",
        mono_lang="eng_Latn",
        other_lang="khk_Cyrl",
        qe_scorer=None,
    )

    assert frame["src"].to_list() == ["one", "two", "three"]
    assert frame["tgt"].to_list() == [CAT_SITS, DOG_RUNS, CAT_SITS_SHORT]


def test_no_chosen_source_gives_an_empty_frame_with_the_schema() -> None:
    selection = _select(_sources("Hello."), [[""]])

    frame = _rows(_sources("Hello."), [[""]], selection, qe_scorer=None)

    assert frame.height == 0
    assert frame.schema == CORPUS_SCHEMA


def test_created_defaults_to_now_in_utc() -> None:
    selection = _select(_sources("a"), [[CAT_SITS]])
    before = datetime.now(UTC) - timedelta(seconds=1)

    frame = _rows(_sources("a"), [[CAT_SITS]], selection, qe_scorer=None, created=None)

    created = frame["created"].to_list()[0]
    assert before <= created <= datetime.now(UTC) + timedelta(seconds=1)
    assert frame.schema["created"] == pl.Datetime("us", "UTC")


def test_building_the_rows_is_deterministic() -> None:
    first = _bt()[1]
    second = _bt()[1]

    assert first.equals(second)


def test_a_selection_for_other_sources_is_a_value_error() -> None:
    selection = Selection(
        chosen=(0, 0),
        qe_scores=(None, None),
        purity=(1.0, 1.0),
        counts={},
        qe_status="skipped: no scorer",
    )

    with pytest.raises(ValueError, match="2 chosen candidates for 1 sources"):
        _rows(_sources("a"), [["b"]], selection, qe_scorer=None)


def test_candidates_for_other_sources_are_a_value_error() -> None:
    selection = _select(_sources("a"), [[CAT_SITS]])

    with pytest.raises(ValueError, match="2 candidate lists for 1 sources"):
        _rows(_sources("a"), [[CAT_SITS], [DOG_RUNS]], selection, qe_scorer=None)


def test_a_kind_other_than_bt_or_ft_is_a_value_error_for_rows_too() -> None:
    selection = _select(_sources("a"), [[CAT_SITS]])

    with pytest.raises(ValueError, match="'rt'"):
        _rows(_sources("a"), [[CAT_SITS]], selection, kind="rt", qe_scorer=None)


@pytest.mark.parametrize("chosen", [1, -1])
def test_a_chosen_index_outside_the_candidates_is_a_value_error(chosen: int) -> None:
    selection = Selection(
        chosen=(chosen,),
        qe_scores=(None,),
        purity=(1.0,),
        counts={},
        qe_status="skipped: no scorer",
    )

    with pytest.raises(ValueError, match="candidate list of 1"):
        _rows(_sources("a"), [[CAT_SITS]], selection, qe_scorer=None)


def test_qe_scores_without_the_name_of_the_scorer_are_a_value_error() -> None:
    selection = _select(_sources("a"), [[CAT_SITS]], scorer=FakeScorer({CAT_SITS: 1.0}))

    with pytest.raises(ValueError, match="qe_scorer"):
        _rows(_sources("a"), [[CAT_SITS]], selection, qe_scorer=None)


def test_a_scorer_name_that_is_not_a_qe_column_is_a_value_error() -> None:
    selection = _select(_sources("a"), [[CAT_SITS]], scorer=FakeScorer())

    with pytest.raises(ValueError, match="'labse'"):
        _rows(_sources("a"), [[CAT_SITS]], selection, qe_scorer="labse")


def test_a_scorer_name_for_a_selection_without_a_scorer_run_is_a_value_error() -> None:
    selection = _select(_sources("a"), [[CAT_SITS]])
    assert selection.qe_status == "skipped: no scorer"

    with pytest.raises(ValueError, match="no QE scorer ran"):
        _rows(_sources("a"), [[CAT_SITS]], selection, qe_scorer="metricx_qe")


def test_a_scorer_that_ran_without_a_winner_still_needs_its_name() -> None:
    selection = _select(_sources("Hello."), [["Hello"]], scorer=FakeScorer())
    assert selection.qe_status == "ok"

    with pytest.raises(ValueError, match="qe_scorer"):
        _rows(_sources("Hello."), [["Hello"]], selection, qe_scorer=None)


QE_SCORER_NAMES = get_args(QeConfig.model_fields["scorer"].annotation)


@pytest.mark.parametrize("name", QE_SCORER_NAMES)
def test_every_scorer_the_qe_config_allows_has_a_score_column(name: str) -> None:
    assert name in SCORES_DTYPE.to_schema()
    selection = _select(
        _sources("a"),
        [[CAT_SITS]],
        cfg=_cfg(QeConfig(scorer=name, max_metricx_qe=8.0, min_cometkiwi=0.5)),
        scorer=FakeScorer({CAT_SITS: 0.75}),
    )

    frame = _rows(_sources("a"), [[CAT_SITS]], selection, qe_scorer=name)

    assert _score(frame, name) == [pytest.approx(0.75)]


def test_pairs_that_are_equal_give_rows_with_the_same_id() -> None:
    # Removing duplicate pairs is the filter stage's job, not this module's.
    sources = _sources("one", "one")
    candidates = [[CAT_SITS], [CAT_SITS]]
    selection = _select(sources, candidates)

    frame = _rows(
        sources,
        candidates,
        selection,
        kind="ft",
        mono_lang="eng_Latn",
        other_lang="khk_Cyrl",
        qe_scorer=None,
    )

    assert frame.height == 2
    assert frame["id"].n_unique() == 1


def test_building_rows_does_not_change_its_inputs() -> None:
    sources = SourceRows(ids=PLANTED_SOURCES.ids, texts=PLANTED_SOURCES.texts)
    candidates = copy.deepcopy(PLANTED_CANDIDATES)
    selection = _select(sources, candidates, scorer=FakeScorer(PLANTED_QE))

    _rows(
        sources,
        candidates,
        selection,
        kind="ft",
        mono_lang="eng_Latn",
        other_lang="khk_Cyrl",
    )

    assert candidates == PLANTED_CANDIDATES
    assert sources == PLANTED_SOURCES


def test_selection_is_frozen() -> None:
    selection = _select(_sources("a"), [[CAT_SITS]])

    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.chosen = ()  # type: ignore[misc]
