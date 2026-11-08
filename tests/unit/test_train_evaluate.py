"""Tests for `rinchen.train.evaluate`: the development-set evaluation of a trained
variant (D7 spec R8, section 5.7).

No test starts Eole: `predict` is a fake that records its call and returns scripted
token lines, so each test pins what is sent (the token lines and forced prefixes for
both placements, in one call), how the output is cut back into directions, stripped,
decoded and scored, and what is written. The real prediction is
`tests/integration/test_train_evaluate.py`.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import sentencepiece

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.metrics.bleu_chrf import corpus_scores
from rinchen.train.evaluate import DevResult, evaluate_variant
from rinchen.train.mixture import DevSet
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN
from tests.support.tiny_parallel import build_toy, train_toy_spm

NNBSP = chr(0x202F)
LINE_SEPARATOR = chr(0x2028)
TAGS = {
    "khk_Cyrl": "<2khk_Cyrl>",
    "eng_Latn": "<2eng_Latn>",
    "rus_Cyrl": "<2rus_Cyrl>",
}
BT = "<bt>"
CKPT = Path("model/ckpt")


@dataclass(frozen=True)
class Call:
    ckpt_dir: Path
    source_tokens: list[str]
    out_dir: Path
    forced_prefix_tokens: list[str] | None
    threads: int
    timeout: float | None
    stale_files: list[Path]


class FakePredict:
    """Stands in for `predict_tokens`: records the call, returns `lines`."""

    def __init__(self, lines: Sequence[str]) -> None:
        self.lines = list(lines)
        self.calls: list[Call] = []

    def __call__(
        self,
        ckpt_dir: Path,
        source_tokens: Sequence[str],
        *,
        out_dir: Path,
        forced_prefix_tokens: Sequence[str] | None = None,
        threads: int = 2,
        timeout: float | None = None,
        stem: str = "eole",
        stale_files: Sequence[Path] = (),
    ) -> list[str]:
        self.calls.append(
            Call(
                ckpt_dir=ckpt_dir,
                source_tokens=list(source_tokens),
                out_dir=out_dir,
                forced_prefix_tokens=(
                    None if forced_prefix_tokens is None else list(forced_prefix_tokens)
                ),
                threads=threads,
                timeout=timeout,
                stale_files=list(stale_files),
            )
        )
        return list(self.lines)


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("evaluate_spm"))


@pytest.fixture(scope="module")
def nnbsp_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A SentencePiece model with pieces that hold U+202F (the suffix gap of the
    traditional Mongolian script), identity normalisation as the project's, and the
    direction tags as single pieces."""
    directory = tmp_path_factory.mktemp("evaluate_nnbsp_spm")
    words = [f"ном{NNBSP}ын", f"гэр{NNBSP}ийн", f"ус{NNBSP}ны", f"нар{NNBSP}ын"]
    text = directory / "text.txt"
    text.write_text(
        "".join(
            f"{words[i % 4]} {words[(i * 3) % 4]} {words[(i + 1) % 4]}.\n"
            for i in range(300)
        ),
        encoding="utf-8",
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(text),
        model_prefix=str(directory / "nnbsp"),
        vocab_size=26,
        model_type="unigram",
        normalization_rule_name="identity",
        user_defined_symbols=[*TAGS.values(), BT, "<2zho_Hans>"],
        split_by_unicode_script=False,
        character_coverage=1.0,
        minloglevel=2,
    )
    model = directory / "nnbsp.model"
    assert any(NNBSP in piece for piece in _pieces(model, words[0]))
    return model


def _processor(model: Path) -> sentencepiece.SentencePieceProcessor:
    return sentencepiece.SentencePieceProcessor(model_file=str(model))


def _pieces(model: Path, text: str) -> list[str]:
    return list(_processor(model).encode(text, out_type=str))


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


def _toy_dev(
    lines: int = 3, directions: Sequence[str] = (MN_RU, EN_MN, RU_MN, MN_EN)
) -> dict[str, DevSet]:
    """`lines` toy dev lines per direction, the directions entered in the order given
    (not the sorted one)."""
    by_direction: dict[str, list[tuple[str, str]]] = {}
    for row in build_toy().dev:
        by_direction.setdefault(f"{row['src_lang']}-{row['tgt_lang']}", []).append(
            (row["src"], row["tgt"])
        )
    return {
        direction: _dev_set(direction, by_direction[direction][:lines])
        for direction in directions
    }


def _output_line(
    model: Path, text: str, *, tgt_tag: str | None = None, extra: Sequence[str] = ()
) -> str:
    """The token line a model would write for the hypothesis `text`; the forced tag
    first when `tgt_tag` is given."""
    pieces = [*([tgt_tag] if tgt_tag else []), *_pieces(model, text), *extra]
    return " ".join(pieces)


def _script(
    dev: dict[str, DevSet],
    model: Path,
    placement: str,
    texts: dict[str, Sequence[str]],
) -> list[str]:
    """The output lines for the hypothesis texts of each direction, directions in
    sorted order."""
    lines: list[str] = []
    for direction in sorted(dev):
        tag = TAGS[dev[direction].tgt_lang] if placement == "tgt" else None
        lines.extend(
            _output_line(model, text, tgt_tag=tag) for text in texts[direction]
        )
    return lines


def _references(dev: dict[str, DevSet]) -> dict[str, Sequence[str]]:
    return {direction: dev[direction].tgt for direction in dev}


def _hyp_lines(out_dir: Path, direction: str) -> list[str]:
    data = (out_dir / f"hyp.{direction}.txt").read_bytes().decode("utf-8")
    assert data.endswith("\n")
    return data.removesuffix("\n").split("\n")


def _run(
    dev: dict[str, DevSet],
    model: Path,
    placement: str,
    texts: dict[str, Sequence[str]],
    out_dir: Path,
) -> tuple[DevResult, FakePredict]:
    fake = FakePredict(_script(dev, model, placement, texts))
    result = evaluate_variant(
        CKPT,
        dev,
        spm_model=model,
        placement=placement,  # type: ignore[arg-type]
        out_dir=out_dir,
        predict=fake,
    )
    return result, fake


# --- what is sent to Eole ----------------------------------------------------------


def test_the_source_placement_sends_the_tagged_source_lines_of_all_directions_once(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=2)

    result, fake = _run(dev, toy_spm, "src", _references(dev), tmp_path)

    (call,) = fake.calls
    expected: list[str] = []
    for direction, tag in (
        (EN_MN, "<2khk_Cyrl>"),
        (MN_EN, "<2eng_Latn>"),
        (MN_RU, "<2rus_Cyrl>"),
        (RU_MN, "<2khk_Cyrl>"),
    ):
        for source in dev[direction].src:
            expected.append(" ".join([tag, *_pieces(toy_spm, source)]))
    assert call.source_tokens == expected
    assert call.forced_prefix_tokens is None
    assert len(call.source_tokens) == 8
    assert list(result.directions) == [EN_MN, MN_EN, MN_RU, RU_MN]


def test_the_target_placement_sends_untagged_sources_and_one_forced_tag_per_line(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=2)

    _, fake = _run(dev, toy_spm, "tgt", _references(dev), tmp_path)

    (call,) = fake.calls
    expected: list[str] = []
    for direction in (EN_MN, MN_EN, MN_RU, RU_MN):
        for source in dev[direction].src:
            expected.append(" ".join(_pieces(toy_spm, source)))
    assert call.source_tokens == expected
    assert call.forced_prefix_tokens == [
        "<2khk_Cyrl>",
        "<2khk_Cyrl>",
        "<2eng_Latn>",
        "<2eng_Latn>",
        "<2rus_Cyrl>",
        "<2rus_Cyrl>",
        "<2khk_Cyrl>",
        "<2khk_Cyrl>",
    ]


def test_the_checkpoint_the_budget_and_the_directory_reach_predict(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=1, directions=(EN_MN,))
    fake = FakePredict(_script(dev, toy_spm, "src", _references(dev)))

    evaluate_variant(
        CKPT,
        dev,
        spm_model=toy_spm,
        placement="src",
        out_dir=tmp_path / "eval",
        threads=3,
        timeout=12.5,
        predict=fake,
    )

    (call,) = fake.calls
    assert call.ckpt_dir == CKPT
    assert call.out_dir == tmp_path / "eval"
    assert call.threads == 3
    assert call.timeout == 12.5


def test_predict_is_asked_to_remove_the_hypothesis_files_of_an_earlier_run(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=1, directions=(MN_EN, EN_MN))

    _, fake = _run(dev, toy_spm, "src", _references(dev), tmp_path)

    (call,) = fake.calls
    assert sorted(call.stale_files) == [
        tmp_path / f"hyp.{EN_MN}.txt",
        tmp_path / f"hyp.{MN_EN}.txt",
    ]


# --- cutting the output back, stripping, decoding ----------------------------------


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_each_direction_gets_its_own_slice_of_the_output_in_sorted_order(
    tmp_path: Path, toy_spm: Path, placement: str
) -> None:
    dev = _toy_dev(lines=3)

    result, _ = _run(dev, toy_spm, placement, _references(dev), tmp_path)

    for direction in dev:
        assert _hyp_lines(tmp_path, direction) == list(dev[direction].tgt)
        assert result.directions[direction].lines == 3
    assert list(result.directions) == sorted(dev)
    # Hypotheses equal to the references: chrF is 100 (BLEU needs 4-grams, which three
    # short toy sentences may not have, so it is compared with the metric itself).
    for direction, scores in result.directions.items():
        assert scores.chrf == 100.0
        references = list(dev[direction].tgt)
        assert scores.bleu == corpus_scores(references, references).bleu
    assert result.chrf_mean == result.chrf_min == 100.0


def test_the_forced_tag_is_stripped_from_each_line_for_the_target_placement(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=2, directions=(EN_MN, MN_RU))

    result, _ = _run(dev, toy_spm, "tgt", _references(dev), tmp_path)

    for direction in dev:
        for line in _hyp_lines(tmp_path, direction):
            assert not line.startswith("<2")
            assert "<2" not in line
    assert result.tag_leaks == 0
    assert result.chrf_min == 100.0


def test_a_line_that_holds_only_the_forced_tag_is_an_empty_hypothesis(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=2, directions=(EN_MN,))
    texts = {EN_MN: ["", dev[EN_MN].tgt[1]]}

    result, _ = _run(dev, toy_spm, "tgt", texts, tmp_path)

    assert _hyp_lines(tmp_path, EN_MN) == ["", dev[EN_MN].tgt[1]]
    assert (
        result.directions[EN_MN].chrf
        == corpus_scores(["", dev[EN_MN].tgt[1]], list(dev[EN_MN].tgt)).chrf
    )


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda rest: f"<2eng_Latn> {rest}", id="another direction's tag"),
        pytest.param(lambda rest: rest, id="a text piece first"),
        pytest.param(lambda rest: "", id="an empty line"),
    ],
)
def test_a_first_token_that_is_not_the_forced_tag_is_an_upstream_error_naming_the_line(
    tmp_path: Path, toy_spm: Path, damage: Callable[[str], str]
) -> None:
    dev = _toy_dev(lines=3, directions=(EN_MN, MN_RU))
    lines = _script(dev, toy_spm, "tgt", _references(dev))
    # The second line (counting from 1) of khk_Cyrl-rus_Cyrl, the fifth of the output.
    lines[4] = damage(lines[4].removeprefix("<2rus_Cyrl> "))
    fake = FakePredict(lines)

    with pytest.raises(UpstreamError) as caught:
        evaluate_variant(
            CKPT,
            dev,
            spm_model=toy_spm,
            placement="tgt",
            out_dir=tmp_path,
            predict=fake,
        )

    message = str(caught.value)
    assert MN_RU in message
    assert "line 2" in message
    assert "<2rus_Cyrl>" in message
    assert EN_MN not in message
    # The text of the output is not echoed.
    assert "Это" not in message
    assert not list(tmp_path.glob("hyp.*"))


def test_an_empty_output_line_is_an_empty_hypothesis_for_the_source_placement(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=3, directions=(EN_MN,))
    references = list(dev[EN_MN].tgt)
    texts = {EN_MN: [references[0], "", references[2]]}

    result, _ = _run(dev, toy_spm, "src", texts, tmp_path)

    assert _hyp_lines(tmp_path, EN_MN) == [references[0], "", references[2]]
    scores = corpus_scores([references[0], "", references[2]], references)
    assert result.directions[EN_MN].chrf == scores.chrf
    assert result.directions[EN_MN].bleu == scores.bleu
    assert 0.0 < result.directions[EN_MN].chrf < 100.0


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_a_piece_with_a_suffix_gap_is_not_split_when_decoding(
    tmp_path: Path, nnbsp_spm: Path, placement: str
) -> None:
    source = f"гэр{NNBSP}ийн"
    target = f"ном{NNBSP}ын гэр{NNBSP}ийн"
    dev = {RU_MN: _dev_set(RU_MN, [(source, target)])}

    result, _ = _run(dev, nnbsp_spm, placement, {RU_MN: [target]}, tmp_path)

    # str.split() would cut the pieces for `ном<U+202F>ын` and `гэр<U+202F>ийн` apart.
    assert _hyp_lines(tmp_path, RU_MN) == [target]
    assert result.directions[RU_MN].chrf == 100.0


# --- scores ------------------------------------------------------------------------


def test_the_scores_of_a_direction_are_the_corpus_scores_of_its_decoded_hypotheses(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=4, directions=(EN_MN, MN_EN))
    # Each hypothesis is the reference of the previous line: close, but not equal.
    texts = {
        direction: [dev[direction].tgt[-1], *dev[direction].tgt[:-1]]
        for direction in dev
    }

    result, _ = _run(dev, toy_spm, "src", texts, tmp_path)

    for direction in dev:
        expected = corpus_scores(texts[direction], list(dev[direction].tgt))
        assert result.directions[direction].chrf == expected.chrf
        assert result.directions[direction].bleu == expected.bleu
        assert 0.0 < expected.chrf < 100.0
        assert result.directions[direction].lines == 4
        # The signatures are the same for every direction: recorded once.
        assert result.chrf_signature == expected.chrf_signature
        assert result.bleu_signature == expected.bleu_signature
    assert result.chrf_signature.startswith("nrefs:1")
    assert "tok:13a" in result.bleu_signature


def test_the_macro_mean_and_the_minimum_of_chrf_are_taken_over_directions(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=4)
    references = _references(dev)
    texts = {
        EN_MN: list(references[EN_MN]),  # 100
        MN_EN: [references[MN_EN][-1], *references[MN_EN][:-1]],  # in between
        MN_RU: ["", "", "", ""],  # 0
        RU_MN: list(references[RU_MN]),  # 100
    }

    result, _ = _run(dev, toy_spm, "src", texts, tmp_path)

    chrfs = {d: corpus_scores(texts[d], list(references[d])).chrf for d in dev}
    assert chrfs[MN_RU] == 0.0
    assert chrfs[EN_MN] == chrfs[RU_MN] == 100.0
    assert 0.0 < chrfs[MN_EN] < 100.0
    assert result.chrf_min == 0.0
    assert result.chrf_mean == pytest.approx(statistics.fmean(chrfs.values()), abs=1e-9)
    assert result.chrf_mean != statistics.median(chrfs.values())
    assert [result.directions[d].chrf for d in sorted(dev)] == [
        chrfs[d] for d in sorted(dev)
    ]


def test_the_directions_weigh_equally_in_the_macro_mean_whatever_their_size(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = {
        EN_MN: _toy_dev(lines=6)[EN_MN],
        MN_EN: _toy_dev(lines=1)[MN_EN],
    }
    texts = {EN_MN: list(dev[EN_MN].tgt), MN_EN: [""]}

    result, _ = _run(dev, toy_spm, "src", texts, tmp_path)

    assert result.chrf_mean == 50.0
    assert result.chrf_min == 0.0


# --- tag leaks ---------------------------------------------------------------------


def test_a_hypothesis_with_a_direction_tag_or_the_bt_tag_is_a_tag_leak(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=6, directions=(EN_MN,))
    references = dev[EN_MN].tgt
    lines = [
        _output_line(toy_spm, references[0]),
        _output_line(toy_spm, references[1], extra=["<2khk_Cyrl>"]),
        _output_line(toy_spm, references[2], extra=[BT]),
        _output_line(toy_spm, references[3], extra=["<2eng_Latn>", BT]),
        # A tag of a registry language that is not in the dev set is a leak too.
        _output_line(toy_spm, references[4], extra=["<2zho_Hans>"]),
        _output_line(toy_spm, references[5]),
    ]
    fake = FakePredict(lines)

    result = evaluate_variant(
        CKPT,
        dev,
        spm_model=toy_spm,
        placement="src",
        out_dir=tmp_path,
        predict=fake,
    )

    assert result.tag_leaks == 4
    hypotheses = _hyp_lines(tmp_path, EN_MN)
    assert "<2khk_Cyrl>" in hypotheses[1]
    assert BT in hypotheses[2]


def test_tag_leaks_are_counted_over_all_directions_and_the_forced_tag_is_no_leak(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=2, directions=(EN_MN, MN_EN))
    lines = [
        # eng_Latn-khk_Cyrl: the forced tag, the translation, a second tag.
        _output_line(toy_spm, dev[EN_MN].tgt[0], tgt_tag="<2khk_Cyrl>", extra=[BT]),
        _output_line(toy_spm, dev[EN_MN].tgt[1], tgt_tag="<2khk_Cyrl>"),
        # khk_Cyrl-eng_Latn
        _output_line(toy_spm, dev[MN_EN].tgt[0], tgt_tag="<2eng_Latn>"),
        _output_line(
            toy_spm, dev[MN_EN].tgt[1], tgt_tag="<2eng_Latn>", extra=["<2eng_Latn>"]
        ),
    ]
    fake = FakePredict(lines)

    result = evaluate_variant(
        CKPT,
        dev,
        spm_model=toy_spm,
        placement="tgt",
        out_dir=tmp_path,
        predict=fake,
    )

    assert result.tag_leaks == 2


def test_a_run_without_a_stray_tag_has_no_tag_leaks(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=2)

    result, _ = _run(dev, toy_spm, "src", _references(dev), tmp_path)

    assert result.tag_leaks == 0
    assert isinstance(result.tag_leaks, int)


# --- the hypothesis files ----------------------------------------------------------


def test_the_hypothesis_files_are_lf_terminated_utf8_under_the_out_dir(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=3)
    out_dir = tmp_path / "eval[1]*?" / "dev"

    _run(dev, toy_spm, "src", _references(dev), out_dir)

    for direction in dev:
        path = out_dir / f"hyp.{direction}.txt"
        data = path.read_bytes()
        assert data.endswith(b"\n")
        assert b"\r" not in data
        assert data.count(b"\n") == 3
        assert data.decode("utf-8").removesuffix("\n").split("\n") == list(
            dev[direction].tgt
        )
    assert sorted(path.name for path in out_dir.glob("*.part")) == []


def test_other_line_separators_and_line_feeds_cannot_split_a_hypothesis_file_line(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=2, directions=(EN_MN,))
    references = dev[EN_MN].tgt
    # Byte-fallback pieces decode to U+2028, CR and LF, which a model can write.
    separator = ["<0xE2>", "<0x80>", "<0xA8>"]
    lines = [
        _output_line(toy_spm, references[0], extra=[*separator, "<0x0D>"]),
        _output_line(toy_spm, references[1], extra=["<0x0A>"]),
    ]
    fake = FakePredict(lines)

    result = evaluate_variant(
        CKPT,
        dev,
        spm_model=toy_spm,
        placement="src",
        out_dir=tmp_path,
        predict=fake,
    )

    first, second = _hyp_lines(tmp_path, EN_MN)
    assert first == f"{references[0]}{LINE_SEPARATOR}\r"
    assert second == f"{references[1]} "
    assert result.directions[EN_MN].lines == 2


def test_the_hypothesis_files_replace_files_of_an_earlier_run(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=1, directions=(EN_MN,))
    (tmp_path / f"hyp.{EN_MN}.txt").write_text("old\nold\n", encoding="utf-8")

    _run(dev, toy_spm, "src", _references(dev), tmp_path)

    assert _hyp_lines(tmp_path, EN_MN) == list(dev[EN_MN].tgt)


# --- the result --------------------------------------------------------------------


def test_the_result_is_json_ready_and_keeps_the_scores_unrounded(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=4, directions=(EN_MN, MN_EN))
    texts = {
        direction: [dev[direction].tgt[-1], *dev[direction].tgt[:-1]]
        for direction in dev
    }

    result, _ = _run(dev, toy_spm, "src", texts, tmp_path)

    data = result.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert set(data) == {
        "directions",
        "chrf_mean",
        "chrf_min",
        "tag_leaks",
        "chrf_signature",
        "bleu_signature",
    }
    assert list(data["directions"]) == [EN_MN, MN_EN]
    for direction in dev:
        expected = corpus_scores(texts[direction], list(dev[direction].tgt))
        assert data["directions"][direction] == {
            "lines": 4,
            "chrf": expected.chrf,
            "bleu": expected.bleu,
        }
    assert data["chrf_mean"] == result.chrf_mean
    assert data["chrf_min"] == result.chrf_min
    assert data["tag_leaks"] == 0
    assert data["chrf_signature"] == result.chrf_signature
    # Not rounded anywhere: some score has more than 4 decimals.
    assert any(
        len(repr(score).partition(".")[2]) > 4
        for entry in data["directions"].values()
        for score in (entry["chrf"], entry["bleu"])
    )


# --- errors ------------------------------------------------------------------------


def test_an_unknown_placement_is_a_value_error_and_predict_is_not_called(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev(lines=1)
    fake = FakePredict([])
    placement: Any = "suffix"

    with pytest.raises(ValueError, match="placement"):
        evaluate_variant(
            CKPT,
            dev,
            spm_model=toy_spm,
            placement=placement,
            out_dir=tmp_path,
            predict=fake,
        )

    assert not fake.calls


def test_no_dev_set_and_an_empty_dev_set_are_data_errors(
    tmp_path: Path, toy_spm: Path
) -> None:
    fake = FakePredict([])

    with pytest.raises(DataError, match="dev"):
        evaluate_variant(
            CKPT,
            {},
            spm_model=toy_spm,
            placement="src",
            out_dir=tmp_path,
            predict=fake,
        )
    dev = _toy_dev(lines=1, directions=(EN_MN,))
    dev[MN_EN] = _dev_set(MN_EN, [])
    with pytest.raises(DataError, match=MN_EN):
        evaluate_variant(
            CKPT,
            dev,
            spm_model=toy_spm,
            placement="src",
            out_dir=tmp_path,
            predict=fake,
        )

    assert not fake.calls


def test_a_model_that_cannot_be_used_is_a_config_error_before_predict_runs(
    tmp_path: Path, toy_spm: Path
) -> None:
    fake = FakePredict([])
    dev = _toy_dev(lines=1)

    with pytest.raises(ConfigError, match="SentencePiece model"):
        evaluate_variant(
            CKPT,
            dev,
            spm_model=tmp_path / "missing.model",
            placement="src",
            out_dir=tmp_path / "eval",
            predict=fake,
        )

    assert not fake.calls
    assert not (tmp_path / "eval").exists()


def test_an_out_dir_that_cannot_be_created_fails_before_predict_runs(
    tmp_path: Path, toy_spm: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    dev = _toy_dev(lines=1, directions=(EN_MN,))
    fake = FakePredict(_script(dev, toy_spm, "src", _references(dev)))

    with pytest.raises(OSError):
        evaluate_variant(
            CKPT,
            dev,
            spm_model=toy_spm,
            placement="src",
            out_dir=blocker / "eval",
            predict=fake,
        )

    assert not fake.calls


def test_a_direction_name_that_is_not_a_file_name_is_a_value_error(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = {"../elsewhere": _dev_set(EN_MN, [("This is the book.", "Энэ бол ном.")])}
    fake = FakePredict([])

    with pytest.raises(ValueError, match="direction"):
        evaluate_variant(
            CKPT,
            dev,
            spm_model=toy_spm,
            placement="src",
            out_dir=tmp_path / "eval",
            predict=fake,
        )

    assert not fake.calls
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.parametrize("delta", [-1, 1])
def test_an_output_with_another_number_of_lines_than_the_source_is_an_upstream_error(
    tmp_path: Path, toy_spm: Path, delta: int
) -> None:
    dev = _toy_dev(lines=2, directions=(EN_MN,))
    lines = _script(dev, toy_spm, "src", _references(dev))
    lines = lines[:-1] if delta < 0 else [*lines, lines[0]]

    with pytest.raises(UpstreamError, match="2 source lines"):
        evaluate_variant(
            CKPT,
            dev,
            spm_model=toy_spm,
            placement="src",
            out_dir=tmp_path,
            predict=FakePredict(lines),
        )

    assert not list(tmp_path.glob("hyp.*"))


def test_what_predict_raises_propagates_and_no_hypothesis_file_is_written(
    tmp_path: Path, toy_spm: Path
) -> None:
    def failing(*args: object, **kwargs: object) -> list[str]:
        raise UpstreamError("eole predict exited with code 1")

    with pytest.raises(UpstreamError, match="exited with code 1"):
        evaluate_variant(
            CKPT,
            _toy_dev(lines=1),
            spm_model=toy_spm,
            placement="src",
            out_dir=tmp_path,
            predict=failing,
        )

    assert not list(tmp_path.glob("hyp.*"))
