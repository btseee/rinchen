"""Tests for the additions of `tests.support.fake_ct2` that the shared guard rules and the
engine tests lean on: the piece-type questions of `FakeProcessor`, and the scoring, the
compute type and the recorded keywords of `ScriptedCt2`."""

from __future__ import annotations

import pytest

from tests.support.fake_ct2 import SPIECE, FakeProcessor, ScriptedCt2

PIECES = ["<s>", "</s>", "<2khk_Cyrl>", SPIECE + "a", "<0x41>", "<0xE0>"]


def ids(processor: FakeProcessor) -> dict[str, int]:
    return {piece: processor.piece_to_id(piece) for piece in PIECES}


# -- FakeProcessor: the piece-type questions


def test_the_unknown_piece_is_the_only_unknown_piece() -> None:
    processor = FakeProcessor(PIECES)

    assert processor.is_unknown(processor.unk_id())
    assert processor.is_unknown(processor.piece_to_id("not a piece"))
    assert not any(processor.is_unknown(i) for i in ids(processor).values())


def test_the_control_pieces_are_the_two_a_model_of_ours_has() -> None:
    processor = FakeProcessor(PIECES)

    assert [
        piece for piece, i in ids(processor).items() if processor.is_control(i)
    ] == [
        "<s>",
        "</s>",
    ]
    assert not processor.is_control(processor.unk_id())


def test_the_byte_pieces_are_the_ones_spelled_as_a_byte() -> None:
    processor = FakeProcessor([*PIECES, "<0xZZ>", "<0x4>"])

    assert [
        piece
        for piece in [*PIECES, "<0xZZ>", "<0x4>"]
        if processor.is_byte(processor.piece_to_id(piece))
    ] == ["<0x41>", "<0xE0>"]
    assert not processor.is_byte(processor.unk_id())


def test_a_piece_the_processor_was_not_given_is_none_of_the_three_but_unknown() -> None:
    processor = FakeProcessor(PIECES)
    missing = processor.piece_to_id("<0x42>")

    assert missing == processor.unk_id()
    assert (processor.is_control(missing), processor.is_byte(missing)) == (False, False)


def test_a_subclass_that_renames_the_unknown_piece_keeps_the_ids_straight() -> None:
    class Renamed(FakeProcessor):
        def id_to_piece(self, piece_id: int) -> str:
            return (
                "<other>"
                if piece_id == self.unk_id()
                else super().id_to_piece(piece_id)
            )

    processor = Renamed(PIECES)

    assert processor.is_unknown(processor.unk_id())
    assert not processor.is_control(processor.unk_id())
    assert not processor.is_byte(processor.unk_id())


# -- ScriptedCt2: the recorded keywords


def test_suppress_sequences_and_disable_unk_are_recorded_with_the_call() -> None:
    fake = ScriptedCt2({("a",): [(["b"], -0.5)]})

    fake.translate_batch(
        [["a"]],
        suppress_sequences=[["x"], ["y"]],
        disable_unk=True,
        beam_size=4,
        return_end_token=True,
    )
    fake.translate_batch([["a"]])

    first, second = fake.calls
    assert first.suppress_sequences == [["x"], ["y"]]
    assert first.disable_unk is True
    assert first.options == {
        "suppress_sequences": [["x"], ["y"]],
        "disable_unk": True,
        "beam_size": 4,
        "return_end_token": True,
    }
    assert second.suppress_sequences is None
    assert second.disable_unk is False


def test_the_recorded_keywords_are_a_copy_of_what_the_call_passed() -> None:
    fake = ScriptedCt2({("a",): [(["b"], -0.5)]})
    suppression = [["x"]]

    fake.translate_batch([["a"]], suppress_sequences=suppression)
    suppression.append(["y"])

    assert fake.calls[0].suppress_sequences == [["x"]]


# -- ScriptedCt2: the compute type


def test_the_compute_type_is_float32_unless_the_test_says_another() -> None:
    assert ScriptedCt2({}).compute_type == "float32"
    assert ScriptedCt2({}, compute_type="int8_float32").compute_type == "int8_float32"


# -- ScriptedCt2: score_batch


def test_score_batch_answers_each_pair_from_the_second_script() -> None:
    fake = ScriptedCt2(
        {},
        score_script={
            (("s", "t"), ("a", "b")): [-0.25, -0.5, -0.125],
            (("s", "t"), ("c",)): [-1.0, -0.0625],
        },
    )

    results = fake.score_batch([["s", "t"], ["s", "t"]], [["a", "b"], ["c"]])

    assert [result.log_probs for result in results] == [
        [-0.25, -0.5, -0.125],
        [-1.0, -0.0625],
    ]


def test_score_batch_appends_the_end_token_as_the_library_does() -> None:
    fake = ScriptedCt2({}, score_script={(("s",), ("a", "b")): [-1.0, -2.0, -3.0]})

    (result,) = fake.score_batch([["s"]], [["a", "b"]])

    assert result.tokens == ["a", "b", "</s>"]
    assert len(result.tokens) == len(result.log_probs)


def test_score_batch_end_token_is_the_one_the_test_names() -> None:
    fake = ScriptedCt2(
        {}, score_script={(("s",), ("a",)): [-1.0, -2.0]}, end_token="<eos>"
    )

    (result,) = fake.score_batch([["s"]], [["a"]])

    assert result.tokens == ["a", "<eos>"]


def test_score_batch_accepts_a_function_of_the_pair() -> None:
    fake = ScriptedCt2(
        {}, score_script=lambda source, target: [-0.5] * (len(target) + 1)
    )

    (result,) = fake.score_batch([["s"]], [["a", "b"]])

    assert result.log_probs == [-0.5, -0.5, -0.5]


def test_score_batch_records_every_call_with_its_keywords() -> None:
    fake = ScriptedCt2(
        {}, score_script=lambda source, target: [-0.5] * (len(target) + 1)
    )

    fake.score_batch([["s"]], [["a"]], batch_type="tokens", max_batch_size=8)

    (call,) = fake.score_calls
    assert call.source == [["s"]]
    assert call.target == [["a"]]
    assert call.options == {"batch_type": "tokens", "max_batch_size": 8}
    assert fake.calls == []


def test_a_pair_the_second_script_lacks_is_a_bug_of_the_test() -> None:
    fake = ScriptedCt2({}, score_script={(("s",), ("a",)): [-1.0, -2.0]})

    with pytest.raises(AssertionError, match="no score"):
        fake.score_batch([["s"]], [["other"]])


def test_score_batch_without_a_second_script_is_a_bug_of_the_test() -> None:
    with pytest.raises(AssertionError, match="no score"):
        ScriptedCt2({}).score_batch([["s"]], [["a"]])


def test_score_batch_raises_the_error_of_the_fake_after_recording_the_call() -> None:
    fake = ScriptedCt2({}, error=RuntimeError("boom"), score_script={})

    with pytest.raises(RuntimeError, match="boom"):
        fake.score_batch([["s"]], [["a"]])

    assert len(fake.score_calls) == 1


def test_score_batch_needs_one_target_per_source() -> None:
    fake = ScriptedCt2({}, score_script={(("s",), ("a",)): [-1.0, -2.0]})

    with pytest.raises(AssertionError, match="one target per source"):
        fake.score_batch([["s"], ["s"]], [["a"]])


def test_the_existing_translate_batch_is_unchanged_by_the_new_fields() -> None:
    fake = ScriptedCt2({("a",): [(["b", "c"], -0.5)]})

    (result,) = fake.translate_batch([["a"]], return_scores=True)

    assert (result.hypotheses, result.scores) == ([["b", "c"]], [-0.5])
    assert fake.score_calls == []
