"""The scores file of D13 (spec T8): the seam through which per-segment scores of COMET and MetricX
enter, produced elsewhere. A JSON Lines file whose first line is the header
`{"schema": "rinchen.eval.scores/1", "metric", "suite_sha256", "hypotheses_sha256"}` and then one
`{"direction", "id", "score"}` line for every segment the run scored.

The header must name the suite and the hypotheses of THIS run (a score of other hypotheses says
nothing about these), every expected segment needs exactly one finite score of at most 1000 in
magnitude, and a metric given for part of the segments (or for part of the directions) is refused.
A refusal is a `DataError` that names the line and, when they are well formed, the direction and
the id, and never what the file held. Nothing here says whether a score is right.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.errors import DataError
from rinchen.eval import segment_scores
from rinchen.eval.outputs import OutputKey
from rinchen.eval.segment_scores import (
    MAX_SCORES_BYTES,
    SCORES_SCHEMA,
    read_scores,
    scores_to_bytes,
)
from tests.support.eval_fakes import OUTPUTS_SHA256, SUITE_SHA256, golden_outputs
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

# Not a well-formed id, a digest or a language code, so a message that echoed it would show.
MARKER = "SECRET MARKER 62"
EXPECTED = list(golden_outputs("alpha"))
FIRST, SECOND = EXPECTED[0], EXPECTED[10]
OTHER_DIGEST = "e5" * 32


def scores_of(keys: list[OutputKey]) -> dict[OutputKey, float]:
    return {key: round(0.5 + index / 100, 2) for index, key in enumerate(keys)}


def header(**changes: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": SCORES_SCHEMA,
        "metric": "comet22",
        "suite_sha256": SUITE_SHA256,
        "hypotheses_sha256": OUTPUTS_SHA256,
    }
    document.update(changes)
    return document


def line(key: OutputKey, score: object = 0.5, **changes: object) -> dict[str, object]:
    document: dict[str, object] = {
        "direction": key.direction,
        "id": key.id,
        "score": score,
    }
    document.update(changes)
    return document


def encode(documents: list[object]) -> bytes:
    return "".join(f"{json.dumps(d, ensure_ascii=False)}\n" for d in documents).encode()


def good_documents() -> list[object]:
    return [header(), *(line(key, score) for key, score in scores_of(EXPECTED).items())]


def read(data: bytes, **changes: object) -> dict[OutputKey, float]:
    arguments: dict[str, object] = {
        "metric": "comet22",
        "suite_sha256": SUITE_SHA256,
        "hypotheses_sha256": OUTPUTS_SHA256,
        "expected": EXPECTED,
    }
    arguments.update(changes)
    return read_scores(data, **arguments)  # type: ignore[arg-type]


def refused(data: bytes, **changes: object) -> str:
    with pytest.raises(DataError) as caught:
        read(data, **changes)
    message = str(caught.value)
    assert message.startswith("not a scores file: ")
    assert MARKER not in message
    return message


def edited(change: Callable[[list[object]], None]) -> bytes:
    documents = good_documents()
    change(documents)
    return encode(documents)


# ---- a good file -------------------------------------------------------------------------------


def test_a_good_file_is_read_in_the_order_of_the_expected_keys() -> None:
    found = read(encode(good_documents()))

    assert found == scores_of(EXPECTED)
    assert list(found) == EXPECTED
    assert all(type(score) is float for score in found.values())


def test_the_lines_may_come_in_any_order_the_result_follows_the_expected_keys() -> None:
    documents = good_documents()
    shuffled = [documents[0], *reversed(documents[1:])]

    assert list(read(encode(shuffled))) == EXPECTED


def test_each_metric_of_the_seam_is_accepted() -> None:
    for metric in ("comet22", "metricx24", "metricx24_qe"):
        data = encode([header(metric=metric), *good_documents()[1:]])

        assert read(data, metric=metric) == scores_of(EXPECTED)


def test_an_integer_score_is_a_number_and_the_bounds_are_inclusive() -> None:
    documents = good_documents()
    documents[1] = line(EXPECTED[0], 1)
    documents[2] = line(EXPECTED[1], 1000)
    documents[3] = line(EXPECTED[2], -1000)

    found = read(encode(documents))

    assert found[EXPECTED[0]] == 1.0
    assert type(found[EXPECTED[0]]) is float
    assert found[EXPECTED[1]] == 1000.0
    assert found[EXPECTED[2]] == -1000.0


def test_a_file_that_ends_without_a_line_feed_is_read() -> None:
    assert read(encode(good_documents()).rstrip(b"\n")) == scores_of(EXPECTED)


def test_nothing_expected_reads_a_file_of_a_header_only() -> None:
    assert read(encode([header()]), expected=[]) == {}


# ---- what the writer writes --------------------------------------------------------------------


def test_the_writer_writes_the_header_then_the_lines_with_sorted_keys() -> None:
    scores = {FIRST: 0.912, SECOND: -0.25}

    data = scores_to_bytes("comet22", SUITE_SHA256, OUTPUTS_SHA256, scores)

    assert (
        data
        == (
            '{"hypotheses_sha256": "' + OUTPUTS_SHA256 + '", "metric": "comet22", '
            '"schema": "rinchen.eval.scores/1", "suite_sha256": "'
            + SUITE_SHA256
            + '"}\n'
            '{"direction": "khk_Cyrl-eng_Latn", "id": "g01", "score": 0.912}\n'
            '{"direction": "eng_Latn-khk_Cyrl", "id": "g01", "score": -0.25}\n'
        ).encode()
    )


def test_what_the_writer_writes_the_reader_reads() -> None:
    scores = scores_of(EXPECTED)

    data = scores_to_bytes("metricx24", SUITE_SHA256, OUTPUTS_SHA256, scores)

    assert read(data, metric="metricx24") == scores


@given(
    st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        min_size=20,
        max_size=20,
    )
)
@settings(max_examples=60, derandomize=True)
def test_any_finite_scores_survive_the_round_trip_exactly(values: list[float]) -> None:
    scores = dict(zip(EXPECTED, values, strict=True))

    data = scores_to_bytes("comet22", SUITE_SHA256, OUTPUTS_SHA256, scores)

    assert read(data) == scores


def test_the_writer_refuses_what_the_reader_would_refuse() -> None:
    for bad in (float("nan"), float("inf"), 1000.5):
        with pytest.raises(ValueError, match="score"):
            scores_to_bytes("comet22", SUITE_SHA256, OUTPUTS_SHA256, {FIRST: bad})
    with pytest.raises(ValueError, match="metric"):
        scores_to_bytes("bleu", SUITE_SHA256, OUTPUTS_SHA256, {FIRST: 0.5})
    for digest in ("ab" * 31 + "a", "AB" * 32, ""):
        with pytest.raises(ValueError, match="sha256"):
            scores_to_bytes("comet22", digest, OUTPUTS_SHA256, {FIRST: 0.5})
        with pytest.raises(ValueError, match="sha256"):
            scores_to_bytes("comet22", SUITE_SHA256, digest, {FIRST: 0.5})


# ---- the header names this run -----------------------------------------------------------------


def test_a_header_naming_another_suite_is_refused() -> None:
    data = edited(lambda d: d.__setitem__(0, header(suite_sha256=OTHER_DIGEST)))

    message = refused(data)

    assert "line 1" in message
    assert "suite_sha256" in message
    assert OTHER_DIGEST not in message


def test_a_header_naming_other_hypotheses_is_refused() -> None:
    # A score of other hypotheses says nothing about these ones.
    data = edited(lambda d: d.__setitem__(0, header(hypotheses_sha256=OTHER_DIGEST)))

    message = refused(data)

    assert "line 1" in message
    assert "hypotheses_sha256" in message
    assert OTHER_DIGEST not in message


def test_a_header_naming_another_metric_is_refused() -> None:
    message = refused(edited(lambda d: d.__setitem__(0, header(metric="metricx24"))))

    assert "line 1" in message
    assert "metric" in message
    assert "metricx24" not in message


def test_a_header_with_a_planted_value_never_comes_back() -> None:
    for field in ("metric", "suite_sha256", "hypotheses_sha256", "schema"):
        refused(edited(lambda d, f=field: d.__setitem__(0, header(**{f: MARKER}))))


def test_an_unknown_schema_is_refused() -> None:
    message = refused(
        edited(lambda d: d.__setitem__(0, header(schema="rinchen.eval.scores/2")))
    )

    assert "line 1" in message
    assert "schema" in message


def test_a_file_with_no_header_is_refused() -> None:
    for data in (
        b"",
        encode(good_documents()[1:]),
        encode([line(FIRST)] + good_documents()[:1]),
    ):
        message = refused(data)

        assert "line 1" in message or "empty" in message


def test_a_header_with_a_field_too_many_or_too_few_is_refused() -> None:
    extra = {**header(), "extra": 1}
    short = {k: v for k, v in header().items() if k != "metric"}
    for document in (extra, short):
        message = refused(edited(lambda d, doc=document: d.__setitem__(0, doc)))
        assert "line 1" in message


def test_a_header_that_is_not_an_object_is_refused() -> None:
    for document in ([], "header", 3, None):
        refused(edited(lambda d, doc=document: d.__setitem__(0, doc)))


def test_a_repeated_key_in_the_header_is_refused() -> None:
    data = good_documents()
    text = encode(data)
    repeated = text.replace(
        b'"metric": "comet22"', b'"metric": "comet22", "metric": "comet22"', 1
    )

    message = refused(repeated)

    assert "line 1" in message


# ---- every score once --------------------------------------------------------------------------


def test_a_missing_score_is_refused_with_the_first_missing_segment() -> None:
    message = refused(edited(lambda d: d.pop(5)))

    assert "1 of 20" in message
    assert "'khk_Cyrl-eng_Latn'" in message
    assert "'g05'" in message


def test_a_metric_given_for_part_of_the_directions_is_refused() -> None:
    first_direction_only = encode(good_documents()[:11])

    message = refused(first_direction_only)

    assert "10 of 20" in message
    assert "'eng_Latn-khk_Cyrl'" in message
    assert "'g01'" in message


def test_a_file_of_a_header_only_lacks_every_score() -> None:
    message = refused(encode([header()]))

    assert "20 of 20" in message


def test_a_repeated_score_is_refused() -> None:
    message = refused(edited(lambda d: d.append(line(EXPECTED[3], 0.9))))

    assert "line 22" in message
    assert "more than once" in message
    assert "'g04'" in message


def test_a_score_for_an_unknown_id_or_direction_is_refused() -> None:
    unknown_id = refused(
        edited(lambda d: d.append(line(OutputKey(FIRST.direction, "g99"))))
    )
    unknown_direction = refused(
        edited(lambda d: d.append(line(OutputKey("eng_Latn-rus_Cyrl", "g01"))))
    )

    assert "line 22" in unknown_id
    assert "line 22" in unknown_direction
    assert "not a segment" in unknown_id


def test_an_id_or_a_direction_that_is_not_well_formed_is_never_named() -> None:
    for key in (
        OutputKey(FIRST.direction, MARKER),
        OutputKey(MARKER, FIRST.id),
        OutputKey(MARKER, MARKER),
    ):
        message = refused(edited(lambda d, k=key: d.append(line(k))))

        assert "line 22" in message


def test_a_well_formed_id_that_belongs_to_no_segment_is_named_with_its_direction() -> (
    None
):
    # The id matched its pattern, so the message may name it; it is never a text.
    message = refused(
        edited(lambda d: d.append(line(OutputKey(FIRST.direction, "g99"))))
    )

    assert "'khk_Cyrl-eng_Latn'" in message
    assert "'g99'" in message


# ---- each score is a finite number of at most 1000 --------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(float("nan"), id="NaN"),
        pytest.param(float("inf"), id="Infinity"),
        pytest.param(float("-inf"), id="-Infinity"),
        pytest.param(1000.5, id="1000.5"),
        pytest.param(-1000.5, id="-1000.5"),
        pytest.param(10**400, id="huge-int"),
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param(MARKER, id="string"),
        pytest.param("0.5", id="numeric-string"),
        pytest.param(None, id="null"),
        pytest.param([0.5], id="list"),
        pytest.param({"score": 0.5}, id="object"),
    ],
)
def test_a_score_that_is_not_a_finite_number_of_at_most_1000_is_refused(
    bad: object,
) -> None:
    # `json.dumps` writes NaN and Infinity as bare words: the reader parses them and refuses
    # the score, so the message can still name the segment.
    data = edited(lambda d: d.__setitem__(4, line(EXPECTED[3], bad)))

    message = refused(data)

    assert "line 5" in message
    assert "'g04'" in message
    assert "score" in message


def test_a_bare_nan_in_the_text_of_the_file_is_read_and_the_score_refused() -> None:
    data = encode(good_documents()).replace(b'"score": 0.52', b'"score": NaN', 1)

    message = refused(data)

    assert "line 4" in message


def test_a_line_with_a_missing_or_an_unknown_field_is_refused() -> None:
    cases = [
        {"direction": FIRST.direction, "id": FIRST.id},
        {"direction": FIRST.direction, "score": 0.5},
        {"id": FIRST.id, "score": 0.5},
        {**line(FIRST), "status": "ok"},
    ]
    for document in cases:
        message = refused(edited(lambda d, doc=document: d.__setitem__(1, doc)))

        assert "line 2" in message


def test_a_direction_or_an_id_that_is_not_a_string_is_refused() -> None:
    for document in (
        {"direction": 3, "id": FIRST.id, "score": 0.5},
        {"direction": FIRST.direction, "id": 3, "score": 0.5},
        {"direction": None, "id": None, "score": 0.5},
    ):
        message = refused(edited(lambda d, doc=document: d.__setitem__(1, doc)))

        assert "line 2" in message


def test_a_repeated_key_in_a_score_line_is_refused() -> None:
    data = encode(good_documents()).replace(
        b'"id": "g01", "score"', b'"id": "g01", "id": "g02", "score"', 1
    )

    message = refused(data)

    assert "line 2" in message


def test_a_line_that_is_not_an_object_or_not_json_is_refused() -> None:
    good = encode(good_documents()).split(b"\n")
    for replacement in (b"[1, 2]", b"3", b'"text"', b"{not json", b"", b"null"):
        lines = [*good]
        lines[2] = replacement
        data = b"\n".join(lines)

        message = refused(data)

        assert "line 3" in message


def test_invalid_utf_8_is_refused_without_a_line() -> None:
    message = refused(b"\xff\xfe" + encode(good_documents()))

    assert "UTF-8" in message


def test_an_empty_line_inside_the_file_is_refused_by_its_number() -> None:
    lines = encode(good_documents()).split(b"\n")
    lines.insert(3, b"")

    assert "line 4" in refused(b"\n".join(lines))


# ---- the cap -----------------------------------------------------------------------------------


def test_a_file_over_the_cap_is_refused_before_it_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parser_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("the file was parsed")

    monkeypatch.setattr(segment_scores, "MAX_SCORES_BYTES", 100)
    monkeypatch.setattr(segment_scores.json, "loads", parser_called)

    message = refused(encode(good_documents()))

    assert "larger than" in message


def test_the_cap_is_128_mebibytes() -> None:
    assert MAX_SCORES_BYTES == 128 * 2**20


# ---- what the caller must get right ------------------------------------------------------------


def test_a_metric_that_is_no_segment_score_metric_is_a_value_error() -> None:
    for metric in ("bleu", "chrf_pp", "comet", ""):
        with pytest.raises(ValueError, match="segment scores"):
            read(encode(good_documents()), metric=metric)


def test_a_digest_that_is_not_a_sha256_is_a_value_error() -> None:
    for name in ("suite_sha256", "hypotheses_sha256"):
        for digest in ("ab" * 31, "AB" * 32, "", None):
            with pytest.raises(ValueError, match="sha256"):
                read(encode(good_documents()), **{name: digest})
