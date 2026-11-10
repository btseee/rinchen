"""The hypotheses of D12 (spec T12): what a system wrote for each request of a suite, the file
that carries them (JSON Lines, one object per request), and the runner that asks an engine for
them with ONE call.

The golden file `tests/fixtures/robustness/golden_hypotheses.jsonl` answers the 33 requests of
the golden suite. Every refusal names the line, the section, the id and the arm and never what
the file held: each test plants `MARKER` in the value a refusal is about, so a message that
echoed it would show."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import pytest

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.eval import robust_hypotheses
from rinchen.eval.robust_hypotheses import (
    MAX_HYPOTHESES_BYTES,
    Hypothesis,
    Translates,
    hypotheses_from_bytes,
    hypotheses_to_bytes,
    read_hypotheses,
    request_place,
    run_requests,
    write_hypotheses,
)
from rinchen.eval.robust_suite import ARMS, Request, RequestKey, suite_requests
from rinchen.infer.engine_types import DecodeOptions, Row, Translation
from rinchen.spm.symbols import user_defined_symbols
from tests.support.engines import (
    ENG,
    KHK,
    engine_environment,  # noqa: F401 - the fixture of the engine test
    scripted_engine,
)
from tests.support.robust_golden import (
    GOLDEN_HYPOTHESES,
    GOLDEN_HYPOTHESES_SHA256,
    golden_hypotheses,
    golden_requests,
    golden_suite,
)

MARKER = "SECRET-MARKER-52"
# `MARKER` itself has the shape of an id; this one has not (an id never holds a "!").
ODD_ID = f"{MARKER}!"
STATUSES = ("ok", "empty", "too_long", "truncated", "no_output")
GOLDEN_LINES = GOLDEN_HYPOTHESES.read_text(encoding="utf-8").split("\n")[:-1]
KEY_N1 = RequestKey("numbers", "n1", "main")


def refused(data: bytes) -> str:
    with pytest.raises(DataError) as info:
        hypotheses_from_bytes(data, golden_requests())
    message = str(info.value)
    assert MARKER not in message
    return message


def as_file(lines: Sequence[str]) -> bytes:
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def line_of(**changes: Any) -> str:
    document: dict[str, Any] = {
        "arm": "main",
        "id": "n1",
        "section": "numbers",
        "status": "ok",
        "text": "Үнэ",
    }
    document.update(changes)
    return json.dumps(document, ensure_ascii=False)


def line_for(section: str, row_id: str, arm: str = "main") -> str:
    for line in GOLDEN_LINES:
        document = json.loads(line)
        if (document["section"], document["id"], document["arm"]) == (
            section,
            row_id,
            arm,
        ):
            return line
    raise AssertionError("no such line")


def changed(index: int, **changes: Any) -> str:
    """Golden line `index` (counted from 0) with its fields changed."""
    document = json.loads(GOLDEN_LINES[index])
    document.update(changes)
    return json.dumps(document, ensure_ascii=False)


def with_line(index: int, replacement: str | None) -> bytes:
    """The golden file with line `index` (counted from 0) replaced, or removed."""
    lines = list(GOLDEN_LINES)
    if replacement is None:
        del lines[index]
    else:
        lines[index] = replacement
    return as_file(lines)


class ScriptedEngine:
    """An object with `translate` that answers from a function, and records its calls."""

    def __init__(
        self, answer: Any = None, statuses: Sequence[str] | None = None
    ) -> None:
        self.calls: list[tuple[list[Row], DecodeOptions | None]] = []
        self._answer = answer
        self._statuses = statuses

    def translate(
        self, rows: Sequence[Row], options: DecodeOptions | None = None
    ) -> list[Translation]:
        self.calls.append((list(rows), options))
        if self._answer is not None:
            return list(self._answer(rows))
        return [
            Translation(
                text=f"[{row.text}]",
                status=self._statuses[i] if self._statuses else "ok",  # type: ignore[arg-type]
                alternatives=(),
                src=row.src or "und",
                tgt=row.tgt,
                script_purity=None,
                retried=False,
                zero_shot=False,
            )
            for i, row in enumerate(rows)
        ]


# ---- the hypothesis ------------------------------------------------------------------------


def test_a_hypothesis_is_a_frozen_text_and_status() -> None:
    hypothesis = Hypothesis("Үнэ", "ok")

    assert (hypothesis.text, hypothesis.status) == ("Үнэ", "ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        hypothesis.text = "x"  # type: ignore[misc]
    assert not hasattr(hypothesis, "__dict__")


@pytest.mark.parametrize("status", STATUSES)
def test_a_hypothesis_accepts_the_five_statuses_of_the_engine(status: str) -> None:
    assert Hypothesis("", status).status == status  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["fine", "OK", "", "error", "ok "])
def test_a_hypothesis_refuses_any_other_status(status: str) -> None:
    with pytest.raises(ValueError, match="status"):
        Hypothesis("x", status)  # type: ignore[arg-type]


# ---- request_place ------------------------------------------------------------------------


def test_a_request_is_named_by_section_id_and_arm() -> None:
    assert (
        request_place("script", "s3", "main") == "section 'script', id 's3', arm 'main'"
    )
    assert request_place("mong", "m1", "converted") == (
        "section 'mong', id 'm1', arm 'converted'"
    )


@pytest.mark.parametrize(
    ("section", "row_id", "arm", "named"),
    [
        (MARKER, "n1", "main", "id 'n1', arm 'main'"),
        ("numbers", ODD_ID, "main", "section 'numbers', arm 'main'"),
        ("numbers", "n1", MARKER, "section 'numbers', id 'n1'"),
        ("numbers", "a b", "main", "section 'numbers', arm 'main'"),
        ("numbers", "x" * 65, "main", "section 'numbers', arm 'main'"),
        (7, "n1", "main", "id 'n1', arm 'main'"),
        ("numbers", None, "main", "section 'numbers', arm 'main'"),
    ],
)
def test_only_a_well_formed_part_of_a_request_is_named(
    section: object, row_id: object, arm: object, named: str
) -> None:
    place = request_place(section, row_id, arm)

    assert place.endswith(named)
    assert MARKER not in place


def test_a_request_none_of_whose_parts_is_well_formed_is_named_as_a_request() -> None:
    assert request_place(MARKER, ODD_ID, MARKER) == "a request"


# ---- the golden file ---------------------------------------------------------------------


def test_the_golden_file_reads_one_hypothesis_per_request_in_request_order() -> None:
    requests = golden_requests()

    found = golden_hypotheses()

    assert len(requests) == 33
    assert list(found) == [request.key for request in requests]
    assert found[KEY_N1] == Hypothesis("Үнэ 1 250,50 доллар.", "ok")
    assert found[RequestKey("numbers", "n8", "main")] == Hypothesis("", "empty")
    assert found[RequestKey("script", "s3", "main")] == Hypothesis("", "empty")
    assert found[RequestKey("mong", "m3", "mong")] == Hypothesis("", "empty")
    assert Counter(h.status for h in found.values()) == {"ok": 30, "empty": 3}


def test_read_hypotheses_gives_the_digest_of_the_bytes_it_parsed() -> None:
    found, digest = read_hypotheses(GOLDEN_HYPOTHESES, golden_requests())

    assert digest == GOLDEN_HYPOTHESES_SHA256
    assert digest == hashlib.sha256(GOLDEN_HYPOTHESES.read_bytes()).hexdigest()
    assert found == golden_hypotheses()


def test_read_hypotheses_reads_the_file_once(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = golden_requests()
    opened: list[Path] = []
    real_open = Path.open

    def spying_open(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        opened.append(self)
        return real_open(self, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(Path, "open", spying_open)

    read_hypotheses(GOLDEN_HYPOTHESES, requests)

    assert opened == [GOLDEN_HYPOTHESES]


def test_the_order_of_the_lines_of_a_file_does_not_matter() -> None:
    reversed_file = as_file(GOLDEN_LINES[::-1])

    found = hypotheses_from_bytes(reversed_file, golden_requests())

    assert list(found) == [request.key for request in golden_requests()]
    assert found == golden_hypotheses()


def test_the_final_line_feed_is_optional() -> None:
    data = "\n".join(GOLDEN_LINES).encode("utf-8")

    assert hypotheses_from_bytes(data, golden_requests()) == golden_hypotheses()


def test_a_missing_status_reads_ok() -> None:
    document = json.loads(GOLDEN_LINES[0])
    del document["status"]
    lines = [json.dumps(document, ensure_ascii=False), *GOLDEN_LINES[1:]]

    found = hypotheses_from_bytes(as_file(lines), golden_requests())

    assert found[KEY_N1] == Hypothesis("Үнэ 1 250,50 доллар.", "ok")


# ---- writing --------------------------------------------------------------------------------


def test_the_golden_file_round_trips_byte_for_byte() -> None:
    assert hypotheses_to_bytes(golden_hypotheses()) == GOLDEN_HYPOTHESES.read_bytes()


def test_write_hypotheses_writes_the_golden_bytes_and_returns_their_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hypotheses.jsonl"

    digest = write_hypotheses(golden_hypotheses(), path)

    assert path.read_bytes() == GOLDEN_HYPOTHESES.read_bytes()
    assert digest == GOLDEN_HYPOTHESES_SHA256
    assert list(tmp_path.iterdir()) == [path]


def test_write_hypotheses_makes_the_file_appear_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = robust_hypotheses.replace_file
    calls: list[tuple[Path, int]] = []

    def spy(path: Path, content: bytes, **kwargs: Any) -> None:
        calls.append((path, len(content)))
        real(path, content, **kwargs)

    monkeypatch.setattr(robust_hypotheses, "replace_file", spy)
    path = tmp_path / "hypotheses.jsonl"

    write_hypotheses(golden_hypotheses(), path)

    assert calls == [(path, GOLDEN_HYPOTHESES.stat().st_size)]


def test_a_line_is_sorted_keys_with_text_written_as_it_is() -> None:
    data = hypotheses_to_bytes({KEY_N1: Hypothesis("Үнэ", "ok")})

    assert (
        data
        == (
            '{"arm": "main", "id": "n1", "section": "numbers", "status": "ok", '
            '"text": "Үнэ"}\n'
        ).encode()
    )


def test_the_lines_follow_the_order_of_the_mapping() -> None:
    first = RequestKey("numbers", "n2", "main")
    mapping = {first: Hypothesis("a", "ok"), KEY_N1: Hypothesis("b", "ok")}

    lines = hypotheses_to_bytes(mapping).decode().split("\n")

    assert [json.loads(line)["id"] for line in lines[:-1]] == ["n2", "n1"]


def test_an_awkward_text_stays_one_line_and_comes_back() -> None:
    odd = (chr(0x2028), chr(0x85), chr(0), chr(0x1F600), user_defined_symbols()[-100])
    nasty = 'a "quote" \\ back\nline\ttab ' + " ".join(odd)
    requests = [Request(KEY_N1, "x", "eng_Latn", "khk_Cyrl", None)]

    data = hypotheses_to_bytes({KEY_N1: Hypothesis(nasty, "truncated")})
    found = hypotheses_from_bytes(data, requests)

    assert data.count(b"\n") == 1
    assert found[KEY_N1] == Hypothesis(nasty, "truncated")


def test_an_empty_mapping_writes_an_empty_file() -> None:
    assert hypotheses_to_bytes({}) == b""


def test_writing_a_lone_surrogate_names_the_request_and_not_the_text(
    tmp_path: Path,
) -> None:
    text = f"{MARKER} " + chr(0xD800)

    with pytest.raises(DataError) as info:
        write_hypotheses({KEY_N1: Hypothesis(text, "ok")}, tmp_path / "h.jsonl")

    assert "section 'numbers', id 'n1', arm 'main'" in str(info.value)
    assert MARKER not in str(info.value)
    assert not (tmp_path / "h.jsonl").exists()


def test_writing_more_than_the_reader_accepts_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hypotheses = golden_hypotheses()
    monkeypatch.setattr(robust_hypotheses, "MAX_HYPOTHESES_BYTES", 100)

    with pytest.raises(DataError, match="larger than"):
        write_hypotheses(hypotheses, tmp_path / "h.jsonl")

    assert not (tmp_path / "h.jsonl").exists()


# ---- refusals of a file ------------------------------------------------------------------


def test_a_missing_request_is_named_with_the_count_of_those_missing() -> None:
    message = refused(with_line(GOLDEN_LINES.index(line_for("script", "s3")), None))

    assert "section 'script', id 's3', arm 'main'" in message
    assert "1 request" in message


def test_the_first_missing_request_in_request_order_is_the_one_named() -> None:
    lines = [
        line
        for line in GOLDEN_LINES
        if json.loads(line)["id"] not in ("p2", "t1", "m3")
    ]

    message = refused(as_file(lines))

    assert "section 'placeholders', id 'p2', arm 'main'" in message
    assert "4 request" in message  # p2, t1 and the two arms of m3
    assert "t1" not in message


def test_an_empty_file_lacks_every_request() -> None:
    message = refused(b"")

    assert "section 'numbers', id 'n1', arm 'main'" in message
    assert "33 request" in message


def test_a_request_the_suite_does_not_have_is_refused_by_name() -> None:
    message = refused(as_file([*GOLDEN_LINES, line_of(id="zz9")]))

    assert "section 'numbers', id 'zz9', arm 'main'" in message
    assert "line 34" in message


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"arm": "converted"}, "arm 'converted'"),
        ({"arm": "cyrl"}, "arm 'cyrl'"),
        ({"section": "terms"}, "section 'terms'"),
    ],
)
def test_a_known_key_that_is_no_request_of_the_suite_is_refused(
    changes: dict[str, str], named: str
) -> None:
    message = refused(as_file([*GOLDEN_LINES, line_of(**changes)]))

    assert named in message


@pytest.mark.parametrize(
    "changes",
    [
        {"id": ODD_ID},
        {"section": MARKER},
        {"arm": MARKER},
        {"id": f"{MARKER} with spaces"},
        {"id": "ё" * 3},
    ],
)
def test_an_unknown_request_with_an_odd_name_is_refused_without_quoting_it(
    changes: dict[str, str],
) -> None:
    message = refused(as_file([*GOLDEN_LINES, line_of(**changes)]))

    assert "line 34" in message
    assert "not a request of the suite" in message


def test_a_request_answered_twice_is_refused_by_name() -> None:
    message = refused(as_file([*GOLDEN_LINES, GOLDEN_LINES[2]]))

    assert "section 'numbers', id 'n3', arm 'main'" in message
    assert "line 34" in message
    assert "more than once" in message


@pytest.mark.parametrize("status", ["fine", "OK", "", 7, None, ["ok"], MARKER])
def test_a_status_outside_the_five_is_refused(status: object) -> None:
    message = refused(with_line(0, changed(0, status=status)))

    assert "section 'numbers', id 'n1', arm 'main'" in message
    assert "line 1" in message
    assert "status" in message


def test_an_unknown_key_is_refused_without_naming_it() -> None:
    message = refused(with_line(0, changed(0, **{MARKER: 1})))

    assert "section 'numbers', id 'n1', arm 'main'" in message
    assert "not part of the format" in message


@pytest.mark.parametrize("field", ["section", "id", "arm", "text"])
def test_a_missing_field_is_refused_by_its_name(field: str) -> None:
    document = json.loads(GOLDEN_LINES[3])
    del document[field]

    message = refused(with_line(3, json.dumps(document, ensure_ascii=False)))

    assert f"field {field!r}" in message
    assert "is missing" in message
    assert "line 4" in message


@pytest.mark.parametrize("field", ["section", "id", "arm", "text", "status"])
@pytest.mark.parametrize("value", [5, None, True, ["a"], {"a": 1}])
def test_a_field_of_the_wrong_type_is_refused(field: str, value: object) -> None:
    message = refused(with_line(3, changed(3, **{field: value})))

    assert f"field {field!r}" in message
    assert "line 4" in message


@pytest.mark.parametrize(
    "line",
    [f"{MARKER} not json", "{", "", f'{{"text": "{MARKER}"', f"'{MARKER}'"],
)
def test_a_line_that_is_not_json_is_refused_by_its_number(line: str) -> None:
    message = refused(with_line(5, line))

    assert "line 6" in message
    assert "not valid JSON" in message


@pytest.mark.parametrize(
    "line", [f'["{MARKER}"]', f'"{MARKER}"', "7", "null", "true", "[]", "NaN"]
)
def test_a_line_that_is_not_an_object_is_refused_by_its_number(line: str) -> None:
    message = refused(with_line(5, line))

    assert "line 6" in message
    assert "not a JSON object" in message


def test_a_repeated_key_in_a_line_is_refused() -> None:
    repeated = (
        '{"arm": "main", "id": "n1", "section": "numbers", "status": "ok", '
        f'"text": "a", "text": "{MARKER}"}}'
    )

    message = refused(with_line(0, repeated))

    assert "line 1" in message
    assert "more than once" in message


def test_a_lone_surrogate_in_a_text_is_refused_by_name() -> None:
    escaped = (
        '{"arm": "main", "id": "n1", "section": "numbers", "status": "ok", '
        f'"text": "{MARKER}\\ud800"}}'
    )

    message = refused(with_line(0, escaped))

    assert "section 'numbers', id 'n1', arm 'main'" in message
    assert "lone surrogate" in message


def test_bytes_that_are_not_utf8_are_refused() -> None:
    data = GOLDEN_HYPOTHESES.read_bytes() + bytes([0xFF, 0xFE, 0x41])

    message = refused(data)

    assert "UTF-8" in message


def test_a_blank_line_inside_the_file_is_refused_by_its_number() -> None:
    lines = [*GOLDEN_LINES[:4], "", *GOLDEN_LINES[4:]]

    message = refused(as_file(lines))

    assert "line 5" in message
    assert "not valid JSON" in message


def test_a_file_over_the_cap_is_refused_before_it_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robust_hypotheses, "MAX_HYPOTHESES_BYTES", 100)

    message = refused(GOLDEN_HYPOTHESES.read_bytes())

    assert "larger than" in message


def test_the_cap_is_a_bound_on_what_is_read_not_on_what_is_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = golden_requests()
    path = tmp_path / "big.jsonl"
    path.write_bytes(b"x" * 1000)
    monkeypatch.setattr(robust_hypotheses, "MAX_HYPOTHESES_BYTES", 100)
    reads: list[int] = []
    real_open = Path.open

    class Spy:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def __enter__(self) -> Self:
            self._handle.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self._handle.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            reads.append(size)
            return self._handle.read(size)  # type: ignore[no-any-return]

    monkeypatch.setattr(
        Path, "open", lambda self, *a, **k: Spy(real_open(self, *a, **k))
    )

    with pytest.raises(DataError, match="larger than"):
        read_hypotheses(path, requests)

    assert reads == [101]


def test_a_file_of_exactly_the_cap_is_read_and_one_byte_more_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = len(GOLDEN_HYPOTHESES.read_bytes())
    requests = golden_requests()

    monkeypatch.setattr(robust_hypotheses, "MAX_HYPOTHESES_BYTES", size)
    assert hypotheses_from_bytes(GOLDEN_HYPOTHESES.read_bytes(), requests)

    monkeypatch.setattr(robust_hypotheses, "MAX_HYPOTHESES_BYTES", size - 1)
    with pytest.raises(DataError, match="larger than"):
        hypotheses_from_bytes(GOLDEN_HYPOTHESES.read_bytes(), requests)


def test_the_arms_of_a_request_are_the_ones_the_runner_makes() -> None:
    converted = {
        request.key.id: "x"
        for request in golden_requests()
        if request.key.section == "mong"
    }

    arms = {request.key.arm for request in suite_requests(golden_suite(), converted)}

    assert ARMS == ("main", "cyrl", "mong", "converted")
    assert arms == set(ARMS)


def test_the_cap_is_128_mib() -> None:
    assert MAX_HYPOTHESES_BYTES == 128 * 1024 * 1024


def test_requests_that_repeat_a_key_are_a_programming_error() -> None:
    request = golden_requests()[0]

    with pytest.raises(ValueError, match="more than once"):
        hypotheses_from_bytes(b"", [request, request])


@pytest.mark.parametrize("target", ["missing", "directory"])
def test_a_file_that_cannot_be_read_is_a_data_error(
    tmp_path: Path, target: str
) -> None:
    path = tmp_path / "nothing.jsonl" if target == "missing" else tmp_path

    with pytest.raises(DataError, match="cannot read the hypotheses file"):
        read_hypotheses(path, golden_requests())


# ---- the runner ---------------------------------------------------------------------------


def test_the_runner_makes_one_call_with_the_rows_of_the_requests_in_order() -> None:
    requests = golden_requests()
    engine = ScriptedEngine()
    options = DecodeOptions(beam_size=2, max_decoding_length=64, script_mask="always")

    run_requests(requests, engine, options)

    assert len(engine.calls) == 1
    rows, passed = engine.calls[0]
    assert passed is options
    assert rows == [
        Row(request.text, request.src_lang, request.tgt_lang, request.script)
        for request in requests
    ]
    assert Row("This is a book.", "eng_Latn", "khk_Cyrl", "Mong") in rows
    assert Row("Water.", None, "khk_Cyrl", None) in rows


def test_the_runner_returns_one_hypothesis_per_request_in_request_order() -> None:
    requests = golden_requests()

    found = run_requests(requests, ScriptedEngine(), DecodeOptions())

    assert list(found) == [request.key for request in requests]
    assert found[KEY_N1] == Hypothesis("[The price is 1,250.50 dollars.]", "ok")
    assert len(found) == len(requests)


def test_a_status_that_is_not_ok_is_returned_as_it_is_and_never_raises() -> None:
    requests = golden_requests()[:5]
    statuses = ("empty", "too_long", "truncated", "no_output", "ok")
    engine = ScriptedEngine(statuses=statuses)

    found = run_requests(requests, engine, DecodeOptions())

    assert [hypothesis.status for hypothesis in found.values()] == list(statuses)


@pytest.mark.parametrize("answers", [0, 2, 4, 33])
def test_a_short_or_long_answer_is_an_upstream_error_naming_the_counts(
    answers: int,
) -> None:
    requests = golden_requests()[:3]

    def answer(rows: Sequence[Row]) -> list[Translation]:
        template = ScriptedEngine().translate(golden_rows(max(answers, 1)))
        return template[:answers]

    with pytest.raises(UpstreamError) as info:
        run_requests(requests, ScriptedEngine(answer), DecodeOptions())

    assert f"{answers} " in str(info.value)
    assert "3 requests" in str(info.value)


def golden_rows(count: int) -> list[Row]:
    return [
        Row(request.text, request.src_lang, request.tgt_lang, request.script)
        for request in golden_requests()[:count]
    ]


def test_an_error_of_the_engine_propagates_unchanged() -> None:
    error = ConfigError("the model does not translate into x")

    def answer(rows: Sequence[Row]) -> list[Translation]:
        raise error

    with pytest.raises(ConfigError) as info:
        run_requests(golden_requests(), ScriptedEngine(answer), DecodeOptions())

    assert info.value is error


def test_requests_that_repeat_a_key_are_not_run() -> None:
    request = golden_requests()[0]
    engine = ScriptedEngine()

    with pytest.raises(ValueError, match="more than once"):
        run_requests([request, request], engine, DecodeOptions())

    assert engine.calls == []


@pytest.mark.usefixtures("engine_environment")
def test_a_real_engine_fits_the_seam(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)
    engine: Translates = fixture.engine
    requests = [
        Request(KEY_N1, "hello world", ENG, KHK, None),
        Request(RequestKey("numbers", "n2", "main"), "...", ENG, KHK, None),
    ]

    found = run_requests(requests, engine, DecodeOptions(beam_size=1))

    assert found[KEY_N1] == Hypothesis("hello world", "ok")
    assert found[RequestKey("numbers", "n2", "main")] == Hypothesis("", "empty")
