"""The suite file of D12 (spec T11, T12), the parts that are not a refusal: the golden suite
of `tests/fixtures/robustness/` round-trips byte for byte, a file is read ONCE and its digest
is the one of the bytes that were parsed, and `suite_requests` lists the requests the runner
sends in a fixed order. Every refusal of the reader is in `test_eval_robust_suite_refusals`.

The golden file is hand-written with known answers for every metric; every expectation here
is a literal of it."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from rinchen.errors import DataError
from rinchen.eval import robust_suite
from rinchen.eval.robust_spans import EntityTable
from rinchen.eval.robust_suite import (
    MAX_SUITE_BYTES,
    SECTIONS,
    SUITE_SCHEMA,
    Request,
    RequestKey,
    Suite,
    SuiteRow,
    TermPair,
    read_suite,
    suite_requests,
    write_suite,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "robustness"
GOLDEN_SUITE = GOLDEN / "golden_suite.json"
GOLDEN_SHA256 = "2767112dc06c61b62fbc66ad7336b3dffeff81c33ce7e7e124875a44503e75b0"
MARKER = "SECRET-MARKER-41"
MONGOLIAN_LETTERS = range(0x1820, 0x1879)


def golden_bytes() -> bytes:
    return GOLDEN_SUITE.read_bytes()


def golden_suite() -> Suite:
    return Suite.from_bytes(golden_bytes())


def row(row_id: str, **changes: object) -> SuiteRow:
    fields: dict[str, object] = {
        "id": row_id,
        "src_lang": "eng_Latn",
        "tgt_lang": "khk_Cyrl",
        "source": "It costs 15.",
        "reference": "Үнэ 15.",
        "category": "int",
    }
    fields.update(changes)
    return SuiteRow(**fields)  # type: ignore[arg-type]


def small_suite() -> Suite:
    return Suite(
        name="small",
        entities=EntityTable({}),
        sections={"numbers": (row("n1"), row("n2"))},
    )


# ---- the constants ---------------------------------------------------------------------


def test_the_constants_are_the_ones_of_the_spec() -> None:
    assert SUITE_SCHEMA == "rinchen.robustness.suite/1"
    assert SECTIONS == (
        "numbers",
        "placeholders",
        "entities",
        "terms",
        "script",
        "mong",
    )
    assert MAX_SUITE_BYTES == 8 * 1024 * 1024
    assert robust_suite.MAX_TEXT_CHARS == 2000


def test_the_golden_file_is_the_one_the_spike_wrote() -> None:
    assert hashlib.sha256(golden_bytes()).hexdigest() == GOLDEN_SHA256
    assert len(golden_bytes()) < 50 * 1024


# ---- the golden suite -------------------------------------------------------------------


def test_read_suite_returns_the_suite_and_the_digest_of_the_bytes_it_parsed() -> None:
    suite, digest = read_suite(GOLDEN_SUITE)

    assert digest == GOLDEN_SHA256
    assert suite.name == "golden"
    assert suite.row_counts() == {
        "numbers": 8,
        "placeholders": 6,
        "entities": 4,
        "terms": 3,
        "script": 6,
        "mong": 3,
    }
    assert list(suite.row_counts()) == list(SECTIONS)


def test_the_golden_entity_table() -> None:
    table = golden_suite().entities

    assert set(table.spellings) == {"bat", "ulaanbaatar"}
    assert table.spellings_of("ulaanbaatar", "eng_Latn") == (
        "Ulaanbaatar",
        "Ulan Bator",
    )
    assert table.spellings_of("ulaanbaatar", "khk_Cyrl") == ("Улаанбаатар",)
    assert table.spellings_of("bat", "eng_Latn") == ("Bat",)
    assert table.spellings_of("bat", "khk_Cyrl") == ("Бат",)


def rows_of(suite: Suite, section: str) -> dict[str, SuiteRow]:
    return {r.id: r for r in suite.sections[section]}


def test_golden_rows_come_back_as_written() -> None:
    suite = golden_suite()

    assert rows_of(suite, "numbers")["n1"] == SuiteRow(
        id="n1",
        src_lang="eng_Latn",
        tgt_lang="khk_Cyrl",
        source="The price is 1,250.50 dollars.",
        reference="Үнэ 1 250,50 доллар.",
        category="grouped",
    )
    assert [r.id for r in suite.sections["numbers"]] == [f"n{i}" for i in range(1, 9)]
    assert rows_of(suite, "entities")["e1"].entities == ("bat", "ulaanbaatar")
    assert rows_of(suite, "entities")["e3"].tgt_lang == "eng_Latn"
    assert rows_of(suite, "terms")["t2"].terms == (
        TermPair("table", "ширээ"),
        TermPair("window", "цонх"),
    )
    assert rows_of(suite, "terms")["t3"].reference == "Сандал энд байна."


def test_golden_script_rows_carry_their_script_and_a_missing_source_language() -> None:
    script_rows = rows_of(golden_suite(), "script")

    assert script_rows["s3"].src_lang is None
    assert script_rows["s3"].category == "untrained"
    assert script_rows["s1"].script is None
    assert script_rows["s4"].script == "Mong"
    assert script_rows["s5"].script == "Mong"
    assert script_rows["s6"].tgt_lang == "eng_Latn"
    assert all(r.reference is None for r in script_rows.values())


def test_golden_mong_rows_hold_the_source_in_traditional_script() -> None:
    mong_rows = rows_of(golden_suite(), "mong")

    assert set(mong_rows) == {"m1", "m2", "m3"}
    for mong_row in mong_rows.values():
        assert mong_row.src_lang == "khk_Cyrl"
        assert mong_row.source_mong is not None
        assert any(ord(c) in MONGOLIAN_LETTERS for c in mong_row.source_mong)
    assert mong_rows["m1"].source == "Энэ бол ном."
    assert mong_rows["m1"].reference == "This is a book."


def test_the_golden_bytes_are_what_to_bytes_writes() -> None:
    assert golden_suite().to_bytes() == golden_bytes()


def test_to_bytes_writes_sorted_keys_two_space_indent_text_and_one_final_lf() -> None:
    data = small_suite().to_bytes()

    text = data.decode("utf-8")
    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert "\r" not in text
    assert "Үнэ" in text, "text is written as text, never as \\u escapes"
    assert (
        text
        == json.dumps(json.loads(text), sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    )
    assert text.startswith('{\n  "entities": {},\n  "name": "small",\n')
    assert '\n  "schema": "rinchen.robustness.suite/1",\n' in text


def test_a_written_suite_reads_back_equal_with_the_digest_of_the_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "suite.json"

    written = write_suite(golden_suite(), path)
    suite, digest = read_suite(path)

    assert written == digest == GOLDEN_SHA256
    assert path.read_bytes() == golden_bytes()
    assert suite == golden_suite()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["suite.json"]


def test_write_suite_replaces_a_file_that_is_there(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    path.write_bytes(b"old")

    digest = write_suite(small_suite(), path)

    assert path.read_bytes() == small_suite().to_bytes()
    assert digest == hashlib.sha256(small_suite().to_bytes()).hexdigest()


def test_write_suite_goes_through_the_replacement_that_leaves_a_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "suite.json"
    calls: list[tuple[Path, bytes]] = []
    real_replace = robust_suite.replace_file

    def recording(target: Path, content: bytes) -> None:
        calls.append((target, content))
        real_replace(target, content)

    monkeypatch.setattr(robust_suite, "replace_file", recording)

    write_suite(small_suite(), path)

    assert calls == [(path, small_suite().to_bytes())]


def test_read_suite_reads_the_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "suite.json"
    path.write_bytes(golden_bytes())
    opened: list[Path] = []
    real_open = Path.open

    def spying_open(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        opened.append(self)
        return real_open(self, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(Path, "open", spying_open)

    read_suite(path)

    assert opened == [path]


def test_the_digest_is_the_one_of_the_bytes_that_were_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "suite.json"
    path.write_bytes(golden_bytes())
    parsed: list[bytes] = []
    real_from_bytes = Suite.from_bytes

    def recording(data: bytes) -> Suite:
        parsed.append(data)
        return real_from_bytes(data)

    monkeypatch.setattr(Suite, "from_bytes", staticmethod(recording))

    _, digest = read_suite(path)

    assert len(parsed) == 1
    assert hashlib.sha256(parsed[0]).hexdigest() == digest


def test_a_file_of_the_largest_size_is_read_and_one_byte_more_is_refused(
    tmp_path: Path,
) -> None:
    padding = b" " * (MAX_SUITE_BYTES - len(golden_bytes()))
    largest = golden_bytes() + padding
    path = tmp_path / "largest.json"
    path.write_bytes(largest)

    suite, digest = read_suite(path)

    assert suite == golden_suite()
    assert digest == hashlib.sha256(largest).hexdigest()
    assert Suite.from_bytes(largest) == golden_suite()
    with pytest.raises(DataError, match="8 MiB"):
        Suite.from_bytes(largest + b" ")


def test_a_file_over_the_cap_is_refused_before_anything_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "huge.json"
    path.write_bytes(
        golden_bytes() + b" " * (MAX_SUITE_BYTES - len(golden_bytes()) + 1)
    )

    def never_parse(*args: object, **kwargs: object) -> object:
        raise AssertionError("a file over the cap was parsed")

    monkeypatch.setattr(json, "loads", never_parse)

    with pytest.raises(DataError, match="8 MiB"):
        read_suite(path)


def test_a_file_that_is_missing_or_a_directory_is_a_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="cannot read"):
        read_suite(tmp_path / "missing.json")
    with pytest.raises(DataError, match="cannot read"):
        read_suite(tmp_path)


def test_write_suite_refuses_what_the_reader_would_refuse_for_its_size(
    tmp_path: Path,
) -> None:
    big = "x" * robust_suite.MAX_TEXT_CHARS
    rows = tuple(
        row(f"n{index}", source=big, reference=big)
        for index in range(MAX_SUITE_BYTES // (2 * len(big)) + 1)
    )
    suite = Suite(name="big", entities=EntityTable({}), sections={"numbers": rows})

    with pytest.raises(DataError, match="8 MiB"):
        write_suite(suite, tmp_path / "big.json")

    assert list(tmp_path.iterdir()) == []


# ---- the suite value --------------------------------------------------------------------


def test_row_counts_lists_the_sections_the_suite_has_in_the_order_of_the_format() -> (
    None
):
    suite = Suite(
        name="partial",
        entities=EntityTable({}),
        sections={"mong": (), "numbers": (row("n1"),)},
    )

    assert suite.row_counts() == {"numbers": 1, "mong": 0}
    assert list(suite.row_counts()) == ["numbers", "mong"]


def test_a_suite_is_frozen_and_its_sections_are_read_only_tuples() -> None:
    suite = Suite(
        name="small",
        entities=EntityTable({}),
        sections={"numbers": [row("n1")]},  # type: ignore[dict-item]
    )

    assert suite.sections["numbers"] == (row("n1"),)
    with pytest.raises(TypeError):
        suite.sections["terms"] = ()  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        suite.name = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        suite.sections["numbers"][0].source = "x"  # type: ignore[misc]


def test_a_suite_with_a_section_the_format_does_not_have_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="no such section"):
        Suite(
            name="typo",
            entities=EntityTable({}),
            sections={"number": (row("n1"),)},
        )


@pytest.mark.parametrize(
    ("section", "changes", "field"),
    [
        ("numbers", {"entities": ("bat",)}, "entities"),
        ("placeholders", {"terms": (TermPair("a", "b"),)}, "terms"),
        ("entities", {"script": "Mong"}, "script"),
        ("terms", {"source_mong": "x"}, "source_mong"),
        ("script", {"entities": ("bat",)}, "entities"),
        ("mong", {"script": "Cyrl"}, "script"),
    ],
)
def test_a_field_the_section_does_not_write_is_refused_not_dropped(
    section: str, changes: dict[str, object], field: str
) -> None:
    suite = Suite(
        name="stray",
        entities=EntityTable({}),
        sections={section: (row("r1", **changes),)},
    )

    with pytest.raises(ValueError) as raised:
        suite.to_bytes()

    message = str(raised.value)
    assert section in message and "index 0" in message and field in message


# ---- suite_requests ---------------------------------------------------------------------

GOLDEN_KEYS = (
    [("numbers", f"n{i}", "main") for i in range(1, 9)]
    + [("placeholders", f"p{i}", "main") for i in range(1, 7)]
    + [("entities", f"e{i}", "main") for i in range(1, 5)]
    + [("terms", f"t{i}", "main") for i in range(1, 4)]
    + [("script", f"s{i}", "main") for i in range(1, 7)]
    + [
        ("mong", mong_id, arm)
        for mong_id in ("m1", "m2", "m3")
        for arm in ("cyrl", "mong")
    ]
)


def by_key(requests: list[Request]) -> dict[tuple[str, str, str], Request]:
    return {tuple(request.key): request for request in requests}


def test_the_golden_suite_makes_33_requests_in_the_order_of_the_format() -> None:
    requests = suite_requests(golden_suite())

    assert len(requests) == 33
    assert [tuple(r.key) for r in requests] == GOLDEN_KEYS
    assert requests[0].key == ("numbers", "n1", "main")
    assert requests[-1].key == ("mong", "m3", "mong")
    assert isinstance(requests[0].key, RequestKey)


def test_a_main_request_carries_the_source_the_languages_and_the_script_of_its_row() -> (
    None
):
    requests = by_key(suite_requests(golden_suite()))

    assert requests[("numbers", "n1", "main")] == Request(
        key=RequestKey("numbers", "n1", "main"),
        text="The price is 1,250.50 dollars.",
        src_lang="eng_Latn",
        tgt_lang="khk_Cyrl",
        script=None,
    )
    assert requests[("script", "s3", "main")].src_lang is None
    assert requests[("script", "s3", "main")].text == "Water."
    assert requests[("script", "s4", "main")].script == "Mong"
    assert requests[("script", "s5", "main")].script == "Mong"
    assert requests[("script", "s1", "main")].script is None
    assert requests[("script", "s6", "main")].tgt_lang == "eng_Latn"
    assert requests[("entities", "e3", "main")].tgt_lang == "eng_Latn"
    assert requests[("terms", "t2", "main")].text == "The table and the window."


def test_a_mong_row_makes_a_cyrillic_request_and_a_traditional_script_request() -> None:
    suite = golden_suite()
    requests = by_key(suite_requests(suite))
    source_mong = rows_of(suite, "mong")["m1"].source_mong

    cyrl = requests[("mong", "m1", "cyrl")]
    mong = requests[("mong", "m1", "mong")]

    assert cyrl.text == "Энэ бол ном."
    assert (cyrl.src_lang, cyrl.tgt_lang, cyrl.script) == ("khk_Cyrl", "eng_Latn", None)
    assert mong.text == source_mong
    assert (mong.src_lang, mong.tgt_lang, mong.script) == ("mn_Mong", "eng_Latn", None)


def test_a_converted_text_adds_one_request_after_the_two_arms_of_its_row() -> None:
    requests = suite_requests(golden_suite(), converted={"m1": "Энэ бол ном."})

    assert len(requests) == 34
    keys = [tuple(r.key) for r in requests]
    position = keys.index(("mong", "m1", "converted"))
    assert keys[position - 2 : position] == [
        ("mong", "m1", "cyrl"),
        ("mong", "m1", "mong"),
    ]
    assert keys[position + 1] == ("mong", "m2", "cyrl")
    extra = requests[position]
    assert extra.text == "Энэ бол ном."
    assert (extra.src_lang, extra.tgt_lang, extra.script) == (
        "khk_Cyrl",
        "eng_Latn",
        None,
    )


def test_no_converted_text_and_an_empty_mapping_make_the_same_requests() -> None:
    suite = golden_suite()

    assert (
        suite_requests(suite, {})
        == suite_requests(suite, None)
        == suite_requests(suite)
    )


@pytest.mark.parametrize("converted_id", ["n1", "m9", MARKER])
def test_a_converted_id_that_is_no_mong_row_is_refused_without_naming_it(
    converted_id: str,
) -> None:
    with pytest.raises(DataError) as raised:
        suite_requests(golden_suite(), converted={"m1": "Энэ", converted_id: "text"})

    assert re.search(r"converted holds 1 id.*'mong' section", str(raised.value))
    assert MARKER not in str(raised.value)


def test_a_converted_text_for_a_suite_without_a_mong_section_is_refused() -> None:
    with pytest.raises(DataError, match="converted"):
        suite_requests(small_suite(), converted={"m1": "text"})


def test_a_suite_with_some_sections_makes_requests_for_those_only() -> None:
    suite = Suite(
        name="partial",
        entities=EntityTable({}),
        sections={"script": (), "numbers": (row("n1"), row("n2"))},
    )

    assert [tuple(r.key) for r in suite_requests(suite)] == [
        ("numbers", "n1", "main"),
        ("numbers", "n2", "main"),
    ]


def test_a_request_is_frozen() -> None:
    request = suite_requests(small_suite())[0]

    with pytest.raises(dataclasses.FrozenInstanceError):
        request.text = "x"  # type: ignore[misc]
