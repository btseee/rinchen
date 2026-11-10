"""The suite value of D13 (spec T2 and T3): what a `Suite` holds and cannot be made to do, the
views of a direction (`view`, `selection_view`: the devtest guard), the 16 standard directions,
the bytes the writer makes and how a file is read and written. The reader's refusals are in
`test_eval_suite_refusals`, the golden suite in `test_eval_suite_golden`. Every expectation is a
literal."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.eval import suites
from rinchen.eval.suites import (
    DirectionView,
    Suite,
    SuiteRow,
    SuiteSource,
    read_suite,
    standard_directions,
    write_suite,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SECRET = "SECRET-SENTENCE-91"
REVISION = "0123456789abcdef0123456789abcdef01234567"


def source(**changes: object) -> SuiteSource:
    fields: dict[str, object] = {
        "dataset": None,
        "revision": None,
        "pinned": False,
        "files": {},
        "missing_languages": (),
    }
    fields.update(changes)
    return SuiteSource(**fields)  # type: ignore[arg-type]


def row(row_id: str, doc: str | None = None, **text: str) -> SuiteRow:
    return SuiteRow(id=row_id, doc=doc, text=text)


def suite_of(
    rows: list[SuiteRow],
    directions: tuple[str, ...] = ("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl"),
    **changes: object,
) -> Suite:
    fields: dict[str, object] = {
        "name": "mini",
        "kind": "fixture",
        "benchmark": "mini",
        "split": "devtest",
        "source": source(),
        "directions": directions,
        "rows": rows,
    }
    fields.update(changes)
    return Suite(**fields)  # type: ignore[arg-type]


def mixed() -> Suite:
    """Five rows: both languages (a doc), English only, Khalkha only, both (no doc), English
    and German."""
    return suite_of(
        [
            row("r1", "d1", eng_Latn="one", khk_Cyrl="neg"),
            row("r2", None, eng_Latn="two"),
            row("r3", "d2", khk_Cyrl="gurav"),
            row("r4", None, eng_Latn="four", khk_Cyrl="dorov"),
            row("r5", "d3", eng_Latn="five", deu_Latn="fuenf"),
        ],
        ("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl", "eng_Latn-deu_Latn"),
    )


# ---- the value ----------------------------------------------------------------------------


def test_a_suite_cannot_be_changed_after_it_is_made() -> None:
    suite = mixed()

    with pytest.raises(dataclasses.FrozenInstanceError):
        suite.name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        suite.rows[0].text["eng_Latn"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        suite.source.files["a"] = "b"  # type: ignore[index]
    assert isinstance(suite.rows, tuple) and isinstance(suite.directions, tuple)


def test_a_suite_made_of_lists_holds_tuples_and_a_copy_of_the_text() -> None:
    text = {"eng_Latn": "one", "khk_Cyrl": "neg"}
    files = {"a.txt": "ab" * 32}
    suite = suite_of(
        [SuiteRow(id="r1", doc=None, text=text)],
        ["khk_Cyrl-eng_Latn"],  # type: ignore[arg-type]
        source=source(files=files, missing_languages=["deu_Latn"]),
    )

    text["eng_Latn"] = "changed"
    files["b.txt"] = "cd" * 32

    assert suite.directions == ("khk_Cyrl-eng_Latn",)
    assert suite.rows[0].text["eng_Latn"] == "one"
    assert dict(suite.source.files) == {"a.txt": "ab" * 32}
    assert suite.source.missing_languages == ("deu_Latn",)


def test_a_suite_cannot_be_hashed_like_the_suite_of_d12() -> None:
    suite = mixed()

    for value in (suite, suite.rows[0], suite.source):
        with pytest.raises(TypeError):
            hash(value)


def test_two_suites_with_the_same_content_are_equal_and_one_letter_makes_them_differ() -> (
    None
):
    first = suite_of([row("r1", None, eng_Latn="one", khk_Cyrl="neg")])
    same = suite_of([row("r1", None, khk_Cyrl="neg", eng_Latn="one")])
    other = suite_of([row("r1", None, eng_Latn="one", khk_Cyrl="nem")])

    assert first == same
    assert first != other


def test_purpose_follows_the_split() -> None:
    purposes = {
        split: suite_of([], split=split).purpose for split in ("dev", "devtest", "test")
    }

    assert purposes == {"dev": "selection", "devtest": "report", "test": "report"}


def test_languages_counts_the_sentences_of_each_language_in_sorted_order() -> None:
    counts = mixed().languages()

    assert counts == {"deu_Latn": 1, "eng_Latn": 4, "khk_Cyrl": 3}
    assert list(counts) == ["deu_Latn", "eng_Latn", "khk_Cyrl"]


# ---- the view of a direction --------------------------------------------------------------


def test_a_view_has_the_rows_that_have_both_languages_in_the_order_of_the_file() -> (
    None
):
    view = mixed().view("khk_Cyrl-eng_Latn")

    assert view == DirectionView(
        direction="khk_Cyrl-eng_Latn",
        src_lang="khk_Cyrl",
        tgt_lang="eng_Latn",
        ids=("r1", "r4"),
        docs=None,
        sources=("neg", "dorov"),
        references=("one", "four"),
    )


def test_the_reverse_direction_reads_the_same_rows_the_other_way_round() -> None:
    view = mixed().view("eng_Latn-khk_Cyrl")

    assert (view.src_lang, view.tgt_lang) == ("eng_Latn", "khk_Cyrl")
    assert view.ids == ("r1", "r4")
    assert view.sources == ("one", "four")
    assert view.references == ("neg", "dorov")


def test_a_row_that_lacks_a_language_is_skipped_from_that_direction_only() -> None:
    suite = mixed()

    assert suite.view("eng_Latn-deu_Latn").ids == ("r5",)
    assert suite.view("khk_Cyrl-eng_Latn").ids == ("r1", "r4")
    assert [r.id for r in suite.rows] == ["r1", "r2", "r3", "r4", "r5"]


def test_the_rows_of_a_view_keep_the_order_of_the_file_and_not_the_order_of_the_ids() -> (
    None
):
    suite = suite_of(
        [
            row("c", None, eng_Latn="3", khk_Cyrl="3k"),
            row("a", None, eng_Latn="1", khk_Cyrl="1k"),
            row("b", None, eng_Latn="2", khk_Cyrl="2k"),
        ]
    )

    assert suite.view("eng_Latn-khk_Cyrl").ids == ("c", "a", "b")
    assert suite.view("eng_Latn-khk_Cyrl").sources == ("3", "1", "2")


def test_documents_are_given_when_every_row_of_the_direction_has_one() -> None:
    suite = suite_of(
        [
            row("r1", "d1", eng_Latn="1", khk_Cyrl="1k"),
            row("r2", None, eng_Latn="2"),
            row("r3", "d2", eng_Latn="3", khk_Cyrl="3k"),
        ]
    )

    assert suite.view("khk_Cyrl-eng_Latn").docs == ("d1", "d2")


def test_documents_are_none_when_one_row_of_the_direction_has_none() -> None:
    suite = suite_of(
        [
            row("r1", "d1", eng_Latn="1", khk_Cyrl="1k"),
            row("r2", None, eng_Latn="2", khk_Cyrl="2k"),
            row("r3", "d2", eng_Latn="3", khk_Cyrl="3k"),
        ]
    )

    assert suite.view("khk_Cyrl-eng_Latn").docs is None
    assert suite.view("eng_Latn-khk_Cyrl").docs is None


def test_a_direction_with_no_row_has_empty_columns_and_no_documents() -> None:
    view = suite_of([row("r1", "d1", eng_Latn="1")]).view("khk_Cyrl-eng_Latn")

    assert view.ids == () and view.sources == () and view.references == ()
    assert view.docs is None


def test_a_view_is_made_of_tuples_of_equal_length() -> None:
    view = mixed().view("khk_Cyrl-eng_Latn")

    for column in (view.ids, view.sources, view.references):
        assert isinstance(column, tuple)
        assert len(column) == 2


def test_a_direction_the_suite_does_not_declare_is_a_config_error() -> None:
    suite = suite_of(
        [row("r1", None, eng_Latn=SECRET, khk_Cyrl=SECRET)], ("khk_Cyrl-eng_Latn",)
    )

    with pytest.raises(ConfigError) as raised:
        suite.view("eng_Latn-khk_Cyrl")

    message = str(raised.value)
    assert "mini" in message and "eng_Latn-khk_Cyrl" in message
    assert SECRET not in message


@pytest.mark.parametrize("direction", ["", "khk_Cyrl", f"{SECRET}-eng_Latn", "x-y-z"])
def test_a_direction_that_is_not_well_formed_is_not_quoted(direction: str) -> None:
    with pytest.raises(ConfigError) as raised:
        mixed().view(direction)

    message = str(raised.value)
    assert "no direction" in message
    assert SECRET not in message


# ---- the devtest guard --------------------------------------------------------------------


@pytest.mark.parametrize("split", ["devtest", "test"])
def test_a_selection_cannot_read_a_suite_that_is_for_reporting(split: str) -> None:
    suite = suite_of(
        [row("r1", None, eng_Latn=SECRET, khk_Cyrl=SECRET)],
        ("khk_Cyrl-eng_Latn",),
        split=split,
    )

    with pytest.raises(ConfigError) as raised:
        suite.selection_view("khk_Cyrl-eng_Latn")

    message = str(raised.value)
    assert "devtest" in message and "reporting" in message
    assert SECRET not in message


def test_a_selection_cannot_read_a_devtest_suite_even_for_a_direction_it_lacks() -> (
    None
):
    suite = suite_of([row("r1", None, eng_Latn="1", khk_Cyrl="1k")])

    with pytest.raises(ConfigError, match="devtest"):
        suite.selection_view("khk_Cyrl-deu_Latn")


@pytest.mark.parametrize("kind", ["benchmark", "fixture"])
def test_a_selection_reads_the_rows_of_a_dev_suite_as_view_does(kind: str) -> None:
    suite = dataclasses.replace(mixed(), split="dev", kind=kind)

    for direction in suite.directions:
        assert suite.selection_view(direction) == suite.view(direction)
    assert suite.selection_view("khk_Cyrl-eng_Latn").sources == ("neg", "dorov")


def test_a_selection_of_a_dev_suite_still_refuses_an_undeclared_direction() -> None:
    suite = dataclasses.replace(mixed(), split="dev")

    with pytest.raises(ConfigError, match="no direction"):
        suite.selection_view("khk_Cyrl-tur_Latn")


def test_a_human_suite_is_never_a_selection() -> None:
    suite = dataclasses.replace(mixed(), kind="human", split="test")

    with pytest.raises(ConfigError, match="reporting"):
        suite.selection_view("khk_Cyrl-eng_Latn")


# ---- the standard directions --------------------------------------------------------------


def test_the_standard_directions_are_the_sixteen_of_khalkha_and_its_eight_partners() -> (
    None
):
    directions = standard_directions()

    assert directions == (
        "khk_Cyrl-deu_Latn",
        "deu_Latn-khk_Cyrl",
        "khk_Cyrl-eng_Latn",
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-jpn_Jpan",
        "jpn_Jpan-khk_Cyrl",
        "khk_Cyrl-kaz_Cyrl",
        "kaz_Cyrl-khk_Cyrl",
        "khk_Cyrl-kor_Hang",
        "kor_Hang-khk_Cyrl",
        "khk_Cyrl-rus_Cyrl",
        "rus_Cyrl-khk_Cyrl",
        "khk_Cyrl-tur_Latn",
        "tur_Latn-khk_Cyrl",
        "khk_Cyrl-zho_Hans",
        "zho_Hans-khk_Cyrl",
    )


def test_no_standard_direction_holds_traditional_script_or_a_pair_of_partners() -> None:
    directions = standard_directions()

    assert len(set(directions)) == 16
    assert not any("mn_Mong" in direction for direction in directions)
    assert all("khk_Cyrl" in direction.split("-") for direction in directions)


def test_the_partners_come_from_the_synthetic_data_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        suites,
        "supported_partner_languages",
        lambda: frozenset({"tur_Latn", "eng_Latn"}),
    )

    assert standard_directions() == (
        "khk_Cyrl-eng_Latn",
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-tur_Latn",
        "tur_Latn-khk_Cyrl",
    )


# ---- the bytes ----------------------------------------------------------------------------


MINI_BYTES = """{
  "benchmark": "mini",
  "directions": [
    "khk_Cyrl-eng_Latn"
  ],
  "kind": "fixture",
  "name": "mini",
  "rows": [
    {
      "doc": null,
      "id": "a1",
      "text": {
        "eng_Latn": "Hi",
        "khk_Cyrl": "Сайн"
      }
    }
  ],
  "schema": "rinchen.eval.suite/1",
  "source": {
    "dataset": null,
    "files": {},
    "missing_languages": [],
    "pinned": false,
    "revision": null
  },
  "split": "devtest"
}
"""


def test_the_writer_makes_sorted_keys_two_space_indent_text_and_one_final_lf() -> None:
    suite = suite_of(
        [SuiteRow(id="a1", doc=None, text={"khk_Cyrl": "Сайн", "eng_Latn": "Hi"})],
        ("khk_Cyrl-eng_Latn",),
    )

    assert suite.to_bytes() == MINI_BYTES.encode("utf-8")


def test_the_writer_keeps_the_order_of_the_directions_and_of_the_rows() -> None:
    suite = suite_of(
        [
            row("z", None, eng_Latn="1", khk_Cyrl="1k"),
            row("a", None, eng_Latn="2", khk_Cyrl="2k"),
        ],
        ("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl"),
    )

    written = json.loads(suite.to_bytes())

    assert written["directions"] == ["khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl"]
    assert [item["id"] for item in written["rows"]] == ["z", "a"]


def test_a_suite_with_every_field_set_reads_back_equal_and_writes_the_same_bytes() -> (
    None
):
    suite = suite_of(
        [
            row("r1", "d1", eng_Latn="one", khk_Cyrl="neg"),
            row("r2", None, eng_Latn="two", khk_Cyrl="hoyor", deu_Latn="zwei"),
        ],
        ("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl", "eng_Latn-deu_Latn"),
        kind="benchmark",
        benchmark="flores_plus",
        split="dev",
        source=source(
            dataset="openlanguagedata/flores_plus",
            revision=REVISION,
            pinned=True,
            files={
                "khk_Cyrl/shard-0.parquet": "ab" * 32,
                "eng_Latn/shard-0.parquet": "cd" * 32,
            },
            missing_languages=("kor_Hang", "tur_Latn"),
        ),
    )

    data = suite.to_bytes()

    assert Suite.from_bytes(data) == suite
    assert Suite.from_bytes(data).to_bytes() == data


# ---- writing and reading a file -----------------------------------------------------------


def test_a_written_suite_reads_back_equal_with_the_digest_of_the_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "suite.json"
    suite = mixed()

    written = write_suite(suite, path)
    read, digest = read_suite(path)

    assert written == digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read == suite
    assert path.read_bytes() == suite.to_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["suite.json"]


def test_write_suite_replaces_a_file_that_is_there(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    path.write_bytes(b"old")

    digest = write_suite(mixed(), path)

    assert path.read_bytes() == mixed().to_bytes()
    assert digest == hashlib.sha256(mixed().to_bytes()).hexdigest()


def test_write_suite_goes_through_the_replacement_that_leaves_a_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "suite.json"
    calls: list[tuple[Path, bytes]] = []
    real_replace = suites.replace_file

    def recording(target: Path, content: bytes) -> None:
        calls.append((target, content))
        real_replace(target, content)

    monkeypatch.setattr(suites, "replace_file", recording)

    write_suite(mixed(), path)

    assert calls == [(path, mixed().to_bytes())]


def test_a_crash_while_writing_leaves_the_earlier_file_and_no_part_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "suite.json"
    path.write_bytes(b"earlier suite")

    def crash(self: Path, target: object) -> Path:
        raise OSError("the disk went away")

    monkeypatch.setattr(Path, "replace", crash)

    with pytest.raises(OSError, match="the disk went away"):
        write_suite(mixed(), path)

    assert path.read_bytes() == b"earlier suite"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["suite.json"]


def test_write_suite_refuses_a_suite_larger_than_the_reader_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suites, "MAX_SUITE_BYTES", 100)

    with pytest.raises(DataError, match="larger than"):
        write_suite(mixed(), tmp_path / "big.json")

    assert list(tmp_path.iterdir()) == []


def test_read_suite_reads_the_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "suite.json"
    path.write_bytes(mixed().to_bytes())
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
    path.write_bytes(mixed().to_bytes())
    parsed: list[bytes] = []
    real_from_bytes = Suite.from_bytes

    def recording(data: bytes) -> Suite:
        parsed.append(data)
        return real_from_bytes(data)

    monkeypatch.setattr(Suite, "from_bytes", staticmethod(recording))

    _, digest = read_suite(path)

    assert len(parsed) == 1
    assert hashlib.sha256(parsed[0]).hexdigest() == digest


def test_the_digest_is_the_one_of_the_file_even_when_the_file_is_not_in_canonical_form(
    tmp_path: Path,
) -> None:
    path = tmp_path / "suite.json"
    one_line = json.dumps(json.loads(mixed().to_bytes()), ensure_ascii=False).encode()
    path.write_bytes(one_line)

    suite, digest = read_suite(path)

    assert one_line != mixed().to_bytes()
    assert suite == mixed()
    assert digest == hashlib.sha256(one_line).hexdigest()
    assert digest != hashlib.sha256(mixed().to_bytes()).hexdigest()


def test_a_file_that_is_not_a_suite_is_a_data_error(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    path.write_bytes(b"{}")

    with pytest.raises(DataError, match="not an evaluation suite"):
        read_suite(path)
