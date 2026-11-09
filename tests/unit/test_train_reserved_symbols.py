"""Tests for `eole_run.add_reserved_symbols`: the vocabulary invariant (D9, T1).

Every Eole vocabulary file must hold every reserved symbol of the SentencePiece model,
or a token that the data never showed, `<sep>` for one, is `<unk>` to the model. The
helper appends what is missing to the file `eole build_vocab` wrote. These tests pin
what it does to the bytes of that file; the callers (`run_training`, `launch`) have
their tests beside their own.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rinchen.errors import DataError
from rinchen.spm.symbols import user_defined_symbols
from rinchen.train import eole_run
from rinchen.train.eole_run import add_reserved_symbols

TAB = "\t"


@pytest.fixture
def vocab(tmp_path: Path) -> Path:
    return tmp_path / "run" / "vocab.src"


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---- what is appended -----------------------------------------------------------


def test_the_missing_symbols_are_appended_in_the_given_order_with_a_count_of_one(
    vocab: Path,
) -> None:
    _write(vocab, b"a\t5\nb\t3\n")

    appended = add_reserved_symbols(vocab, ["<y>", "a", "<x>"])

    assert appended == ("<y>", "<x>")
    assert vocab.read_bytes() == b"a\t5\nb\t3\n<y>\t1\n<x>\t1\n"


def test_nothing_missing_leaves_the_file_unwritten(
    vocab: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"<x>\t9\na\t5\n<y>\t2\n"
    _write(vocab, content)
    before = vocab.stat()

    def refuse(path: Path, data: bytes) -> None:
        raise AssertionError("the file was rewritten")

    monkeypatch.setattr(eole_run, "replace_file", refuse)

    appended = add_reserved_symbols(vocab, ["<x>", "<y>"])

    assert appended == ()
    assert vocab.read_bytes() == content
    after = vocab.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


@pytest.mark.parametrize("mode", [0o640, 0o600, 0o444, 0o664])
def test_the_file_keeps_its_permissions(vocab: Path, mode: int) -> None:
    _write(vocab, b"a\t5\n")
    vocab.chmod(mode)

    add_reserved_symbols(vocab, ["<x>"])

    assert vocab.stat().st_mode & 0o7777 == mode
    assert vocab.read_bytes() == b"a\t5\n<x>\t1\n"


def test_a_vocabulary_that_is_a_link_becomes_a_regular_file_with_its_targets_mode(
    vocab: Path, tmp_path: Path
) -> None:
    # The rename replaces the link, not the file it points to, which is left as it was.
    target = _write(tmp_path / "elsewhere" / "vocab.txt", b"a\t5\n")
    target.chmod(0o640)
    vocab.parent.mkdir(parents=True)
    vocab.symlink_to(target)

    add_reserved_symbols(vocab, ["<x>"])

    assert not vocab.is_symlink()
    assert vocab.stat().st_mode & 0o7777 == 0o640
    assert vocab.read_bytes() == b"a\t5\n<x>\t1\n"
    assert target.read_bytes() == b"a\t5\n"


def test_a_second_call_appends_nothing(vocab: Path) -> None:
    _write(vocab, b"a\t5\n")
    add_reserved_symbols(vocab, ["<x>", "<y>"])
    once = vocab.read_bytes()

    assert add_reserved_symbols(vocab, ["<x>", "<y>"]) == ()

    assert vocab.read_bytes() == once


def test_a_symbol_given_twice_is_appended_once(vocab: Path) -> None:
    _write(vocab, b"a\t5\n")

    appended = add_reserved_symbols(vocab, ["<x>", "<y>", "<x>"])

    assert appended == ("<x>", "<y>")
    assert vocab.read_bytes() == b"a\t5\n<x>\t1\n<y>\t1\n"


def test_no_symbols_leave_the_file_alone(vocab: Path) -> None:
    _write(vocab, b"a\t5")

    assert add_reserved_symbols(vocab, []) == ()

    assert vocab.read_bytes() == b"a\t5"


# ---- what counts as present -----------------------------------------------------


def test_only_the_first_column_of_a_line_is_the_token(vocab: Path) -> None:
    # `<x>` is inside another token, in a count column and a part of a longer symbol:
    # none of those is the token `<x>`.
    _write(vocab, b"a<x>b\t3\nfoo\t<x>\n<x>_long\t4\n")

    appended = add_reserved_symbols(vocab, ["<x>"])

    assert appended == ("<x>",)
    assert vocab.read_bytes().endswith(b"<x>_long\t4\n<x>\t1\n")


def test_a_symbol_that_is_the_start_of_another_token_is_still_missing(
    vocab: Path,
) -> None:
    _write(vocab, b"<2eng_Latn>\t4\n")

    appended = add_reserved_symbols(vocab, ["<2eng", "<2eng_Latn>"])

    assert appended == ("<2eng",)
    assert vocab.read_bytes() == b"<2eng_Latn>\t4\n<2eng\t1\n"


def test_a_symbol_that_is_the_start_of_a_symbol_in_the_file_is_missing_the_other_way(
    vocab: Path,
) -> None:
    _write(vocab, b"<2eng\t4\n")

    assert add_reserved_symbols(vocab, ["<2eng_Latn>"]) == ("<2eng_Latn>",)


def test_the_placeholders_with_their_white_square_brackets_match_exactly(
    vocab: Path,
) -> None:
    open_bracket, close_bracket = chr(0x27E6), chr(0x27E7)
    present = f"{open_bracket}7{close_bracket}"
    other_brackets = "[8]"  # ASCII look-alikes are other tokens
    _write(vocab, f"{present}\t2\n{other_brackets}\t2\n".encode())

    appended = add_reserved_symbols(
        vocab,
        [present, f"{open_bracket}8{close_bracket}"],
    )

    assert appended == (f"{open_bracket}8{close_bracket}",)


# ---- the shape of the file ------------------------------------------------------


def test_a_file_without_a_final_newline_gets_one_before_the_symbols(
    vocab: Path,
) -> None:
    _write(vocab, b"a\t5\nb\t3")

    add_reserved_symbols(vocab, ["<x>"])

    assert vocab.read_bytes() == b"a\t5\nb\t3\n<x>\t1\n"


def test_crlf_line_endings_are_kept_and_the_new_lines_end_in_lf(vocab: Path) -> None:
    # The existing bytes are not normalised; Eole drops a `\r` at the end of a line.
    _write(vocab, b"a\t5\r\n<x>\t3\r\n")

    appended = add_reserved_symbols(vocab, ["<x>", "<y>"])

    assert appended == ("<y>",)
    assert vocab.read_bytes() == b"a\t5\r\n<x>\t3\r\n<y>\t1\n"


def test_a_symbol_ending_a_crlf_line_without_a_count_is_present(vocab: Path) -> None:
    _write(vocab, b"a\r\n<x>\r\n")

    assert add_reserved_symbols(vocab, ["<x>"]) == ()


def test_bytes_that_are_not_utf8_are_kept(vocab: Path) -> None:
    _write(vocab, b"\xff\xfe\t3\na\t2\n")

    add_reserved_symbols(vocab, ["<x>"])

    assert vocab.read_bytes() == b"\xff\xfe\t3\na\t2\n<x>\t1\n"


def test_a_file_without_counts_gets_the_bare_symbols(vocab: Path) -> None:
    # Eole reads a file whose first line has no count as one token per line, and then
    # `<x><TAB>1` would be a token of its own.
    _write(vocab, b"a\nb\n")

    appended = add_reserved_symbols(vocab, ["b", "<x>"])

    assert appended == ("<x>",)
    assert vocab.read_bytes() == b"a\nb\n<x>\n"


def test_a_file_with_counts_after_a_space_gets_tab_separated_counts(
    vocab: Path,
) -> None:
    _write(vocab, b"a 5\nb 3\n")

    appended = add_reserved_symbols(vocab, ["b", "<x>"])

    assert appended == ("<x>",)
    assert vocab.read_bytes() == b"a 5\nb 3\n<x>\t1\n"


# ---- the real symbols ------------------------------------------------------------


def test_every_reserved_symbol_of_the_project_is_added_to_a_vocabulary_without_them(
    vocab: Path,
) -> None:
    symbols = user_defined_symbols()
    _write(vocab, b"\xe2\x96\x81a\t9\n" + f"{symbols[3]}\t2\n".encode())

    appended = add_reserved_symbols(vocab, symbols)

    assert appended == tuple(symbols[:3] + symbols[4:])
    tokens = [line.split(TAB)[0] for line in vocab.read_text("utf-8").splitlines()]
    assert sorted(tokens[1:]) == sorted(symbols)
    assert len(tokens) == len(set(tokens))


# ---- failure --------------------------------------------------------------------


def test_a_failed_write_leaves_the_original_and_no_part_file(
    vocab: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(vocab, b"a\t5\n")

    def fail(self: Path, target: str | os.PathLike[str]) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail)

    with pytest.raises(OSError, match="disk full"):
        add_reserved_symbols(vocab, ["<x>"])

    assert vocab.read_bytes() == b"a\t5\n"
    assert sorted(path.name for path in vocab.parent.iterdir()) == ["vocab.src"]


def test_a_missing_vocabulary_file_is_a_data_error_and_nothing_is_created(
    vocab: Path,
) -> None:
    with pytest.raises(DataError, match="vocabulary file") as info:
        add_reserved_symbols(vocab, ["<x>"])

    assert str(vocab) in str(info.value)
    assert not vocab.parent.exists()


def test_a_vocabulary_that_is_a_directory_is_a_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="vocabulary file"):
        add_reserved_symbols(tmp_path, ["<x>"])


@pytest.mark.parametrize(
    "content",
    [b"", b"\n", b"\n\n\n", b"\r\n", b" \t \n", b"  ", b"\n\r\n  \n"],
    ids=[
        "no bytes",
        "one LF",
        "LFs",
        "one CRLF",
        "spaces and a tab",
        "spaces",
        "blank lines of every kind",
    ],
)
def test_a_vocabulary_with_no_token_is_a_data_error_and_is_left_as_it_is(
    vocab: Path, content: bytes
) -> None:
    # `eole build_vocab` that was cut short leaves such a file; with the symbols alone
    # added, every real piece would be `<unk>` to the model, and Eole itself refuses an
    # empty vocabulary.
    _write(vocab, content)

    with pytest.raises(DataError, match="holds no token") as info:
        add_reserved_symbols(vocab, ["<x>", "<y>"])

    assert str(vocab) in str(info.value)
    assert vocab.read_bytes() == content
    assert sorted(path.name for path in vocab.parent.iterdir()) == ["vocab.src"]


def test_a_vocabulary_with_no_token_is_refused_even_when_no_symbol_is_asked_for(
    vocab: Path,
) -> None:
    _write(vocab, b"")

    with pytest.raises(DataError, match="holds no token"):
        add_reserved_symbols(vocab, [])


def test_one_token_among_blank_lines_is_a_vocabulary(vocab: Path) -> None:
    _write(vocab, b"\n  \na\t5\n\n")

    appended = add_reserved_symbols(vocab, ["<x>"])

    assert appended == ("<x>",)


@pytest.mark.parametrize(
    "bad",
    ["", "a\tb", "a\nb", "a\rb", "\t", "<x>\n"],
    ids=["empty", "tab", "lf", "cr", "only a tab", "trailing lf"],
)
def test_a_symbol_that_would_break_the_line_format_is_refused_before_any_write(
    vocab: Path, bad: str
) -> None:
    _write(vocab, b"a\t5\n")

    with pytest.raises(ValueError, match="symbol"):
        add_reserved_symbols(vocab, ["<x>", bad])

    assert vocab.read_bytes() == b"a\t5\n"
