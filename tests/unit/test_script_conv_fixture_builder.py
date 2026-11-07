"""The fixture builder: golden selection on planted references, and independence
from the lexicon it regenerates.

Mongolian and invisible characters are written as `chr()` calls so the exact
code points stay reviewable.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from build_script_conv_fixtures import GOLDEN_ROWS, build

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import DataError
from rinchen.script_conv.fallback import development_split, transliterate_fallback
from tests.support.script_reference import (
    FIXTURE_DIR,
    REFERENCE_PATH,
    read_tsv_rows,
)

SCRIPT_A = chr(0x1820)
SCRIPT_MVS = chr(0x180E)
# A lexicon spelling with a vowel separator: a character the fallback never emits.
BEYOND_FALLBACK = SCRIPT_A + SCRIPT_MVS + SCRIPT_A

LEXICON_PATH = REPO_ROOT / "src" / "rinchen" / "script_conv" / "lexicon.tsv"
GOLDEN_FILE = "rules_golden.tsv"


def _planted_words() -> tuple[list[str], list[str]]:
    """Three-letter words, split into development and held-out, both sorted."""
    words = sorted(
        "".join(letters) for letters in itertools.product("абвгдзлмнр", repeat=3)
    )
    development = [word for word in words if development_split(word)]
    held_out = [word for word in words if not development_split(word)]
    return development, held_out


def _write_reference(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = ["cyrillic\ttraditional"] + [f"{word}\t{script}" for word, script in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _plant(
    tmp_path: Path, *, development: int, held_out_correct: int, held_out_wrong: int
) -> tuple[Path, list[str], list[str]]:
    """A reference with the given numbers of rows, and the words by role.

    Development rows carry the fallback spelling, except the last five, which carry a
    spelling with a vowel separator. Held-out rows are either spelled exactly like
    the fallback (correct) or one letter longer (wrong).
    """
    development_words, held_out_words = _planted_words()
    chosen_development = development_words[:development]
    correct = held_out_words[:held_out_correct]
    wrong = held_out_words[held_out_correct : held_out_correct + held_out_wrong]
    rows = [
        (
            word,
            BEYOND_FALLBACK
            if index >= development - 5
            else transliterate_fallback(word),
        )
        for index, word in enumerate(chosen_development)
    ]
    rows += [(word, transliterate_fallback(word)) for word in correct]
    rows += [(word, transliterate_fallback(word) + SCRIPT_A) for word in wrong]
    reference = tmp_path / "reference.tsv"
    _write_reference(reference, sorted(rows))
    return reference, chosen_development, correct


def _golden(directory: Path) -> list[tuple[str, str, str]]:
    return [
        (row["cyrillic"], row["expected"], row["path"])
        for row in read_tsv_rows(directory / GOLDEN_FILE)
    ]


def test_fewer_than_seventy_fallback_words_are_made_up_with_lexicon_rows(
    tmp_path: Path,
) -> None:
    reference, development, correct = _plant(
        tmp_path, development=110, held_out_correct=20, held_out_wrong=30
    )

    build(reference, tmp_path / "out" / "lexicon.tsv", tmp_path / "out")

    rows = _golden(tmp_path / "out")
    assert len(rows) == GOLDEN_ROWS
    fallback_rows = [row for row in rows if row[2] == "fallback"]
    lexicon_rows = [row for row in rows if row[2] == "lexicon"]
    assert [word for word, _, _ in fallback_rows] == correct
    assert len(lexicon_rows) == GOLDEN_ROWS - len(correct)
    assert {word for word, _, _ in lexicon_rows} <= set(development)
    assert len({word for word, _, _ in rows}) == GOLDEN_ROWS


def test_lexicon_rows_the_fallback_cannot_spell_are_taken_first(
    tmp_path: Path,
) -> None:
    reference, development, _ = _plant(
        tmp_path, development=110, held_out_correct=20, held_out_wrong=30
    )

    build(reference, tmp_path / "out" / "lexicon.tsv", tmp_path / "out")

    lexicon_words = {
        word for word, _, path in _golden(tmp_path / "out") if path == "lexicon"
    }
    assert set(development[-5:]) <= lexicon_words


def test_more_than_seventy_fallback_words_are_cut_to_seventy(tmp_path: Path) -> None:
    reference, _, correct = _plant(
        tmp_path, development=60, held_out_correct=80, held_out_wrong=0
    )

    build(reference, tmp_path / "out" / "lexicon.tsv", tmp_path / "out")

    rows = _golden(tmp_path / "out")
    assert len(rows) == GOLDEN_ROWS
    assert [word for word, _, path in rows if path == "fallback"] == correct[:70]
    assert sum(path == "lexicon" for _, _, path in rows) == 30


def test_a_reference_too_small_for_one_hundred_golden_rows_is_a_data_error(
    tmp_path: Path,
) -> None:
    reference, _, _ = _plant(
        tmp_path, development=30, held_out_correct=10, held_out_wrong=10
    )

    with pytest.raises(DataError, match=r"100 golden rows"):
        build(reference, tmp_path / "out" / "lexicon.tsv", tmp_path / "out")


def test_a_reference_too_small_for_golden_rows_leaves_every_output_untouched(
    tmp_path: Path,
) -> None:
    reference, _, _ = _plant(
        tmp_path, development=30, held_out_correct=10, held_out_wrong=10
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    earlier = {
        out_dir / "lexicon.tsv": "earlier lexicon\n",
        out_dir / GOLDEN_FILE: "earlier golden\n",
        out_dir / "tiny_words.tsv": "earlier tiny words\n",
    }
    for path, text in earlier.items():
        path.write_text(text, encoding="utf-8", newline="\n")

    with pytest.raises(DataError, match=r"100 golden rows"):
        build(reference, out_dir / "lexicon.tsv", out_dir)

    assert {
        path: path.read_text(encoding="utf-8") for path in out_dir.iterdir()
    } == earlier


def test_a_reference_too_small_for_golden_rows_creates_no_output_directory(
    tmp_path: Path,
) -> None:
    reference, _, _ = _plant(
        tmp_path, development=30, held_out_correct=10, held_out_wrong=10
    )

    with pytest.raises(DataError, match=r"100 golden rows"):
        build(reference, tmp_path / "lex" / "lexicon.tsv", tmp_path / "fixtures")

    assert not (tmp_path / "lex").exists()
    assert not (tmp_path / "fixtures").exists()


_REGENERATE_WITHOUT_THE_LEXICON = textwrap.dedent(
    """
    import sys
    from importlib import resources
    from pathlib import Path

    scripts_dir, reference, out_dir = sys.argv[1:]
    real_files = resources.files


    class _Hidden:
        def __init__(self, inner):
            self._inner = inner

        def joinpath(self, *names):
            if names == ("lexicon.tsv",):
                raise FileNotFoundError("lexicon.tsv is hidden")
            return self._inner.joinpath(*names)


    resources.files = lambda package: _Hidden(real_files(package))
    sys.path.insert(0, scripts_dir)
    import build_script_conv_fixtures as builder

    out = Path(out_dir)
    builder.build(Path(reference), out / "lexicon.tsv", out)
    assert "rinchen.script_conv.rules" not in sys.modules, "the builder imported rules"
    """
)


@pytest.mark.subprocess
def test_the_builder_regenerates_the_lexicon_while_the_old_one_is_unreadable(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _REGENERATE_WITHOUT_THE_LEXICON,
            str(REPO_ROOT / "scripts"),
            str(REFERENCE_PATH),
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (out_dir / "lexicon.tsv").read_bytes() == LEXICON_PATH.read_bytes()
    for name in (GOLDEN_FILE, "tiny_words.tsv"):
        assert (out_dir / name).read_bytes() == (FIXTURE_DIR / name).read_bytes()
