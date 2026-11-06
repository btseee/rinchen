"""Unit tests for `rinchen.spm.corpus` (corpus discovery and balanced sampling).

Sampling tests name every line `<language>-<index>`, so per-language counts and
original order can be read straight off the result. Code points that matter
(U+202F, U+2028, U+0085, U+2029) are spelled by name, never as literals.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError, DataError
from rinchen.spm.corpus import discover_corpus, read_lines, sample_balanced

TINY_CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "spm" / "tiny_corpus"

NNBSP = "\N{NARROW NO-BREAK SPACE}"


def _write_files(directory: Path, files: dict[str, str]) -> Path:
    """Create `directory` holding `files` (name -> text), writing text verbatim."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8", newline="")
    return directory


def _language_lines(language: str, count: int) -> list[str]:
    return [f"{language}-{index}" for index in range(count)]


def _indices_of(sample: Sequence[str], language: str) -> list[int]:
    """Original indices of `language`'s lines in `sample`, in sample order."""
    prefix = f"{language}-"
    return [
        int(line.removeprefix(prefix)) for line in sample if line.startswith(prefix)
    ]


def _count_per_language(sample: Sequence[str], languages: Sequence[str]) -> list[int]:
    return [len(_indices_of(sample, language)) for language in languages]


# --- discover_corpus -------------------------------------------------------


def test_discover_returns_sorted_code_to_path_ignoring_other_extensions(
    tmp_path: Path,
) -> None:
    corpus_dir = _write_files(
        tmp_path / "corpus",
        {
            "rus_Cyrl.txt": "line\n",
            "khk_Cyrl.txt": "line\n",
            "eng_Latn.txt": "line\n",
            "notes.md": "not a corpus\n",
            "eng_Latn.txt.bak": "not a corpus\n",
        },
    )
    (corpus_dir / "archive.txt").mkdir()
    _write_files(corpus_dir / "nested", {"zho_Hans.txt": "line\n"})

    corpus = discover_corpus(corpus_dir)

    assert list(corpus) == ["eng_Latn", "khk_Cyrl", "rus_Cyrl"]
    assert corpus == {
        "eng_Latn": corpus_dir / "eng_Latn.txt",
        "khk_Cyrl": corpus_dir / "khk_Cyrl.txt",
        "rus_Cyrl": corpus_dir / "rus_Cyrl.txt",
    }


@pytest.mark.parametrize("file_name", ["eng_latn.txt", "notes.txt"])
def test_unknown_stem_raises_config_error_naming_file(
    tmp_path: Path, file_name: str
) -> None:
    corpus_dir = _write_files(
        tmp_path / "corpus", {"khk_Cyrl.txt": "line\n", file_name: "line\n"}
    )

    with pytest.raises(ConfigError, match=re.escape(file_name)):
        discover_corpus(corpus_dir)


def test_missing_directory_raises_config_error(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_corpus"

    with pytest.raises(ConfigError, match=re.escape(str(missing))):
        discover_corpus(missing)


def test_path_that_is_a_file_raises_config_error(tmp_path: Path) -> None:
    _write_files(tmp_path, {"corpus.txt": "line\n"})
    not_a_directory = tmp_path / "corpus.txt"

    with pytest.raises(ConfigError, match=re.escape(str(not_a_directory))):
        discover_corpus(not_a_directory)


@pytest.mark.parametrize(
    "files",
    [{}, {"notes.md": "not a corpus\n"}],
    ids=["empty-directory", "only-other-extensions"],
)
def test_no_txt_files_raises_data_error(tmp_path: Path, files: dict[str, str]) -> None:
    corpus_dir = _write_files(tmp_path / "corpus", files)

    with pytest.raises(DataError, match=re.escape(str(corpus_dir))):
        discover_corpus(corpus_dir)


@pytest.mark.parametrize(
    "blank_text",
    ["", "\n\n", "  \n\t\r\n"],
    ids=["zero-bytes", "newlines-only", "whitespace-only"],
)
def test_blank_only_file_raises_data_error_naming_language(
    tmp_path: Path, blank_text: str
) -> None:
    corpus_dir = _write_files(
        tmp_path / "corpus",
        {"eng_Latn.txt": "line\n", "khk_Cyrl.txt": blank_text},
    )

    with pytest.raises(DataError, match="khk_Cyrl"):
        discover_corpus(corpus_dir)


def test_discover_accepts_the_committed_tiny_corpus() -> None:
    corpus = discover_corpus(TINY_CORPUS_DIR)

    assert list(corpus) == ["eng_Latn", "khk_Cyrl", "mn_Mong", "rus_Cyrl"]
    assert all(path == TINY_CORPUS_DIR / f"{code}.txt" for code, path in corpus.items())


# --- read_lines ------------------------------------------------------------


def test_read_lines_strips_crlf_and_skips_blanks(tmp_path: Path) -> None:
    _write_files(tmp_path, {"corpus.txt": "alpha\r\n\r\nbeta\n  \n\t\r\ngamma"})

    assert read_lines(tmp_path / "corpus.txt") == ["alpha", "beta", "gamma"]


def test_read_lines_keeps_nnbsp_at_a_line_edge(tmp_path: Path) -> None:
    _write_files(tmp_path, {"corpus.txt": f"{NNBSP}abc{NNBSP}\r\nplain\n"})

    assert read_lines(tmp_path / "corpus.txt") == [f"{NNBSP}abc{NNBSP}", "plain"]


@pytest.mark.parametrize(
    "separator",
    [
        "\N{LINE SEPARATOR}",
        "\N{PARAGRAPH SEPARATOR}",
        "\N{NEXT LINE}",
        "\x0b",
        "\x0c",
        "\x1e",
        "\r",
    ],
    ids=["U+2028", "U+2029", "U+0085", "VT", "FF", "RS", "lone-CR"],
)
def test_read_lines_splits_only_on_newline(tmp_path: Path, separator: str) -> None:
    _write_files(tmp_path, {"corpus.txt": f"left{separator}right\nnext\n"})

    assert read_lines(tmp_path / "corpus.txt") == [f"left{separator}right", "next"]


# --- sample_balanced -------------------------------------------------------


@pytest.mark.parametrize(
    "input_sentence_size", [5, 100], ids=["at-budget", "under-budget"]
)
def test_all_lines_returned_when_under_budget(input_sentence_size: int) -> None:
    lines_by_lang = {
        "khk_Cyrl": _language_lines("khk_Cyrl", 3),
        "eng_Latn": _language_lines("eng_Latn", 2),
    }

    sample = sample_balanced(
        lines_by_lang, input_sentence_size=input_sentence_size, temperature=0.3, seed=7
    )

    assert sample == [*lines_by_lang["khk_Cyrl"], *lines_by_lang["eng_Latn"]]


def test_empty_mapping_returns_empty_list() -> None:
    assert sample_balanced({}, input_sentence_size=10, temperature=0.3, seed=7) == []


@pytest.mark.parametrize("input_sentence_size", [0, -1])
def test_non_positive_input_sentence_size_raises_value_error(
    input_sentence_size: int,
) -> None:
    with pytest.raises(ValueError, match="input_sentence_size"):
        sample_balanced(
            {"khk_Cyrl": _language_lines("khk_Cyrl", 3)},
            input_sentence_size=input_sentence_size,
            temperature=0.3,
            seed=7,
        )


def test_quotas_follow_temperature() -> None:
    languages = ["khk_Cyrl", "eng_Latn"]
    # A 100:1 size ratio gives quotas 79 and 20; the small language needs at least
    # 20 lines, or its quota would be capped and the temperature effect hidden.
    sizes = [10000, 100]
    lines_by_lang = {
        language: _language_lines(language, size)
        for language, size in zip(languages, sizes, strict=True)
    }

    sample = sample_balanced(
        lines_by_lang, input_sentence_size=100, temperature=0.3, seed=7
    )

    counts = _count_per_language(sample, languages)
    assert counts == [79, 20]
    # 100 of 10100 lines is about 1 percent of the corpus; T=0.3 lifts it to 20 percent.
    assert counts[1] / sum(counts) > sizes[1] / sum(sizes)


def test_quota_capped_by_language_size_without_redistribution() -> None:
    languages = ["khk_Cyrl", "eng_Latn"]
    sizes = [1000, 5]
    lines_by_lang = {
        language: _language_lines(language, size)
        for language, size in zip(languages, sizes, strict=True)
    }

    sample = sample_balanced(
        lines_by_lang, input_sentence_size=100, temperature=0.3, seed=7
    )

    assert _count_per_language(sample, languages) == [83, 5]
    assert len(sample) == 88


def test_same_seed_same_sample_different_seed_differs() -> None:
    lines_by_lang = {
        "khk_Cyrl": _language_lines("khk_Cyrl", 1000),
        "eng_Latn": _language_lines("eng_Latn", 10),
    }

    def draw(seed: int) -> list[str]:
        return sample_balanced(
            lines_by_lang, input_sentence_size=100, temperature=0.3, seed=seed
        )

    assert draw(1) == draw(1)
    assert draw(1) != draw(2)


def test_sample_preserves_within_language_order() -> None:
    languages = ["khk_Cyrl", "eng_Latn"]
    lines_by_lang = {
        "khk_Cyrl": _language_lines("khk_Cyrl", 1000),
        "eng_Latn": _language_lines("eng_Latn", 50),
    }

    sample = sample_balanced(
        lines_by_lang, input_sentence_size=100, temperature=0.3, seed=7
    )

    for language in languages:
        indices = _indices_of(sample, language)
        assert indices == sorted(set(indices))
        assert 0 < len(indices) < len(lines_by_lang[language])


def test_sample_lists_languages_in_mapping_order() -> None:
    lines_by_lang = {
        "khk_Cyrl": _language_lines("khk_Cyrl", 1000),
        "eng_Latn": _language_lines("eng_Latn", 10),
    }

    sample = sample_balanced(
        lines_by_lang, input_sentence_size=100, temperature=0.3, seed=7
    )

    languages_in_sample = [line.rsplit("-", 1)[0] for line in sample]
    assert [language for language, _ in groupby(languages_in_sample)] == [
        "khk_Cyrl",
        "eng_Latn",
    ]


def test_language_sample_does_not_depend_on_mapping_order() -> None:
    khk = _language_lines("khk_Cyrl", 1000)
    eng = _language_lines("eng_Latn", 10)

    khk_first = sample_balanced(
        {"khk_Cyrl": khk, "eng_Latn": eng},
        input_sentence_size=100,
        temperature=0.3,
        seed=7,
    )
    khk_last = sample_balanced(
        {"eng_Latn": eng, "khk_Cyrl": khk},
        input_sentence_size=100,
        temperature=0.3,
        seed=7,
    )

    for language in ("khk_Cyrl", "eng_Latn"):
        assert _indices_of(khk_first, language) == _indices_of(khk_last, language)
