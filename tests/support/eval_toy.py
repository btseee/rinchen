"""The fixture suites of the D13 gate, made from the robustness toy (`tests/support/robust_toy.py`).

`toy_suite(toy, split)` joins the rows of the toy's held-out items into one `Suite` of kind `fixture`,
benchmark `toy`: the name `toy-devtest` for the toy's TEST rows (split `devtest`, 480 rows) and
`toy-dev` for its dev rows (split `dev`, 292 rows). A row is one item, one sentence in each of
`eng_Latn`, `khk_Cyrl` and `rus_Cyrl`, found by the Khalkha text its pair rows share, in the order the
toy lists its items; no row has a document, and the directions are the six ordered pairs of the three
languages (the toy is trained for four of them: `eng_Latn-rus_Cyrl` and `rus_Cyrl-eng_Latn` are not).
The rows of the toy that start in traditional script (`mn_Mong`) belong to no suite here.

`write_toy_suites(directory)` writes the two files as the suite writer does, `write_lines_directory`
makes a directory of text files, one per language (`<code>.txt`, one sentence per line), such as
`rinchen eval suite build` reads for a benchmark that comes as lines; `extra_files` adds files of other
names (holding the Khalkha lines) so that a test can show that a name is refused.

Nothing here loads a model, opens a socket or touches a GPU.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rinchen.eval.suites import Suite, SuiteRow, SuiteSource, write_suite
from tests.support.robust_toy import RobustToy, build_robust_toy

__all__ = [
    "DEVTEST_ROWS",
    "DEV_ROWS",
    "INNER_MONGOLIAN_FILE",
    "LANGUAGES",
    "SIX_DIRECTIONS",
    "Suites",
    "toy_suite",
    "write_lines_directory",
    "write_toy_suites",
]

LANGUAGES: Final = ("eng_Latn", "khk_Cyrl", "rus_Cyrl")
SIX_DIRECTIONS: Final = tuple(
    f"{source}-{target}" for source, target in itertools.permutations(LANGUAGES, 2)
)
# The toy's TEST split: 100 base, 210 one-number, 30 two-number, 100 placeholder and 40 entity
# items; its dev split: 60, 140, 20, 60 and 12.
DEVTEST_ROWS: Final = 480
DEV_ROWS: Final = 292
# The name of a file the builder must refuse (the guards name the variety on purpose).
INNER_MONGOLIAN_FILE: Final = "khk_Mong.txt"
_TOY_SPLIT: Final = {"devtest": "test", "dev": "dev"}
_KEY_LANGUAGE: Final = "khk_Cyrl"


@dataclass(frozen=True)
class Suites:
    """The two suite files of the gate."""

    devtest: Path
    dev: Path


def toy_suite(toy: RobustToy, split: str) -> Suite:
    """The suite of the toy's held-out items for `split` (`devtest` or `dev`).

    Raises:
        ValueError: `split` is not one of the two, or an item lacks one of the three languages,
            or two rows of an item disagree about a sentence.
    """
    if split not in _TOY_SPLIT:
        raise ValueError(f"split must be one of {', '.join(_TOY_SPLIT)}")
    items: dict[str, dict[str, str]] = {}
    for row in toy.rows:
        if (
            row.split != _TOY_SPLIT[split]
            or row.src_lang not in LANGUAGES
            or row.tgt_lang not in LANGUAGES
        ):
            continue
        sentences = {row.src_lang: row.src, row.tgt_lang: row.tgt}
        item = items.setdefault(sentences[_KEY_LANGUAGE], {})
        for language, sentence in sentences.items():
            if item.setdefault(language, sentence) != sentence:
                raise ValueError(
                    f"two rows of one item disagree about the {language} sentence"
                )
    rows: list[SuiteRow] = []
    for index, item in enumerate(items.values(), start=1):
        missing = [language for language in LANGUAGES if language not in item]
        if missing:
            raise ValueError(f"item {index} lacks {', '.join(missing)}")
        rows.append(SuiteRow(id=f"toy-{index:04d}", doc=None, text=item))
    return Suite(
        name=f"toy-{split}",
        kind="fixture",
        benchmark="toy",
        split=split,
        source=SuiteSource(
            dataset=None, revision=None, pinned=False, files={}, missing_languages=()
        ),
        directions=SIX_DIRECTIONS,
        rows=tuple(rows),
    )


def write_toy_suites(directory: Path) -> Suites:
    """Write `toy-devtest.json` and `toy-dev.json` into `directory` (made when missing)."""
    directory.mkdir(parents=True, exist_ok=True)
    toy = build_robust_toy()
    suites = Suites(
        devtest=directory / "toy-devtest.json", dev=directory / "toy-dev.json"
    )
    write_suite(toy_suite(toy, "devtest"), suites.devtest)
    write_suite(toy_suite(toy, "dev"), suites.dev)
    return suites


def write_lines_directory(
    directory: Path, suite: Suite, *, extra_files: tuple[str, ...] = ()
) -> None:
    """Write `<language>.txt` for each language of `suite` (one sentence per line, LF) into
    `directory`, and each of `extra_files` holding the Khalkha lines."""
    directory.mkdir(parents=True, exist_ok=True)
    for language in LANGUAGES:
        _write_lines(directory / f"{language}.txt", suite, language)
    for name in extra_files:
        _write_lines(directory / name, suite, _KEY_LANGUAGE)


def _write_lines(path: Path, suite: Suite, language: str) -> None:
    path.write_text(
        "".join(f"{row.text[language]}\n" for row in suite.rows),
        encoding="utf-8",
        newline="\n",
    )
