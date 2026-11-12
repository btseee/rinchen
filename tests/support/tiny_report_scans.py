"""The content scans of the tiny pipeline's report and its Markdown (D15 Task 6, G6.2).

`violations` names what a document must not hold, as places and counts and never as the text found: a
time of day or a date, a key that names a duration, an absolute path or a Windows path, the digest names
that embed a path (`manifest_sha256`, `hypotheses_sha256`), a line that looks like another program's
output, a value of the environment of the run, and a sentence of the fixtures. The scans are themselves
tested on documents that break each rule.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from typing import Any, Final

from rinchen.config.loader import REPO_ROOT

FIXTURE_DIRECTORY: Final = REPO_ROOT / "tests" / "fixtures" / "parallel"
# A date, or a time of day (also inside `T12:30:00Z`).
_WHEN: Final = re.compile(r"\b20[0-9]{2}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}:[0-9]{2}")
_DURATION_KEY: Final = re.compile(
    r"(?:^|_)(?:seconds?|ms|duration|elapsed|time|timestamp|date)(?:_|$)", re.IGNORECASE
)
# A slash that starts a path (nothing of a word before it), and a drive letter with a backslash.
_PATH: Final = re.compile(r"(?<![A-Za-z0-9_.])/[A-Za-z0-9_.~-]|[A-Za-z]:\\")
_EMBEDS_A_PATH: Final = ("manifest_sha256", "hypotheses_sha256")
_PROGRAM_OUTPUT: Final = (
    "Traceback",
    'File "',
    "INFO",
    "WARNING",
    "DEBUG",
    "ERROR",
    "it/s",
    "s/it",
    "[Errno",
    chr(0x2588),
)
MIN_SENTENCE: Final = 8
WORD_TABLES: Final = ("nouns", "entities")
NOISE_TEXTS: Final = ("src", "tgt")
_LANGUAGE_COLUMN: Final = re.compile(r"[a-z]{2,3}_[A-Z][a-z]{3}")


def keys_of(value: Any) -> Iterator[str]:
    """Every key of every mapping in a JSON value."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from keys_of(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys_of(item)


def strings_of(value: Any) -> Iterator[str]:
    """Every text (key or value) in a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from strings_of(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_of(item)


@cache
def fixture_sentences() -> frozenset[str]:
    """The sentences of the fixture files: every cell of a column that holds a language (the pair
    files, the Mongolian file and the monolingual file) and the two texts of the noise file, when it is
    at least `MIN_SENTENCE` characters long. The nouns and the entities are single words and are not
    read."""
    sentences: set[str] = set()
    for path in sorted(FIXTURE_DIRECTORY.glob("*.tsv")):
        if path.stem in WORD_TABLES:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        header = lines[0].split("\t")
        columns = [
            index
            for index, name in enumerate(header)
            if _LANGUAGE_COLUMN.fullmatch(name) or name in NOISE_TEXTS
        ]
        for line in lines[1:]:
            cells = line.split("\t")
            sentences.update(
                cells[index] for index in columns if len(cells[index]) >= MIN_SENTENCE
            )
    return frozenset(sentences)


def text_violations(text: str, environment: Sequence[str] = ()) -> list[str]:
    """What a rendered text (the JSON or the Markdown, as text) holds that it must not."""
    found = []
    if _WHEN.search(text):
        found.append("a date or a time of day")
    if _PATH.search(text):
        found.append("an absolute path")
    found += [f"the name {name}" for name in _EMBEDS_A_PATH if name in text]
    found += [
        "a line of another program's output" for mark in _PROGRAM_OUTPUT if mark in text
    ]
    found += ["a value of the environment" for value in environment if value in text]
    found += [
        "a sentence of the fixtures"
        for sentence in fixture_sentences()
        if sentence in text
    ]
    return found


def violations(data: bytes, environment: Sequence[str] = ()) -> list[str]:
    """The violations of a report's bytes: the text rules, and a key that names a duration."""
    text = data.decode("utf-8")
    found = text_violations(text, environment)
    try:
        document = json.loads(text)
    except ValueError:
        return found
    found += [
        f"the key {key}" for key in keys_of(document) if _DURATION_KEY.search(key)
    ]
    return found
