"""What the tests of the community files share (`tests/unit/test_github_community_files.py`): where the
files are, how a Markdown text is read (headings, sections, links, paragraphs, paths in code spans), how
the content rules are applied to a text, and how the version-controlled text is walked to find the one
address the owner approved for three files.

Nothing here starts a process, opens a socket or imports the library. The needles of the content rules
are those of `tests/unit/test_d13_content_rules.py`, which spells them as code points so that no file
holds them; the extra needles and the address of the owner are spelled the same way here, and the test
module checks that neither this file nor itself holds one. A failure names a line and a number, never a
word. The walk of the version-controlled text lists directories instead of asking `git`, so that it
needs no process; the data directory, the caches and the virtual environment are not in its list.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from tests.unit.test_d13_content_rules import lines_with_a_needle

ROOT: Final = Path(__file__).resolve().parents[2]

CODE_OF_CONDUCT: Final = ".github/CODE_OF_CONDUCT.md"
CONTRIBUTING: Final = ".github/CONTRIBUTING.md"
SECURITY: Final = ".github/SECURITY.md"
PULL_REQUEST_TEMPLATE: Final = ".github/PULL_REQUEST_TEMPLATE.md"

WIKI_URL: Final = "https://github.com/btseee/rinchen/wiki"
ADVISORY_URL: Final = "https://github.com/btseee/rinchen/security/advisories/new"
COVENANT_URL: Final = (
    "https://www.contributor-covenant.org/version/2/1/code_of_conduct.html"
)
OWNER_ADDRESS: Final = "".join(
    map(
        chr,
        (98, 98, 97, 116, 116, 115, 101, 114, 101, 110, 56, 56)
        + (64, 103, 109, 97, 105, 108, 46, 99, 111, 109),
    )
)
# More needles than the content rules have: the words for a line under a commit message that
# vouches for it, and for an agreement a contributor would be asked to make.
# fmt: off
EXTRA_NEEDLE_CODE_POINTS: Final[tuple[tuple[int, ...], ...]] = (
    (115, 105, 103, 110, 45, 111, 102, 102),
    (115, 105, 103, 110, 101, 100, 45, 111, 102, 102),
    (115, 105, 103, 110, 111, 102, 102),
    (99, 111, 110, 116, 114, 105, 98, 117, 116, 111, 114, 32, 97, 103, 114, 101, 101, 109, 101, 110, 116),
)
# fmt: on
EXTRA_NEEDLE_BASE: Final = 100
# The directories of the repository that git tracks, besides the files of its root (`data/` holds
# ignored content, and the caches and the virtual environment are ignored and not listed). A name that
# does not exist is skipped.
VERSION_CONTROLLED: Final = (
    ".github",
    "configs",
    "docker",
    "docs",
    "images",
    "manifests",
    "plans",
    "reports",
    "scripts",
    "src",
    "tests",
    "vendor",
)
SKIPPED_DIRECTORIES: Final = frozenset(
    {"__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".hypothesis"}
)
BINARY_SUFFIXES: Final = frozenset({".whl", ".png", ".pyc", ".so"})
# Words that name nothing as the text of a link.
VAGUE_LINK_TEXTS: Final = frozenset(
    {"here", "click here", "link", "this link", "this", "read more", "more", "page"}
)
MAX_WORDS_IN_A_PARAGRAPH: Final = 90

_FENCE: Final = re.compile(r"^\s*(```|~~~)")
_HEADING: Final = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LINK: Final = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_IMAGE: Final = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)")
_CODE_SPAN: Final = re.compile(r"`[^`]*`")
_BACKTICKED: Final = re.compile(r"`([^`\n]+)`")
_REPOSITORY_PATH: Final = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*/?")
_LIST_OR_TABLE: Final = re.compile(r"^(\s|[-*+]\s|\||>|#|<|\d+[.)]\s)")
URL: Final = re.compile(r"https?://[^\s)>\]\"']+")


def read_text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def prose(text: str) -> list[tuple[int, str]]:
    """`(line number, line)` of every line of `text` that is outside a fenced code block."""
    lines: list[tuple[int, str]] = []
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            lines.append((number, line))
    return lines


def headings(text: str) -> list[tuple[int, str]]:
    """`(level, title)` of every heading of `text`, outside code blocks."""
    found: list[tuple[int, str]] = []
    for _, line in prose(text):
        match = _HEADING.match(line)
        if match:
            found.append((len(match.group(1)), match.group(2)))
    return found


def slug(title: str) -> str:
    """The anchor GitHub makes of a heading."""
    lowered = title.lower().replace("`", "")
    return re.sub(r"[^a-z0-9 _-]", "", lowered).strip().replace(" ", "-")


def section(text: str, title: str) -> str:
    """The text under the heading `title` up to the next heading of the same or a higher level."""
    lines = text.splitlines()
    start = level = None
    for number, line in prose(text):
        match = _HEADING.match(line)
        if match is None:
            continue
        if start is None:
            if match.group(2) == title:
                start, level = number, len(match.group(1))
        elif level is not None and len(match.group(1)) <= level:
            return "\n".join(lines[start : number - 1])
    assert start is not None, f"no heading {title!r}"
    return "\n".join(lines[start:])


def links(text: str) -> list[tuple[str, str]]:
    """`(text, target)` of every Markdown link of `text` outside code blocks and code spans."""
    found: list[tuple[str, str]] = []
    for _, line in prose(text):
        found += _LINK.findall(_CODE_SPAN.sub("code", line))
    return found


def images(text: str) -> list[tuple[str, str]]:
    """`(alternative text, target)` of every Markdown image of `text`."""
    return _IMAGE.findall(text)


def paragraphs(text: str) -> list[str]:
    """The paragraphs of `text`: runs of plain lines, not lists, tables, headings or markup."""
    blocks: list[list[str]] = [[]]
    for _, line in prose(text):
        if not line.strip() or _LIST_OR_TABLE.match(line):
            blocks.append([])
        else:
            blocks[-1].append(line)
    return [" ".join(block) for block in blocks if block]


def repository_paths(text: str) -> Iterator[str]:
    """The code spans of `text` that name a path under a version-controlled directory."""
    roots = {name for name in VERSION_CONTROLLED if (ROOT / name).is_dir()}
    for span in _BACKTICKED.findall(text):
        if (
            "/" in span
            and _REPOSITORY_PATH.fullmatch(span)
            and span.split("/")[0] in roots
        ):
            yield span


def non_ascii_lines(data: bytes) -> list[int]:
    return [
        number
        for number, line in enumerate(data.split(b"\n"), start=1)
        if any(byte > 127 for byte in line)
    ]


def excluded_words(text: str) -> list[tuple[int, int]]:
    """`(line number, needle number)` of every line that holds a word the project's content rules
    exclude or one of the extra needles (numbered from `EXTRA_NEEDLE_BASE`)."""
    extra = ["".join(map(chr, points)) for points in EXTRA_NEEDLE_CODE_POINTS]
    found = list(lines_with_a_needle(text))
    for number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        found += [
            (number, EXTRA_NEEDLE_BASE + n)
            for n, word in enumerate(extra)
            if word in lowered
        ]
    return sorted(found)


def rules_of_claude(marker: str) -> list[str]:
    """The bullets of `CLAUDE.md` under the line `marker` (`ALWAYS:` or `NEVER:`), without the dash."""
    lines = read_text("CLAUDE.md").splitlines()
    rules: list[str] = []
    for line in lines[lines.index(marker) + 1 :]:
        if not line.startswith("- "):
            break
        rules.append(line[2:])
    return rules


def squeezed(text: str) -> str:
    return " ".join(text.split())


def version_controlled_files() -> Iterator[Path]:
    """The files of the root and every file under `VERSION_CONTROLLED`, caches left out."""
    yield from sorted(path for path in ROOT.iterdir() if path.is_file())
    for name in VERSION_CONTROLLED:
        for directory, subdirectories, files in os.walk(ROOT / name):
            subdirectories[:] = sorted(set(subdirectories) - SKIPPED_DIRECTORIES)
            yield from (Path(directory) / file for file in sorted(files))


def holders_of_the_address() -> list[str]:
    """The version-controlled text files that hold the address of the owner, as paths from the root."""
    local_part = OWNER_ADDRESS.split("@")[0]
    holders: list[str] = []
    for path in version_controlled_files():
        if path.suffix in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if local_part in text.lower():
            holders.append(path.relative_to(ROOT).as_posix())
    return sorted(holders)


def parameter_defaults(function: ast.FunctionDef) -> dict[str, ast.expr | None]:
    """The default of every parameter of `function`, `None` for one that has none."""
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    given = [None] * (len(positional) - len(arguments.defaults)) + arguments.defaults
    found: dict[str, ast.expr | None] = {
        argument.arg: default for argument, default in zip(positional, given)
    }
    found.update(
        {a.arg: d for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults)}
    )
    return found
