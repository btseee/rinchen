"""Check a clone of the project's GitHub Wiki, where the documents of the repository moved.

    uv run python scripts/check_wiki.py WIKI_DIR [--verify-ids-from FILE ...]

The checks, over every `*.md` page of WIKI_DIR (the `.git` directory of a clone is skipped):

- the content rules: no page holds a word of the two subjects the project's rules exclude, except the
  moved records (`is_record_page`), whose original text is kept as it was written. The needles are
  spelled as code points, the same ones as `tests/unit/test_d13_content_rules.py` (a test pins them
  equal), and a problem names the needle by its number, never the word;
- internal links: a link to a page of the Wiki (a bare page name, `#anchor`, `Page#anchor`, `[[Page]]`
  or the absolute Wiki URL) must name a page that exists and, with an anchor, a heading of that page; a
  relative link that is not a page name (a repository path) points at nothing on the Wiki. Links inside
  code fences and code spans are not links;
- characters: a page that is not a moved record holds ASCII only; no page holds a carriage return; every
  code fence is closed;
- page names: letters and digits in parts joined by single hyphens, or `_Sidebar` and `_Footer`; no two
  names that differ only in case; a `Home` page exists;
- with `--verify-ids-from FILE`: every Verification-Log row id (`D-...`, `D13-...`) that a `decision:`
  line of FILE names is a row of the page `Verification-Log`, and FILE names at least one.

It prints one line per problem (`page:line: kind: detail`, line 0 for the page as a whole) and a count,
and exits 0 when there is no problem, 1 when there is one and 2 when WIKI_DIR is not a directory.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

WIKI_URL: Final = "https://github.com/btseee/rinchen/wiki"
HOME: Final = "Home"
SPECIAL_PAGES: Final = frozenset({"_Sidebar", "_Footer"})
VERIFICATION_LOG: Final = "Verification-Log"
# The pages that hold the moved records (runbooks, specs, plans, research, the history of the state, the
# decisions and their index, the verification log and the stack) in their original text. `State` is the
# living status page and is checked like a guide page.
RECORD_PAGE_PATTERNS: Final = (
    "Runbook-*",
    "Spec-*",
    "Plan-*",
    "Research-*",
    "State-History-*",
    "Decision-*",
    "Decisions",
    VERIFICATION_LOG,
    "Stack",
)
# One tuple of code points per needle; the words are never written out.
# fmt: off
NEEDLE_CODE_POINTS: Final[tuple[tuple[int, ...], ...]] = (
    (108, 105, 99, 101, 110),
    (99, 111, 112, 121, 114, 105, 103, 104, 116),
    (108, 101, 103, 97, 108),
    (108, 97, 119, 115, 117, 105, 116),
    (108, 97, 119, 102, 117, 108),
    (108, 97, 119, 121, 101, 114),
    (116, 114, 97, 100, 101, 109, 97, 114, 107),
    (112, 97, 116, 101, 110, 116),
    (108, 105, 97, 98, 105, 108, 105, 116, 121),
    (119, 97, 114, 114, 97, 110, 116, 121),
    (116, 101, 114, 109, 115, 32, 111, 102, 32, 117, 115, 101),
    (116, 101, 114, 109, 115, 32, 111, 102, 32, 115, 101, 114, 118, 105, 99, 101),
    (115, 112, 100, 120),
    (97, 116, 116, 111, 114, 110, 101, 121),
    (114, 111, 121, 97, 108, 116),
    (101, 117, 108, 97),
    (100, 109, 99, 97),
    (106, 117, 114, 105, 115, 100, 105, 99, 116, 105, 111, 110),
)
# fmt: on

_PAGE_NAME: Final = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
_FENCE: Final = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HEADING: Final = re.compile(r"^ {0,3}#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$")
_CODE_SPAN: Final = re.compile(r"(`+)(?:.+?)\1")
_LINK: Final = re.compile(r"\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_WIKI_LINK: Final = re.compile(r"\[\[([^\]]+)\]\]")
_INLINE_LINK: Final = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_SCHEME: Final = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ROW_ID: Final = re.compile(r"\bD(?:[0-9]+)?-[A-Z0-9]+(?:-[A-Z0-9]+)*")


@dataclass(frozen=True, order=True)
class Problem:
    page: str
    line: int
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.page}:{self.line}: {self.kind}: {self.detail}"


def needles() -> tuple[str, ...]:
    return tuple("".join(map(chr, points)) for points in NEEDLE_CODE_POINTS)


def is_record_page(name: str) -> bool:
    """Whether the page `name` (without `.md`) holds a moved record in its original text."""
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in RECORD_PAGE_PATTERNS)


def slug(heading: str) -> str:
    """The anchor the rendered page gives a heading: the text of its links, in lower case, without
    punctuation other than hyphens and underscores, spaces as hyphens."""
    text = _INLINE_LINK.sub(r"\1", heading).lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _text_lines(lines: Sequence[str]) -> Iterator[tuple[int, str]]:
    """`(number, line)` of every line outside a code fence."""
    for number, line, fenced in _fence_states(lines):
        if not fenced:
            yield number, line


def _closes(line: str, opened: str) -> bool:
    """Whether `line` closes the fence `opened`: the same character, at least as many, nothing else."""
    found = _FENCE.match(line)
    return (
        found is not None
        and found.group(1)[0] == opened[0]
        and len(found.group(1)) >= len(opened)
        and line.strip() == found.group(1)
    )


def _fence_states(lines: Sequence[str]) -> Iterator[tuple[int, str, bool]]:
    """`(number, line, fenced)` of every line; the lines of a fence itself are fenced."""
    opened: str | None = None
    for number, line in enumerate(lines, start=1):
        if opened is None:
            found = _FENCE.match(line)
            opened = found.group(1) if found else None
            yield number, line, opened is not None
            continue
        yield number, line, True
        if _closes(line, opened):
            opened = None


def _unclosed_fence(lines: Sequence[str]) -> int | None:
    """The line that opens a fence no later line closes, or None."""
    opened: str | None = None
    opened_at: int | None = None
    for number, line in enumerate(lines, start=1):
        if opened is None:
            found = _FENCE.match(line)
            if found:
                opened, opened_at = found.group(1), number
        elif _closes(line, opened):
            opened, opened_at = None, None
    return opened_at


def _anchors(lines: Sequence[str]) -> set[str]:
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    for _number, line in _text_lines(lines):
        found = _HEADING.match(line)
        if found is None:
            continue
        base = slug(found.group(1))
        count = seen.get(base, 0)
        seen[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def _targets(line: str) -> Iterator[str]:
    """The link targets of a line outside its code spans."""
    plain = _CODE_SPAN.sub(lambda found: " " * len(found.group(0)), line)
    for found in _LINK.finditer(plain):
        yield found.group(1)
    for found in _WIKI_LINK.finditer(plain):
        yield found.group(1).split("|")[-1].strip().replace(" ", "-")


def _link_problem(target: str, page: str, anchors: dict[str, set[str]]) -> str | None:
    """Why `target`, a link on `page`, points at nothing on the Wiki, or None."""
    if target.startswith(WIKI_URL):
        rest = target[len(WIKI_URL) :]
        if rest and not rest.startswith(("/", "#")):
            return None
        target = rest.lstrip("/") or HOME
    elif _SCHEME.match(target):
        return None
    name, _, anchor = target.partition("#")
    name = name or page
    if name not in anchors:
        return f"{target} is not a page"
    if anchor and anchor not in anchors[name]:
        return f"{target} has no such heading"
    return None


def _page_problems(name: str, text: str, anchors: dict[str, set[str]]) -> list[Problem]:
    page = f"{name}.md"
    lines = text.split("\n")
    record = is_record_page(name)
    problems: list[Problem] = []
    carriage = next((n for n, line in enumerate(lines, 1) if "\r" in line), None)
    if carriage is not None:
        problems.append(
            Problem(page, carriage, "crlf", "a carriage return ends the line")
        )
    opened = _unclosed_fence(lines)
    if opened is not None:
        problems.append(Problem(page, opened, "fence", "a code fence is never closed"))
    for number, line in enumerate(lines, start=1):
        if not record and not line.isascii():
            problems.append(Problem(page, number, "ascii", "a character outside ASCII"))
        if not record:
            lowered = line.lower()
            problems += [
                Problem(page, number, "needle", f"needle {index}")
                for index, needle in enumerate(needles())
                if needle in lowered
            ]
    for number, line in _text_lines(lines):
        for target in _targets(line.rstrip("\r")):
            why = _link_problem(target, name, anchors)
            if why is not None:
                problems.append(Problem(page, number, "link", why))
    return problems


def _name_problems(files: Sequence[Path]) -> list[Problem]:
    problems = []
    for path in files:
        legal = path.suffix == ".md" and (
            path.stem in SPECIAL_PAGES or _PAGE_NAME.fullmatch(path.stem) is not None
        )
        if not legal:
            problems.append(Problem(path.name, 0, "page-name", "not a legal page name"))
    folded: dict[str, list[str]] = {}
    for path in files:
        folded.setdefault(path.name.casefold(), []).append(path.name)
    for names in folded.values():
        if len(names) > 1:
            problems += [
                Problem(name, 0, "page-name", "another page differs only in case")
                for name in names
            ]
    if not any(path.name == f"{HOME}.md" for path in files):
        problems.append(
            Problem(f"{HOME}.md", 0, "page-name", "the Wiki has no Home page")
        )
    return problems


def _row_problems(wiki: Path, sources: Sequence[Path]) -> list[Problem]:
    log = wiki / f"{VERIFICATION_LOG}.md"
    rows = log.read_text(encoding="utf-8") if log.is_file() else ""
    problems = []
    for source in sources:
        named = 0
        lines = source.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if "decision:" not in line:
                continue
            for row in _ROW_ID.findall(line):
                named += 1
                if f"| {row} " not in rows:
                    detail = f"{row} is not a row of {VERIFICATION_LOG}"
                    problems.append(Problem(source.name, number, "row", detail))
        if named == 0:
            detail = f"no decision names a row of {VERIFICATION_LOG}"
            problems.append(Problem(source.name, 0, "row", detail))
    return problems


def check_wiki(wiki: Path, verify_ids_from: Sequence[Path] = ()) -> list[Problem]:
    """Every problem of the Wiki clone `wiki`, sorted by page, line and kind."""
    files = sorted(path for path in wiki.iterdir() if path.is_file())
    pages = {
        path.stem: path.read_bytes().decode("utf-8")
        for path in files
        if path.suffix == ".md"
    }
    anchors = {name: _anchors(text.split("\n")) for name, text in pages.items()}
    problems = _name_problems(files)
    for name, text in pages.items():
        problems += _page_problems(name, text, anchors)
    problems += _row_problems(wiki, verify_ids_from)
    return sorted(problems)


def main(argv: Sequence[str] | None = None) -> int:
    assert sys.version_info[:2] == (3, 14), sys.version
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wiki", type=Path, help="a clone of the Wiki")
    parser.add_argument(
        "--verify-ids-from",
        type=Path,
        action="append",
        default=[],
        help="a file whose decision lines name Verification-Log rows",
    )
    options = parser.parse_args(argv)
    if not options.wiki.is_dir():
        print(f"{options.wiki} is not a directory", file=sys.stderr)
        return 2
    problems = check_wiki(options.wiki, options.verify_ids_from)
    for problem in problems:
        print(problem)
    pages = sum(1 for path in options.wiki.iterdir() if path.suffix == ".md")
    print(f"{pages} pages, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
