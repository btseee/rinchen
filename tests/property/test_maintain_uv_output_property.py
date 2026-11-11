"""Property tests of the dry-run parser and of the drift decision (D14, spec M6): whatever a
well-formed output looks like, it parses to exactly the changes it was written from; one line of
text that is none of the classes, inserted anywhere, is counted and turns the decision into a
failure; and for ANY text the parser does not raise and the decision is `ok` only on the positive
line with no change line.

The parser is also compared with a second reader written here with `str.split` and no regular
expression, over well-formed outputs, outputs with an inserted unknown line and outputs with
characters deleted, inserted and replaced: the two must agree on every one. Under the
`no_network` guard like every D14 test module.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.maintain import states
from rinchen.maintain.drift import drift_outcome
from rinchen.maintain.lock import normalize_name
from rinchen.maintain.uv_output import Change, ChangeKind, DryRun, parse_dry_run
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import lock_of

pytestmark = pytest.mark.usefixtures("no_network")

ESC = chr(0x1B)
POSITIVE_LINE = "No lockfile changes detected"
TOLERANT = [HealthCheck.too_slow, HealthCheck.data_too_large]


def cases_settings(examples: int) -> settings:
    return settings(max_examples=examples, suppress_health_check=TOLERANT)


# --- the second reader: split on spaces, compare tokens, no regular expression ------------------

ALNUM = string.ascii_letters + string.digits
NAME_CHARS = ALNUM + "._-"
VERSION_CHARS = ALNUM + ".+!_-"
VERBS: dict[str, ChangeKind] = {
    "Update": "update",
    "Updated": "update",
    "Add": "add",
    "Added": "add",
    "Remove": "remove",
    "Removed": "remove",
}


def reference_lines(text: str) -> list[str]:
    """The non-blank lines of `text`: escape sequences removed, split on newline characters, one
    trailing carriage return dropped, spaces and tabs trimmed."""
    kept = []
    position = 0
    while position < len(text):
        if text[position] == ESC and text[position + 1 : position + 2] == "[":
            end = position + 2
            while end < len(text) and text[end] in "0123456789;":
                end += 1
            if end < len(text) and text[end] in string.ascii_letters:
                position = end + 1
                continue
        kept.append(text[position])
        position += 1
    lines = []
    for line in "".join(kept).split("\n"):
        line = line.removesuffix("\r").strip(" \t")
        if line:
            lines.append(line)
    return lines


def reference_name(token: str) -> str | None:
    if not token or token[0] not in ALNUM or token[-1] not in ALNUM:
        return None
    if any(char not in NAME_CHARS for char in token):
        return None
    normalised = []
    for char in token.lower():
        if char in "._-" and normalised and normalised[-1] == "-":
            continue
        normalised.append("-" if char in "._-" else char)
    return "".join(normalised)


def reference_version(token: str) -> str | None:
    token = token.removeprefix("v")
    if not token or token[0] not in string.digits:
        return None
    return token if all(char in VERSION_CHARS for char in token) else None


def reference_change(tokens: list[str]) -> Change | None:
    if len(tokens) not in (3, 5) or tokens[0] not in VERBS:
        return None
    kind = VERBS[tokens[0]]
    name = reference_name(tokens[1])
    first = reference_version(tokens[2])
    if name is None or first is None:
        return None
    if kind == "update":
        second = reference_version(tokens[4]) if len(tokens) == 5 else None
        if len(tokens) != 5 or tokens[3] != "->" or second is None:
            return None
        return Change("update", name, first, second)
    if len(tokens) != 3:
        return None
    return (
        Change(kind, name, None, first)
        if kind == "add"
        else Change(kind, name, first, None)
    )


def _plain(token: str) -> bool:
    return token != "" and "\t" not in token


def reference_is_boilerplate(tokens: list[str]) -> bool:
    if tokens[0] == "Using":
        return (
            len(tokens) >= 6
            and tokens[1] == "CPython"
            and _plain(tokens[2])
            and tokens[3] == "interpreter"
            and tokens[4] == "at:"
            and " ".join(tokens[5:]) != ""
        )
    if tokens[0] == "Resolved":
        return (
            len(tokens) == 5
            and tokens[1] != ""
            and all(char in string.digits for char in tokens[1])
            and tokens[2] in ("package", "packages")
            and tokens[3] == "in"
            and _plain(tokens[4])
        )
    return False


def reference_parse(text: str) -> DryRun:
    changes: list[Change] = []
    no_changes, warnings, unrecognised = False, 0, 0
    for line in reference_lines(text):
        tokens = line.split(" ")
        if reference_is_boilerplate(tokens):
            continue
        if line == "warning:" or line.startswith("warning: "):
            warnings += 1
        elif line == POSITIVE_LINE:
            no_changes = True
        elif (change := reference_change(tokens)) is not None:
            changes.append(change)
        else:
            unrecognised += 1
    return DryRun(tuple(changes), no_changes, warnings, unrecognised)


# --- the generators --------------------------------------------------------------------------------

NAMES = st.lists(
    st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=6),
    min_size=1,
    max_size=3,
).map("-".join)
VERSIONS = st.tuples(
    st.sampled_from(["", "", "", "1!"]),
    st.lists(st.integers(min_value=0, max_value=999), min_size=1, max_size=4),
    st.sampled_from(["", "", "rc1", "a2", "b1", ".post1", "+cu130", ".dev3+local.7"]),
).map(lambda parts: parts[0] + ".".join(map(str, parts[1])) + parts[2])
PAST_OR_PRESENT = {
    "update": ("Update", "Updated"),
    "add": ("Add", "Added"),
    "remove": ("Remove", "Removed"),
}
BLANKISH = ["", "", "", " ", "\t", "  \t"]
STYLES = [ESC + "[1m", ESC + "[32m", ESC + "[0m", ESC + "[1;32m"]


@dataclass(frozen=True)
class Case:
    """An output as lines, how they end, and what it says."""

    lines: tuple[str, ...]
    end: str
    changes: tuple[Change, ...]
    warnings: int
    change_at: tuple[int, ...]

    @property
    def text(self) -> str:
        return self.end.join(self.lines) + self.end


@st.composite
def boilerplate_lines(draw: st.DrawFn) -> tuple[str, int]:
    """A line of boilerplate, or a warning, and the number of warnings it is."""
    kind = draw(st.sampled_from(["using", "resolved", "warning"]))
    if kind == "using":
        version = draw(st.sampled_from(["3.14.10", "3.12.1+freethreaded", "3.14.0"]))
        path = draw(
            st.sampled_from(["/opt/python/bin/python3", "/a b/c", "C:\\Py\\python.exe"])
        )
        line, warnings = f"Using CPython {version} interpreter at: {path}", 0
    elif kind == "resolved":
        count = draw(st.integers(min_value=0, max_value=9999))
        took = draw(st.sampled_from(["1ms", "2.31s", "0.00s", "3m"]))
        line, warnings = (
            f"Resolved {count} package{'s' if count != 1 else ''} in {took}",
            0,
        )
    else:
        line, warnings = (
            "warning: "
            + draw(st.text(alphabet=string.ascii_letters + " .:", max_size=20)),
            1,
        )
    if draw(st.booleans()):
        line = draw(st.sampled_from(STYLES)) + line + STYLES[2]
    return draw(st.sampled_from(BLANKISH)) + line + draw(
        st.sampled_from(BLANKISH)
    ), warnings


@st.composite
def change_lines(draw: st.DrawFn, kind: ChangeKind, name: str) -> tuple[str, Change]:
    """One change line in a random spelling, and the change it says."""
    verb = draw(st.sampled_from(PAST_OR_PRESENT[kind]))
    old, new = draw(VERSIONS), draw(VERSIONS)
    spelled = draw(
        st.sampled_from(
            [name, name.upper(), name.replace("-", "_"), name.replace("-", ".").title()]
        )
    )
    tokens = [
        verb,
        spelled,
        draw(st.sampled_from(["", "v"])) + (new if kind == "add" else old),
    ]
    if kind == "update":
        tokens += ["->", draw(st.sampled_from(["", "v"])) + new]
    tokens = [
        draw(st.sampled_from(["", "", STYLES[0], STYLES[1]]))
        + token
        + draw(st.sampled_from(["", STYLES[2]]))
        for token in tokens
    ]
    shown = {"update": (old, new), "add": (None, new), "remove": (old, None)}[kind]
    padded = (
        draw(st.sampled_from(BLANKISH))
        + " ".join(tokens)
        + draw(st.sampled_from(BLANKISH))
    )
    return padded, Change(kind, normalize_name(name), shown[0], shown[1])


@st.composite
def cases(draw: st.DrawFn, minimum: int = 1) -> Case:
    names = draw(st.lists(NAMES, min_size=minimum, max_size=6, unique=True))
    changes, lines = [], []
    for name in names:
        kind = draw(st.sampled_from(["update", "add", "remove"]))
        line, change = draw(change_lines(kind, name))
        lines.append(line)
        changes.append(change)
    before = draw(st.lists(boilerplate_lines(), max_size=2))
    after = draw(st.lists(boilerplate_lines(), max_size=2))
    warnings = sum(counted for _, counted in (*before, *after))
    ordered = [text for text, _ in before] + lines + [text for text, _ in after]
    where = tuple(range(len(before), len(before) + len(lines)))
    end = draw(st.sampled_from(["\n", "\r\n"]))
    return Case(tuple(ordered), end, tuple(changes), warnings, where)


RECOGNISED_PREFIXES = (
    "Using",
    "Resolved",
    "warning:",
    "No lockfile",
    "Update",
    "Add",
    "Remove",
)
JUNK = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",), blacklist_characters=ESC + "\n\r"
    ),
    min_size=1,
    max_size=40,
).filter(
    lambda line: (
        line.strip(" \t") != ""
        and not line.strip(" \t").startswith(RECOGNISED_PREFIXES)
    )
)


def lock_matching(changes: Sequence[Change]) -> dict[str, list[str]]:
    """The lock that makes every change of a case consistent: old versions are locked, adds are
    not."""
    return {change.name: [change.old] for change in changes if change.old is not None}


def decide(text: str, changes: Sequence[Change]) -> tuple[str, str | None]:
    result = drift_outcome(
        exit_code=0,
        timed_out=False,
        text=text,
        lock=lock_of(lock_matching(changes)),
        rules=[],
        timeout_seconds=300,
    )
    return result.state, result.reason


# --- well-formed outputs ---------------------------------------------------------------------------


@cases_settings(1_000)
@given(case=cases())
def test_a_well_formed_output_parses_to_exactly_the_changes_it_was_written_from(
    case: Case,
) -> None:
    expected = DryRun(case.changes, False, case.warnings, 0)
    assert parse_dry_run(case.text) == expected
    assert reference_parse(case.text) == expected


@cases_settings(1_000)
@given(case=cases())
def test_a_well_formed_output_that_agrees_with_the_lock_is_a_drift_with_one_entry_per_change(
    case: Case,
) -> None:
    result = drift_outcome(
        exit_code=0,
        timed_out=False,
        text=case.text,
        lock=lock_of(lock_matching(case.changes)),
        rules=[],
        timeout_seconds=300,
    )
    assert (result.state, result.reason) == ("drift", None)
    assert [(entry.package, entry.change) for entry in result.entries] == sorted(
        (change.name, change.kind) for change in case.changes
    )
    assert result.warnings == case.warnings


# --- one line of text that is none of the classes ------------------------------------------------


@cases_settings(1_000)
@given(case=cases(), junk=JUNK, where=st.integers(min_value=0, max_value=100))
def test_one_unknown_line_anywhere_is_counted_and_the_decision_is_never_ok_or_drift(
    case: Case, junk: str, where: int
) -> None:
    position = where % (len(case.lines) + 1)
    lines = (*case.lines[:position], junk, *case.lines[position:])
    text = case.end.join(lines) + case.end

    run = parse_dry_run(text)
    assert run.unrecognised >= 1
    assert run == DryRun(case.changes, False, case.warnings, 1)
    state, reason = decide(text, case.changes)
    assert (state, reason) == ("fail", states.fail_unrecognised(1))


@cases_settings(1_000)
@given(case=cases(minimum=0), junk=JUNK)
def test_an_unknown_line_next_to_the_positive_line_is_never_ok(
    case: Case, junk: str
) -> None:
    text = "\n".join([POSITIVE_LINE, junk, *case.lines]) + "\n"
    assert decide(text, case.changes)[0] == "fail"


# --- any text -------------------------------------------------------------------------------------

POOL = [
    POSITIVE_LINE,
    ESC + "[1m" + POSITIVE_LINE + ESC + "[0m",
    "  " + POSITIVE_LINE + "\t",
    POSITIVE_LINE + ".",
    "Update numpy v2.5.4 -> v2.5.5",
    "Added tomli-w 1.2.0",
    "Remove sortedcontainers v2.4.0",
    "Update numpy v2.5.4",
    "Update numpy v2.5.4 <- v2.5.5",
    "warning: x",
    "Using CPython 3.14.10 interpreter at: /x",
    "Resolved 187 packages in 1ms",
    "",
    "error: no",
]
ANY_TEXT = st.one_of(
    st.text(max_size=200),
    st.lists(st.one_of(st.sampled_from(POOL), st.text(max_size=30)), max_size=8).map(
        "\n".join
    ),
)


@cases_settings(2_000)
@given(text=ANY_TEXT)
def test_no_text_makes_the_parser_raise_and_ok_needs_the_positive_line_and_no_change(
    text: str,
) -> None:
    run = parse_dry_run(text)
    assert run == reference_parse(text)
    result = drift_outcome(
        exit_code=0,
        timed_out=False,
        text=text,
        lock=lock_of({"numpy": ["2.5.4"], "sortedcontainers": ["2.4.0"]}),
        rules=[],
        timeout_seconds=300,
    )
    has_positive_line = POSITIVE_LINE in reference_lines(text)
    if result.state == "ok":
        assert has_positive_line
        assert run.changes == ()
        assert run.unrecognised == 0
        assert result.entries == ()
    elif result.state == "drift":
        assert not has_positive_line
        assert run.unrecognised == 0
        assert len(result.entries) == len(run.changes) > 0
    else:
        assert result.entries == ()
        assert result.reason is not None


@cases_settings(1_000)
@given(text=st.lists(st.sampled_from(POOL), max_size=6).map("\n".join))
def test_exit_status_one_is_never_ok_or_drift_for_any_text(text: str) -> None:
    result = drift_outcome(
        exit_code=1,
        timed_out=False,
        text=text,
        lock=lock_of({"numpy": ["2.5.4"]}),
        rules=[],
        timeout_seconds=300,
    )
    assert (result.state, result.reason) == ("fail", states.fail_uv_exit(1))


# --- the two readers agree, on well-formed and on damaged outputs ------------------------------------


@cases_settings(1_000)
@given(case=cases(), junk=JUNK, where=st.integers(min_value=0, max_value=100))
def test_the_two_readers_agree_on_an_output_with_an_inserted_unknown_line(
    case: Case, junk: str, where: int
) -> None:
    position = where % (len(case.lines) + 1)
    lines = (*case.lines[:position], junk, *case.lines[position:])
    text = case.end.join(lines) + case.end
    assert parse_dry_run(text) == reference_parse(text)


@cases_settings(2_000)
@given(
    case=cases(),
    edits=st.lists(
        st.tuples(
            st.sampled_from(["delete", "insert", "replace"]),
            st.integers(min_value=0, max_value=10_000),
            st.sampled_from(list(" \t-><=vV.+!_0123456789abcxyzAZ:[;" + ESC)),
        ),
        min_size=1,
        max_size=3,
    ),
)
def test_the_two_readers_agree_on_an_output_with_characters_deleted_inserted_or_replaced(
    case: Case, edits: list[tuple[str, int, str]]
) -> None:
    text = case.text
    for action, where, char in edits:
        position = where % (len(text) + 1)
        if action == "insert":
            text = text[:position] + char + text[position:]
        elif action == "delete":
            text = text[:position] + text[position + 1 :]
        else:
            text = text[:position] + char + text[position + 1 :]
    assert parse_dry_run(text) == reference_parse(text)


@cases_settings(1_000)
@given(
    case=cases(),
    pick=st.integers(min_value=0, max_value=5),
    mode=st.sampled_from(["version", "arrow", "name"]),
)
def test_the_two_readers_agree_when_a_change_line_loses_a_version_or_an_arrow_or_a_name(
    case: Case, pick: int, mode: str
) -> None:
    index = case.change_at[pick % len(case.change_at)]
    tokens = case.lines[index].split()
    if mode == "version":
        tokens = tokens[:2] + tokens[3:]
    elif mode == "name":
        tokens = tokens[:1] + tokens[2:]
    lines = list(case.lines)
    lines[index] = " ".join(tokens)
    if mode == "arrow":
        lines[index] = lines[index].replace("->", "<-")
    text = case.end.join(lines) + case.end
    assert parse_dry_run(text) == reference_parse(text)
