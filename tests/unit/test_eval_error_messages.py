"""No message of an error of `rinchen.eval` interpolates a text of a suite or of a hypothesis
(D12, Task 5).

A text of a suite (a source, a reference, a term, a spelling) or of a hypothesis (what a system
wrote) must never come back in an exception message, a log line or a manifest: it may be a
document, or a secret. The way to keep that true as the code changes is to look at every place
where an error is built from something that is not a literal: this module finds each by reading
the source (an AST scan) and compares the list with the one below, in which every entry was
reviewed and says why it holds no such text. A new interpolation fails the test until someone has
looked at it and added it here with its reason; an entry that no longer matches code fails too, so
the list cannot rot.

The scan follows the four error classes of the project, `ValueError` and `TypeError`, the
helpers that make a refusal (`refusal` of the suite reader, `_refused` of the hypotheses reader)
and the command's exit for an output it cannot write (`exit_cannot_write`): the helpers are where
the message is assembled, so their callers' arguments are listed too.

What a reviewed entry may hold: an index, a count, a constant, a field name of the format, a
section of the format, a registry code, an id that matched its pattern, a path or an option of the
operator, the digest of a file. Never a source, a reference, a term, a spelling or a hypothesis.

The scan also reads the three modules of the command `rinchen eval robustness` (`cli/`), whose
messages come from the same files: they name paths of the operator, options, counts and the
parts of a request that `request_place` chose, never what a file holds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import rinchen
from tests.support.error_messages import ERROR_NAMES, Finding, interpolations

PACKAGE = Path(rinchen.__file__).parent / "eval"
COMMAND = Path(rinchen.__file__).parent / "cli"
COMMAND_MODULES: Final = (
    "eval_robustness.py",
    "eval_robustness_output.py",
    "eval_robustness_plan.py",
)
NAMES: Final = ERROR_NAMES | {
    "refusal",
    "_refused",
    "ValueError",
    "TypeError",
    "exit_cannot_write",
}

PLACE = (
    "the place the reader chose: a section of the format, an index and an id that matched "
    "its pattern, never a text"
)
FIELD = "a field of the format, chosen by the reader and never taken from the file"
COUNT = "a count"
CONSTANT = "a constant of the module"
OPERATOR_PATH = "a path of the operator, not a text of a suite or of a hypothesis"
REQUEST = (
    "a request of the suite named by section, id and arm; `request_place` names a part "
    "only when it is well formed (a section of the format, an id that matched the pattern, "
    "an arm of the format) and no text of a file otherwise"
)
LINE_PLACE = "the number of a line and a request named by `request_place`: well-formed parts only"
CALLER_PLACE = (
    "the place of a refusal, made by the callers that are listed here, see them"
)

# (file, function, expression, why it holds no text of a suite or of a hypothesis)
REVIEWED: Final[tuple[tuple[str, str, str, str], ...]] = (
    # ---- the hypotheses ----
    (
        "eval/robust_hypotheses.py",
        "Hypothesis.__post_init__",
        "', '.join(_STATUSES)",
        "the five statuses of the engine, constants",
    ),
    (
        "eval/robust_hypotheses.py",
        "run_requests",
        "len(answers)",
        COUNT,
    ),
    (
        "eval/robust_hypotheses.py",
        "run_requests",
        "len(requests)",
        COUNT,
    ),
    (
        "eval/robust_hypotheses.py",
        "read_hypotheses",
        "path",
        OPERATOR_PATH,
    ),
    (
        "eval/robust_hypotheses.py",
        "hypotheses_from_bytes",
        "place",
        LINE_PLACE,
    ),
    (
        "eval/robust_hypotheses.py",
        "hypotheses_from_bytes",
        "len(missing)",
        COUNT,
    ),
    (
        "eval/robust_hypotheses.py",
        "hypotheses_from_bytes",
        "request_place(*missing[0])",
        "the first request of the suite that the file lacks: " + REQUEST,
    ),
    (
        "eval/robust_hypotheses.py",
        "hypotheses_to_bytes",
        "request_place(*key)",
        "the key of a hypothesis being written: " + REQUEST,
    ),
    (
        "eval/robust_hypotheses.py",
        "_too_large",
        "MAX_HYPOTHESES_BYTES // 2 ** 20",
        CONSTANT,
    ),
    (
        "eval/robust_hypotheses.py",
        "_refused",
        "place",
        (
            "the argument of the helper: its callers pass a line number, a request named by "
            "`request_place` or a literal, all listed here"
        ),
    ),
    (
        "eval/robust_hypotheses.py",
        "_refused",
        "problem",
        (
            "the argument of the helper: its callers pass a literal, a field of the format "
            "or the five statuses, all listed here"
        ),
    ),
    (
        "eval/robust_hypotheses.py",
        "_read_line",
        "where",
        "`line N`: the number of a line, an integer",
    ),
    (
        "eval/robust_hypotheses.py",
        "_read_line",
        "place",
        LINE_PLACE,
    ),
    (
        "eval/robust_hypotheses.py",
        "_read_line",
        "field",
        FIELD,
    ),
    (
        "eval/robust_hypotheses.py",
        "_read_line",
        "', '.join(_STATUSES)",
        "the five statuses of the engine, constants",
    ),
    # ---- the result ----
    (
        "eval/robustness.py",
        "RobustnessSystem.__post_init__",
        "field",
        "`model_sha256` or `manifest_sha256`, literals of a fixed tuple",
    ),
    (
        "eval/robustness.py",
        "_require_exactly",
        "len(missing)",
        COUNT,
    ),
    (
        "eval/robustness.py",
        "_require_exactly",
        "request_place(*missing[0])",
        "the first request of the suite that has no hypothesis: " + REQUEST,
    ),
    (
        "eval/robustness.py",
        "_require_exactly",
        "len(stray)",
        COUNT,
    ),
    (
        "eval/robustness.py",
        "_require_exactly",
        "request_place(*stray[0])",
        "the key of a hypothesis that answers no request, which may come from anywhere: "
        + REQUEST,
    ),
    # ---- the sections and the metrics ----
    (
        "eval/robust_sections.py",
        "_reference",
        "section",
        "`numbers`, `placeholders` or `mong`, literals of the callers",
    ),
    (
        "eval/robust_mong.py",
        "_require_one_text_per_reference",
        "arm",
        "`cyrl`, `mong` or `converted`, literals of the caller",
    ),
    (
        "eval/robust_mong.py",
        "_require_one_text_per_reference",
        "len(texts)",
        COUNT,
    ),
    (
        "eval/robust_mong.py",
        "_require_one_text_per_reference",
        "len(references)",
        COUNT,
    ),
    (
        "eval/robust_spans.py",
        "EntityTable.spellings_of",
        "entity_id",
        (
            "an id of the entity table of the suite (the reader allows `[a-z0-9_]{1,32}` only) "
            "that a row listed; the spelling it lacks is not quoted"
        ),
    ),
    (
        "eval/robust_spans.py",
        "EntityTable.spellings_of",
        "language",
        "the target language of a row, a code of the language registry",
    ),
    # ---- the command: `rinchen eval robustness` ----
    (
        "cli/eval_robustness_plan.py",
        "plan_of",
        "_SYSTEM_RULE",
        "the characters a system name may hold, a constant of the module",
    ),
    (
        "cli/eval_robustness.py",
        "_require_the_files_that_were_hashed",
        "name",
        "`model.bin` or `manifest.json`, the literals of the function's own tuple",
    ),
    (
        "cli/eval_robustness_output.py",
        "commit",
        "target",
        OPERATOR_PATH,
    ),
    (
        "cli/eval_robustness_output.py",
        "commit",
        "exc",
        (
            "the OSError of a write the command made itself, from bytes it produced: it names a "
            "path of the operator and the reason the system gave, no text of a suite or of a "
            "hypothesis"
        ),
    ),
    (
        "cli/eval_robustness_output.py",
        "commit",
        "_OUTPUTS",
        CONSTANT,
    ),
    # ---- the suite file ----
    (
        "eval/robust_suite.py",
        "Suite.__post_init__",
        "', '.join(SECTIONS)",
        "the six sections of the format, constants",
    ),
    (
        "eval/robust_suite.py",
        "read_suite",
        "path",
        OPERATOR_PATH,
    ),
    (
        "eval/robust_suite.py",
        "_too_large",
        "MAX_SUITE_BYTES // 2 ** 20",
        CONSTANT,
    ),
    (
        "eval/robust_suite.py",
        "_document",
        "None",
        "the absent field of a refusal about the whole document, a constant",
    ),
    (
        "eval/robust_suite.py",
        "_document",
        "SUITE_SCHEMA",
        CONSTANT,
    ),
    (
        "eval/robust_suite.py",
        "_document",
        "key",
        (
            "a top-level key of the format (`schema`, `name`, `entities`, `sections`), taken "
            "from the reader's own tuple"
        ),
    ),
    (
        "eval/robust_suite.py",
        "_entity_table",
        "entry",
        "`entity table, entry index N`: an index, an integer",
    ),
    (
        "eval/robust_suite.py",
        "_spellings_by_language",
        "entry",
        "the entry index and an id that matched `[a-z0-9_]{1,32}` before it was named",
    ),
    (
        "eval/robust_suite.py",
        "_spellings",
        "entry",
        "the entry index and an id that matched `[a-z0-9_]{1,32}` before it was named",
    ),
    (
        "eval/robust_suite.py",
        "_spellings",
        "language",
        "a language code of the registry (the caller refused any other)",
    ),
    (
        "eval/robust_suite.py",
        "_spellings",
        "_MAX_SPELLINGS",
        CONSTANT,
    ),
    (
        "eval/robust_suite.py",
        "suite_requests",
        "stray",
        COUNT,
    ),
    # ---- the rows of the suite file ----
    (
        "eval/robust_suite_rules.py",
        "refusal",
        "where",
        (
            "the helper that builds every refusal of the reader: the place and the field its "
            "callers chose, listed here"
        ),
    ),
    (
        "eval/robust_suite_rules.py",
        "refusal",
        "problem",
        (
            "the helper's argument: a literal, or a constant or a count its callers made, "
            "listed here"
        ),
    ),
    (
        "eval/robust_suite_rules.py",
        "object_at",
        "place",
        CALLER_PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "object_at",
        "field",
        FIELD,
    ),
    (
        "eval/robust_suite_rules.py",
        "object_at",
        "problem",
        (
            "a literal, or the name of a repeated key that belongs to the format or is a code of "
            "the language registry (`known`): any other repeated key is not named"
        ),
    ),
    (
        "eval/robust_suite_rules.py",
        "checked_name",
        "place",
        CALLER_PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "checked_name",
        "field",
        FIELD,
    ),
    (
        "eval/robust_suite_rules.py",
        "checked_name",
        "_ID_RULE",
        CONSTANT,
    ),
    (
        "eval/robust_suite_rules.py",
        "checked_text",
        "place",
        CALLER_PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "checked_text",
        "field",
        FIELD,
    ),
    (
        "eval/robust_suite_rules.py",
        "checked_text",
        "limit",
        "the length limit of the format, an integer",
    ),
    (
        "eval/robust_suite_rules.py",
        "read_section",
        "section",
        "a section of the format, from the reader's own tuple",
    ),
    (
        "eval/robust_suite_rules.py",
        "read_section",
        "None",
        "the absent field of a refusal about a whole section, a constant",
    ),
    (
        "eval/robust_suite_rules.py",
        "read_section",
        "row_place(section, index, row.id)",
        PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "read_row",
        "place",
        PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "_row_object",
        "place",
        PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "_row_object",
        "key",
        "a field of the section's own table of fields, taken from the reader's tuple",
    ),
    (
        "eval/robust_suite_rules.py",
        "_row_object",
        "None",
        "the absent field of a refusal about a whole row, a constant",
    ),
    (
        "eval/robust_suite_rules.py",
        "_language",
        "place",
        PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "_language",
        "field",
        "`src_lang` or `tgt_lang`, literals of the callers",
    ),
    (
        "eval/robust_suite_rules.py",
        "_language",
        "problem",
        "one of the module's constants that describe a language code, passed by the callers",
    ),
    (
        "eval/robust_suite_rules.py",
        "_require_few_numbers",
        "place",
        PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "_require_few_numbers",
        "MAX_NUMBER_TOKENS",
        CONSTANT,
    ),
    (
        "eval/robust_suite_rules.py",
        "_entity_ids",
        "place",
        PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "_entity_ids",
        "field",
        "`entities[N]`: a field of the format and the index of the id in the row's list",
    ),
    (
        "eval/robust_suite_rules.py",
        "_term_pairs",
        "place",
        PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "_term_pairs",
        "field",
        "`terms[N]` or `terms[N].src` or `.tgt`: fields of the format and an index",
    ),
    (
        "eval/robust_suite_rules.py",
        "_term_pairs",
        "key",
        "`src` or `tgt`, literals of the reader's own tuple",
    ),
    (
        "eval/robust_suite_rules.py",
        "_script",
        "place",
        PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "_mong_source",
        "place",
        PLACE,
    ),
    (
        "eval/robust_suite_rules.py",
        "row_document",
        "row_place(section, index)",
        "a section of the format and an index, without an id",
    ),
    (
        "eval/robust_suite_rules.py",
        "row_document",
        "field",
        "an optional field of the format, from the reader's own table of defaults",
    ),
)


def scanned() -> list[Finding]:
    found: list[Finding] = []
    for path in sorted(PACKAGE.glob("*.py")):
        relative = f"eval/{path.name}"
        found += interpolations(path.read_text(encoding="utf-8"), relative, names=NAMES)
    for name in COMMAND_MODULES:
        found += interpolations(
            (COMMAND / name).read_text(encoding="utf-8"), f"cli/{name}", names=NAMES
        )
    return found


def keys_of(entries: list[Finding]) -> set[tuple[str, str, str]]:
    return {(f.path, f.function, f.expression) for f in entries}


def test_every_interpolation_in_an_error_of_eval_has_been_reviewed() -> None:
    reviewed = {
        (path, function, expression) for path, function, expression, _ in REVIEWED
    }

    assert sorted(keys_of(scanned()) - reviewed) == [], (
        "unreviewed: add it to REVIEWED with its reason"
    )


def test_every_reviewed_entry_still_matches_a_line_of_code() -> None:
    reviewed = {
        (path, function, expression) for path, function, expression, _ in REVIEWED
    }

    assert sorted(reviewed - keys_of(scanned())) == [], "stale: remove it from REVIEWED"


def test_every_reviewed_entry_says_why_and_is_listed_once() -> None:
    keys = [(path, function, expression) for path, function, expression, _ in REVIEWED]

    assert len(keys) == len(set(keys))
    assert all(len(reason.split()) >= 2 for *_, reason in REVIEWED)


def test_no_reviewed_entry_quotes_a_text() -> None:
    # The kind of expression that would put a text of a suite or of a hypothesis into a message.
    suspicious = (
        "text",
        "texts",
        "source",
        "reference",
        "hypothesis",
        "hypotheses",
        "sentence",
        "line",
        "word",
        "value",
        "spelling",
        "term",
    )
    quoting = [
        (path, function, expression)
        for path, function, expression, _ in REVIEWED
        if any(f"{name}[" in expression or expression == name for name in suspicious)
    ]

    assert quoting == []


def test_the_scan_covers_every_module_of_eval() -> None:
    covered = {f.path for f in scanned()}

    assert {
        "eval/robust_hypotheses.py",
        "eval/robustness.py",
        "eval/robust_sections.py",
        "eval/robust_suite.py",
        "eval/robust_suite_rules.py",
        "eval/robust_spans.py",
        "eval/robust_mong.py",
        "cli/eval_robustness.py",
        "cli/eval_robustness_output.py",
        "cli/eval_robustness_plan.py",
    } <= covered


# ---- the scan itself ----------------------------------------------------------------------


def test_the_scan_follows_the_names_it_is_given() -> None:
    source = (
        "def one(x):\n"
        "    raise refusal(x, 'a')\n"
        "def two(x):\n"
        "    raise ValueError(f'a {x} b')\n"
        "def three(x):\n"
        "    raise DataError(f'c {x}')\n"
    )

    default = [(f.function, f.expression) for f in interpolations(source, "x.py")]
    wider = [
        (f.function, f.expression) for f in interpolations(source, "x.py", names=NAMES)
    ]

    assert default == [("three", "x")]
    assert wider == [("one", "x"), ("two", "x"), ("three", "x")]
