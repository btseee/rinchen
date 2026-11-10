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
        "`model_sha256`, `manifest_sha256` or `hypotheses_sha256`, literals of a fixed tuple",
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
    # ---- the bootstrap (D13): numbers only, no text of a suite ever reaches it ----
    (
        "eval/bootstrap.py",
        "BootstrapSettings.__post_init__",
        "MIN_RESAMPLES",
        CONSTANT,
    ),
    # ---- the evaluation suite file (D13): the same rule as the robustness suite ----
    (
        "eval/suite_rules.py",
        "refusal",
        "where",
        (
            "the helper that builds every refusal of the suite reader: the place and the field "
            "its callers chose, listed here"
        ),
    ),
    (
        "eval/suite_rules.py",
        "refusal",
        "problem",
        (
            "the helper's argument: a literal, or a constant, a count, a code of the registry or "
            "a string of the fixed form of a code that its callers made, listed here"
        ),
    ),
    (
        "eval/suite_rules.py",
        "object_at",
        "place",
        CALLER_PLACE,
    ),
    (
        "eval/suite_rules.py",
        "object_at",
        "field",
        FIELD,
    ),
    (
        "eval/suite_rules.py",
        "object_at",
        "problem",
        (
            "a literal, or the name of a repeated key that belongs to the format or is a code of "
            "the language registry (`known`): any other repeated key is not named"
        ),
    ),
    (
        "eval/suite_rules.py",
        "checked_name",
        "place",
        CALLER_PLACE,
    ),
    (
        "eval/suite_rules.py",
        "checked_name",
        "field",
        FIELD,
    ),
    (
        "eval/suite_rules.py",
        "checked_name",
        "_NAME_RULE",
        CONSTANT,
    ),
    (
        "eval/suite_rules.py",
        "checked_row_id",
        "place",
        PLACE,
    ),
    (
        "eval/suite_rules.py",
        "checked_row_id",
        "field",
        FIELD,
    ),
    (
        "eval/suite_rules.py",
        "checked_row_id",
        "_ROW_ID_RULE",
        CONSTANT,
    ),
    (
        "eval/suite_rules.py",
        "checked_language",
        "place",
        CALLER_PLACE,
    ),
    (
        "eval/suite_rules.py",
        "checked_language",
        "field",
        FIELD,
    ),
    (
        "eval/suite_rules.py",
        "checked_language",
        "_LANGUAGE_RULE",
        CONSTANT,
    ),
    (
        "eval/suite_rules.py",
        "checked_language",
        "_FORBIDDEN_RULE",
        CONSTANT,
    ),
    (
        "eval/suite_rules.py",
        "checked_language",
        "forbidden",
        (
            "an entry of FORBIDDEN_LANGUAGES, a constant of the module, reached only when the "
            "value equals it: the rule names the code and quotes nothing of the file"
        ),
    ),
    (
        "eval/suite_rules.py",
        "checked_language",
        "code",
        (
            "a code of the registry that the caller excluded (a constant of the module), reached "
            "only when the value equals it"
        ),
    ),
    (
        "eval/suite_rules.py",
        "checked_language",
        "shaped",
        (
            "a string of the fixed form of a language code (three lowercase letters, an "
            "underscore, a capital letter and three lowercase letters: seven characters, never a "
            "sentence) that the registry lacks"
        ),
    ),
    (
        "eval/suite_rules.py",
        "checked_sentence",
        "place",
        PLACE,
    ),
    (
        "eval/suite_rules.py",
        "checked_sentence",
        "field",
        "`text.<code>`: a field of the format and a code of the registry",
    ),
    (
        "eval/suite_rules.py",
        "checked_sentence",
        "MAX_SENTENCE_CHARS",
        CONSTANT,
    ),
    (
        "eval/suite_rules.py",
        "split_direction",
        "place",
        (
            "the argument of the function: `direction` or the place its callers chose (the "
            "reader's `suite`, an option of the operator), never the direction itself"
        ),
    ),
    (
        "eval/suite_rules.py",
        "split_direction",
        "field",
        "`directions[N]` or None: a field of the format and the position of the direction",
    ),
    (
        "eval/suite_rules.py",
        "split_direction",
        "_DIRECTION_RULE",
        CONSTANT,
    ),
    (
        "eval/suite_rules.py",
        "read_source",
        "key",
        "a field of the source's own tuple, a literal of the reader",
    ),
    (
        "eval/suite_rules.py",
        "read_source",
        "None",
        "the absent field of a refusal about the whole source, a constant",
    ),
    (
        "eval/suite_rules.py",
        "_files",
        "field",
        "`files[N]`: a field of the format and the position of the entry in the file",
    ),
    (
        "eval/suite_rules.py",
        "_files",
        "_FILE_NAME_RULE",
        CONSTANT,
    ),
    (
        "eval/suite_rules.py",
        "read_directions",
        "field",
        "`directions[N]`: a field of the format and the position of the direction in the list",
    ),
    (
        "eval/suite_rules.py",
        "read_directions",
        "_DIRECTION_RULE",
        CONSTANT,
    ),
    (
        "eval/suite_rules.py",
        "require_rows_for_directions",
        "index",
        "the position of a direction in the list, an integer",
    ),
    (
        "eval/suite_rules.py",
        "read_rows",
        "MAX_ROWS",
        CONSTANT,
    ),
    (
        "eval/suite_rules.py",
        "read_rows",
        "row_place(index, row.id)",
        PLACE,
    ),
    (
        "eval/suite_rules.py",
        "_read_row",
        "place",
        PLACE,
    ),
    (
        "eval/suite_rules.py",
        "_read_row",
        "key",
        "a field of the row's own tuple (`id`, `doc`, `text`), a literal of the reader",
    ),
    (
        "eval/suite_rules.py",
        "_read_row",
        "None",
        "the absent field of a refusal about a whole row, a constant",
    ),
    (
        "eval/suite_rules.py",
        "_read_text",
        "place",
        PLACE,
    ),
    (
        "eval/suites.py",
        "Suite.view",
        "self.name",
        (
            "the name of the suite: it matched the id pattern when the suite was read, and a "
            "suite built in memory is named by its builder"
        ),
    ),
    (
        "eval/suites.py",
        "Suite.view",
        "_named(direction)",
        (
            "the direction asked for, only when it has the form of two language codes and "
            "nothing is quoted otherwise: a direction is an option of the operator, never a "
            "text of a suite"
        ),
    ),
    (
        "eval/suites.py",
        "Suite.from_bytes",
        "kind",
        "one of the three kinds of the format: `_member` refused any other value first",
    ),
    (
        "eval/suites.py",
        "Suite.from_bytes",
        "' or '.join(SPLITS_OF_KIND[kind])",
        "the splits a kind may have, constants of the module",
    ),
    (
        "eval/suites.py",
        "read_suite",
        "path",
        OPERATOR_PATH,
    ),
    (
        "eval/suites.py",
        "_too_large",
        "MAX_SUITE_BYTES // 2 ** 20",
        CONSTANT,
    ),
    (
        "eval/suites.py",
        "_member",
        "field",
        "`kind` or `split`, literals of the callers",
    ),
    (
        "eval/suites.py",
        "_member",
        "', '.join(allowed)",
        "the kinds or the splits of the format, constants",
    ),
    (
        "eval/suites.py",
        "_document",
        "None",
        "the absent field of a refusal about the whole document, a constant",
    ),
    (
        "eval/suites.py",
        "_document",
        "SUITE_SCHEMA",
        CONSTANT,
    ),
    (
        "eval/suites.py",
        "_document",
        "key",
        (
            "a top-level key of the format (`schema`, `name`, `kind`, ...), taken from the "
            "reader's own tuple"
        ),
    ),
    # ---- the readers of the sources of a suite (D13, Task 4) ----
    (
        "eval/suite_sources.py",
        "lines_rows",
        "language",
        "a registry code: `checked_language` has passed it before any message is built",
    ),
    ("eval/suite_sources.py", "lines_rows", "number", "the number of a line"),
    ("eval/suite_sources.py", "build_multiway", "HUB_LANGUAGE", CONSTANT),
    (
        "eval/suite_sources.py",
        "build_multiway",
        "language",
        "a key of `languages`, which `checked_language` has passed before any message is built",
    ),
    ("eval/suite_sources.py", "build_multiway", "len(rows)", COUNT),
    ("eval/suite_sources.py", "build_multiway", "len(hub)", COUNT),
    (
        "eval/suite_sources.py",
        "_by_id",
        "language",
        "a registry code, passed by `build_multiway` before it is called",
    ),
    ("eval/suite_sources.py", "_by_id", "len(found)", COUNT),
    ("eval/suite_sources.py", "_by_id", "len(rows)", COUNT),
    ("eval/suite_sources.py", "_partner_directions", "HUB_LANGUAGE", CONSTANT),
    (
        "eval/suite_sources.py",
        "human_rows",
        "src",
        "a registry code of a row: `_item` has passed it through `checked_language`",
    ),
    (
        "eval/suite_sources.py",
        "human_rows",
        "tgt",
        "a registry code of a row: `_item` has passed it through `checked_language`",
    ),
    (
        "eval/suite_sources.py",
        "_item",
        "place",
        "`export row index` and the index of the row in the export, never a cell",
    ),
    (
        "eval/suite_sources.py",
        "_item",
        "column",
        "a column of the export, taken from the module's own tuple `HUMAN_COLUMNS`",
    ),
    (
        "eval/suite_sources.py",
        "_item",
        "None",
        "the absent field of a refusal about the whole row, a constant",
    ),
    ("eval/suite_sources.py", "_item", "_HUMAN_SPLIT", CONSTANT),
    (
        "eval/suite_sources.py",
        "_optional_document",
        "place",
        "`export row index` and the index of the row in the export, never a cell",
    ),
    ("eval/suite_sources.py", "_optional_document", "_DOC_ID_RULE", CONSTANT),
    (
        "eval/suite_sources.py",
        "_optional_segment",
        "place",
        "`export row index` and the index of the row in the export, never a cell",
    ),
    (
        "eval/suite_sources.py",
        "_require_one_set",
        "index",
        "the index of a row of the export",
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


def test_the_scan_covers_the_modules_of_the_evaluation_suite() -> None:
    covered = {f.path for f in scanned()}

    assert {"eval/suite_rules.py", "eval/suites.py"} <= covered


def test_the_scan_covers_the_readers_of_the_sources_of_a_suite() -> None:
    covered = {f.path for f in scanned()}

    assert "eval/suite_sources.py" in covered


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
