"""No message of `rinchen eval suite`, of `rinchen eval run` and of the configuration they read holds
a text of a suite, of a source or of a system's output (D13, Tasks 4 and 7; later tasks extend this
file with their command modules).

`tests/unit/test_eval_error_messages.py` reads every module of `rinchen.eval`, the suite readers
and `suite_sources` among them, and compares each place where an error is built from something that
is not a literal with a reviewed list. This file does the same for what lives outside `rinchen.eval`:
the command modules `cli/eval_suite*.py` and `config/eval_suites.py`, whose messages name paths of
the operator, options, counts, the languages of the registry and the places a reader chose (a line,
a row index, a field of the format), never what a file held. The second half plants a marker in
every text of every source a build can read, runs each way the command fails and asserts that the
marker is in no standard output, no standard error and no log line.

A reviewed entry may hold: an index, a count, a constant, a field name, a registry code, an id
that matched its pattern, a path or an option of the operator, the class name of an upstream error.
Never a sentence, a document address, a cell of a file or the message of an upstream error."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest

import rinchen
from tests.support.error_messages import ERROR_NAMES, Finding, interpolations
from tests.support.eval_suite_cli import (
    LANGUAGES,
    FakeLoadDataset,
    SuiteWorkspace,
    export_row,
    golden_export,
    install_fake_load_dataset,
    invoke,
    make_workspace,
    shard_rows,
    write_export,
    write_language_shards,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SOURCE_ROOT = Path(rinchen.__file__).parent
CLI = SOURCE_ROOT / "cli"
MARKER = "SECRET-MARKER-83"
NAMES: Final = ERROR_NAMES | {"exit_cannot_write", "refusal"}

OPERATOR_PATH = "a path of the operator, not a text of a source"
OS_ERROR = "the OSError of the operating system for a path of the operator"
CONSTANT = "a constant of the module"
COUNT = "a count or the number of a line or a row"
LANGUAGE = "a registry code: it has passed the check against the registry"
OPTION = "an option of the operator that the configuration was asked about, not a text of a file"
UPSTREAM_CLASS = (
    "the class name of an upstream error, never its message (which may hold a token)"
)
DIRECTION_PLACE = (
    "`direction_place` names a direction only when it has the shape of one, else says it is "
    "not well formed"
)
CONFIGURED = (
    "a name, a split or a column that the configuration holds and the code has checked"
)

# (file, function, expression, why it holds no text of a source)
REVIEWED: Final[tuple[tuple[str, str, str, str], ...]] = (
    # ---- cli/eval_suite.py ----
    ("cli/eval_suite.py", "fetch_command", "benchmark", CONFIGURED),
    ("cli/eval_suite.py", "fetch_command", "target", OPERATOR_PATH),
    ("cli/eval_suite.py", "fetch_command", "exc", OS_ERROR),
    ("cli/eval_suite_plan.py", "pin_of", "_REVISION_RULE", CONSTANT),
    ("cli/eval_suite.py", "_fetch_language", "destination", OPERATOR_PATH),
    ("cli/eval_suite.py", "_fetch_language", "exc", OS_ERROR),
    ("cli/eval_suite.py", "_fetch_language", "language", LANGUAGE),
    ("cli/eval_suite.py", "_fetch_language", "split", CONFIGURED),
    ("cli/eval_suite.py", "_fetch_language", "dataset", CONFIGURED),
    ("cli/eval_suite.py", "_fetch_language", "type(error).__name__", UPSTREAM_CLASS),
    ("cli/eval_suite_output.py", "write_fetch_manifest", "target", OPERATOR_PATH),
    ("cli/eval_suite_output.py", "write_fetch_manifest", "exc", OS_ERROR),
    ("cli/eval_suite_plan.py", "plan_of", "_NAME_RULE", CONSTANT),
    ("cli/eval_suite.py", "_build_human", "', '.join(EXPORT_SUFFIXES)", CONSTANT),
    ("cli/eval_suite.py", "_build_multiway", "origin", OPERATOR_PATH),
    ("cli/eval_suite.py", "_build_multiway", "root", OPERATOR_PATH),
    ("cli/eval_suite.py", "_build_multiway", "HUB_LANGUAGE", CONSTANT),
    ("cli/eval_suite_output.py", "commit_build", "target", OPERATOR_PATH),
    ("cli/eval_suite_output.py", "commit_build", "exc", OS_ERROR),
    ("cli/eval_suite.py", "inspect_command", "path", OPERATOR_PATH),
    ("cli/eval_suite.py", "inspect_command", "MAX_SUITE_BYTES // 2 ** 20", CONSTANT),
    ("cli/eval_suite_plan.py", "benchmark_of", "name", OPTION),
    (
        "cli/eval_suite_plan.py",
        "benchmark_of",
        "', '.join(sorted(suites.benchmarks))",
        CONFIGURED,
    ),
    ("cli/eval_suite_plan.py", "benchmark_of", "split", OPTION),
    ("cli/eval_suite_plan.py", "benchmark_of", "', '.join(bench.splits)", CONFIGURED),
    # ---- cli/eval_suite_files.py ----
    ("cli/eval_suite_files.py", "language_inputs", "root", OPERATOR_PATH),
    ("cli/eval_suite_files.py", "language_inputs", "entry", OPERATOR_PATH),
    ("cli/eval_suite_files.py", "language_inputs", "SHARD_PATTERN", CONSTANT),
    ("cli/eval_suite_files.py", "language_inputs", "benchmark", CONFIGURED),
    ("cli/eval_suite_files.py", "_language_of", "path", OPERATOR_PATH),
    ("cli/eval_suite_files.py", "_language_of", "code", LANGUAGE),
    ("cli/eval_suite_files.py", "_language_of", "benchmark", CONFIGURED),
    (
        "cli/eval_suite_files.py",
        "_read_parquet",
        "where",
        "a language of the registry and the file name of a shard, as listed in the source directory",
    ),
    (
        "cli/eval_suite_files.py",
        "_read_parquet",
        "type(error).__name__",
        UPSTREAM_CLASS,
    ),
    (
        "cli/eval_suite_files.py",
        "_column",
        "where",
        "a language of the registry and the file name of a shard, as listed in the source directory",
    ),
    ("cli/eval_suite_files.py", "_column", "name", CONFIGURED),
    (
        "cli/eval_suite_files.py",
        "_bad_cell",
        "where",
        "a language of the registry and the file name of a shard, as listed in the source directory",
    ),
    ("cli/eval_suite_files.py", "_bad_cell", "column", CONFIGURED),
    (
        "cli/eval_suite_files.py",
        "_bad_cell",
        "problem",
        "the words `must be text` or `must be text or a whole number`, chosen by the callers",
    ),
    ("cli/eval_suite_files.py", "_bad_cell", "index", COUNT),
    ("cli/eval_suite_files.py", "export_rows", "path", OPERATOR_PATH),
    ("cli/eval_suite_files.py", "_json_lines", "number", COUNT),
    ("cli/eval_suite_files.py", "_tsv", "reader.line_num", COUNT),
    (
        "cli/eval_suite_files.py",
        "_tsv",
        "type(error).__name__",
        "the class name of the csv error",
    ),
    ("cli/eval_suite_files.py", "_tsv_row", "index", COUNT),
    (
        "cli/eval_suite_files.py",
        "read_fetch_manifest",
        "type(error).__name__",
        "the class name of the validation error of a manifest, never its message",
    ),
    ("cli/eval_suite_files.py", "read_fetch_manifest", "FETCH_STAGE", CONSTANT),
    ("cli/eval_suite_files.py", "read_fetch_manifest", "benchmark", CONFIGURED),
    ("cli/eval_suite_files.py", "read_fetch_manifest", "split", CONFIGURED),
    ("cli/eval_suite_files.py", "check_against_fetch", "language", LANGUAGE),
    (
        "cli/eval_suite_files.py",
        "check_against_fetch",
        "key[1]",
        "the file name of a shard in the source directory: a path of the operator",
    ),
    (
        "cli/eval_suite_files.py",
        "check_against_fetch",
        "name",
        "a file name that the manifest of a fetch lists: a path of the operator's own manifest",
    ),
    # ---- config/eval_suites.py ----
    (
        "config/eval_suites.py",
        "load_eval_suites_config",
        "error",
        (
            "the message of PyYAML about the operator's own configuration file, which may "
            "quote a line of it: never a text of a suite or of a source"
        ),
    ),
    (
        "config/eval_suites.py",
        "load_eval_suites_config",
        "type(raw).__name__",
        "the class name of the top level of the configuration",
    ),
    (
        "config/eval_suites.py",
        "load_eval_suites_config",
        "problems",
        (
            "the field paths and the messages of the validation, which never quote a value "
            "(`include_input=False`; the messages of the module's own validators quote none)"
        ),
    ),
    # ---- cli/eval_run*.py and cli/eval_systems.py (Task 7) ----
    ("cli/eval_run.py", "_require_declared", "direction_place(entry)", DIRECTION_PLACE),
    ("cli/eval_run.py", "_require_run_dir_apart", "run_dir", OPERATOR_PATH),
    ("cli/eval_run.py", "_require_run_dir_apart", "held", OPERATOR_PATH),
    (
        "cli/eval_run.py",
        "_require_the_files_that_were_hashed",
        "name",
        "`model.bin` or `manifest.json`, the two names the loop above lists",
    ),
    ("cli/eval_run_inputs.py", "read_robustness_input", "ROBUSTNESS_STAGE", CONSTANT),
    (
        "cli/eval_run_inputs.py",
        "read_robustness_input",
        "name",
        "`robustness.json` or `report.json`, the two names of the files D12 writes",
    ),
    (
        "cli/eval_run_output.py",
        "documents_of",
        "MAX_OUTPUTS_BYTES // 2 ** 20",
        CONSTANT,
    ),
    ("cli/eval_run_output.py", "commit", "target", OPERATOR_PATH),
    ("cli/eval_run_output.py", "commit", "exc", OS_ERROR),
    ("cli/eval_run_output.py", "commit", "_OUTPUTS", CONSTANT),
    ("cli/eval_run_plan.py", "plan_of", "_SYSTEM_RULE", CONSTANT),
    ("cli/eval_run_plan.py", "plan_of", "_SYSTEM_OPTIONS", CONSTANT),
    ("cli/eval_run_plan.py", "plan_of", "len(chosen)", COUNT),
    (
        "cli/eval_run_plan.py",
        "plan_of",
        "', '.join(engine_options)",
        "the names of options the operator gave, from a fixed table",
    ),
    ("cli/eval_run_plan.py", "_usable_name", "what", CONSTANT),
    ("cli/eval_run_plan.py", "_usable_name", "_SYSTEM_RULE", CONSTANT),
    (
        "cli/eval_run_plan.py",
        "_scores_of",
        "', '.join(SEGMENT_SCORE_METRICS)",
        "the three metric names of the registry",
    ),
    ("cli/eval_systems.py", "generator_system", "name", OPTION),
    (
        "cli/eval_systems.py",
        "generator_system",
        "', '.join(sorted(generators))",
        "the names of the generators the operator's own configuration holds",
    ),
    (
        "cli/eval_systems.py",
        "_generators_of",
        "where",
        "the line and column PyYAML gives for the operator's own configuration, two integers",
    ),
    (
        "cli/eval_systems.py",
        "_generators_of",
        "type(raw).__name__",
        "the class name of the top level of the configuration",
    ),
    (
        "cli/eval_systems.py",
        "_generators_of",
        "problems",
        (
            "the field paths and the messages of the validation, which never quote a value "
            "(`include_input=False`; the messages of the module's own validators quote none)"
        ),
    ),
    ("cli/eval_systems.py", "_evaluation_config", "name", OPTION),
    ("cli/eval_systems.py", "vocabulary_guard", "entry.path", OPERATOR_PATH),
    ("cli/eval_systems.py", "vocabulary_guard", "family", CONSTANT),
    ("cli/eval_systems.py", "vocabulary_guard", "language", LANGUAGE),
    (
        "cli/eval_systems.py",
        "vocabulary_guard",
        "shown",
        "the token the family's table (D6's code tables) holds for a registry code, in parentheses and only when it is not the code: never an entry of the vocabulary",
    ),
    # ---- cli/eval_report*.py ----
    (
        "cli/eval_report.py",
        "_refuse_dev_runs",
        "one.record.suite.name",
        "the name of the suite of a run, an id: `run_from_bytes` checked it against the pattern",
    ),
    (
        "cli/eval_report.py",
        "_suite_of",
        "one.record.system.name",
        "the name of the system of a run, an id: `run_from_bytes` checked it against the pattern",
    ),
    (
        "cli/eval_report.py",
        "_suite_of",
        "one.record.suite.name",
        "the name of the suite of a run, an id: `run_from_bytes` checked it against the pattern",
    ),
    ("cli/eval_report_inputs.py", "read_run_record", "directory", OPERATOR_PATH),
    ("cli/eval_report_inputs.py", "read_run_record", "RUN_FILE", CONSTANT),
    (
        "cli/eval_report_inputs.py",
        "read_run_record",
        "name",
        (
            "a relative path inside the run directory that `run.json` lists (`run_from_bytes` "
            "refuses `..`, a leading `/` and a backslash): a file name of the operator's run, not a "
            "text of a suite or of an output"
        ),
    ),
    ("cli/eval_report_inputs.py", "_manifest_of", "directory", OPERATOR_PATH),
    (
        "cli/eval_report_inputs.py",
        "_require_a_finished_run",
        "directory",
        OPERATOR_PATH,
    ),
    ("cli/eval_report_inputs.py", "_require_a_finished_run", "RUN_STAGE", CONSTANT),
    ("cli/eval_report_output.py", "commit", "target", OPERATOR_PATH),
    ("cli/eval_report_output.py", "commit", "exc", OS_ERROR),
    ("cli/eval_report_output.py", "commit", "_OUTPUTS", CONSTANT),
    # ---- cli/release*.py and export/release.py (Task 9) ----
    ("cli/release_stage_inputs.py", "_manifest_of", "directory", OPERATOR_PATH),
    (
        "cli/release_stage_inputs.py",
        "_require_a_finished_report",
        "directory",
        OPERATOR_PATH,
    ),
    (
        "cli/release_stage_inputs.py",
        "_require_a_finished_report",
        "REPORT_STAGE",
        CONSTANT,
    ),
    (
        "cli/release_stage_inputs.py",
        "_require_a_finished_report",
        "_STATUS_KEY",
        CONSTANT,
    ),
    ("cli/release_stage_inputs.py", "read_report_dir", "directory", OPERATOR_PATH),
    (
        "cli/release_stage_inputs.py",
        "read_report_dir",
        "name",
        "`report.json` or `report.md`, the file names of the two entries of the loop above",
    ),
    ("cli/release_stage_output.py", "commit", "target", OPERATOR_PATH),
    ("cli/release_stage_output.py", "commit", "exc", OS_ERROR),
    ("cli/release_stage_output.py", "commit", "_OUTPUTS", CONSTANT),
    (
        "export/release.py",
        "require_report_for_export",
        "system",
        (
            "the system named by the operator's `--system`, which `check_system_name` has "
            "checked against the id pattern: it names no text of a report"
        ),
    ),
    (
        "export/release.py",
        "require_report_for_export",
        "entry.kind",
        "the kind of a system, one of three constants: the strict reader of the report checked it",
    ),
    (
        "export/release.py",
        "require_report_for_export",
        "what",
        "`model` or `manifest`, the two words of the loop above",
    ),
    (
        "export/release.py",
        "require_report_for_export",
        "short(found)",
        "the first twelve hexadecimal digits of a digest the strict reader of the report checked",
    ),
    (
        "export/release.py",
        "require_report_for_export",
        "short(expected)",
        "the first twelve hexadecimal digits of the digest of the export's file",
    ),
)


def scanned_paths() -> list[Path]:
    return [
        *sorted(CLI.glob("eval_suite*.py")),
        SOURCE_ROOT / "config" / "eval_suites.py",
        *sorted(CLI.glob("eval_report*.py")),
        CLI / "release.py",
        *sorted(CLI.glob("release_stage*.py")),
        SOURCE_ROOT / "export" / "release.py",
        *sorted((SOURCE_ROOT / "export").glob("model_card*.py")),
    ]


def run_paths() -> list[Path]:
    """The modules of `rinchen eval run`: their messages name the words `token` and `login`
    (the vocabulary guard is about language TOKENS), so the environment scan below is narrower for
    them."""
    return [*sorted(CLI.glob("eval_run*.py")), CLI / "eval_systems.py"]


def scanned() -> list[Finding]:
    found: list[Finding] = []
    for path in [*scanned_paths(), *run_paths()]:
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        found += interpolations(path.read_text(encoding="utf-8"), relative, names=NAMES)
    return found


def keys_of(entries: list[Finding]) -> set[tuple[str, str, str]]:
    return {(f.path, f.function, f.expression) for f in entries}


def test_every_interpolation_in_an_error_of_the_suite_commands_has_been_reviewed() -> (
    None
):
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
    suspicious = (
        "text",
        "texts",
        "source",
        "sentence",
        "line",
        "cell",
        "row",
        "value",
        "doc",
        "url",
    )
    quoting = [
        (path, function, expression)
        for path, function, expression, _ in REVIEWED
        if any(f"{name}[" in expression or expression == name for name in suspicious)
    ]

    assert quoting == []


def test_the_scan_covers_the_modules_of_the_suite_commands_and_the_configuration() -> (
    None
):
    covered = {path.relative_to(SOURCE_ROOT).as_posix() for path in scanned_paths()}

    assert {
        "cli/eval_suite.py",
        "cli/eval_suite_files.py",
        "config/eval_suites.py",
    } <= covered


def test_the_scan_covers_the_modules_of_the_report_command() -> None:
    covered = {path.relative_to(SOURCE_ROOT).as_posix() for path in scanned_paths()}

    assert {
        "cli/eval_report.py",
        "cli/eval_report_inputs.py",
        "cli/eval_report_output.py",
    } <= covered


def test_the_scan_covers_the_modules_of_the_run_command() -> None:
    covered = {path.relative_to(SOURCE_ROOT).as_posix() for path in run_paths()}

    assert {
        "cli/eval_run.py",
        "cli/eval_run_output.py",
        "cli/eval_run_plan.py",
        "cli/eval_run_inputs.py",
        "cli/eval_systems.py",
    } <= covered


@pytest.mark.parametrize("path", run_paths(), ids=lambda path: path.name)
def test_the_run_modules_never_read_the_environment_or_a_token_file(path: Path) -> None:
    # `RINCHEN_MODEL` is the option's `envvar`, which typer reads; the modules themselves never do,
    # and nothing here logs in or looks for a Hub token (no Hub is contacted by this command).
    forbidden = ("environ", "getenv", "hf_home", "login", "hf_token", "huggingface")

    names = {name.lower() for name in code_names(path)}

    assert [
        name for name in sorted(names) if any(word in name for word in forbidden)
    ] == []


# ---- the modules never look at the environment or at a token ------------------------------


def code_names(path: Path) -> set[str]:
    """Every identifier and every string of the code of a module that is not a docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            found.update(node.name.split("."))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            found.add(node.value)
    return found


@pytest.mark.parametrize("path", scanned_paths(), ids=lambda path: path.name)
def test_the_modules_never_read_the_environment_or_a_token(path: Path) -> None:
    forbidden = ("environ", "getenv", "token", "hf_home", "login", "hf_token")

    names = {name.lower() for name in code_names(path)}

    assert [
        name for name in sorted(names) if any(word in name for word in forbidden)
    ] == []


def test_the_name_scan_finds_what_it_is_told_to(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        '"""A docstring that says token."""\n'
        "import os\n"
        "from huggingface_hub import login\n"
        "value = os.environ['HF_TOKEN']\n"
        "other = os.getenv('X')\n",
        encoding="utf-8",
    )

    names = {name.lower() for name in code_names(probe)}

    assert {"environ", "getenv", "login", "hf_token"} <= names
    assert "a docstring that says token." not in names


# ---- a marker in every text of every source -----------------------------------------------


def marked_shards(workspace: SuiteWorkspace) -> None:
    """Every sentence and address of every language holds the marker."""
    for language in LANGUAGES:
        rows = shard_rows(language)
        for row in rows:
            row["text"] = f"{MARKER} {row['text']}"
            row["url"] = f"https://example.org/{MARKER}"
        write_language_shards(workspace.shards, language, rows)


def shards_with_fewer_rows(workspace: SuiteWorkspace) -> list[str]:
    marked_shards(workspace)
    rows = shard_rows("eng_Latn")[:3]
    for row in rows:
        row["text"] = f"{MARKER} {row['text']}"
    write_language_shards(workspace.shards, "eng_Latn", rows)
    return workspace.build_args()


def shards_with_a_blank_sentence(workspace: SuiteWorkspace) -> list[str]:
    marked_shards(workspace)
    rows = shard_rows("eng_Latn")
    for row in rows:
        row["text"] = f"{MARKER} {row['text']}"
    rows[2]["text"] = "  "
    write_language_shards(workspace.shards, "eng_Latn", rows)
    return workspace.build_args()


def shards_with_a_number_for_text(workspace: SuiteWorkspace) -> list[str]:
    marked_shards(workspace)
    rows = shard_rows("rus_Cyrl")
    for row in rows:
        row["text"] = 7
        row["url"] = f"https://example.org/{MARKER}"
    write_language_shards(workspace.shards, "rus_Cyrl", rows)
    return workspace.build_args()


def shards_with_a_null_id(workspace: SuiteWorkspace) -> list[str]:
    marked_shards(workspace)
    rows = shard_rows("eng_Latn")
    for row in rows:
        row["text"] = f"{MARKER} {row['text']}"
    rows[1]["id"] = None
    write_language_shards(workspace.shards, "eng_Latn", rows)
    return workspace.build_args()


def shards_with_an_id_that_is_not_one(workspace: SuiteWorkspace) -> list[str]:
    marked_shards(workspace)
    rows = shard_rows("eng_Latn", ids=("a b", "b", "c", "d"))
    for row in rows:
        row["text"] = f"{MARKER} {row['text']}"
    rows[0]["id"] = f"{MARKER} with spaces"
    write_language_shards(workspace.shards, "eng_Latn", rows)
    return workspace.build_args()


def shards_that_are_not_parquet(workspace: SuiteWorkspace) -> list[str]:
    marked_shards(workspace)
    workspace.shard("eng_Latn").write_bytes(MARKER.encode() * 10)
    return workspace.build_args()


def lines_files(workspace: SuiteWorkspace) -> Path:
    source = workspace.root / "ntrex"
    source.mkdir()
    for language in LANGUAGES:
        (source / f"{language}.txt").write_text(
            "".join(f"{MARKER} line {number}\n" for number in range(4)),
            encoding="utf-8",
        )
    return source


def lines_args(workspace: SuiteWorkspace, source: Path) -> list[str]:
    return workspace.build_args(benchmark="ntrex", source=source)


def lines_with_a_blank_line(workspace: SuiteWorkspace) -> list[str]:
    source = lines_files(workspace)
    (source / "eng_Latn.txt").write_text(
        f"{MARKER}\n\n{MARKER}\n{MARKER}\n", encoding="utf-8"
    )
    return lines_args(workspace, source)


def lines_with_windows_line_ends(workspace: SuiteWorkspace) -> list[str]:
    source = lines_files(workspace)
    (source / "eng_Latn.txt").write_bytes(f"{MARKER}\r\n{MARKER}\r\n".encode())
    return lines_args(workspace, source)


def lines_that_are_not_utf_8(workspace: SuiteWorkspace) -> list[str]:
    source = lines_files(workspace)
    (source / "eng_Latn.txt").write_bytes(MARKER.encode() + b"\xff\n")
    return lines_args(workspace, source)


def lines_of_another_length(workspace: SuiteWorkspace) -> list[str]:
    source = lines_files(workspace)
    (source / "eng_Latn.txt").write_text(f"{MARKER}\n{MARKER}\n", encoding="utf-8")
    return lines_args(workspace, source)


def lines_with_a_line_break_inside(workspace: SuiteWorkspace) -> list[str]:
    source = lines_files(workspace)
    (source / "eng_Latn.txt").write_text(
        "".join(f"{MARKER}{chr(0x2028)}x\n" for _ in range(4)), encoding="utf-8"
    )
    return lines_args(workspace, source)


def human_args(workspace: SuiteWorkspace, export: Path) -> list[str]:
    return [
        "eval",
        "suite",
        "build",
        "--benchmark",
        "human",
        "--from",
        str(export),
        "--out-dir",
        str(workspace.out_dir),
    ]


def marked_export() -> list[dict[str, Any]]:
    rows = golden_export()
    for row in rows:
        row["src"] = f"{MARKER} {row['src']}"
        row["tgt"] = f"{MARKER} {row['tgt']}"
    return rows


def human_from_the_web(workspace: SuiteWorkspace) -> list[str]:
    rows = marked_export()
    rows[1]["origin"] = f"{MARKER}-web"
    return human_args(workspace, write_export(workspace.root / "e.jsonl", rows))


def human_with_two_sets(workspace: SuiteWorkspace) -> list[str]:
    rows = marked_export()
    rows[2]["set"] = f"{MARKER}-other"
    return human_args(workspace, write_export(workspace.root / "e.jsonl", rows))


def human_with_a_bad_language(workspace: SuiteWorkspace) -> list[str]:
    rows = marked_export()
    rows[2]["src_lang"] = f"{MARKER}-lang"
    return human_args(workspace, write_export(workspace.root / "e.jsonl", rows))


def human_with_a_bad_document(workspace: SuiteWorkspace) -> list[str]:
    rows = marked_export()
    rows[0]["doc_id"] = f"{MARKER} has spaces"
    return human_args(workspace, write_export(workspace.root / "e.jsonl", rows))


def human_with_a_blank_source(workspace: SuiteWorkspace) -> list[str]:
    rows = marked_export()
    rows[3]["src"] = " "
    return human_args(workspace, write_export(workspace.root / "e.jsonl", rows))


def human_with_a_column_the_export_has_not(workspace: SuiteWorkspace) -> list[str]:
    rows = marked_export()
    rows[3][MARKER] = MARKER
    return human_args(workspace, write_export(workspace.root / "e.jsonl", rows))


def human_that_is_not_json(workspace: SuiteWorkspace) -> list[str]:
    export = workspace.root / "e.jsonl"
    export.write_text(f"{MARKER} is not json\n", encoding="utf-8")
    return human_args(workspace, export)


def human_with_a_repeated_key(workspace: SuiteWorkspace) -> list[str]:
    export = workspace.root / "e.jsonl"
    export.write_text(f'{{"src": "{MARKER}", "src": "{MARKER}"}}\n', encoding="utf-8")
    return human_args(workspace, export)


def human_json_that_is_not_an_object(workspace: SuiteWorkspace) -> list[str]:
    export = workspace.root / "e.jsonl"
    export.write_text(json.dumps([MARKER]) + "\n", encoding="utf-8")
    return human_args(workspace, export)


def human_tsv_with_a_bad_segment_number(workspace: SuiteWorkspace) -> list[str]:
    rows = marked_export()
    rows[0]["seg_idx"] = MARKER
    export = write_export(workspace.root / "e.tsv", rows)
    return human_args(workspace, export)


def human_tsv_with_another_header(workspace: SuiteWorkspace) -> list[str]:
    export = workspace.root / "e.tsv"
    export.write_text(f"{MARKER}\tsrc\n{MARKER}\t{MARKER}\n", encoding="utf-8")
    return human_args(workspace, export)


def human_tsv_with_a_short_row(workspace: SuiteWorkspace) -> list[str]:
    export = write_export(workspace.root / "e.tsv", marked_export())
    export.write_text(
        export.read_text(encoding="utf-8") + f"{MARKER}\t{MARKER}\n", encoding="utf-8"
    )
    return human_args(workspace, export)


def human_parquet_that_is_not_parquet(workspace: SuiteWorkspace) -> list[str]:
    export = workspace.root / "e.parquet"
    export.write_bytes(MARKER.encode())
    return human_args(workspace, export)


def human_with_both_ways(workspace: SuiteWorkspace) -> list[str]:
    rows = marked_export()
    rows.append(
        export_row("d9", 1, MARKER, MARKER, src_lang="khk_Cyrl", tgt_lang="eng_Latn")
    )
    return human_args(workspace, write_export(workspace.root / "e.jsonl", rows))


def human_with_a_long_sentence(workspace: SuiteWorkspace) -> list[str]:
    rows = marked_export()
    rows[0]["src"] = MARKER * 500
    return human_args(workspace, write_export(workspace.root / "e.jsonl", rows))


SCENARIOS: Final[dict[str, Callable[[SuiteWorkspace], list[str]]]] = {
    "shards-fewer-rows": shards_with_fewer_rows,
    "shards-blank-sentence": shards_with_a_blank_sentence,
    "shards-number-for-text": shards_with_a_number_for_text,
    "shards-null-id": shards_with_a_null_id,
    "shards-id-not-an-id": shards_with_an_id_that_is_not_one,
    "shards-not-parquet": shards_that_are_not_parquet,
    "lines-blank-line": lines_with_a_blank_line,
    "lines-windows-line-ends": lines_with_windows_line_ends,
    "lines-not-utf-8": lines_that_are_not_utf_8,
    "lines-other-length": lines_of_another_length,
    "lines-line-break-inside": lines_with_a_line_break_inside,
    "human-from-the-web": human_from_the_web,
    "human-two-sets": human_with_two_sets,
    "human-bad-language": human_with_a_bad_language,
    "human-bad-document": human_with_a_bad_document,
    "human-blank-source": human_with_a_blank_source,
    "human-unknown-column": human_with_a_column_the_export_has_not,
    "human-not-json": human_that_is_not_json,
    "human-repeated-key": human_with_a_repeated_key,
    "human-not-an-object": human_json_that_is_not_an_object,
    "human-tsv-bad-segment": human_tsv_with_a_bad_segment_number,
    "human-tsv-other-header": human_tsv_with_another_header,
    "human-tsv-short-row": human_tsv_with_a_short_row,
    "human-parquet-not-parquet": human_parquet_that_is_not_parquet,
    "human-both-ways": human_with_both_ways,
    "human-long-sentence": human_with_a_long_sentence,
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_no_way_for_a_build_to_fail_prints_a_text_of_its_sources(
    scenario: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    workspace = make_workspace(tmp_path)
    args = SCENARIOS[scenario](workspace)

    result = invoke(args)

    assert result.exit_code == 3, result.output
    assert result.stderr.startswith("error: "), result.stderr
    assert MARKER not in result.stdout
    assert MARKER not in result.stderr
    assert MARKER not in caplog.text
    assert not workspace.out_dir.exists()


def test_a_fetch_that_fails_upstream_prints_the_class_of_the_error_and_no_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    install_fake_load_dataset(
        monkeypatch,
        FakeLoadDataset(
            {language: shard_rows("eng_Latn") for language in LANGUAGES},
            error=ValueError(f"{MARKER} in the message of the upstream error"),
        ),
    )

    result = invoke(
        [
            "eval",
            "suite",
            "fetch",
            "--benchmark",
            "flores_plus",
            "--split",
            "dev",
            "--out-dir",
            str(tmp_path / "out"),
            "--revision",
            "a" * 40,
        ]
    )

    assert result.exit_code == 4, result.output
    assert "ValueError" in result.stderr
    assert MARKER not in result.output
    assert MARKER not in caplog.text


def test_a_marker_in_a_configuration_value_is_not_quoted_by_the_commands(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    config = tmp_path / "c.yaml"
    config.write_text(
        "benchmarks:\n"
        "  flores_plus:\n"
        "    reader: parquet_shards\n"
        f"    revision: {MARKER}\n"
        "    splits: [devtest]\n"
        "    columns: {id: id, text: text}\n"
        "    languages: [khk_Cyrl, eng_Latn]\n",
        encoding="utf-8",
    )

    result = invoke(workspace.build_args("--config", str(config)))

    assert result.exit_code == 2, result.output
    assert MARKER not in result.output
