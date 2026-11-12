"""The D15 gate's hygiene (R14): what the pipeline and the gate that runs it may and may not do, read from
their source with the `ast` module and from their text. Nothing here starts a process or reads a model:
the scans run over files, so they cost seconds, and each scan is itself tested on sources that break the
rule in `tests/unit/test_tiny_gate_hygiene_scans.py` and `tests/unit/test_tiny_layering.py` (whose scans
this module runs again over the same files, as a gate and not as a unit test).

    the pipeline (`rinchen.tiny`, `rinchen.config.tiny`, the commands `cli/tiny*.py`) asks no device a
    question and names no device value (no `torch.cuda`, no device count, no vendor query program, no
    `device="auto"`); the loopback listener of `rinchen.tiny.fakes` is the only module with a socket;
    no module of `rinchen.tiny` imports the command layer or any layer above the stages; no module
    branches on the name of a profile
    the gate: ONE function starts every process, carries the guards in its call (the session, the
    environment it was given, the refusal before it, the guards as a literal of its module) and bounds
    every wait; no other file of the gate starts a process; no argument of the gate names a device other
    than the CPU
    the content: no file D15 adds (code, configuration, document, fixture, test) holds a word the
    project's content rules exclude; a failure names the file, the line and the number of the needle and
    never the word
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from rinchen.config.loader import REPO_ROOT
from tests.support.maintain_gate_hygiene import device_questions
from tests.support.tiny_gate_hygiene import (
    content_files,
    device_value_arguments,
    guard_findings,
    needle_lines,
    process_starts,
    untimed_waits,
)
from tests.unit.test_tiny_layering import (
    COMMAND_MODULES,
    MODULES,
    SERVER_ALLOWED,
    SERVER_MODULE,
    device_references,
    layer_violations,
    network_references,
    profile_branches,
)

SUPPORT: Final = REPO_ROOT / "tests" / "support"
INTEGRATION: Final = REPO_ROOT / "tests" / "integration"
STARTER_MODULE: Final = SUPPORT / "tiny_gate.py"
# The files of the gate: the helpers and the modules that use them.
GATE_FILES: Final = [
    *sorted(SUPPORT.glob("tiny_gate*.py")),
    *sorted(INTEGRATION.glob("test_tiny_gate*.py")),
]
PIPELINE: Final = [*MODULES, *COMMAND_MODULES]
MINIMUM_PIPELINE_FILES: Final = 30


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def test_the_scans_reach_the_whole_pipeline_and_the_whole_gate() -> None:
    names = {_relative(path) for path in [*PIPELINE, *GATE_FILES]}

    assert len(set(PIPELINE)) >= MINIMUM_PIPELINE_FILES
    for wanted in (
        "src/rinchen/tiny/pipeline.py",
        "src/rinchen/tiny/fakes.py",
        "src/rinchen/config/tiny.py",
        "src/rinchen/cli/tiny.py",
        "src/rinchen/cli/tiny_output.py",
        "tests/support/tiny_gate.py",
        "tests/support/tiny_gate_processes.py",
        "tests/support/tiny_gate_hygiene.py",
        "tests/integration/test_tiny_gate.py",
        "tests/integration/test_tiny_gate_determinism.py",
        "tests/integration/test_tiny_gate_controls.py",
        "tests/integration/test_tiny_gate_hygiene.py",
    ):
        assert wanted in names, wanted


# ---- the pipeline ------------------------------------------------------------------------------------


def test_no_module_of_the_pipeline_asks_a_device_question_or_names_a_device() -> None:
    found = {
        _relative(path): [
            *device_references(_read(path)),
            *device_questions(_read(path)),
            *device_value_arguments(_read(path)),
        ]
        for path in PIPELINE
    }

    assert {name: finds for name, finds in found.items() if finds} == {}


def test_the_loopback_listener_is_the_only_socket_of_the_pipeline() -> None:
    found = {
        _relative(path): network_references(
            _read(path), SERVER_ALLOWED if path == SERVER_MODULE else ()
        )
        for path in PIPELINE
    }

    assert {name: finds for name, finds in found.items() if finds} == {}
    assert network_references(_read(SERVER_MODULE)) != []


def test_no_module_of_the_library_imports_the_command_layer_or_a_layer_above() -> None:
    found = {_relative(path): layer_violations(_read(path)) for path in MODULES}

    assert {name: finds for name, finds in found.items() if finds} == {}


def test_no_module_of_the_pipeline_branches_on_the_name_of_a_profile() -> None:
    found = {_relative(path): profile_branches(_read(path)) for path in PIPELINE}

    assert {name: finds for name, finds in found.items() if finds} == {}


# ---- the gate ------------------------------------------------------------------------------------------


def test_the_one_function_that_starts_a_process_carries_the_guards_in_its_call() -> (
    None
):
    source = _read(STARTER_MODULE)

    assert guard_findings(source) == []
    assert process_starts(source) == ["Popen"]


def test_no_other_file_of_the_gate_starts_a_process_or_waits_without_a_bound() -> None:
    others = [path for path in GATE_FILES if path != STARTER_MODULE]

    assert others
    assert {
        _relative(path): process_starts(_read(path))
        for path in others
        if process_starts(_read(path))
    } == {}
    assert {
        _relative(path): untimed_waits(_read(path))
        for path in others
        if untimed_waits(_read(path))
    } == {}


def test_no_file_of_the_gate_names_a_device_other_than_the_cpu() -> None:
    found = {
        _relative(path): [
            *device_value_arguments(_read(path)),
            *device_questions(_read(path)),
        ]
        for path in GATE_FILES
    }

    assert {name: finds for name, finds in found.items() if finds} == {}


# ---- the content -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", content_files(), ids=lambda path: path.as_posix())
def test_no_file_d15_adds_holds_a_word_the_content_rules_exclude(path: Path) -> None:
    assert needle_lines(_read(REPO_ROOT / path)) == []
