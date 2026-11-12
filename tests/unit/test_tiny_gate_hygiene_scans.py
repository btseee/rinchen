"""The scans of the D15 gate's hygiene (`tests/support/tiny_gate_hygiene.py`), each tested on sources that
break the rule and on sources that keep it (D15 Task 9, R14). They read source with the `ast` module, so a
word in a comment or a docstring never counts and an alias cannot hide a name.

The words of a device question are written in pieces: the scans of the gate read the tests as well.
"""

from __future__ import annotations

import pytest

from tests.support.tiny_gate_hygiene import (
    content_files,
    device_value_arguments,
    guard_findings,
    needle_lines,
    process_starts,
)

_GOOD = """
import subprocess

ENVIRONMENT_GUARDS = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "UV_OFFLINE": "1",
}


def require_guards(environment):
    pass


def start(argv, environment):
    require_guards(environment)
    return subprocess.Popen(
        argv, env=environment, start_new_session=True, stdin=subprocess.DEVNULL
    )
"""


def test_a_module_that_starts_a_child_the_guarded_way_has_no_finding() -> None:
    assert guard_findings(_GOOD) == []


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        (
            _GOOD.replace("start_new_session=True", "start_new_session=False"),
            "start_new_session",
        ),
        (_GOOD.replace(", start_new_session=True", ""), "start_new_session"),
        (_GOOD.replace("env=environment, ", ""), "env="),
        (_GOOD.replace("    require_guards(environment)\n", ""), "require_guards"),
        (
            _GOOD.replace(
                "    require_guards(environment)\n    return subprocess.Popen(",
                "    child = subprocess.Popen(",
            ).replace(
                "stdin=subprocess.DEVNULL\n    )",
                "stdin=subprocess.DEVNULL\n    )\n    require_guards(environment)\n    return child",
            ),
            "require_guards",
        ),
        (_GOOD.replace('"UV_OFFLINE": "1",\n', ""), "UV_OFFLINE"),
        (_GOOD.replace('"CUDA_VISIBLE_DEVICES": "",\n', ""), "CUDA_VISIBLE_DEVICES"),
        (_GOOD.replace('"HF_HUB_OFFLINE": "1"', '"HF_HUB_OFFLINE": "0"'), "HF_HUB"),
        (
            _GOOD.replace(
                "ENVIRONMENT_GUARDS = {", "ENVIRONMENT_GUARDS = dict({"
            ).replace('"UV_OFFLINE": "1",\n}', '"UV_OFFLINE": "1",\n})'),
            "literal",
        ),
        (_GOOD + "\nsubprocess.run(['x'])\n", "run"),
        (_GOOD + "\nsubprocess.check_output(['x'])\n", "check_output"),
        (_GOOD + "\nimport os\nos.system('x')\n", "system"),
        (_GOOD + "\nimport os\nos.popen('x')\n", "popen"),
        (_GOOD + "\nimport os\nos.execv('x', ['x'])\n", "execv"),
        (_GOOD + "\nimport os\nos.spawnl(0, 'x')\n", "spawnl"),
        (
            _GOOD + "\nsubprocess.Popen(['x'], env={}, start_new_session=True)\n",
            "require_guards",
        ),
    ],
    ids=[
        "session-false",
        "no-session",
        "no-env",
        "no-guard-call",
        "guard-after-the-start",
        "no-uv-offline",
        "no-device-switch",
        "wrong-hub-value",
        "guards-not-a-literal",
        "run",
        "check-output",
        "system",
        "popen",
        "exec",
        "spawn",
        "second-start-unguarded",
    ],
)
def test_the_guard_scan_sees_each_way_a_start_can_miss_a_guard(
    source: str, fragment: str
) -> None:
    findings = guard_findings(source)

    assert findings != []
    assert any(fragment in finding for finding in findings), findings


def test_a_wait_without_a_timeout_is_a_finding() -> None:
    source = _GOOD + "\n\ndef wait(child):\n    child.communicate()\n"

    assert any("timeout" in finding for finding in guard_findings(source))


def test_a_wait_with_a_timeout_is_not() -> None:
    source = _GOOD + "\n\ndef wait(child):\n    child.communicate(timeout=3)\n"

    assert guard_findings(source) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import subprocess\nsubprocess.Popen(['x'])", ["Popen"]),
        ("import subprocess\nsubprocess.run(['x'], timeout=1)", ["run"]),
        ("import os\nos.system('x')", ["system"]),
        ("from subprocess import Popen\nPopen(['x'])", ["Popen"]),
        ("import subprocess as sp\nsp.Popen(['x'])", ["Popen"]),
        ("import os\nos.killpg(1, 9)", []),
        ("x = 1", []),
    ],
)
def test_the_start_scan_names_every_call_that_starts_a_process(
    source: str, expected: list[str]
) -> None:
    assert process_starts(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "f(device='" + "au" + "to')",
        "f(device='" + "cu" + "da')",
        'Translator(path, device="' + "AU" + 'TO")',
        "run(a, device=" + "'cu" + "da:0')",
    ],
)
def test_the_device_value_scan_sees_a_device_that_is_not_the_cpu(source: str) -> None:
    assert device_value_arguments(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "f(device='cpu')",
        "f(device=device)",
        "f(automatic=True)",
        "x = 'the " + "au" + "to" + "mation of it'",
        "def f(device: str = 'cpu'): pass",
    ],
)
def test_the_device_value_scan_leaves_the_cpu_and_other_words_alone(
    source: str,
) -> None:
    assert device_value_arguments(source) == []


def test_a_default_that_is_not_the_cpu_is_found_in_a_signature_too() -> None:
    source = "def f(device: str = '" + "au" + "to'): pass"

    assert device_value_arguments(source) != []


def test_a_text_without_a_needle_has_no_line_and_a_planted_one_is_named_by_number() -> (
    None
):
    from tests.unit.test_d13_content_rules import needles

    word = needles()[3]

    assert needle_lines("fine\nstill fine\n") == []
    assert needle_lines(f"fine\nthis holds {word} here\n") == [(2, 3)]


def test_the_files_to_scan_reach_every_part_of_d15_and_no_file_twice() -> None:
    names = {path.as_posix() for path in content_files()}

    assert len(names) == len(content_files())
    assert len(names) >= 150
    for wanted in (
        "src/rinchen/tiny/pipeline.py",
        "src/rinchen/cli/tiny.py",
        "src/rinchen/config/tiny.py",
        "configs/tiny/tiny.yaml",
        "tests/fixtures/parallel/eng_Latn-khk_Cyrl.tsv",
        "tests/fixtures/tiny/golden_report.md",
        "tests/support/tiny_gate.py",
        "tests/integration/test_tiny_gate.py",
        "tests/unit/test_tiny_gate_helpers.py",
        "scripts/build_parallel_fixtures.py",
    ):
        assert wanted in names, wanted
