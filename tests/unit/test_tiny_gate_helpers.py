"""The helpers of the D15 gate (`tests/support/tiny_gate.py`), tested WITHOUT starting anything real
(D15 Task 9): no `rinchen`, no `uv`, no `eole`, no shell. A recorder takes the place of
`subprocess.Popen` inside the helper's module, so a refusal is shown by the recorder never being
called, and a start by what the recorder was given.

What is pinned: the floors are literals; the body of the `tiny` recipe is read from the justfile; the
environment of a child holds the four switches (the empty device list, the Hub and the datasets
offline, uv offline) and never a token, and a start that would hand a child anything else is refused
before a process exists; a child starts in a session of its own with its input closed; a wait is bounded
and takes the child's whole group and nothing else with it; the scans of the documents find what they
claim to find.

The words of a device question are written in pieces: the scans of the gate read the tests as well.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Final

import pytest

from rinchen.config.loader import REPO_ROOT
from tests.support import tiny_gate
from tests.support.fake_uv import read_calls
from tests.support.tiny_gate import (
    ENVIRONMENT_GUARDS,
    MARKER_VARIABLE,
    MAX_INT8_CHRF_DROP,
    MAX_PARITY_DELTA,
    MAX_WALL_TIME_SECONDS,
    MIN_PARITY_EXACT,
    MIN_PLACEHOLDER_ACC,
    MIN_SCRIPT_PURITY,
    MIN_TRAIN_CHRF,
    absolute_paths,
    duration_keys,
    finish,
    fixture_text_hits,
    fixture_texts,
    guarded_environment,
    leaked_values,
    make_shim,
    part_states,
    read_manifest,
    read_report,
    recipe_command,
    require_guards,
    run_recipe,
    run_rinchen,
    run_tiny,
    secret_like_strings,
    send_to_group,
    stage_of,
    start_guarded,
    start_tiny,
    threshold_of,
    timestamps,
    wait_for_file,
)

PIDS: Final = 424242
FIXTURES: Final = REPO_ROOT / "tests" / "fixtures" / "parallel"
JUSTFILE_TEXT: Final = (
    "gate:\n    uv run ruff format --check .\n\n"
    "tiny:\n    uv run --frozen --no-sync rinchen tiny run --profile tiny\n\n"
    "accept:\n    echo nothing\n"
)


class RecordingPopen:
    """Stands in for `subprocess.Popen`: remembers how it was called and plays a child that already
    ended with the exit code in `exit_code` (or one that never ends, for the timeout tests)."""

    calls: ClassVar[list[dict[str, Any]]] = []
    killed_groups: ClassVar[list[tuple[int, int]]] = []
    marked_killed: ClassVar[list[str]] = []
    exit_code: ClassVar[int] = 0
    hangs: ClassVar[bool] = False

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        type(self).calls.append({"argv": argv, **kwargs})
        self.pid = PIDS
        self.returncode: int | None = None if type(self).hangs else type(self).exit_code
        self.waited: list[Any] = []

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        self.waited.append(timeout)
        if type(self).hangs and self.returncode is None:
            raise subprocess.TimeoutExpired("recorded", timeout or 0)
        return b"first\nsecond\n", b"a warning\n"

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = -9 if self.returncode is None else self.returncode
        return self.returncode


@pytest.fixture(autouse=True)
def nothing_real_starts(monkeypatch: pytest.MonkeyPatch) -> type[RecordingPopen]:
    """Every start of a process in the helper's module goes to the recorder, and the kill of a group
    and of the marked processes go to lists: the tests below cannot reach a real process."""
    RecordingPopen.calls = []
    RecordingPopen.killed_groups = []
    RecordingPopen.marked_killed = []
    RecordingPopen.exit_code = 0
    RecordingPopen.hangs = False
    monkeypatch.setattr(tiny_gate.subprocess, "Popen", RecordingPopen)
    monkeypatch.setattr(
        tiny_gate.os,
        "killpg",
        lambda pid, number: RecordingPopen.killed_groups.append((pid, number)),
    )
    monkeypatch.setattr(
        tiny_gate.processes,
        "kill_the_run",
        lambda marker: RecordingPopen.marked_killed.append(marker),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return RecordingPopen


def _start(tmp_path: Path, **options: Any) -> dict[str, Any]:
    start_guarded(["a-program", "an-argument"], cwd=tmp_path, **options)
    assert len(RecordingPopen.calls) == 1
    return RecordingPopen.calls[0]


# ---- the floors ----------------------------------------------------------------------------------


def test_the_floors_are_the_literals_of_the_master_plan() -> None:
    assert MIN_TRAIN_CHRF == 50.0
    assert MIN_PARITY_EXACT == 0.95
    assert MAX_PARITY_DELTA == 1e-3
    assert MAX_INT8_CHRF_DROP == 1.0
    assert MIN_SCRIPT_PURITY == 1.0
    assert MIN_PLACEHOLDER_ACC == 1.0
    assert MAX_WALL_TIME_SECONDS == 1200


# ---- the recipe ----------------------------------------------------------------------------------


def test_the_body_of_the_recipe_is_read_from_the_shipped_justfile() -> None:
    assert (
        recipe_command(REPO_ROOT / "justfile")
        == "uv run --frozen --no-sync rinchen tiny run --profile tiny"
    )


def test_the_body_is_the_indented_line_under_the_recipe_name(tmp_path: Path) -> None:
    justfile = tmp_path / "justfile"
    justfile.write_text(JUSTFILE_TEXT, encoding="utf-8")

    assert recipe_command(justfile) == (
        "uv run --frozen --no-sync rinchen tiny run --profile tiny"
    )


@pytest.mark.parametrize(
    "text",
    [
        "gate:\n    echo one\n",
        "tiny:\n",
        "tiny:\n\naccept:\n    echo one\n",
        "tiny:\n    echo one\n    echo two\n",
        "tiny arg:\n    echo one\n",
        "tiny: gate\n    echo one\n",
    ],
    ids=[
        "no-recipe",
        "empty-body",
        "empty-body-then-another",
        "two-lines",
        "a-parameter",
        "a-dependency",
    ],
)
def test_a_recipe_that_is_not_one_command_line_is_refused(
    tmp_path: Path, text: str
) -> None:
    justfile = tmp_path / "justfile"
    justfile.write_text(text, encoding="utf-8")

    with pytest.raises(AssertionError, match="tiny"):
        recipe_command(justfile)


# ---- the environment of a child --------------------------------------------------------------------


def test_the_environment_holds_the_four_switches_and_the_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = guarded_environment()

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HF_HUB_OFFLINE"] == environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["UV_OFFLINE"] == "1"
    for name, value in ENVIRONMENT_GUARDS.items():
        assert environment[name] == value
    assert len(environment[MARKER_VARIABLE]) >= 16


def test_every_environment_has_a_marker_of_its_own() -> None:
    assert (
        guarded_environment()[MARKER_VARIABLE] != guarded_environment()[MARKER_VARIABLE]
    )


def test_the_environment_never_holds_a_token_or_the_variables_that_steer_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "HF_TOKEN",
        "GITHUB_TOKEN",
        "SOME_SECRET",
        "A_PASSWORD",
        "AN_API_KEY",
        "PYTEST_CURRENT_TEST",
        "COV_CORE_SOURCE",
    ):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("KEPT_VARIABLE", "kept")

    environment = guarded_environment()

    assert [
        name
        for name in environment
        if name.endswith(("_TOKEN", "_SECRET", "_PASSWORD", "_API_KEY"))
    ] == []
    assert not [name for name in environment if name.startswith(("PYTEST_", "COV_"))]
    assert environment["KEPT_VARIABLE"] == "kept"


def test_the_shim_directory_comes_first_on_the_path(tmp_path: Path) -> None:
    environment = guarded_environment(shim_dir=tmp_path / "bin")

    assert environment["PATH"].split(os.pathsep)[0] == str(tmp_path / "bin")
    assert environment["PATH"].endswith(os.environ["PATH"])


def test_the_extra_variables_are_added_last(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = guarded_environment(
        env_extra={"PYTHONHASHSEED": "3", "LC_ALL": "C", "TZ": "Asia/Ulaanbaatar"}
    )

    assert environment["PYTHONHASHSEED"] == "3"
    assert environment["LC_ALL"] == "C"
    assert environment["TZ"] == "Asia/Ulaanbaatar"


@pytest.mark.parametrize(
    "name",
    [
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "UV_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_TOKEN",
        "ANY_SECRET",
        "ANY_PASSWORD",
        "ANY_API_KEY",
        MARKER_VARIABLE,
    ],
)
def test_the_extra_variables_cannot_change_a_guard_or_carry_a_token(name: str) -> None:
    with pytest.raises(AssertionError, match=name):
        guarded_environment(env_extra={name: "x"})


# ---- the refusal to start ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "UV_OFFLINE"],
)
def test_a_start_is_refused_unless_every_switch_is_in_the_environment(
    name: str,
) -> None:
    environment = guarded_environment()
    del environment[name]

    with pytest.raises(AssertionError, match=name):
        require_guards(environment)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("CUDA_VISIBLE_DEVICES", "-1"),
        ("HF_HUB_OFFLINE", "0"),
        ("HF_DATASETS_OFFLINE", ""),
        ("UV_OFFLINE", "false"),
    ],
)
def test_a_start_is_refused_when_a_switch_has_another_value(
    name: str, value: str
) -> None:
    environment = guarded_environment()
    environment[name] = value

    with pytest.raises(AssertionError, match=name):
        require_guards(environment)


@pytest.mark.parametrize(
    "name", ["HF_TOKEN", "GITHUB_TOKEN", "X_SECRET", "X_PASSWORD", "X_API_KEY"]
)
def test_a_start_is_refused_when_a_token_would_reach_the_child(name: str) -> None:
    environment = guarded_environment()
    environment[name] = "value"

    with pytest.raises(AssertionError, match=name):
        require_guards(environment)


def test_a_start_is_refused_when_the_test_process_itself_sees_a_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
        start_guarded(["a-program"], cwd=tmp_path)

    assert RecordingPopen.calls == []


def test_a_start_is_refused_when_the_test_process_has_no_device_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")

    with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
        start_guarded(["a-program"], cwd=tmp_path)

    assert RecordingPopen.calls == []


@pytest.mark.parametrize(
    "name", ["CUDA_VISIBLE_DEVICES", "UV_OFFLINE", "HF_TOKEN", "X_API_KEY"]
)
def test_a_start_with_a_guard_overridden_or_a_token_given_never_reaches_a_process(
    name: str, tmp_path: Path
) -> None:
    with pytest.raises(AssertionError, match=name):
        start_guarded(["a-program"], cwd=tmp_path, env_extra={name: "1"})

    assert RecordingPopen.calls == []


def test_the_shim_is_written_for_the_repository_with_an_empty_call_log(
    tmp_path: Path,
) -> None:
    directory = make_shim(tmp_path / "bin")

    assert directory == tmp_path / "bin"
    assert os.access(directory / "uv", os.X_OK)
    assert f'PROJECT = "{REPO_ROOT}"' in (directory / "uv").read_text(encoding="utf-8")
    assert read_calls(directory) == []


# ---- the start ------------------------------------------------------------------------------------


def test_a_child_starts_in_a_session_of_its_own_with_no_input_and_the_guards(
    tmp_path: Path,
) -> None:
    call = _start(tmp_path, env_extra={"TZ": "UTC"})

    assert call["argv"] == ["a-program", "an-argument"]
    assert call["start_new_session"] is True
    assert call["stdin"] == subprocess.DEVNULL
    assert call["stdout"] == subprocess.PIPE
    assert call["stderr"] == subprocess.PIPE
    assert call["cwd"] == tmp_path
    assert call["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert call["env"]["UV_OFFLINE"] == "1"
    assert call["env"]["TZ"] == "UTC"


def test_an_argument_that_is_a_path_is_handed_over_as_text(tmp_path: Path) -> None:
    start_guarded(["a-program", tmp_path / "file"], cwd=tmp_path)

    assert RecordingPopen.calls[0]["argv"] == ["a-program", str(tmp_path / "file")]


def test_the_started_child_knows_its_marker_and_its_process(tmp_path: Path) -> None:
    started = start_guarded(["a-program"], cwd=tmp_path)

    assert started.marker == RecordingPopen.calls[0]["env"][MARKER_VARIABLE]
    assert started.process.pid == PIDS


def test_the_recipe_runs_through_sh_with_the_shim_first_on_the_path(
    tmp_path: Path,
) -> None:
    shim = tmp_path / "bin"

    invocation = run_recipe(
        "the recipe body", cwd=tmp_path, env_extra=None, shim_dir=shim
    )

    call = RecordingPopen.calls[0]
    assert call["argv"] == ["sh", "-c", "the recipe body"]
    assert call["env"]["PATH"].split(os.pathsep)[0] == str(shim)
    assert call["cwd"] == tmp_path
    assert invocation.returncode == 0
    assert invocation.stdout == ["first", "second"]
    assert invocation.stderr == "a warning\n"


def test_the_command_runs_with_the_interpreter_of_the_tests(tmp_path: Path) -> None:
    run_tiny(["--profile", "tiny", "--dry-run"], cwd=tmp_path)

    assert RecordingPopen.calls[0]["argv"] == [
        sys.executable,
        "-m",
        "rinchen",
        "tiny",
        "run",
        "--profile",
        "tiny",
        "--dry-run",
    ]


def test_any_other_command_runs_the_same_way(tmp_path: Path) -> None:
    run_rinchen(["eval", "robustness", "--help"], cwd=tmp_path)

    assert RecordingPopen.calls[0]["argv"] == [
        sys.executable,
        "-m",
        "rinchen",
        "eval",
        "robustness",
        "--help",
    ]
    assert RecordingPopen.calls[0]["start_new_session"] is True


def test_a_started_command_hands_back_the_process_to_wait_for(tmp_path: Path) -> None:
    started = start_tiny(["--out-dir", "out"], cwd=tmp_path)

    assert started.process.pid == PIDS
    assert RecordingPopen.calls[0]["argv"][3:5] == ["tiny", "run"]


def test_the_exit_code_of_the_child_is_the_one_handed_back(tmp_path: Path) -> None:
    RecordingPopen.exit_code = 4

    assert run_tiny([], cwd=tmp_path).returncode == 4


def test_a_command_that_does_not_end_in_time_is_killed_through_its_group_only(
    tmp_path: Path,
) -> None:
    RecordingPopen.hangs = True

    with pytest.raises(subprocess.TimeoutExpired):
        run_tiny([], cwd=tmp_path, timeout=7)

    assert RecordingPopen.killed_groups == [(PIDS, signal.SIGKILL)]
    marker = RecordingPopen.calls[0]["env"][MARKER_VARIABLE]
    assert RecordingPopen.marked_killed == [marker]


def test_the_wait_is_bounded_by_the_timeout_it_is_given(tmp_path: Path) -> None:
    started = start_guarded(["a-program"], cwd=tmp_path)

    finish(started, timeout=11)

    assert isinstance(started.process, RecordingPopen)
    assert started.process.waited == [11]


def test_a_child_that_ended_is_not_signalled(tmp_path: Path) -> None:
    run_tiny([], cwd=tmp_path)

    assert RecordingPopen.killed_groups == []
    assert RecordingPopen.marked_killed == []


def test_a_wait_for_a_file_returns_when_the_file_exists(tmp_path: Path) -> None:
    RecordingPopen.hangs = True
    started = start_guarded(["a-program"], cwd=tmp_path)
    (tmp_path / "log").write_text("x", encoding="utf-8")

    wait_for_file(tmp_path / "log", started, seconds=5)


def test_a_wait_for_a_file_fails_when_the_command_ended_first(tmp_path: Path) -> None:
    started = start_guarded(["a-program"], cwd=tmp_path)

    with pytest.raises(AssertionError, match="ended"):
        wait_for_file(tmp_path / "log", started, seconds=5)


def test_a_wait_for_a_file_fails_when_the_time_is_up(tmp_path: Path) -> None:
    RecordingPopen.hangs = True
    started = start_guarded(["a-program"], cwd=tmp_path)

    with pytest.raises(AssertionError, match="in time"):
        wait_for_file(tmp_path / "log", started, seconds=0.3)


def test_a_signal_goes_to_the_group_of_the_child_and_to_no_other(
    tmp_path: Path,
) -> None:
    started = start_guarded(["a-program"], cwd=tmp_path)

    send_to_group(started, signal.SIGTERM)

    assert RecordingPopen.killed_groups == [(PIDS, signal.SIGTERM)]


# ---- reading what a run wrote -------------------------------------------------------------------------


def test_the_report_and_the_manifest_are_read_as_documents(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text('{"verdict": "pass"}\n', encoding="utf-8")
    (tmp_path / "manifest.json").write_text('{"stage": "tiny_run"}\n', encoding="utf-8")

    assert read_report(tmp_path) == {"verdict": "pass"}
    assert read_manifest(tmp_path) == {"stage": "tiny_run"}


def test_a_report_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text("[1, 2]\n", encoding="utf-8")

    with pytest.raises(AssertionError):
        read_report(tmp_path)


_REPORT: Final = {
    "stages": [
        {"id": "fetch", "state": "ok", "parts": []},
        {
            "id": "filter",
            "state": "ok",
            "parts": [
                {"id": "lid", "state": "skipped"},
                {"id": "numbers", "state": "ok"},
            ],
        },
    ],
    "thresholds": [{"id": "stages", "verdict": "pass"}],
}


def test_a_stage_a_threshold_and_the_parts_of_a_stage_are_found_by_their_ids() -> None:
    assert stage_of(_REPORT, "fetch")["state"] == "ok"
    assert threshold_of(_REPORT, "stages")["verdict"] == "pass"
    assert part_states(stage_of(_REPORT, "filter")) == {
        "lid": "skipped",
        "numbers": "ok",
    }
    assert part_states(stage_of(_REPORT, "fetch")) == {}


def test_an_id_that_is_not_listed_is_refused() -> None:
    with pytest.raises(ValueError):
        stage_of(_REPORT, "serve")
    with pytest.raises(ValueError):
        threshold_of(_REPORT, "wall_time")


# ---- the scans -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "see /home/someone/work",
        "at /tmp/x/y.json now",
        "(/usr/bin/env)",
        "C:\\Users\\someone",
    ],
)
def test_the_path_scan_sees_an_absolute_path(text: str) -> None:
    assert absolute_paths(text) != []


@pytest.mark.parametrize("text", ["1/2 and n/a", "a/b/c relative", "report.json"])
def test_the_path_scan_leaves_relative_paths_and_fractions_alone(text: str) -> None:
    assert absolute_paths(text) == []


@pytest.mark.parametrize("text", ["on 2026-11-11 it", "at 06:20:41 it"])
def test_the_timestamp_scan_sees_a_date_and_a_time_of_day(text: str) -> None:
    assert timestamps(text) != []


@pytest.mark.parametrize("text", ["version 3.14.10", "0.9923 and 1.4e-06", "1,250.50"])
def test_the_timestamp_scan_leaves_versions_and_figures_alone(text: str) -> None:
    assert timestamps(text) == []


def test_a_duration_key_is_found_at_any_depth_and_named_by_its_path() -> None:
    document = {
        "a": {"b": [{"elapsed_seconds": 1}, {"fine": 2}]},
        "wall_time": 3,
        "created": "x",
    }

    assert duration_keys(document) == [
        "a.b[0].elapsed_seconds",
        "created",
        "wall_time",
    ]


@pytest.mark.parametrize(
    "key",
    [
        "seconds",
        "second",
        "secs",
        "duration",
        "elapsed",
        "timestamp",
        "time",
        "date",
        "created",
        "started",
        "finished",
        "ms",
        "millis",
        "Elapsed",
        "run-time",
        "total elapsed",
    ],
)
def test_each_word_of_a_duration_or_a_date_makes_a_key_one(key: str) -> None:
    assert duration_keys({key: 1}) == [key]


@pytest.mark.parametrize(
    "key", ["timeout", "status_codes", "translate_200_total", "sentences", "id"]
)
def test_a_key_that_only_looks_like_a_duration_is_not_one(key: str) -> None:
    assert duration_keys({key: 1}) == []


def test_the_keys_of_a_report_that_holds_no_duration_are_clean() -> None:
    document = {
        "schema": "rinchen.tiny/1",
        "blocks": {"serve": {"status_codes": [200], "translate_200_total": 1.0}},
        "thresholds": [{"id": "wall_time", "value": None}],
    }

    assert duration_keys(document) == []


def test_a_value_of_the_environment_in_a_text_is_named_and_never_quoted() -> None:
    values = {"HOME": "/home/someone", "USER": "someone", "TERM": "xy"}

    assert leaked_values("a path /home/someone/work", values) == ["HOME", "USER"]
    assert leaked_values("nothing of it", values) == []
    # A value of fewer than four characters says nothing.
    assert leaked_values("xy xy xy", values) == []


@pytest.mark.parametrize(
    "text",
    [
        "token hf_" + "a1b2c3d4e5f6g7h8i9j0k1l2m3",
        "key sk-" + "a1b2c3d4e5f6g7h8i9j0k1l2m3",
        "ghp_" + "a1b2c3d4e5f6g7h8i9j0k1l2m3n4",
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "-----BEGIN " + "PRIVATE KEY-----",
        "HF_" + "TOKEN=abc",
        "MY_API_" + "KEY: abc",
    ],
)
def test_the_secret_scan_sees_a_credential_by_its_shape(text: str) -> None:
    assert secret_like_strings(text) != []


@pytest.mark.parametrize(
    "text",
    [
        "the token-ratio check needs a SentencePiece model",
        '"token_ratio_check": "x"',
        "the password policy is not here",
        "a sk-ill is a skill",
    ],
)
def test_the_secret_scan_leaves_ordinary_words_alone(text: str) -> None:
    assert secret_like_strings(text) == []


def test_the_fixture_texts_are_the_sentences_and_the_non_ascii_words_of_the_files() -> (
    None
):
    texts = fixture_texts(FIXTURES)

    assert len(texts) > 500
    assert all(len(text) >= 4 for text in texts)
    assert all(" " in text and len(text) >= 10 or not text.isascii() for text in texts)
    # No id, split, kind or reason column is a text.
    assert "base" not in texts
    assert "train" not in texts


def test_a_text_of_a_fixture_in_a_document_is_counted_and_never_quoted(
    tmp_path: Path,
) -> None:
    texts = fixture_texts(FIXTURES)
    sentence = next(text for text in texts if " " in text)

    assert fixture_text_hits(f"before {sentence} after", texts) == 1
    assert fixture_text_hits("a document of figures only", texts) == 0


def test_the_fixture_reading_is_of_text_columns_only(tmp_path: Path) -> None:
    # The id, split, kind and reason columns are labels: even a long one that holds a space, or a
    # letter that is not ASCII, is not a text.
    (tmp_path / "pairs.tsv").write_text(
        "id\tsplit\tkind\texpect\tsrc_lang\ttgt_lang\tsrc\ttgt\n"
        "an id with spaces\ttrain\tbase\ta long reason with spaces\t"
        + chr(0x43D)
        * 6
        + "\tother language\tthis is a sentence\tshort\n"
        "x-0002\tdev\t"
        + chr(0x43D) * 6
        + "\tint\tabc\tdef\tanother whole sentence\t"
        + chr(0x43D) * 5
        + "\n",
        encoding="utf-8",
    )

    assert fixture_texts(tmp_path) == (
        "another whole sentence",
        "this is a sentence",
        chr(0x43D) * 5,
    )


def test_the_environment_guards_are_the_four_switches_and_the_hidden_device() -> None:
    assert ENVIRONMENT_GUARDS["CUDA_VISIBLE_DEVICES"] == ""
    assert {
        name: ENVIRONMENT_GUARDS[name]
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "UV_OFFLINE")
    } == {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "UV_OFFLINE": "1"}


def test_a_mapping_of_values_is_all_the_leak_scan_takes() -> None:
    values: Mapping[str, str] = {"cwd": "/work/dir"}

    assert leaked_values("in /work/dir now", values) == ["cwd"]
