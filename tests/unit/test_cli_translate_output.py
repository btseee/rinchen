"""`rinchen translate` (D11, fix wave): what the command checks before it opens an output file
and what it does when its standard output cannot be written.

- The request (the target, the source, the script, documents) is checked with the model BEFORE
  the output file is opened, and for an empty input too: a mistyped `--tgt` used to empty the
  output file before the first text told the engine about it.
- A closed pipe (`rinchen translate ... | head -1`) is exit 1 with one `error:` line; the
  buffer of standard output is pointed at the null device, so that the flush at exit cannot fail
  a second time (a second message and status 120).

The exits and the other refusals are in `test_cli_translate_exits.py`.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import typer

from rinchen.errors import ConfigError
from tests.support.engines import (
    engine_environment,  # noqa: F401 - the fixture of the tests over the real engine
    scripted_engine,
)
from tests.support.launch_cli import assert_one_error
from tests.support.translate_cli import (
    SRC,
    TGT,
    FakeEngine,
    Opened,
    RequestCheck,
    broken_stdout,
    install,
    invoke,
    probe_stdin,
    translate_args,
)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    return install(monkeypatch)


# --- the request is checked before the output file is opened ---------------------------------


def test_the_request_is_checked_with_the_model_before_the_input_is_read(
    opened: Opened, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe_stdin(monkeypatch, b"hello\n", opened.events)

    result = invoke(translate_args("--src", SRC, "--script", "Mong", tgt="khk_Cyrl"))

    assert result.exit_code == 0, result.stderr
    assert opened.engine.checked == [RequestCheck(SRC, "khk_Cyrl", "Mong", False)]
    assert opened.events == ["open", "read", "read", "translate"]


def test_the_request_checked_for_documents_says_so(opened: Opened) -> None:
    result = invoke(translate_args("--doc", "--src", SRC), stdin=b"a\n")

    assert result.exit_code == 0, result.stderr
    assert opened.engine.checked == [RequestCheck(SRC, TGT, None, True)]


def test_a_request_the_model_refuses_is_exit_2_with_the_output_file_and_input_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened = install(
        monkeypatch,
        FakeEngine(request_error=ConfigError("the model does not translate into 'x'")),
    )
    probe_stdin(monkeypatch, b"hello\n", opened.events)
    out = tmp_path / "out.txt"
    out.write_bytes(b"PRECIOUS\n")

    result = invoke(translate_args("--output", str(out)))

    assert_one_error(result, 2, "the model does not translate into 'x'")
    assert out.read_bytes() == b"PRECIOUS\n"
    assert result.stdout == ""
    assert opened.events == ["open"]
    assert opened.engine.calls == []
    assert opened.engine.closed == 1


def test_a_request_the_model_refuses_is_exit_2_whatever_the_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, FakeEngine(request_error=ConfigError("no such target")))

    result = invoke(translate_args(), stdin=b"")

    assert_one_error(result, 2, "no such target")


@pytest.mark.usefixtures("engine_environment")
class TestAnUnknownTargetOverTheRealEngine:
    """The probe of the review: `--tgt khk_Cyrll --output out.txt` over a file that holds
    something precious used to empty it before the target was looked at."""

    def run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *args: str,
        tgt: str,
        stdin: bytes = b"hello\n",
    ) -> tuple[Any, Path]:
        install(monkeypatch, scripted_engine(tmp_path).engine)
        out = tmp_path / "out.txt"
        out.write_bytes(b"PRECIOUS\n")
        result = invoke(
            translate_args(*args, "--output", str(out), tgt=tgt), stdin=stdin
        )
        return result, out

    def test_a_mistyped_target_leaves_the_output_file_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result, out = self.run(monkeypatch, tmp_path, tgt="khk_Cyrll")

        assert_one_error(result, 2, "the model does not translate into 'khk_Cyrll'")
        assert out.read_bytes() == b"PRECIOUS\n"

    def test_a_mistyped_target_is_exit_2_for_an_empty_input_as_well(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result, out = self.run(monkeypatch, tmp_path, tgt="khk_Cyrll", stdin=b"")

        assert_one_error(result, 2, "the model does not translate into 'khk_Cyrll'")
        assert out.read_bytes() == b"PRECIOUS\n"

    def test_documents_for_a_sentence_model_leave_the_output_file_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result, out = self.run(
            monkeypatch, tmp_path, "--doc", "--src", "eng_Latn", tgt=TGT
        )

        assert_one_error(result, 2, "the model does not translate documents")
        assert out.read_bytes() == b"PRECIOUS\n"

    def test_a_source_that_is_the_target_leaves_the_output_file_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result, out = self.run(monkeypatch, tmp_path, "--src", TGT, tgt=TGT)

        assert_one_error(result, 2, "the source and the target language are both")
        assert out.read_bytes() == b"PRECIOUS\n"


def test_a_closed_pipe_is_exit_1_and_the_flush_at_exit_cannot_fail_again(
    opened: Opened, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdout = broken_stdout(monkeypatch)
    try:
        result = invoke(translate_args(), stdin=b"hello\n")

        assert_one_error(result, 1, "cannot write the output to standard output")
        assert "Broken pipe" in result.stderr
        # What the interpreter does with the unwritten bytes when it exits: it used to fail
        # a second time ("Exception ignored while flushing sys.stdout", status 120).
        stdout.stream.flush()
    finally:
        stdout.stream.close()
    assert opened.engine.closed == 1


def test_a_standard_output_that_is_no_descriptor_is_left_as_it_is(
    opened: Opened, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refusing(io.BytesIO):
        def flush(self) -> None:
            raise BrokenPipeError(32, "Broken pipe")

    real = typer.get_binary_stream
    monkeypatch.setattr(
        typer,
        "get_binary_stream",
        lambda name, *a, **k: Refusing() if name == "stdout" else real(name, *a, **k),
    )

    result = invoke(translate_args(), stdin=b"hello\n")

    assert_one_error(result, 1, "cannot write the output to standard output")
