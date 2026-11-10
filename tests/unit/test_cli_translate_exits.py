"""`rinchen translate` (D11, Task 5): what it refuses and the exit code of each failure.

Exit codes: 0 done (problems in the lines do not count unless `--strict`, which is 4 and is
in `test_cli_translate.py`); 2 a bad option or a model our manifest does not vouch for; 3
input that cannot be read or is not UTF-8, or the engine's `DataError`; 4 the library failed;
1 the output cannot be written; 130 interrupted. Every failure is one `error:` line on stderr
that is the last line, and a refusal that comes before the first read writes nothing to
stdout. `open_engine`, the only place the command touches the model, is tested at the end.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest

from rinchen.cli import translate as translate_module
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.infer.engine_types import LoadOptions, RequestTextError
from tests.support.engines import (
    engine_environment,  # noqa: F401 - the fixture of the test over the real engine
    scripted_engine,
    symbol_processor,
)
from tests.support.launch_cli import assert_one_error
from tests.support.model_exports import write_model_export
from tests.support.step_dirs import flip_one_byte
from tests.support.translate_cli import (
    MODEL,
    SRC,
    TGT,
    FakeEngine,
    Opened,
    install,
    invoke,
    probe_stdin,
    translate_args,
)

BAD_UTF8 = b"\xff"
NOT_ASCII_SECRET = "private request text"


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    return install(monkeypatch)


def refused(
    opened: Opened, args: list[str], message: str, *, exit_code: int = 2
) -> None:
    """The run exits with `exit_code`, says `message`, wrote nothing to stdout and did not
    open the model."""
    result = invoke(args, stdin=b"hello\n")

    assert_one_error(result, exit_code, message)
    assert result.stdout == ""
    assert opened.calls == []


# --- exit 2: options ------------------------------------------------------------------------


def test_a_missing_model_is_exit_2(opened: Opened) -> None:
    result = invoke(
        ["translate", "--tgt", TGT],
        stdin=b"hello\n",
        env={"RINCHEN_MODEL": None},  # type: ignore[dict-item]
    )

    assert result.exit_code == 2
    assert "--model" in result.stderr
    assert result.stdout == ""
    assert opened.calls == []


def test_a_missing_target_is_exit_2(opened: Opened) -> None:
    result = invoke(translate_args(tgt=None), stdin=b"hello\n")

    assert result.exit_code == 2
    assert "--tgt" in result.stderr
    assert result.stdout == ""
    assert opened.calls == []


def test_the_model_is_read_from_the_environment(opened: Opened) -> None:
    result = invoke(
        ["translate", "--tgt", TGT],
        stdin=b"hello\n",
        env={"RINCHEN_MODEL": "from/the/environment"},
    )

    assert result.exit_code == 0, result.stderr
    assert opened.calls[0].model == Path(os.path.abspath("from/the/environment"))


def test_the_option_wins_over_the_environment(opened: Opened) -> None:
    result = invoke(translate_args(), stdin=b"hello\n", env={"RINCHEN_MODEL": "other"})

    assert result.exit_code == 0, result.stderr
    assert opened.calls[0].model == Path(os.path.abspath(MODEL))


def test_a_script_for_a_language_that_has_none_is_refused_before_the_model_opens(
    opened: Opened,
) -> None:
    refused(
        opened,
        translate_args("--script", "Mong", tgt="eng_Latn"),
        "a script applies to a Mongolian target only",
    )


def test_a_compute_type_of_auto_is_refused_before_the_model_opens(
    opened: Opened,
) -> None:
    refused(
        opened,
        translate_args("--compute-type", "auto"),
        "the compute type `auto` is refused",
    )


def test_a_device_of_auto_is_refused_before_the_model_opens(opened: Opened) -> None:
    result = invoke(translate_args("--device", "auto"), stdin=b"hello\n")

    assert result.exit_code == 2
    assert "--device" in result.stderr
    assert result.stdout == ""
    assert opened.calls == []


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--beam-size", "0", "--beam-size"),
        ("--n-best", "5", "DecodeOptions.n (5) must not exceed"),
        ("--max-decoding-length", "0", "--max-decoding-length"),
        ("--max-source-pieces", "1024", "DecodeOptions.max_source_pieces (1024)"),
        ("--batch-size", "0", "--batch-size"),
        ("--inter-threads", "0", "--inter-threads"),
        ("--intra-threads", "0", "--intra-threads"),
        ("--script-mask", "sometimes", "--script-mask"),
        ("--format", "xml", "--format"),
        ("--script", "Latn", "--script"),
    ],
)
def test_an_option_out_of_range_is_exit_2_before_the_model_opens(
    opened: Opened, option: str, value: str, message: str
) -> None:
    result = invoke(translate_args(option, value), stdin=b"hello\n")

    assert result.exit_code == 2, result.output
    assert message in result.stderr
    assert result.stdout == ""
    assert opened.calls == []


def test_documents_need_a_source_language(opened: Opened) -> None:
    refused(opened, translate_args("--doc"), "--doc needs --src")


@pytest.mark.parametrize(
    "paths",
    [["--script-converter-ct2", "conv"], ["--script-converter-spm", "conv.model"]],
    ids=["ct2 alone", "spm alone"],
)
def test_the_script_converter_needs_both_its_paths(
    opened: Opened, paths: list[str]
) -> None:
    refused(
        opened,
        translate_args(*paths),
        "--script-converter-ct2 and --script-converter-spm go together",
    )


def test_a_device_of_cuda_is_passed_on_as_the_operators_choice(opened: Opened) -> None:
    result = invoke(translate_args("--device", "cuda"), stdin=b"hello\n")

    assert result.exit_code == 0, result.stderr
    assert opened.calls[0].load == LoadOptions(device="cuda")


def test_the_output_may_not_be_the_input(opened: Opened, tmp_path: Path) -> None:
    shared = tmp_path / "text.txt"
    shared.write_text("hello\n", encoding="utf-8")

    result = invoke(translate_args("--input", str(shared), "--output", str(shared)))

    assert_one_error(result, 2, "--output is the same file as --input")
    assert shared.read_text(encoding="utf-8") == "hello\n"
    assert opened.calls == []


def test_the_output_may_not_be_a_hard_link_to_the_input(
    opened: Opened, tmp_path: Path
) -> None:
    source = tmp_path / "text.txt"
    source.write_text("hello\n", encoding="utf-8")
    other = tmp_path / "other.txt"
    os.link(source, other)

    result = invoke(translate_args("--input", str(source), "--output", str(other)))

    assert_one_error(result, 2, "--output is the same file as --input")
    assert source.read_text(encoding="utf-8") == "hello\n"


def test_the_output_may_not_be_a_link_to_the_input(
    opened: Opened, tmp_path: Path
) -> None:
    source = tmp_path / "text.txt"
    source.write_text("hello\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(source)

    result = invoke(translate_args("--input", str(source), "--output", str(link)))

    assert_one_error(result, 2, "--output is the same file as --input")
    assert source.read_text(encoding="utf-8") == "hello\n"


# --- exit 2: the model ----------------------------------------------------------------------


def test_a_model_the_seam_refuses_is_exit_2_and_the_input_is_never_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusal = ConfigError("the directory is not an export our manifest vouches for")
    opened = install(monkeypatch, error=refusal)
    probe_stdin(monkeypatch, b"hello\n", opened.events)

    result = invoke(translate_args())

    assert_one_error(result, 2, "the directory is not an export")
    assert result.stdout == ""
    assert opened.events == ["open"]


def test_a_model_that_cannot_be_read_is_exit_3(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = install(monkeypatch, error=DataError("cannot load the CTranslate2 model"))

    result = invoke(translate_args(), stdin=b"hello\n")

    assert_one_error(result, 3, "cannot load the CTranslate2 model")
    assert result.stdout == ""
    assert opened.events == ["open"]


def test_a_directory_that_does_not_exist_is_exit_2(tmp_path: Path) -> None:
    result = invoke(
        ["translate", "--model", str(tmp_path / "nothing"), "--tgt", TGT],
        stdin=b"hello\n",
    )

    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert str(tmp_path / "nothing") in result.stderr


def test_a_directory_of_the_old_export_format_is_exit_2_with_the_remedy(
    tmp_path: Path,
) -> None:
    written = write_model_export(tmp_path / "old")
    (written.directory / "rinchen.json").rename(written.directory / "export.json")

    result = invoke(
        ["translate", "--model", str(written.directory), "--tgt", TGT], stdin=b"hello\n"
    )

    assert_one_error(result, 2, "")
    assert "rinchen export convert" in result.stderr
    assert result.stdout == ""


def test_a_directory_whose_file_was_changed_is_exit_2(tmp_path: Path) -> None:
    written = write_model_export(tmp_path / "model")
    flip_one_byte(written.directory / "model.bin")

    result = invoke(
        ["translate", "--model", str(written.directory), "--tgt", TGT], stdin=b"hello\n"
    )

    assert_one_error(result, 2, "")
    assert "model.bin" in result.stderr
    assert result.stdout == ""


# --- exit 3: the input ----------------------------------------------------------------------


def test_invalid_utf_8_in_the_first_chunk_writes_nothing(opened: Opened) -> None:
    result = invoke(translate_args(), stdin=b"fine\nb" + BAD_UTF8 + b"d\nlast\n")

    assert_one_error(
        result,
        3,
        "input line 2 is not valid UTF-8: invalid byte at offset 1 of the line",
    )
    assert result.stdout == ""
    assert opened.engine.calls == []


def test_invalid_utf_8_after_a_whole_chunk_keeps_the_chunks_written(
    opened: Opened,
) -> None:
    result = invoke(
        translate_args("--batch-size", "1"), stdin=b"fine\nb" + BAD_UTF8 + b"d\n"
    )

    assert_one_error(result, 3, "input line 2 is not valid UTF-8")
    assert result.stdout == "<fine>\n"


def test_the_message_of_an_input_error_quotes_no_text(opened: Opened) -> None:
    result = invoke(
        translate_args(), stdin=NOT_ASCII_SECRET.encode() + BAD_UTF8 + b"\n"
    )

    assert result.exit_code == 3
    assert NOT_ASCII_SECRET not in result.output


def test_an_input_file_that_is_not_there_is_exit_3(
    opened: Opened, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.txt"

    result = invoke(translate_args("--input", str(missing)))

    assert_one_error(result, 3, f"cannot read the input {missing}")
    assert result.stdout == ""
    assert opened.engine.calls == []


def test_an_input_that_is_not_there_leaves_an_existing_output_alone(
    opened: Opened, tmp_path: Path
) -> None:
    out = tmp_path / "out.txt"
    out.write_text("old\n", encoding="utf-8")

    result = invoke(
        translate_args("--input", str(tmp_path / "missing.txt"), "--output", str(out))
    )

    assert result.exit_code == 3, result.output
    assert out.read_text(encoding="utf-8") == "old\n"


def test_an_input_that_is_a_directory_is_exit_3(opened: Opened, tmp_path: Path) -> None:
    result = invoke(translate_args("--input", str(tmp_path)))

    assert_one_error(result, 3, f"cannot read the input {tmp_path}")


def test_an_input_file_is_read_instead_of_standard_input(
    opened: Opened, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "in.txt"
    source.write_bytes("Сайн\nбайна\n".encode())
    probe_stdin(monkeypatch, b"never\n", opened.events)

    result = invoke(translate_args("--input", str(source)))

    assert result.exit_code == 0, result.stderr
    assert result.stdout == "<Сайн>\n<байна>\n"
    assert "read" not in opened.events


def test_an_error_of_the_engine_in_the_data_is_exit_3_and_names_the_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, FakeEngine(error=DataError("row 1 cannot be encoded")))

    result = invoke(translate_args(), stdin=b"a\nb\nc\n")

    assert_one_error(result, 3, "lines 1 to 3: row 1 cannot be encoded")
    assert result.stdout == ""


@pytest.mark.usefixtures("engine_environment")
def test_a_reserved_symbol_in_a_line_is_exit_3_naming_the_line_and_not_the_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch, scripted_engine(tmp_path, processor=symbol_processor()).engine)

    result = invoke(
        translate_args(),
        stdin=f"fine\n{NOT_ASCII_SECRET} <2eng_Latn> {NOT_ASCII_SECRET}\nfine\n".encode(),
    )

    assert_one_error(result, 3, "line 2 contains the reserved symbol <2eng_Latn>")
    assert "private" not in result.stderr
    assert result.stdout == ""


@pytest.mark.usefixtures("engine_environment")
def test_a_reserved_symbol_in_a_later_chunk_names_its_line_of_the_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch, scripted_engine(tmp_path, processor=symbol_processor()).engine)

    result = invoke(
        translate_args("--batch-size", "2"),
        stdin=b"a\nb\nc\n<2eng_Latn> d\ne\n",
    )

    # The engine counts from 0 inside its chunk (`row 1`); the input is counted from 1.
    assert_one_error(result, 3, "line 4 contains the reserved symbol <2eng_Latn>")
    assert "row" not in result.stderr
    assert result.stdout == "a\nb\n"


@pytest.mark.usefixtures("engine_environment")
def test_a_reserved_symbol_in_a_document_names_its_number_in_the_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = scripted_engine(
        tmp_path, processor=symbol_processor(), document_level=True
    ).engine
    install(monkeypatch, engine)

    result = invoke(
        translate_args("--doc", "--src", SRC, "--batch-size", "1"),
        stdin=b"fine\n\nfine\n\ngood <sep> bad\n",
    )

    assert_one_error(result, 3, "document 3 contains the reserved symbol <sep>")
    assert "document 0" not in result.stderr
    assert result.stdout.count("\n") == 3


def test_a_refusal_of_a_text_that_is_not_a_row_or_a_document_keeps_the_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(
        monkeypatch,
        FakeEngine(error=RequestTextError("window 3", "cannot be encoded")),
    )

    result = invoke(translate_args(), stdin=b"a\nb\nc\n")

    assert_one_error(result, 3, "lines 1 to 3: window 3 cannot be encoded")


# --- exit 4: the library --------------------------------------------------------------------


def test_a_failure_of_the_library_is_exit_4_and_names_the_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, FakeEngine(error=UpstreamError("CTranslate2 failed on row 0")))

    result = invoke(translate_args(), stdin=b"a\nb\n")

    assert_one_error(result, 4, "lines 1 to 2: CTranslate2 failed on row 0")
    assert result.stdout == ""


def test_a_failure_in_a_later_chunk_keeps_what_was_written_and_names_its_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(
        monkeypatch,
        FakeEngine(error=UpstreamError("CTranslate2 failed"), error_on=2),
    )

    result = invoke(translate_args("--batch-size", "2"), stdin=b"a\nb\nc\nd\ne\n")

    assert_one_error(result, 4, "lines 3 to 4: CTranslate2 failed")
    assert result.stdout == "<a>\n<b>\n"


def test_a_failure_of_one_text_names_that_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(
        monkeypatch,
        FakeEngine(error=UpstreamError("CTranslate2 failed"), error_on=2),
    )

    result = invoke(translate_args("--batch-size", "1"), stdin=b"a\nb\n")

    assert_one_error(result, 4, "line 2: CTranslate2 failed")


def test_an_engine_that_answers_the_wrong_number_of_times_is_exit_4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Short(FakeEngine):
        def translate(self, rows: Any, options: Any = None) -> Any:
            return super().translate(rows, options)[:-1]

    install(monkeypatch, Short())

    result = invoke(translate_args(), stdin=b"a\nb\n")

    assert_one_error(result, 4, "lines 1 to 2: the engine answered 1 times for 2 texts")
    assert result.stdout == ""


def test_a_failure_of_documents_names_the_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, FakeEngine(error=UpstreamError("CTranslate2 failed")))

    result = invoke(translate_args("--doc", "--src", SRC), stdin=b"a\n\nb\n")

    assert_one_error(result, 4, "documents 1 to 2: CTranslate2 failed")


def test_the_refusal_of_the_engine_in_the_request_is_exit_2_without_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, FakeEngine(error=ConfigError("the model does not translate")))

    result = invoke(translate_args(), stdin=b"a\n")

    assert_one_error(result, 2, "the model does not translate")
    assert result.stdout == ""


def test_the_engine_is_closed_after_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = install(monkeypatch, FakeEngine(error=UpstreamError("failed")))

    invoke(translate_args(), stdin=b"a\n")

    assert opened.engine.closed == 1


# --- exit 1: the output ---------------------------------------------------------------------


def test_an_output_that_cannot_be_opened_is_exit_1(
    opened: Opened, tmp_path: Path
) -> None:
    target = tmp_path / "no" / "such" / "dir" / "out.txt"

    result = invoke(translate_args("--output", str(target)), stdin=b"hello\n")

    assert_one_error(result, 1, f"cannot write the output {target}")
    assert result.stdout == ""
    assert opened.engine.calls == []
    assert opened.engine.closed == 1


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full")
def test_an_output_that_fills_up_is_exit_1(opened: Opened) -> None:
    result = invoke(translate_args("--output", "/dev/full"), stdin=b"hello\n")

    assert_one_error(result, 1, "cannot write the output /dev/full")
    assert opened.engine.closed == 1


class FlakyClose(io.BytesIO):
    """An output file that cannot be closed (its last write-back fails)."""

    def close(self) -> None:
        super().close()
        raise OSError("disk gone")


def open_flaky_output(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """`Path.open` of `target` for writing gives a `FlakyClose`; any other path is opened as
    usual."""
    real_open = Path.open

    def open_path(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == target:
            return FlakyClose()
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)


def test_an_output_that_cannot_be_closed_after_a_good_run_is_exit_1(
    opened: Opened, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.txt"
    open_flaky_output(monkeypatch, target)

    result = invoke(translate_args("--output", str(target)), stdin=b"hello\n")

    assert_one_error(result, 1, f"cannot write the output {target}: disk gone")


def test_an_output_that_cannot_be_closed_does_not_hide_the_failure_of_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch, FakeEngine(error=UpstreamError("CTranslate2 failed")))
    target = tmp_path / "out.txt"
    open_flaky_output(monkeypatch, target)

    result = invoke(translate_args("--output", str(target)), stdin=b"hello\n")

    assert_one_error(result, 4, "line 1: CTranslate2 failed")


def test_an_output_file_is_written_whole_and_stdout_stays_empty(
    opened: Opened, tmp_path: Path
) -> None:
    out = tmp_path / "out.txt"

    result = invoke(translate_args("--output", str(out)), stdin="hello\nworld\n")

    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""
    assert out.read_bytes() == b"<hello>\n<world>\n"
    assert result.stderr.startswith("translated 2 lines: ok 2")


def test_an_output_file_of_an_empty_input_is_made_empty(
    opened: Opened, tmp_path: Path
) -> None:
    out = tmp_path / "out.txt"
    out.write_text("old\n", encoding="utf-8")

    result = invoke(translate_args("--output", str(out)), stdin=b"")

    assert result.exit_code == 0, result.stderr
    assert out.read_bytes() == b""


def test_an_output_file_is_complete_before_the_exit_of_strict(
    opened: Opened, tmp_path: Path
) -> None:
    out = tmp_path / "out.txt"

    result = invoke(translate_args("--output", str(out), "--strict"), stdin="a\n\nb\n")

    assert result.exit_code == 4
    assert out.read_bytes() == b"<a>\n\n<b>\n"


# --- exit 130 -------------------------------------------------------------------------------


def test_an_interrupt_is_exit_130_and_keeps_the_chunks_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = install(monkeypatch, FakeEngine(error=KeyboardInterrupt(), error_on=2))

    result = invoke(translate_args("--batch-size", "1"), stdin=b"a\nb\n")

    assert result.exit_code == 130, result.output
    assert result.stdout == "<a>\n"
    assert opened.engine.closed == 1


# --- the help -------------------------------------------------------------------------------


def test_the_top_level_help_lists_translate() -> None:
    result = invoke(["--help"])

    assert result.exit_code == 0
    assert "translate" in result.stdout


def test_the_help_of_translate_lists_every_option() -> None:
    result = invoke(["translate", "--help"])

    assert result.exit_code == 0
    for option in (
        "--model",
        "--tgt",
        "--src",
        "--script",
        "--beam-size",
        "--n-best",
        "--max-decoding-length",
        "--max-source-pieces",
        "--batch-size",
        "--script-mask",
        "--device",
        "--compute-type",
        "--inter-threads",
        "--intra-threads",
        "--doc",
        "--format",
        "--input",
        "--output",
        "--strict",
        "--script-converter-ct2",
        "--script-converter-spm",
    ):
        assert option in result.stdout, option


# --- open_engine ----------------------------------------------------------------------------


class StubEngine:
    """Stands for `Engine`: records `open` and `close`."""

    opened: ClassVar[list[tuple[Path, LoadOptions, Any]]] = []
    closed: ClassVar[int] = 0

    @classmethod
    def open(
        cls, directory: Path, load: LoadOptions, *, script_converter: Any = None
    ) -> StubEngine:
        cls.opened.append((directory, load, script_converter))
        return cls()

    def close(self) -> None:
        StubEngine.closed += 1


class StubConverter:
    """Stands for D5's `ScriptConverter`."""

    made: ClassVar[list[tuple[Path, Path, str]]] = []
    error: ClassVar[BaseException | None] = None

    def __init__(self, ct2_dir: Path, spm_model: Path, *, device: str = "cpu") -> None:
        StubConverter.made.append((ct2_dir, spm_model, device))
        if StubConverter.error is not None:
            raise StubConverter.error


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    StubEngine.opened, StubEngine.closed = [], 0
    StubConverter.made, StubConverter.error = [], None
    monkeypatch.setattr(translate_module, "Engine", StubEngine)
    monkeypatch.setattr(translate_module, "ScriptConverter", StubConverter)


@pytest.mark.usefixtures("stubs")
def test_open_engine_opens_the_export_with_the_load_options_and_no_converter() -> None:
    load = LoadOptions(compute_type="int8", intra_threads=2)

    engine = translate_module.open_engine(Path("model"), load, None)

    assert isinstance(engine, StubEngine)
    assert StubEngine.opened == [(Path("model"), load, None)]
    assert StubConverter.made == []


@pytest.mark.usefixtures("stubs")
def test_open_engine_loads_the_script_converter_on_the_device_of_the_engine() -> None:
    load = LoadOptions(device="cuda")

    translate_module.open_engine(
        Path("model"), load, (Path("conv"), Path("conv.model"))
    )

    assert StubConverter.made == [(Path("conv"), Path("conv.model"), "cuda")]
    ((_, _, converter),) = StubEngine.opened
    assert isinstance(converter, StubConverter)


@pytest.mark.usefixtures("stubs")
def test_open_engine_stops_when_the_converter_cannot_be_loaded() -> None:
    StubConverter.error = DataError("no such model")

    with pytest.raises(DataError, match="no such model"):
        translate_module.open_engine(
            Path("model"), LoadOptions(), (Path("conv"), Path("conv.model"))
        )

    assert StubEngine.opened == []


@pytest.mark.usefixtures("stubs")
def test_open_engine_lets_the_refusal_of_the_export_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> StubEngine:
        raise ConfigError("not an export")

    monkeypatch.setattr(StubEngine, "open", refuse)

    with pytest.raises(ConfigError, match="not an export"):
        translate_module.open_engine(Path("model"), LoadOptions(), None)
