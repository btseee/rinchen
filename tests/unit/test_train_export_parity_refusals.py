"""`rinchen.train.export_parity`: what a strict parity run refuses, and how it fails when Eole
or CTranslate2 does not do what it must. A run that is refused before Eole starts leaves no
file and starts nothing; the code of the module is scanned for what must never be in it (a
GPU, an automatic device or compute type, an import of a layer above it).

No test starts Eole or loads a model (see `test_train_export_parity.py`); the default loader of
the library is tested against a stand-in for the `ctranslate2` module.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import sentencepiece

import rinchen.train.export_parity as module
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.export.record import SPM_FILE
from rinchen.train.export_parity import ParitySource, strict_parity
from tests.support.fake_ct2 import SPIECE
from tests.support.fake_parity_eole import (
    TAG,
    WEIGHTS_SHA256,
    ParityCase,
    ParityCt2,
    make_case,
)
from tests.support.step_dirs import SPM_BYTES

SOURCE = Path(module.__file__)


def run(case: ParityCase, **options: Any) -> object:
    arguments: dict[str, Any] = {
        "out_dir": case.out_dir,
        "engine": case.ct2,
        "processor": case.processor,
    }
    return strict_parity(
        case.checkpoint, case.export, case.sources, **(arguments | options)
    )


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParityCase:
    return make_case(tmp_path, monkeypatch)


def assert_nothing_started(case: ParityCase) -> None:
    assert case.eole.launches == []
    assert case.ct2.calls == []
    assert not case.out_dir.exists()


# --- what is refused before anything starts -------------------------------------------------------


def test_no_source_is_a_value_error(case: ParityCase) -> None:
    with pytest.raises(ValueError, match="no source"):
        strict_parity(
            case.checkpoint,
            case.export,
            [],
            out_dir=case.out_dir,
            engine=case.ct2,
            processor=case.processor,
        )

    assert_nothing_started(case)


@pytest.mark.parametrize("threads", [0, -1])
def test_fewer_than_one_thread_is_a_value_error(case: ParityCase, threads: int) -> None:
    with pytest.raises(ValueError, match="threads"):
        run(case, threads=threads)

    assert_nothing_started(case)
    assert case.processor.encoded == []


def test_a_target_the_model_has_no_tag_for_is_refused_naming_it(
    case: ParityCase,
) -> None:
    sources = [*case.sources, ParitySource("w9", "mn_Mong")]

    with pytest.raises(ConfigError, match="mn_Mong"):
        strict_parity(
            case.checkpoint,
            case.export,
            sources,
            out_dir=case.out_dir,
            engine=case.ct2,
            processor=case.processor,
        )

    assert_nothing_started(case)


def test_a_text_without_a_piece_is_refused_naming_its_number_and_not_its_text(
    case: ParityCase,
) -> None:
    sources = [case.sources[0], ParitySource("   ", "khk_Cyrl"), case.sources[1]]

    with pytest.raises(ValueError, match="source 2") as raised:
        strict_parity(
            case.checkpoint,
            case.export,
            sources,
            out_dir=case.out_dir,
            engine=case.ct2,
            processor=case.processor,
        )

    assert "   " not in str(raised.value)
    assert_nothing_started(case)


# --- the checkpoint must be the export's --------------------------------------------------------


def test_a_checkpoint_of_other_weights_than_the_export_s_is_refused_before_eole(
    case: ParityCase,
) -> None:
    (case.checkpoint / "model.00.safetensors").write_bytes(b"other weights")

    with pytest.raises(ConfigError, match="weights") as raised:
        run(case)

    assert WEIGHTS_SHA256[:12] in str(raised.value)
    assert_nothing_started(case)


def test_a_checkpoint_of_the_export_s_weights_is_accepted(case: ParityCase) -> None:
    run(case)

    assert len(case.eole.launches) == 1


def test_the_digest_a_caller_took_is_used_and_the_weights_are_not_hashed_again(
    case: ParityCase,
) -> None:
    # The file is the export's, the digest the caller hands over is not: the digest decides.
    with pytest.raises(ConfigError, match="weights"):
        run(case, weights_sha256="a" * 64)

    assert_nothing_started(case)


def test_a_digest_of_the_export_s_weights_needs_no_weights_file(
    case: ParityCase,
) -> None:
    (case.checkpoint / "model.00.safetensors").unlink()

    run(case, weights_sha256=WEIGHTS_SHA256)

    assert len(case.eole.launches) == 1


def test_a_checkpoint_without_weights_is_refused_when_no_digest_was_given(
    case: ParityCase,
) -> None:
    (case.checkpoint / "model.00.safetensors").unlink()

    with pytest.raises(ConfigError, match="not found"):
        run(case)

    assert_nothing_started(case)


def test_a_checkpoint_path_with_a_glob_character_is_refused_before_eole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch)
    glob_dir = tmp_path / "ck[1]"
    case.checkpoint.rename(glob_dir)

    with pytest.raises(ConfigError, match="glob"):
        strict_parity(
            glob_dir,
            case.export,
            case.sources,
            out_dir=case.out_dir,
            engine=case.ct2,
            processor=case.processor,
        )

    assert case.eole.launches == []


def test_a_file_of_the_export_that_changed_since_it_was_verified_is_refused(
    case: ParityCase,
) -> None:
    (case.export.directory / SPM_FILE).write_bytes(SPM_BYTES + b"\0")

    with pytest.raises(ConfigError, match="changed after"):
        strict_parity(
            case.checkpoint,
            case.export,
            case.sources,
            out_dir=case.out_dir,
            engine=case.ct2,
        )

    assert_nothing_started(case)


# --- Eole does not do what it must --------------------------------------------------------------


def test_eole_failing_is_an_upstream_error_that_leaves_no_report(
    case: ParityCase,
) -> None:
    case.eole.error = UpstreamError("eole predict exited with code 1")

    with pytest.raises(UpstreamError, match="exited with code 1"):
        run(case)

    assert case.ct2.calls == []


def test_eole_writing_no_output_file_is_an_upstream_error(case: ParityCase) -> None:
    case.eole.writes_output = False

    with pytest.raises(UpstreamError, match="no output"):
        run(case)


def test_an_output_of_an_earlier_run_is_not_taken_for_this_one(
    case: ParityCase,
) -> None:
    case.out_dir.mkdir(parents=True)
    (case.out_dir / "parity.tokens.out").write_text(
        "a\t-1.0\t1.0\n" * 3, encoding="utf-8"
    )
    case.eole.writes_output = False

    with pytest.raises(UpstreamError, match="no output"):
        run(case)


@pytest.mark.parametrize("extra", [1, -1])
def test_another_number_of_lines_than_sources_is_an_upstream_error(
    case: ParityCase, extra: int
) -> None:
    if extra > 0:
        case.eole.extra_lines = extra
    else:
        case.eole.answers = case.answers[:-1]

    with pytest.raises(UpstreamError, match="wrote") as raised:
        run(case)

    assert "3" in str(raised.value)
    assert case.ct2.calls == []


def test_a_line_of_another_shape_is_an_upstream_error_naming_its_number(
    case: ParityCase,
) -> None:
    case.eole.raw_lines = [
        f"{SPIECE}a\t-1.0\t1.0",
        f"{SPIECE}secret\t-1.0",
        f"{SPIECE}b\t-1.0\t1.0",
    ]

    with pytest.raises(UpstreamError, match="line 2") as raised:
        run(case)

    assert "secret" not in str(raised.value)


# --- the library does not do what it must --------------------------------------------------------


def test_the_library_answering_a_wrong_number_of_results_is_an_upstream_error(
    case: ParityCase,
) -> None:
    case.ct2.result_offset = -1

    with pytest.raises(UpstreamError, match="results"):
        run(case)


@pytest.mark.parametrize(
    "error", [RuntimeError("boom secret"), ValueError("boom secret")]
)
def test_the_library_failing_to_translate_is_an_upstream_error_without_its_text(
    case: ParityCase, error: Exception
) -> None:
    case.ct2.error = error

    with pytest.raises(UpstreamError) as raised:
        run(case)

    assert "secret" not in str(raised.value)
    assert isinstance(raised.value.__cause__, type(error))


def test_the_library_failing_to_score_is_an_upstream_error(case: ParityCase) -> None:
    def failing(line: tuple[str, ...], target: tuple[str, ...]) -> Sequence[float]:
        raise RuntimeError("cannot score secret")

    case.ct2.score_script = failing

    with pytest.raises(UpstreamError) as raised:
        run(case)

    assert "secret" not in str(raised.value)


def test_the_library_scoring_another_number_of_lines_is_an_upstream_error(
    case: ParityCase,
) -> None:
    real = case.ct2.score_batch
    case.ct2.score_batch = (
        lambda source, target, **options: real(  # type: ignore[method-assign]
            source, target, **options
        )[:-1]
    )

    with pytest.raises(UpstreamError, match="scored"):
        run(case)


def test_a_score_without_log_probabilities_is_an_upstream_error(
    case: ParityCase,
) -> None:
    case.ct2.score_script = lambda line, target: []

    with pytest.raises(UpstreamError, match="log-probabilit"):
        run(case)


# --- the default loader of the library ----------------------------------------------------------


class FakeCtranslate2(ModuleType):
    """A `ctranslate2` module whose `Translator` records how it was made."""

    def __init__(
        self, translator: ParityCt2 | None = None, **failures: Exception
    ) -> None:
        super().__init__("ctranslate2")
        self.made: list[tuple[str, dict[str, Any]]] = []
        self._translator = translator
        self._failure = failures.get("error")
        self.Translator = self._make

    def _make(self, path: str, **options: Any) -> ParityCt2:
        self.made.append((path, options))
        if self._failure is not None:
            raise self._failure
        assert self._translator is not None
        return self._translator


def test_the_default_translator_is_loaded_on_the_cpu_with_what_the_file_stores(
    case: ParityCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = FakeCtranslate2(case.ct2)
    monkeypatch.setitem(sys.modules, "ctranslate2", library)

    strict_parity(
        case.checkpoint,
        case.export,
        case.sources,
        out_dir=case.out_dir,
        threads=3,
        processor=case.processor,
    )

    assert library.made == [
        (
            str(case.export.directory),
            {"device": "cpu", "compute_type": "default", "intra_threads": 3},
        )
    ]


@pytest.mark.parametrize("error", [RuntimeError("no model"), OSError("no file")])
def test_the_default_translator_is_loaded_before_eole_starts(
    case: ParityCase, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setitem(sys.modules, "ctranslate2", FakeCtranslate2(error=error))

    with pytest.raises(DataError, match="cannot load"):
        strict_parity(
            case.checkpoint,
            case.export,
            case.sources,
            out_dir=case.out_dir,
            processor=case.processor,
        )

    assert case.eole.launches == []


def test_a_type_the_library_cannot_run_is_a_config_error(
    case: ParityCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        sys.modules, "ctranslate2", FakeCtranslate2(error=ValueError("unsupported"))
    )

    with pytest.raises(ConfigError, match="compute type"):
        strict_parity(
            case.checkpoint,
            case.export,
            case.sources,
            out_dir=case.out_dir,
            processor=case.processor,
        )


def test_the_default_processor_is_the_export_s_own_sentencepiece_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=1)
    pieces = sentencepiece.SentencePieceProcessor(model_proto=SPM_BYTES).encode(
        "w0", out_type=str
    )
    assert pieces
    case.eole.answers = [([f"{SPIECE}x"], -0.25)]
    case.ct2.script = lambda line: [([f"{SPIECE}x"], -0.5)]
    case.ct2.score_script = lambda line, target: [-0.125, -0.125]

    strict_parity(
        case.checkpoint,
        case.export,
        case.sources,
        out_dir=case.out_dir,
        engine=case.ct2,
    )

    (launch,) = case.eole.launches
    assert launch.source_lines == [" ".join([TAG, *pieces])]


# --- the module itself --------------------------------------------------------------------------


def _names_in(path: Path) -> set[str]:
    """Every name, attribute and string constant of a source."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


# Spelled in two halves so that this file does not hold them either.
GPU_NAMES = (
    "torch" + ".cuda",
    "cuda_" + "available",
    "get_cuda_" + "device_count",
    "nvidia" + "-smi",
)


def test_no_code_or_test_of_the_parity_touches_the_gpu() -> None:
    root = Path(__file__).resolve().parents[2]
    files = [
        SOURCE,
        root / "src/rinchen/cli/export_parity.py",
        root / "tests/support/fake_parity_eole.py",
        root / "tests/support/parity_cli.py",
        *sorted((root / "tests/unit").glob("test_*export_parity*.py")),
    ]
    assert len(files) == 8
    for file in files:
        text = file.read_text(encoding="utf-8")
        assert not [name for name in GPU_NAMES if name in text], file
        imported = _imports_of(ast.parse(text), top_level_only=False)
        assert not {m for m in imported if m.split(".")[0] == "torch"}, file


@pytest.mark.parametrize(
    "file", [SOURCE, SOURCE.parents[2] / "rinchen/cli/export_parity.py"]
)
def test_nothing_asks_for_an_automatic_device_or_compute_type(file: Path) -> None:
    assert "auto" not in _names_in(file)


def _imports_of(tree: ast.Module, *, top_level_only: bool) -> set[str]:
    """The modules a source imports (`from m import n` names `m`), wherever the import is, or
    only in the statements at the top of the module."""
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    found: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    return found


def test_the_module_imports_no_layer_above_train_and_loads_no_library_at_import() -> (
    None
):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    everywhere = _imports_of(tree, top_level_only=False)
    at_the_top = _imports_of(tree, top_level_only=True)

    for layer in ("rinchen.cli", "rinchen.script_conv", "rinchen.serve"):
        assert not {m for m in everywhere if m == layer or m.startswith(f"{layer}.")}
    assert "ctranslate2" in everywhere
    assert "ctranslate2" not in at_the_top


def test_importing_the_module_does_not_load_ctranslate2() -> None:
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    code = (
        "import sys; import rinchen.train.export_parity; "
        "assert 'ctranslate2' not in sys.modules"
    )

    outcome = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
        check=False,
    )

    assert outcome.returncode == 0, outcome.stderr
