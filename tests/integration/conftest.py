"""Session fixtures of the converter integration tests.

`tiny_run` trains the tiny converter once per test session, from the tiny reference
list, and predicts the training sources in both directions; `tiny_ct2` exports that
checkpoint to CTranslate2. Every test of the session that needs a trained or
exported model takes these two, so the run is paid for once. Everything is written
under pytest's temporary directories. Each Eole or CTranslate2 command is bounded
by `COMMAND_TIMEOUT_SECONDS`.

The `d11_*` fixtures are the D11 gates': `d11_gpu_hidden` hides every device for the
session, `d11_workspace` is the toy corpus and SentencePiece model, `d11_teacher` the two
tiny teachers (tag in the source, tag as the forced target prefix), `d11_exports` their
CTranslate2 exports, `d11_tiny_export_dir` the float32 one of the source-tag teacher (the tiny
model the serving gate serves) and `d11_document_model` a document model continued from the
first teacher and exported. Each runs its stage once per session, the first time a test asks.
The serving gate's own fixtures (`d11_serving_log`, `d11_test_log`, `d11_sentences` and the two
reference engines) live in `tests/support/serving_gate_fixtures.py`.

The `d12_*` fixtures are the robustness gate's (`tests/integration/test_robustness_gate.py`):
`d12_robust_workspace` is the robustness toy's corpus file and SentencePiece model,
`d12_robust_teacher` the tiny teacher trained on it once for 6,000 steps (about 190 s),
`d12_robust_export` its float32 CTranslate2 export and `d12_suites` the four suite files of the gate.
They run under the same `d11_gpu_hidden`.

The `d13_*` fixtures are the evaluation and release gates' (`tests/integration/test_eval_gate.py`,
`test_eval_gate_controls.py` and `test_release_gate.py`): `d13_offline` switches the Hub libraries to
offline for the session, removes any token from the environment and blocks the Hub libraries for the
commands, `d13_suites` writes the two fixture suites of the toy and `d13_runs` makes the three runs the
gates share (the D12 export, the D11 teacher and the export's outputs fed back as a file) with the real
`rinchen eval run`. They run under `d11_gpu_hidden`.

The `tiny_*` fixtures are the D15 gate's (`tests/integration/test_tiny_gate*.py`): `tiny_gpu_hidden` hides
every device from the test process (the helpers refuse to start a command otherwise), `tiny_shim` writes the
`uv` shim the recipe finds, `tiny_run_dir` runs the body of the `tiny` recipe ONCE per pytest process (about
two to three minutes: the modules of the gate share it) and `tiny_report` is its parsed `report.json`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rinchen.script_conv.char_model import check_char_model, train_char_model
from rinchen.script_conv.config import (
    dump_config,
    render_converter_config,
    validate_converter_config,
)
from rinchen.script_conv.data import build_corpus, read_pairs_tsv
from rinchen.script_conv.export import export_ct2
from rinchen.script_conv.train import predict, run_training
from tests.support.d11_gate import (
    Exports,
    Teachers,
    converted_exports,
    trained_teachers,
)
from tests.support.d11_gate_doc import DocumentModel, build_document_model
from tests.support.distill_gate_stages import Teacher, Workspace, make_workspace
from tests.support.engines import SilentLog
from tests.support.eval_gate import (
    HUB_BLOCKED_VARIABLE,
    Runs,
    block_hub_libraries,
    make_runs,
)
from tests.support.eval_toy import Suites as ToySuites
from tests.support.eval_toy import write_toy_suites
from tests.support.robust_gate import (
    Suites,
    export_robust,
    robust_workspace,
    train_robust_teacher,
    write_suites,
)
from tests.support.script_reference import FIXTURE_DIR
from tests.support.serving_gate_fixtures import (  # noqa: F401 - re-exported for pytest
    d11_float32_engine,
    d11_int8_engine,
    d11_no_server_is_left,
    d11_sentences,
    d11_serving_log,
    d11_test_log,
)
from tests.support.tiny_converter import (
    COMMAND_TIMEOUT_SECONDS,
    DIRECTION_TAGS,
    TinyRun,
    read_lines,
)
from tests.support.tiny_gate import RecipeRun, make_shim, read_report, run_the_recipe

_VALID_FRACTION = 0.1
_SEED = 1234
_CHAR_VOCAB_SIZE = 200


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> TinyRun:
    root = tmp_path_factory.mktemp("converter_tiny")
    run_dir = root / "run"
    pairs = read_pairs_tsv(FIXTURE_DIR / "tiny_words.tsv", origin="reference")
    corpus = build_corpus(
        pairs, root / "corpus", valid_fraction=_VALID_FRACTION, seed=_SEED
    )

    texts = [pair.cyrillic for pair in pairs] + [pair.script for pair in pairs]
    spm_model = train_char_model(
        texts, root / "converter_char", vocab_size=_CHAR_VOCAB_SIZE
    )
    check_char_model(spm_model, texts)

    cfg = render_converter_config(
        "tiny", data_dir=corpus.directory, spm_model=spm_model, run_dir=run_dir
    )
    validate_converter_config(cfg, profile="tiny")
    config_path = root / "converter_tiny.yaml"
    dump_config(cfg, config_path)

    started = time.perf_counter()
    ckpt_dir = run_training(config_path, run_dir, timeout=COMMAND_TIMEOUT_SECONDS)
    train_seconds = time.perf_counter() - started

    predictions = {
        direction: read_lines(
            predict(
                ckpt_dir,
                corpus.files[f"{direction}.train.src"],
                tag=tag,
                spm_model=spm_model,
                out_file=run_dir / f"{direction}.train.pred",
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
        )
        for direction, tag in DIRECTION_TAGS.items()
    }
    return TinyRun(
        corpus=corpus,
        spm_model=spm_model,
        config_path=config_path,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        train_seconds=train_seconds,
        predictions=predictions,
    )


@pytest.fixture(scope="session")
def tiny_ct2(tiny_run: TinyRun, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The CTranslate2 model directory exported from `tiny_run`'s checkpoint."""
    return export_ct2(
        tiny_run.ckpt_dir,
        tmp_path_factory.mktemp("converter_ct2") / "model",
        timeout=COMMAND_TIMEOUT_SECONDS,
    )


@pytest.fixture(scope="session")
def d11_gpu_hidden() -> Iterator[None]:
    """No device is visible to any process of the D11 gates: Eole, the converter and the
    CTranslate2 translators of the tests inherit the empty list."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        yield


@pytest.fixture(scope="module")
def d11_quiet_engine_log() -> Iterator[None]:
    """The engine's first log line would configure the root logger for the rest of the
    session, and `test_logging` expects to be the first to do so: silence it for the module."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("rinchen.utils.logging.get", lambda name: SilentLog())
        yield


@pytest.fixture(scope="session")
def d11_workspace(
    d11_gpu_hidden: None, tmp_path_factory: pytest.TempPathFactory
) -> Workspace:
    return make_workspace(tmp_path_factory.mktemp("d11_gate"))


@pytest.fixture(scope="session")
def d11_teacher(d11_workspace: Workspace) -> Teachers:
    return trained_teachers(d11_workspace)


@pytest.fixture(scope="session")
def d11_exports(d11_workspace: Workspace, d11_teacher: Teachers) -> Exports:
    return converted_exports(d11_workspace, d11_teacher)


@pytest.fixture(scope="session")
def d11_tiny_export_dir(d11_exports: Exports) -> Path:
    """The float32 export of the source-tag teacher: the tiny model the serving gate serves."""
    return d11_exports.float32.directory


@pytest.fixture(scope="session")
def d11_document_model(
    d11_workspace: Workspace, d11_teacher: Teachers
) -> DocumentModel:
    return build_document_model(d11_workspace, d11_teacher.source)


@pytest.fixture(scope="session")
def d12_robust_workspace(
    d11_gpu_hidden: None, tmp_path_factory: pytest.TempPathFactory
) -> Workspace:
    return robust_workspace(tmp_path_factory.mktemp("d12_gate"))


@pytest.fixture(scope="session")
def d12_robust_teacher(d12_robust_workspace: Workspace) -> Teacher:
    """The tiny teacher trained on the robustness toy: paid for once per session."""
    return train_robust_teacher(d12_robust_workspace)


@pytest.fixture(scope="session")
def d12_robust_export(
    d12_robust_workspace: Workspace,
    d12_robust_teacher: Teacher,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """The float32 export of the robustness teacher (tag in the source)."""
    return export_robust(
        d12_robust_workspace,
        d12_robust_teacher,
        tmp_path_factory.mktemp("d12_export") / "model",
    ).directory


@pytest.fixture(scope="session")
def d12_suites(tmp_path_factory: pytest.TempPathFactory) -> Suites:
    """The main suite and the three control suites of the robustness toy, as files."""
    return write_suites(tmp_path_factory.mktemp("d12_suites"))


@pytest.fixture(scope="session")
def d13_offline(
    d11_gpu_hidden: None, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    """The Hub libraries are offline for every process of the D13 gates, their home is an empty
    directory, no token is in the environment and a directory of blocked libraries (`huggingface_hub`
    and `datasets` that raise `ImportError`) is named for the commands to put first on their path."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("HF_DATASETS_OFFLINE", "1")
        patch.setenv("HF_HOME", str(tmp_path_factory.mktemp("d13_hf_home")))
        patch.delenv("HF_TOKEN", raising=False)
        blocked = block_hub_libraries(tmp_path_factory.mktemp("d13_hub_blocked"))
        patch.setenv(HUB_BLOCKED_VARIABLE, str(blocked))
        yield


@pytest.fixture(scope="session")
def d13_suites(
    d13_offline: None, tmp_path_factory: pytest.TempPathFactory
) -> ToySuites:
    """The fixture suites `toy-devtest` and `toy-dev`, made from the robustness toy."""
    return write_toy_suites(tmp_path_factory.mktemp("d13_suites"))


@pytest.fixture(scope="session")
def d13_runs(
    d13_suites: ToySuites,
    d12_robust_export: Path,
    d11_tiny_export_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Runs:
    """`rinchen eval run` over the devtest suite for the D12 export (`a-model`), the D11 teacher
    (`d11-teacher`) and the export's outputs as a file (`a-file`), under one output root."""
    return make_runs(
        d13_suites,
        d12_robust_export,
        d11_tiny_export_dir,
        tmp_path_factory.mktemp("d13_runs"),
    )


@pytest.fixture(scope="session")
def tiny_gpu_hidden() -> Iterator[None]:
    """No device is visible to any process of the D15 gate: the helpers that start a command refuse
    to start it unless the test process itself holds the empty list."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        yield


@pytest.fixture(scope="session")
def tiny_shim(tiny_gpu_hidden: None, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The directory of the `uv` shim of the run of the recipe (and of its call log)."""
    return make_shim(tmp_path_factory.mktemp("tiny_shim"))


@pytest.fixture(scope="session")
def tiny_run_dir(
    tiny_gpu_hidden: None, tiny_shim: Path, tmp_path_factory: pytest.TempPathFactory
) -> RecipeRun:
    """The body of the `tiny` recipe, run once through `sh` in a temporary directory: paid for once
    per pytest process, by the first test of the gate that asks."""
    return run_the_recipe(tmp_path_factory.mktemp("tiny_gate_run"), shim_dir=tiny_shim)


@pytest.fixture(scope="session")
def tiny_report(tiny_run_dir: RecipeRun) -> dict[str, Any]:
    """`report.json` of `tiny_run_dir`, parsed. Fails with what the recipe printed when there is none."""
    assert (tiny_run_dir.out_dir / "report.json").is_file(), (
        f"the recipe exited {tiny_run_dir.invocation.returncode} and wrote no report: "
        f"{tiny_run_dir.invocation.stderr[-2000:]}"
    )
    return read_report(tiny_run_dir.out_dir)
