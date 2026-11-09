"""The D11 export gates: strict parity of an export with its checkpoint, the quantisations, and
the proof that no GPU is used. Everything runs through the real `rinchen` commands with the real
Eole and the real CTranslate2 on the CPU; nothing here touches a GPU (`CUDA_VISIBLE_DEVICES` is
the empty string for the session, Eole trains with `gpu_ranks: []`, the converter runs with the
device hidden and every CTranslate2 translator of the tests is built on `device="cpu"` with the
compute type the file stores).

The data are the toy corpora of `tests/support/tiny_parallel.py`: templated sentences, so every
number below tests machinery and ranks nothing (the toy saturates at chrF 98.2).

Session fixtures (`tests/integration/conftest.py`) run the expensive stages once, in the order the
scenarios need, and take a snapshot of what the tests look at:

    T  the source-tag teacher (2,000 steps, averaged) and a target-tag teacher (1,000 steps,
       averaged), through D10's stage
    X  `rinchen export convert` of the source teacher at float32, int8 and int8_float16 and of
       the target teacher at float32
    P  `rinchen export parity` of the float32 exports (50 dev sentences of each of the 4
       directions) and of the int8 export, and the library's parity of the target export
       with the forced tag's log-probability KEPT (the trap the library avoids)

Gates, all thresholds literals of this file and never lowered (a miss is answered with more
steps of the tiny config):

    G1  float32, tag in the source: exact share at least 0.95, mean |delta| at most 1e-3
    G2  float32, tag as the forced target prefix: the same two, and without the prefix drop the
        mean |delta| per token is above 0.01 (the gate would catch its omission)
    G3  quantisation: int8 greedy chrF at least float32's minus 1.0, `model.bin` smaller, the
        int8 parity not gated by the log-probability bound and at least 0.95 exact;
        `int8_float16` converts and loads on the CPU as `int8_float32`; an explicit
        `int8_float16` and `auto` are refused
    G7  no GPU: an AST scan of the D11 code and tests, the Eole logs, the effective compute types

Measured on this machine (CPU, two threads, Eole 0.6.4, CTranslate2 4.8.4, 2026-11-09; the
machine was shared with other work, so a time moves by a third from one run to the next, a count
or a score never does; the D11 spike measured the same counts and scores):

    T  source teacher: launch 71 s (145 s on a loaded machine), average 1 s; target teacher:
       launch 40 s, average 1 s; both 4 saves or 2, no non-finite loss
    X  `export convert` 3.8 to 5.3 s each; model.bin 2,034,311 bytes at float32 (both
       placements), 1,488,691 at int8 (73.2 %), 844,931 at int8_float16 (41.5 %)
    P  `export parity` 5.7 s (source float32), 6.0 s (target float32), 5.8 s (int8)
    G1 200 of 200 exact, mean |delta| 7.19e-07 per token (bound 1e-3), max 3.43e-06
    G2 200 of 200 exact, mean |delta| 1.37e-06 (max 5.05e-06); with the forced tag's
       log-probability kept: 200 of 200 exact and a mean |delta| of 0.0433 per token, 31,600
       times larger, 43 times the bound
    G3 greedy chrF over the 400 dev rows: float32 98.207, int8 98.268 (+0.061),
       int8_float16 98.197 (-0.010); int8 parity 199 of 200 exact, mean |delta| 1.14e-03
       (above the float32 bound, which a quantised export is not held to), max 1.6e-02;
       effective compute types float32, int8_float32, int8_float32; the target teacher, a
       1,000-step model, reaches chrF 58.3 (a number, no gate)
    G7 the Eole logs say `GPU ranks: []` and `Starting training in cpu mode`; no D11 file asks a
       device question; every CTranslate2 translator of the tests is on `cpu`

The whole of this module and `test_engine_surfaces.py` takes 8 min 0 s on a machine with a load
average of 12 to 30 (the teachers 3.5 min, the document model 2.2 min, the D5 converter 1.7
min).
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError
from rinchen.export.record import EXPORT_FILES
from rinchen.infer.engine_types import LoadOptions
from rinchen.train.export_parity import ParityReport
from tests.support.d11_gate import (
    PARITY_LINES_PER_DIRECTION,
    Exports,
    Parity,
    Teachers,
    forbidden_gpu_findings,
    greedy_chrf,
    open_engine,
    parity_without_the_prefix_drop,
    run_parity,
)
from tests.support.distill_gate import sha256_of, source_findings
from tests.support.distill_gate_stages import DIRECTIONS, Workspace, manifest_of

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.usefixtures("d11_gpu_hidden", "d11_quiet_engine_log"),
]

# The gates. They are never lowered: a miss is answered with more steps.
MIN_EXACT_SHARE: Final = 0.95
MAX_MEAN_DELTA: Final = 1e-3
# Without the prefix drop the delta per token is about 0.05 on the toy (spike: 0.48 per sentence).
MIN_DELTA_WITHOUT_THE_DROP: Final = 0.01
MAX_INT8_CHRF_DROP: Final = 1.0
SENTENCES: Final = 200


# ---------------------------------------------------------------------------------
# The fixtures: each runs one stage of `tests/support/d11_gate.py` once per module.


@pytest.fixture(scope="module")
def source_parity(
    d11_workspace: Workspace, d11_teacher: Teachers, d11_exports: Exports
) -> Parity:
    return run_parity(
        d11_workspace,
        d11_exports.float32,
        d11_teacher.source.averaged,
        d11_workspace.root / "parity_src_float32",
    )


@pytest.fixture(scope="module")
def target_parity(
    d11_workspace: Workspace, d11_teacher: Teachers, d11_exports: Exports
) -> Parity:
    return run_parity(
        d11_workspace,
        d11_exports.target_float32,
        d11_teacher.target.averaged,
        d11_workspace.root / "parity_tgt_float32",
    )


@pytest.fixture(scope="module")
def int8_parity(
    d11_workspace: Workspace, d11_teacher: Teachers, d11_exports: Exports
) -> Parity:
    return run_parity(
        d11_workspace,
        d11_exports.int8,
        d11_teacher.source.averaged,
        d11_workspace.root / "parity_src_int8",
    )


@pytest.fixture(scope="module")
def kept_prefix(
    d11_workspace: Workspace, d11_teacher: Teachers, d11_exports: Exports
) -> ParityReport:
    return parity_without_the_prefix_drop(
        d11_workspace,
        d11_exports.target_float32,
        d11_teacher.target.averaged,
        d11_workspace.root / "parity_tgt_kept_prefix",
    )


def _assert_the_command_passed(parity: Parity, *, gated: bool) -> dict[str, Any]:
    """The facts of a passing `rinchen export parity` run that do not depend on the model."""
    invocation = parity.invocation
    assert invocation.returncode == 0, invocation.stderr
    result = parity.result
    assert result["sentences"] == SENTENCES
    assert result["lines"] == dict.fromkeys(DIRECTIONS, PARITY_LINES_PER_DIRECTION)
    assert result["passes"] is True
    assert result["gated"] is gated
    assert result["exact_share"] >= MIN_EXACT_SHARE
    assert result["min_exact_share"] == MIN_EXACT_SHARE
    assert result["max_mean_delta"] == MAX_MEAN_DELTA
    bound = "gated" if gated else "not gated: quantized"
    assert invocation.stdout[0].startswith(
        f"parity: {result['exact']}/{SENTENCES} exact"
    )
    assert invocation.stdout[0].endswith(f"({bound}) pass")
    assert invocation.stdout[1] == f"result: {parity.out_dir / 'parity.json'}"
    manifest = manifest_of(parity.out_dir)
    assert manifest.stage == "export_parity"
    assert manifest.status["parity"] == "pass"
    return result


def _record_parity(record_property: Any, name: str, parity: Parity) -> None:
    result = parity.result
    record_property(f"{name}_exact", f"{result['exact']}/{result['sentences']}")
    record_property(f"{name}_mean_abs_delta", f"{result['mean_abs_delta']:.3g}")
    record_property(f"{name}_max_abs_delta", f"{result['max_abs_delta']:.3g}")
    record_property(f"{name}_seconds", round(parity.invocation.seconds, 1))


# =================================================================================
# T and X: what the gates are made of.


def test_both_teachers_trained_and_every_export_is_vouched_for_by_its_manifest(
    d11_teacher: Teachers, d11_exports: Exports, record_property: Any
) -> None:
    source, target = d11_teacher.source, d11_teacher.target
    record_property("source_teacher_launch_seconds", round(source.launch.seconds, 1))
    record_property("target_teacher_launch_seconds", round(target.launch.seconds, 1))
    assert source.saved_steps == [500, 1000, 1500, 2000]
    assert target.saved_steps == [500, 1000]
    assert source.curve.non_finite_step is None
    assert target.curve.non_finite_step is None

    exports = {
        ("src", "float32"): d11_exports.float32,
        ("src", "int8"): d11_exports.int8,
        ("src", "int8_float16"): d11_exports.int8_float16,
        ("tgt", "float32"): d11_exports.target_float32,
    }
    for (placement, quantization), exported in exports.items():
        record_property(
            f"export_{placement}_{quantization}_seconds",
            round(exported.invocation.seconds, 1),
        )
        record_property(
            f"export_{placement}_{quantization}_model_bytes",
            (exported.directory / "model.bin").stat().st_size,
        )
        assert sorted(path.name for path in exported.directory.iterdir()) == sorted(
            (*EXPORT_FILES, "manifest.json")
        )
        assert exported.record["tag_placement"] == placement
        assert exported.record["quantization"] == quantization
        assert exported.record["document_level"] is False
        assert exported.record["directions"] == list(DIRECTIONS)
        # The placement is derived from the saved config and checked against the statement.
        assert "tag placement: checked against the saved config" in (
            exported.invocation.stdout
        )
        assert exported.manifest.stage == "export"
        assert exported.manifest.status["quantization"] == quantization


# =================================================================================
# G1: strict parity, float32, the tag in the source.


def test_float32_export_of_the_source_tag_teacher_passes_strict_parity(
    source_parity: Parity, record_property: Any
) -> None:
    result = _assert_the_command_passed(source_parity, gated=True)
    _record_parity(record_property, "src_float32", source_parity)

    assert result["placement"] == "src"
    assert result["quantization"] == "float32"
    assert result["mean_abs_delta"] <= MAX_MEAN_DELTA


def test_the_parity_result_names_the_export_and_the_weights_it_compared(
    source_parity: Parity, d11_exports: Exports
) -> None:
    export = d11_exports.float32
    result = source_parity.result

    assert result["export_name"] == export.record["name"]
    assert result["model_sha256"] == sha256_of(export.directory / "model.bin")
    assert (
        result["checkpoint_weights_sha256"]
        == export.record["checkpoint"]["weights_sha256"]
    )


def test_the_parity_of_an_export_with_the_weights_of_another_model_is_refused(
    d11_workspace: Workspace, d11_teacher: Teachers, d11_exports: Exports
) -> None:
    out_dir = d11_workspace.root / "parity_wrong_weights"
    parity = run_parity(
        d11_workspace,
        d11_exports.float32,
        d11_teacher.target.averaged,
        out_dir,
    )

    assert parity.invocation.returncode == 2
    assert parity.invocation.stdout == []
    assert "a parity of two different models says nothing" in parity.invocation.stderr
    assert not out_dir.exists()


# =================================================================================
# G2: strict parity, float32, the tag as the forced prefix of the target.


def test_float32_export_of_the_target_tag_teacher_passes_strict_parity(
    target_parity: Parity, record_property: Any
) -> None:
    result = _assert_the_command_passed(target_parity, gated=True)
    _record_parity(record_property, "tgt_float32", target_parity)

    assert result["placement"] == "tgt"
    assert result["quantization"] == "float32"
    assert result["mean_abs_delta"] <= MAX_MEAN_DELTA


def test_without_the_prefix_drop_the_log_probability_delta_is_far_above_the_gate(
    kept_prefix: ParityReport, target_parity: Parity, record_property: Any
) -> None:
    record_property(
        "tgt_kept_prefix_mean_abs_delta", f"{kept_prefix.mean_abs_delta:.3g}"
    )
    record_property(
        "tgt_kept_prefix_exact", f"{kept_prefix.exact}/{kept_prefix.sentences}"
    )

    # The same translations: only the forced tag's own log-probability differs.
    assert kept_prefix.exact_share >= MIN_EXACT_SHARE
    assert kept_prefix.exact == target_parity.result["exact"]
    assert kept_prefix.mean_abs_delta > MIN_DELTA_WITHOUT_THE_DROP
    assert kept_prefix.mean_abs_delta > 1000 * target_parity.result["mean_abs_delta"]
    assert not kept_prefix.passes


# =================================================================================
# G3: the quantisations.


@pytest.fixture(scope="module")
def chrf(d11_workspace: Workspace, d11_exports: Exports) -> dict[str, float]:
    """chrF of the engine's greedy translations of the 400 dev rows, by export."""
    scores = {}
    for name, exported in (
        ("float32", d11_exports.float32),
        ("int8", d11_exports.int8),
        ("int8_float16", d11_exports.int8_float16),
    ):
        engine = open_engine(exported.directory)
        try:
            scores[name] = greedy_chrf(engine, d11_workspace)
        finally:
            engine.close()
    return scores


def test_the_int8_export_translates_within_one_chrf_point_of_float32(
    chrf: dict[str, float], record_property: Any
) -> None:
    for name, score in chrf.items():
        record_property(f"chrf_{name}", round(score, 3))
    record_property("chrf_int8_minus_float32", round(chrf["int8"] - chrf["float32"], 3))

    assert chrf["float32"] >= 90.0
    assert chrf["int8"] >= chrf["float32"] - MAX_INT8_CHRF_DROP
    assert chrf["int8_float16"] >= chrf["float32"] - MAX_INT8_CHRF_DROP


def test_the_int8_model_file_is_smaller_and_int8_float16_smaller_still(
    d11_exports: Exports, record_property: Any
) -> None:
    sizes = {
        name: (exported.directory / "model.bin").stat().st_size
        for name, exported in (
            ("float32", d11_exports.float32),
            ("int8", d11_exports.int8),
            ("int8_float16", d11_exports.int8_float16),
        )
    }
    for name, size in sizes.items():
        record_property(f"model_bytes_{name}", size)
    record_property(
        "model_bytes_int8_share", round(sizes["int8"] / sizes["float32"], 3)
    )

    assert sizes["int8"] < sizes["float32"]
    assert sizes["int8_float16"] < sizes["int8"]


def test_the_int8_export_is_held_to_the_exact_share_alone(
    int8_parity: Parity, record_property: Any
) -> None:
    result = _assert_the_command_passed(int8_parity, gated=False)
    _record_parity(record_property, "src_int8", int8_parity)

    assert result["quantization"] == "int8"
    assert result["placement"] == "src"
    # The log-probability delta is reported and is no part of the gate: it was measured above
    # the float32 bound in the D11 spike (1.45e-3), so a quantised export is gated by chrF.
    assert result["mean_abs_delta"] > 0.0


def test_compute_types_are_read_from_the_engine_and_a_wrong_choice_is_refused(
    d11_exports: Exports, record_property: Any
) -> None:
    effective = {}
    for name, exported in (
        ("float32", d11_exports.float32),
        ("int8", d11_exports.int8),
        ("int8_float16", d11_exports.int8_float16),
    ):
        engine = open_engine(exported.directory)
        try:
            effective[name] = engine.info.compute_type
            assert engine.info.device == "cpu"
            assert engine.info.quantization == name
        finally:
            engine.close()
    for name, compute_type in effective.items():
        record_property(f"effective_compute_type_{name}", compute_type)

    # A CPU runs int8_float16 weights as int8_float32 (the converter warned at load).
    assert effective == {
        "float32": "float32",
        "int8": "int8_float32",
        "int8_float16": "int8_float32",
    }

    # An explicit type the CPU cannot run, and `auto` (which would quantise a float32 export to
    # int8 on a CPU and pick the GPU when one is visible), are refused.
    with pytest.raises(ConfigError, match="int8_float16"):
        open_engine(d11_exports.float32.directory, compute_type="int8_float16")
    with pytest.raises(ConfigError, match="auto"):
        LoadOptions(compute_type="auto")
    with pytest.raises(ConfigError, match="auto"):
        LoadOptions(device="auto")  # type: ignore[arg-type]


# =================================================================================
# G7: no GPU.

# The code and the tests D11 added: nothing in them asks a device question or names the GPU's
# tools (`ctranslate2.get_cuda_device_count()` once slipped into the design spike).
_SOURCE_FILES: Final = (
    *sorted((REPO_ROOT / "src" / "rinchen" / "export").glob("*.py")),
    *sorted((REPO_ROOT / "src" / "rinchen" / "infer").glob("*.py")),
    *(
        REPO_ROOT / "src" / "rinchen" / "cli" / name
        for name in (
            "export.py",
            "export_parity.py",
            "translate.py",
            "translate_io.py",
            "distill_export.py",
        )
    ),
    REPO_ROOT / "src" / "rinchen" / "train" / "export_parity.py",
)
_TEST_FILES: Final = (
    *sorted((REPO_ROOT / "tests" / "support").glob("d11_gate*.py")),
    REPO_ROOT / "tests" / "integration" / "test_export_parity.py",
    REPO_ROOT / "tests" / "integration" / "test_engine_surfaces.py",
    *sorted((REPO_ROOT / "tests" / "unit").glob("test_export_*.py")),
    *sorted((REPO_ROOT / "tests" / "unit").glob("test_infer_*.py")),
    *sorted((REPO_ROOT / "tests" / "unit").glob("test_translate*.py")),
    *sorted((REPO_ROOT / "tests" / "unit").glob("test_train_export_parity*.py")),
)


def test_no_gpu_question_is_asked_and_every_translator_of_the_tests_is_on_the_cpu() -> (
    None
):
    assert len(_SOURCE_FILES) >= 15
    for path in (*_SOURCE_FILES, *_TEST_FILES):
        assert forbidden_gpu_findings(path) == [], path
    for path in _TEST_FILES:
        devices = source_findings(path).translator_devices
        assert set(devices) <= {"'cpu'"}, path
    # The module that builds a translator for the measurements has them, and each is on cpu.
    support = source_findings(REPO_ROOT / "tests" / "support" / "d11_gate.py")
    assert support.translator_devices == ["'cpu'"] * 3


def test_the_eole_runs_trained_and_predicted_on_the_cpu(
    d11_teacher: Teachers, source_parity: Parity, target_parity: Parity
) -> None:
    for teacher in (d11_teacher.source, d11_teacher.target):
        assert "GPU ranks: []" in teacher.log
        assert "Starting training in cpu mode" in teacher.log
        assert "nccl" not in teacher.log.lower()
        assert "cuda:" not in teacher.log
    for parity in (source_parity, target_parity):
        log = (parity.out_dir / "parity.log").read_text("utf-8", errors="replace")
        assert "cuda:" not in log
        assert "nccl" not in log.lower()


def test_the_converter_logs_name_no_device(d11_exports: Exports) -> None:
    for exported in (
        d11_exports.float32,
        d11_exports.int8,
        d11_exports.int8_float16,
        d11_exports.target_float32,
    ):
        log = exported.converter_log.read_text("utf-8", errors="replace").lower()
        assert "cuda" not in log
