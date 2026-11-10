"""`rinchen eval run`: the order in which it hashes and reads, and what a run that is accepted but
then fails or is stopped does and leaves behind (D13, Task 7): every input hashed one by one before
the first is read, a file that changed between its hash and its read (2), an error of the engine
or of a translator with the exit code of its class (2, 3, 4), an engine that loaded other files than
the ones hashed, an interrupt (130, the engine closed, nothing written and the last run as it was)
and the refusals that belong to a baseline (an unknown generator, `auto`, a vocabulary that lacks the
family's tokens).

What is refused before the run, with nothing opened, is in `test_cli_eval_run_refusals.py`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError, UpstreamError
from tests.support.eval_cli import (
    EvalWorkspace,
    Seams,
    assert_refused,
    install,
    invoke,
    make_workspace,
    robustness_directories,
    scores_bytes,
)
from tests.support.eval_fakes import (
    GOLDEN_OUTPUTS_FILES,
    FakeCandidateTranslator,
    answering,
    golden_outputs,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def workspace(tmp_path: Path) -> EvalWorkspace:
    return make_workspace(tmp_path)


@pytest.fixture
def seams(workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch) -> Seams:
    return install(monkeypatch, workspace.engine(), FakeCandidateTranslator())


# --- a file that changed after it was hashed --------------------------------------------------------------


@pytest.mark.parametrize("which", ["suite", "hypotheses", "metrics"])
def test_an_input_that_changed_between_the_hash_and_the_read_is_a_config_error(
    workspace: EvalWorkspace,
    seams: Seams,
    monkeypatch: pytest.MonkeyPatch,
    which: str,
) -> None:
    from rinchen.cli import eval_run

    target = {
        "suite": workspace.suite,
        "hypotheses": workspace.alpha,
        "metrics": workspace.metrics,
    }[which]
    real = eval_run.verified_content

    def swapping(file: Path, digest: str) -> bytes:
        if file == target:
            file.write_bytes(file.read_bytes() + b"\n")
        return real(file, digest)

    monkeypatch.setattr(eval_run, "verified_content", swapping)

    result = invoke(
        workspace.hypotheses_args("--metrics-config", str(workspace.metrics))
    )

    assert_one_error(result, 2, "")
    assert "changed after it was hashed" in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_a_generators_config_that_changed_between_the_hash_and_the_read_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rinchen.cli import eval_run

    real = eval_run.verified_content

    def swapping(file: Path, digest: str) -> bytes:
        if file == workspace.generators:
            file.write_bytes(file.read_bytes() + b"\n# changed\n")
        return real(file, digest)

    monkeypatch.setattr(eval_run, "verified_content", swapping)

    result = invoke(workspace.generator_args("vllm_toy"))

    assert_one_error(result, 2, "")
    assert "changed after it was hashed" in result.stderr
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize(
    "which", ["robustness.json", "report.json", "manifest.json", "scores"]
)
def test_a_robustness_or_scores_file_that_changed_between_the_hash_and_the_read_is_refused(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:

    from rinchen.cli import eval_run_inputs

    robust = robustness_directories(workspace, monkeypatch)
    scores = workspace.root / "in" / "comet22.jsonl"
    scores.write_bytes(
        scores_bytes(
            "comet22",
            hashlib.sha256(workspace.suite.read_bytes()).hexdigest(),
            hashlib.sha256(GOLDEN_OUTPUTS_FILES["alpha"].read_bytes()).hexdigest(),
            0.5,
        )
    )
    target = scores if which == "scores" else robust.named_alpha / which
    real = eval_run_inputs.verified_content

    def swapping(file: Path, digest: str) -> bytes:
        if file == target:
            file.write_bytes(file.read_bytes() + b" ")
        return real(file, digest)

    monkeypatch.setattr(eval_run_inputs, "verified_content", swapping)

    result = invoke(
        workspace.hypotheses_args(
            "--robustness",
            str(robust.named_alpha),
            "--segment-scores",
            f"comet22={scores}",
        )
    )

    assert_one_error(result, 2, "")
    assert "changed after it was hashed" in result.stderr
    assert not workspace.out_dir.exists()


def record_hashes_and_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record, in order, every file the command hashes (`hash <name>`) and every file it reads
    back from the bytes that were hashed (`read <name>`), wherever it does so."""
    from rinchen.cli import eval_run, eval_run_inputs

    events: list[str] = []
    real_hash, real_read = eval_run_inputs.hash_input, eval_run.verified_content

    def hashing(path: Path, **keywords: Any) -> Any:
        events.append(f"hash {path.name}")
        return real_hash(path, **keywords)

    def reading(file: Path, digest: str) -> bytes:
        events.append(f"read {file.name}")
        return real_read(file, digest)

    monkeypatch.setattr(eval_run_inputs, "hash_input", hashing)
    monkeypatch.setattr(eval_run_inputs, "verified_content", reading)
    monkeypatch.setattr(eval_run, "verified_content", reading)
    return events


def test_every_input_is_hashed_one_by_one_before_the_first_of_them_is_read(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = record_hashes_and_reads(monkeypatch)

    result = invoke(
        workspace.hypotheses_args("--metrics-config", str(workspace.metrics))
    )

    assert result.exit_code == 0, result.output
    assert events[:3] == [
        "hash golden_suite.json",
        "hash metrics.yaml",
        "hash alpha.jsonl",
    ]
    assert [e for e in events if e.startswith("hash")] == events[:3]
    assert {e for e in events[3:]} == {
        "read golden_suite.json",
        "read metrics.yaml",
        "read alpha.jsonl",
    }


def test_the_robustness_files_and_the_scores_are_hashed_before_anything_is_read(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:

    robust = robustness_directories(workspace, monkeypatch)
    scores = workspace.root / "in" / "comet22.jsonl"
    scores.write_bytes(
        scores_bytes(
            "comet22",
            hashlib.sha256(workspace.suite.read_bytes()).hexdigest(),
            hashlib.sha256(GOLDEN_OUTPUTS_FILES["alpha"].read_bytes()).hexdigest(),
            0.5,
        )
    )
    events = record_hashes_and_reads(monkeypatch)

    result = invoke(
        workspace.hypotheses_args(
            "--metrics-config",
            str(workspace.metrics),
            "--robustness",
            str(robust.named_alpha),
            "--segment-scores",
            f"comet22={scores}",
        )
    )

    assert result.exit_code == 0, result.output
    assert events[:7] == [
        "hash golden_suite.json",
        "hash metrics.yaml",
        "hash alpha.jsonl",
        "hash robustness.json",
        "hash report.json",
        "hash manifest.json",
        "hash comet22.jsonl",
    ]
    assert not any(event.startswith("hash") for event in events[7:])
    assert len(events) > 7


def test_the_export_is_vouched_for_before_anything_else_is_hashed(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rinchen.cli import eval_run_inputs

    events: list[str] = []
    real_verify, real_hash = eval_run_inputs.verify_export, eval_run_inputs.hash_input

    def verifying(directory: Path) -> Any:
        events.append("verify")
        return real_verify(directory)

    def hashing(path: Path, **keywords: Any) -> Any:
        events.append(f"hash {path.name}")
        return real_hash(path, **keywords)

    monkeypatch.setattr(eval_run_inputs, "verify_export", verifying)
    monkeypatch.setattr(eval_run_inputs, "hash_input", hashing)
    install(monkeypatch, workspace.engine())

    assert invoke(workspace.model_args()).exit_code == 0

    assert events[:3] == ["verify", "hash golden_suite.json", "hash metrics.yaml"]


# --- the engine -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ConfigError("the model does not write the target"), 2),
        (DataError("a text holds a lone surrogate"), 3),
        (UpstreamError("CTranslate2 failed"), 4),
    ],
)
def test_an_error_of_the_engine_has_its_exit_code_leaves_nothing_and_closes_the_engine(
    workspace: EvalWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    code: int,
) -> None:
    engine = workspace.engine(error=error)
    install(monkeypatch, engine)

    result = invoke(workspace.model_args())

    assert_one_error(result, code, str(error))
    assert result.stdout == ""
    assert engine.closed == 1
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ConfigError("no such device"), 2),
        (DataError("no such file"), 3),
        (UpstreamError("x"), 4),
    ],
)
def test_an_engine_that_cannot_be_opened_is_refused_with_nothing_written(
    workspace: EvalWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    code: int,
) -> None:
    seams = install(monkeypatch, workspace.engine(), open_error=error)

    result = invoke(workspace.model_args())

    assert_one_error(result, code, str(error))
    assert seams.events == ["open"]
    assert not workspace.out_dir.exists()


def test_an_engine_that_answers_fewer_translations_than_requests_is_an_upstream_error(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = list(golden_outputs("alpha").values())[:-1]
    engine = workspace.engine(answer=answering(outputs))
    install(monkeypatch, engine)

    result = invoke(workspace.model_args())

    assert_one_error(result, 4, "the engine answered 19 translations for 20 requests")
    assert engine.closed == 1
    assert not workspace.out_dir.exists()


def test_an_interrupt_during_the_run_exits_130_closes_the_engine_and_writes_nothing(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine(error=KeyboardInterrupt())
    install(monkeypatch, engine)

    result = invoke(workspace.model_args())

    assert result.exit_code == 130, result.output
    assert engine.closed == 1
    assert not workspace.out_dir.exists()


def test_an_interrupt_leaves_the_files_and_the_manifest_of_the_last_run_as_they_were(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    assert invoke(workspace.model_args()).exit_code == 0
    before = snapshot(workspace.out_dir)
    engine = workspace.engine(error=KeyboardInterrupt())
    install(monkeypatch, engine)

    result = invoke(workspace.model_args())

    assert result.exit_code == 130
    assert engine.closed == 1
    assert snapshot(workspace.out_dir) == before


def test_a_failed_run_leaves_the_last_results_and_their_manifest_untouched(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    assert invoke(workspace.model_args()).exit_code == 0
    before = snapshot(workspace.out_dir)
    install(monkeypatch, workspace.engine(error=UpstreamError("CTranslate2 failed")))

    result = invoke(workspace.model_args())

    assert result.exit_code == 4
    assert snapshot(workspace.out_dir) == before


# --- the baselines ---------------------------------------------------------------------------------------


def test_an_unknown_generator_is_a_config_error_listing_the_configured_names(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.generator_args("nope"))

    assert result.exit_code == 2, result.output
    for name in ("nllb_toy", "madlad_toy", "vllm_toy", "llamacpp_toy"):
        assert name in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_a_ct2_generator_on_auto_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.generator_args("nllb_auto"))

    assert result.exit_code == 2, result.output
    assert "auto" in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_a_family_whose_vocabulary_lacks_its_tokens_is_refused_naming_the_family_and_a_language(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.generator_args("nllb_bad_vocabulary"))

    assert result.exit_code == 2, result.output
    assert "nllb" in result.stderr
    assert any(code in result.stderr for code in ("eng_Latn", "khk_Cyrl", "rus_Cyrl"))
    assert_refused(workspace, seams, result, 2)


def test_a_baseline_directory_without_a_vocabulary_is_refused(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    (workspace.root / "nllb" / "shared_vocabulary.json").unlink()

    result = invoke(workspace.generator_args("nllb_toy"))

    assert result.exit_code == 2, result.output
    assert "shared_vocabulary.json" in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_a_generators_config_that_is_missing_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams, tmp_path: Path
) -> None:
    arguments = workspace.generator_args("vllm_toy")
    arguments[arguments.index("--generators-config") + 1] = str(tmp_path / "none.yaml")

    result = invoke(arguments)

    assert_one_error(result, 2, "generators config not found")
    assert_refused(workspace, seams, result, 2)


def test_an_error_of_the_translator_has_its_exit_code(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    translator = FakeCandidateTranslator(
        answer=lambda texts, src, tgt: (_ for _ in ()).throw(
            UpstreamError("the server is down")
        )
    )
    install(monkeypatch, None, translator)

    result = invoke(workspace.generator_args("vllm_toy"))

    assert_one_error(result, 4, "the server is down")
    assert not workspace.out_dir.exists()
