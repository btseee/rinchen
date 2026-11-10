"""Shared setup of the tests of `rinchen eval run` (D13, Task 7).

The command touches a model in two places only, `open_engine` and `build_translator` of
`rinchen.cli.eval_systems`; `install` replaces both by seams that record their calls and hand out a
`FakeRunEngine` and a `FakeCandidateTranslator` of the test's own. Nothing here loads a model,
imports `ctranslate2`, opens a socket or touches a GPU.

An `EvalWorkspace` is the files of one run under a temporary directory: copies of the golden
suite and of the golden outputs of `alpha` and `beta`, a fabricated export that `verify_export`
vouches for, a metrics configuration, a generators configuration over two directories that stand
in for converted NLLB and MADLAD models (what each family needs in its vocabulary, `family_tokens`)
and an output directory that does not exist. `robustness_directories` runs D12's own command over
D12's golden suite and leaves the directories `--robustness` takes. `renames` and `fail_replace`
(from `tests.support.robust_cli`) see every file the command writes by its rename.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pytest
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli.main import app
from rinchen.config.synth import GeneratorConfig
from rinchen.eval.outputs import OutputKey, outputs_to_bytes
from rinchen.infer.engine_types import DecodeOptions, LoadOptions, ModelInfo, Row
from rinchen.synth.ct2_baselines import MADLAD_CODES, NLLB_CODES
from tests.support.eval_fakes import (
    FIXTURES,
    GOLDEN_OUTPUTS_FILES,
    GOLDEN_SUITE_FILE,
    FakeSuiteEngine,
    RowsCall,
    golden_outputs,
)
from tests.support.eval_golden import (
    ALPHA_ENG,
    GOLDEN_DIRECTIONS,
    GOLDEN_IDS,
)
from tests.support.model_exports import WrittenModel, write_model_export
from tests.support.robust_cli import (
    FakeRobustEngine,
    answers_from_golden,
    model_info,
)
from tests.support.robust_cli import install as install_robust
from tests.support.robust_cli import (
    make_workspace as make_robust_workspace,
)

runner = CliRunner()

SYSTEMS_MODULE: Final = "rinchen.cli.eval_systems"
MANIFEST_FILE: Final = "manifest.json"
RUN_FILE: Final = "run.json"
OUTPUTS_FILE: Final = "hypotheses.jsonl"
FIRST: Final = "khk_Cyrl-eng_Latn"
SECOND: Final = "eng_Latn-khk_Cyrl"
THIRD: Final = "eng_Latn-rus_Cyrl"
MARKER: Final = "SECRET-MARKER-91"
SPM_BYTES: Final = b"a sentencepiece model of the test"

# The stdout of the golden `alpha` run (Task 2's table: the intervals are the document unit).
ALPHA_LINES: Final = (
    "suite: golden",
    "system: alpha",
    "purpose: report",
    f"{FIRST}: chrF++ 73.08 [54.91, 85.48] (10 segments, 10 ok)",
    f"{SECOND}: chrF++ 68.06 [50.67, 80.98] (10 segments, 10 ok)",
)


def invoke(args: Sequence[str], *, env: dict[str, str] | None = None) -> Result:
    """Run `rinchen` with `args` in this process; standard output and standard error stay
    apart (`result.stdout`, `result.stderr`)."""
    return runner.invoke(app, list(args), env=env)


def run_dir_of(out_dir: Path, system: str, suite: str = "golden") -> Path:
    """Where a run of `system` over `suite` writes: `<out>/<suite>/<system>`."""
    return out_dir / suite / system


def read_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


# ---- the seams ---------------------------------------------------------------------------------


@dataclass
class FakeRunEngine(FakeSuiteEngine):
    """The engine as the command sees it (`info`, `translate`, `close`) with an `error` that
    `translate` raises instead of answering (after recording the call) and the `events` of the
    run in order."""

    error: BaseException | None = None
    events: list[str] = field(default_factory=list)

    def translate(
        self, rows: Sequence[Row], options: DecodeOptions | None = None
    ) -> Any:
        self.events.append("translate")
        if self.error is not None:
            self.calls.append(RowsCall(list(rows), options))
            raise self.error
        return super().translate(rows, options)

    def close(self) -> None:
        super().close()
        self.events.append("close")


def alpha_engine(info: ModelInfo | None = None, **options: Any) -> FakeRunEngine:
    """A fake engine that answers the twenty requests of the golden suite with what `alpha`
    wrote, whatever the rows are."""
    from tests.support.eval_fakes import answering

    options.setdefault("answer", answering(list(golden_outputs("alpha").values())))
    engine = FakeRunEngine(**options)
    if info is not None:
        engine.info = info
    return engine


@dataclass(frozen=True)
class OpenCall:
    """What the command asked `open_engine` for."""

    model: Path
    load: LoadOptions


@dataclass
class Seams:
    """The two seams: their calls, the engine and the translator they hand out and the events
    of both (`open`, `build`, then what the engine saw)."""

    engine: Any
    translator: Any
    opens: list[OpenCall] = field(default_factory=list)
    builds: list[GeneratorConfig] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def install(
    monkeypatch: pytest.MonkeyPatch,
    engine: Any = None,
    translator: Any = None,
    *,
    open_error: BaseException | None = None,
    build_error: BaseException | None = None,
) -> Seams:
    """Replace `open_engine` and `build_translator` of `rinchen.cli.eval_systems` by seams
    that record their calls and hand out `engine` and `translator`, or raise the given error.
    The engine's own `events` list is the seams' list."""
    seams = Seams(engine, translator)
    if hasattr(engine, "events"):
        seams.events = engine.events

    def open_engine(model: Path, load: LoadOptions) -> Any:
        seams.events.append("open")
        seams.opens.append(OpenCall(model, load))
        if open_error is not None:
            raise open_error
        return seams.engine

    def build_translator(config: GeneratorConfig) -> Any:
        seams.events.append("build")
        seams.builds.append(config)
        if build_error is not None:
            raise build_error
        return seams.translator

    monkeypatch.setattr(f"{SYSTEMS_MODULE}.open_engine", open_engine)
    monkeypatch.setattr(f"{SYSTEMS_MODULE}.build_translator", build_translator)
    return seams


# ---- the baselines -----------------------------------------------------------------------------


def family_tokens(family: str) -> list[str]:
    """What the vocabulary of a converted model of `family` holds for the nine languages."""
    codes = NLLB_CODES if family == "nllb" else MADLAD_CODES
    return ["<unk>", "<s>", "</s>", *codes.values()]


def write_family_directory(
    directory: Path, tokens: Sequence[str], *, spm_name: str = "sentencepiece.bpe.model"
) -> Path:
    """A directory that stands in for a converted baseline: the converter's three files, with
    the vocabulary `tokens`, and the SentencePiece model. Returns the model directory."""
    directory.mkdir(parents=True)
    (directory / "model.bin").write_bytes(
        b"fabricated ctranslate2 weights of a baseline"
    )
    (directory / "shared_vocabulary.json").write_text(json.dumps(list(tokens)), "utf-8")
    (directory / "config.json").write_text(json.dumps({"eos_token": "</s>"}), "utf-8")
    (directory / spm_name).write_bytes(SPM_BYTES)
    return directory


def generators_text(root: Path) -> str:
    """The generators configuration of the workspace: two CTranslate2 baselines over good
    directories, one over a vocabulary that lacks the languages, one on `auto`, and two chat
    servers; every sampling setting is the synthetic-data default the run must override."""
    nllb, madlad, bad = root / "nllb", root / "madlad", root / "nllb-bad"
    ct2 = "    kind: ct2\n    n: 4\n    beam_size: 4\n    max_tokens: 128\n"
    return (
        "generators:\n"
        f"  nllb_toy:\n{ct2}    family: nllb\n    model_dir: {nllb}\n"
        f"    spm_model: {nllb / 'sentencepiece.bpe.model'}\n"
        f"  madlad_toy:\n{ct2}    family: madlad\n    model_dir: {madlad}\n"
        f"    spm_model: {madlad / 'sentencepiece.bpe.model'}\n"
        f"  nllb_bad_vocabulary:\n{ct2}    family: nllb\n    model_dir: {bad}\n"
        f"    spm_model: {bad / 'sentencepiece.bpe.model'}\n"
        f"  nllb_auto:\n{ct2}    family: nllb\n    device: auto\n    model_dir: {nllb}\n"
        f"    spm_model: {nllb / 'sentencepiece.bpe.model'}\n"
        "  vllm_toy:\n"
        "    kind: vllm\n    base_url: http://127.0.0.1:8000/v1\n    model: Hy-MT2-7B\n"
        "    n: 4\n    temperature: 0.7\n    top_p: 0.95\n    max_tokens: 200\n"
        "  llamacpp_toy:\n"
        "    kind: llamacpp\n    base_url: http://127.0.0.1:8080/v1\n    n: 4\n"
        "    temperature: 0.7\n    top_p: 0.95\n"
    )


METRICS_TEXT: Final = """\
bootstrap:
  resamples: 1000
  seed: 20261110
  alpha: 0.05
primary_metric: chrf_pp
comparison_metrics: [chrf_pp, bleu]
baseline_system: null
targets: []
"""


# ---- the workspace -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalWorkspace:
    """The files of one run; `out_dir` does not exist."""

    root: Path
    suite: Path
    alpha: Path
    beta: Path
    metrics: Path
    generators: Path
    export: WrittenModel
    out_dir: Path

    @property
    def run_dir(self) -> Path:
        """Where the golden suite's run of `alpha` (the default name of a file `alpha.jsonl`
        copy) is written."""
        return run_dir_of(self.out_dir, "alpha")

    def base(
        self, *extra: str, suite: Path | None = None, out_dir: Path | None = None
    ) -> list[str]:
        return [
            "eval",
            "run",
            "--suite",
            str(self.suite if suite is None else suite),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def hypotheses_args(
        self,
        *extra: str,
        hypotheses: Path | None = None,
        suite: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        """`eval run --suite S --out-dir O --hypotheses H` and `extra`; the default system name
        is the stem of the file, `alpha`."""
        return self.base(
            "--hypotheses",
            str(self.alpha if hypotheses is None else hypotheses),
            *extra,
            suite=suite,
            out_dir=out_dir,
        )

    def model_args(
        self,
        *extra: str,
        model: Path | None = None,
        suite: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        return self.base(
            "--model",
            str(self.export.directory if model is None else model),
            *extra,
            suite=suite,
            out_dir=out_dir,
        )

    def generator_args(
        self,
        name: str,
        *extra: str,
        suite: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        return self.base(
            "--generator",
            name,
            "--generators-config",
            str(self.generators),
            *extra,
            suite=suite,
            out_dir=out_dir,
        )

    def engine(self, **options: Any) -> FakeRunEngine:
        """A fake engine that answers with `alpha`'s outputs and whose `info` is that of the
        fabricated export, trained for the two golden directions."""
        return alpha_engine(model_info(self.export), **options)


def make_workspace(root: Path) -> EvalWorkspace:
    """The golden suite and outputs, a fabricated export trained for the two golden
    directions, the two configurations and two family directories under `root`."""
    inputs = root / "in"
    inputs.mkdir(parents=True)
    suite = shutil.copyfile(GOLDEN_SUITE_FILE, inputs / "golden_suite.json")
    alpha = shutil.copyfile(GOLDEN_OUTPUTS_FILES["alpha"], inputs / "alpha.jsonl")
    beta = shutil.copyfile(GOLDEN_OUTPUTS_FILES["beta"], inputs / "beta.jsonl")
    write_family_directory(root / "nllb", family_tokens("nllb"))
    write_family_directory(root / "madlad", family_tokens("madlad"))
    write_family_directory(root / "nllb-bad", ["<unk>", "<s>", "</s>", "<2eng_Latn>"])
    metrics = inputs / "metrics.yaml"
    metrics.write_text(METRICS_TEXT, "utf-8")
    generators = inputs / "generators.yaml"
    generators.write_text(generators_text(root), "utf-8")
    return EvalWorkspace(
        root=root,
        suite=Path(suite),
        alpha=Path(alpha),
        beta=Path(beta),
        metrics=metrics,
        generators=generators,
        export=write_model_export(root / "model", directions=sorted(GOLDEN_DIRECTIONS)),
        out_dir=root / "out",
    )


def three_language_suite(path: Path) -> Path:
    """The golden suite with a Russian sentence on every row and a third direction,
    `eng_Latn-rus_Cyrl`, which a model trained for the two golden ones cannot run."""
    document = json.loads(GOLDEN_SUITE_FILE.read_text("utf-8"))
    document["directions"].append(THIRD)
    for row in document["rows"]:
        row["text"]["rus_Cyrl"] = "русский " + row["id"]
    path.write_text(json.dumps(document, ensure_ascii=False), "utf-8")
    return path


def alpha_texts(direction: str) -> tuple[str, ...]:
    """What `alpha` wrote for `direction` of the golden suite."""
    return tuple(
        output.text
        for key, output in golden_outputs("alpha").items()
        if key.direction == direction
    )


def outputs_file(path: Path, texts: Mapping[str, Sequence[str]]) -> Path:
    """A hypotheses file holding `texts` (a text per row, by direction) as the product writes
    it, every status `ok`."""
    from rinchen.eval.outputs import Output

    outputs = {
        OutputKey(direction, row_id): Output(text, "ok")
        for direction, direction_texts in texts.items()
        for row_id, text in zip(GOLDEN_IDS, direction_texts, strict=True)
    }
    path.write_bytes(outputs_to_bytes(outputs))
    return path


def one_direction_file(path: Path) -> Path:
    """A hypotheses file that holds only the direction `khk_Cyrl-eng_Latn` of `alpha`."""
    return outputs_file(path, {FIRST: ALPHA_ENG})


# ---- D12's files -------------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessDirectories:
    """The output directories of D12's `rinchen eval robustness` over its golden suite: one
    run of a model (the fabricated export of the workspace) and one of a hypotheses file
    named `alpha`."""

    model: Path
    named_alpha: Path
    other_model: Path


def robustness_directories(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> RobustnessDirectories:
    """Run D12's command three times: for the export of the workspace with a fake engine, for a
    hypotheses file under the name `alpha` and for another fabricated export. Each is exit 0."""
    robust = make_robust_workspace(workspace.root / "robust-ws")
    model_out = workspace.root / "robust-model"
    engine = FakeRobustEngine(
        info=model_info(workspace.export), answer=answers_from_golden
    )
    install_robust(monkeypatch, engine)
    first = invoke(
        [
            "eval",
            "robustness",
            "--suite",
            str(robust.suite),
            "--model",
            str(workspace.export.directory),
            "--out-dir",
            str(model_out),
        ]
    )
    assert first.exit_code == 0, first.output
    named_out = workspace.root / "robust-alpha"
    second = invoke(
        [
            "eval",
            "robustness",
            "--suite",
            str(robust.suite),
            "--hypotheses",
            str(robust.hypotheses),
            "--system",
            "alpha",
            "--out-dir",
            str(named_out),
        ]
    )
    assert second.exit_code == 0, second.output
    other = write_model_export(
        workspace.root / "other-model",
        directions=sorted(GOLDEN_DIRECTIONS),
        name="other",
    )
    other_engine = FakeRobustEngine(info=model_info(other), answer=answers_from_golden)
    install_robust(monkeypatch, other_engine)
    other_out = workspace.root / "robust-other"
    third = invoke(
        [
            "eval",
            "robustness",
            "--suite",
            str(robust.suite),
            "--model",
            str(other.directory),
            "--out-dir",
            str(other_out),
        ]
    )
    assert third.exit_code == 0, third.output
    return RobustnessDirectories(
        model_out, named_alpha=named_out, other_model=other_out
    )


def rewrite_robustness_file(
    directory: Path, name: str, edit: Callable[[dict[str, Any]], None]
) -> None:
    """Change the JSON document `name` of a robustness directory with `edit` and make the
    manifest of that directory vouch for the new bytes, so that only the document itself is at
    fault."""
    from rinchen.manifest import store

    path = directory / name
    document = json.loads(path.read_text("utf-8"))
    edit(document)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "utf-8",
    )
    manifest = store.read(directory)
    assert manifest is not None
    manifest.outputs = [
        item.model_copy(
            update={
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
        if Path(item.path).name == name
        else item
        for item in manifest.outputs
    ]
    store.write(directory, manifest)


# ---- the scores of the seam --------------------------------------------------------------------


def scores_bytes(
    metric: str,
    suite_sha256: str,
    hypotheses_sha256: str,
    value: Callable[[OutputKey], float] | float = 0.75,
) -> bytes:
    """A scores file for every segment of the golden `alpha` outputs (both directions), each
    scored `value` (a number or a function of the key)."""
    from rinchen.eval.segment_scores import scores_to_bytes

    scores = {
        key: (value(key) if callable(value) else value)
        for key in golden_outputs("alpha")
    }
    return scores_to_bytes(metric, suite_sha256, hypotheses_sha256, scores)


def golden_fixture(name: str) -> Path:
    return FIXTURES / name


def assert_refused(
    workspace: EvalWorkspace, seams: Seams, result: Any, code: int
) -> None:
    """Refused with `code`: nothing on standard output, the engine never opened, no translator
    built and no output directory made."""
    assert result.exit_code == code, result.output
    assert result.stdout == ""
    assert seams.opens == []
    assert seams.builds == []
    assert not workspace.out_dir.exists()
