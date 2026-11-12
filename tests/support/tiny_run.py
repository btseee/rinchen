"""A recording fake runner for the CLI stages of the tiny pipeline, and the inputs the stages are run
over (D15, brief G4.4).

`RecordingRunner` is a `CommandRunner` that starts no process. It records every call (the argument
list, the working directory, the environment, the log and the timeout) and plays the command: it
checks the inputs the real command would need (a file that is not there, a corpus that is not a
corpus, a config the real loader refuses, an output directory that exists, a missing
`--prepare-only`, `--no-qe` or `--tag-placement src`) and refuses with the exit code 2 and a line in the
log, so a stage that wires a path wrongly fails on it instead of passing; then it writes the files
the real command writes, from the recordings of the spike under `tests/fixtures/tiny_run/` (the
`parity.json`, the `counts.json`, the five lines of `convert-script eval`) and from the real inputs
(the corpus rows it is given, the matrix it loads, the steps of the config it is launched with).

A test makes it fail a command (an exit code, a timeout, a refusal to start, an exception) by the
command's key, `<group> <command>` and for an export also its quantization: `spm train`,
`export convert int8`; `after` runs a function once the command has written its files (a test
removes one). A command that is not played raises `UnscriptedCall`. `noisy` fills every log with text
that is not what any command prints, to show that no stage reads one; `contact_server` makes the fake
`synth run` do what the real child does, one chat request to the listener of the config for each text;
`vocab_pieces` and `launched_steps` make the files a command writes differ from the config it was given
(the pieces of the vocabulary, the steps of the launch), to show where a stage's facts come from.

`copy_inputs` makes the output directory of a test hold what the library stages of the pipeline
wrote (the corpus and the filtered corpus, produced once by the real stages), `stage_function` binds
the seed the pipeline would hand the stages that need one, and `ready_for` and `run_one` run the
stages before the one under test over a runner that plays every command. `library_outputs` and `out_dir`
are the fixtures of the tests that use them: a module imports both.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.client
import json
import shutil
import sys
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import polars as pl
import pytest
import yaml

from rinchen.config.ablation import id_number, load_ablation
from rinchen.config.loader import REPO_ROOT
from rinchen.config.synth import VllmConfig, load_synth_config
from rinchen.config.tiny import TinyConfig
from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.schema import MONO_SCHEMA, corpus_frame
from rinchen.corpus.writers import write_corpus
from rinchen.errors import ConfigError, DataError
from rinchen.maintain.commands import CommandResult
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.script_conv.char_model import load_char_model_config
from rinchen.tiny import data_stages, model_stages, model_stages_rest
from rinchen.tiny.context import StageContext, StageResult, stage_dir
from rinchen.train.variants import expand_variants
from tests.support.tiny_stage_inputs import (
    make_context,
    offline_in,
    shipped_config,
    shipped_fixtures,
)

RECORDINGS: Final = REPO_ROOT / "tests" / "fixtures" / "tiny_run"
SEED: Final = 1234
VARIANT: Final = "t1_bt0.2_src_s1234"
NOISE: Final = "WARNING 3 4 five cer nan {broken json\n"
FLAGS: Final = frozenset({"--no-qe", "--prepare-only"})
WEIGHTS: Final = "model.00.safetensors"
CREATED: Final = datetime(2026, 11, 11, tzinfo=UTC)
MODEL_STAGES: Final = (
    "spm",
    "converter",
    "synth",
    "prepare",
    "train",
    "average",
    "export",
    "parity",
)
# The command (by key) that each stage runs, in order: what a test expects of `RecordingRunner.keys`.
COMMANDS_OF: Final[Mapping[str, tuple[str, ...]]] = {
    "spm": ("spm train", "spm check"),
    "converter": (
        "convert-script build-data",
        "convert-script train",
        "convert-script export",
        "convert-script eval",
    ),
    "synth": ("synth run",),
    "prepare": ("train ablate",),
    "train": ("train launch",),
    "average": ("train average",),
    "export": ("export convert float32", "export convert int8"),
    "parity": ("export parity",),
}


class UnscriptedCall(AssertionError):
    """The code under test ran a command that the runner does not play."""


class Refused(Exception):
    """The command the runner plays would have refused its inputs (exit code 2)."""


@dataclass(frozen=True, slots=True)
class Call:
    """One call of `run`, as it was made, with the key of the command it was."""

    key: str
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    log: Path
    timeout: int


class Invocation:
    """The words, the positional arguments and the options of an argument list."""

    def __init__(self, argv: Sequence[str]) -> None:
        self.words = (argv[3], argv[4])
        tail = list(argv[5:])
        self.positionals: list[str] = []
        self.options: dict[str, list[str]] = {}
        index = 0
        while index < len(tail) and not tail[index].startswith("--"):
            self.positionals.append(tail[index])
            index += 1
        while index < len(tail):
            name = tail[index]
            index += 1
            if name in FLAGS or index >= len(tail) or tail[index].startswith("--"):
                self.options.setdefault(name, []).append("")
            else:
                self.options.setdefault(name, []).append(tail[index])
                index += 1

    def has(self, name: str) -> bool:
        return name in self.options

    def one(self, name: str) -> str:
        values = self.options.get(name, [])
        if len(values) != 1:
            raise Refused(f"{name} must be given once")
        return values[0]

    def many(self, name: str) -> list[str]:
        return self.options.get(name, [])

    def path(self, name: str) -> Path:
        return Path(self.one(name))

    def number(self, name: str, minimum: float = 1) -> float:
        try:
            value = float(self.one(name))
        except ValueError as error:
            raise Refused(f"{name} is not a number") from error
        if value < minimum:
            raise Refused(f"{name} is too small")
        return value


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise Refused(f"{path.name} is not a file")
    return path


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise Refused(f"{path.name} is not a directory")
    return path


def require_new(path: Path) -> Path:
    if path.exists():
        raise Refused(f"{path.name} exists")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_text(name: str) -> str:
    return (RECORDINGS / name).read_text(encoding="utf-8")


class RecordingRunner:
    """A `CommandRunner` that plays the commands of the CLI stages; see the module docstring."""

    def __init__(
        self,
        *,
        fail: Mapping[str, int] | None = None,
        time_out: Collection[str] = (),
        not_started: Collection[str] = (),
        raises: Mapping[str, BaseException] | None = None,
        logs: Mapping[str, str] | None = None,
        write_on_failure: Collection[str] = (),
        after: Mapping[str, Callable[[Invocation], None]] | None = None,
        parity: Mapping[str, Any] | None = None,
        vocab_pieces: int | None = None,
        launched_steps: float | None = None,
        noisy: bool = False,
        contact_server: bool = False,
    ) -> None:
        self.calls: list[Call] = []
        self.requests_made = 0
        self._vocab_pieces = vocab_pieces
        self._launched_steps = launched_steps
        self._fail = dict(fail or {})
        self._time_out = set(time_out)
        self._not_started = set(not_started)
        self._raises = dict(raises or {})
        self._logs = dict(logs or {})
        self._write_on_failure = set(write_on_failure)
        self._after = dict(after or {})
        self._parity = dict(parity or {})
        self._noisy = noisy
        self._contact_server = contact_server
        self._log_text = ""

    @property
    def keys(self) -> list[str]:
        return [call.key for call in self.calls]

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log: Path,
        timeout: int,
    ) -> CommandResult:
        if list(argv[:3]) != [_python(), "-m", "rinchen"]:
            raise AssertionError(
                "a command of the pipeline is a call of python -m rinchen"
            )
        invocation = Invocation(argv)
        key = self._key(invocation)
        self.calls.append(Call(key, tuple(argv), cwd, dict(env), log, timeout))
        assert cwd.is_dir(), "the working directory of a command exists"
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log_text = self._logs.get(key, self._default_log(key))
        try:
            return self._play(key, invocation, log)
        finally:
            log.write_text(self._log_text, encoding="utf-8")

    def _play(self, key: str, call: Invocation, log: Path) -> CommandResult:
        if key in self._raises:
            raise self._raises[key]
        if key in self._not_started:
            return CommandResult(exit_code=None, timed_out=False, started=False)
        if key in self._time_out:
            return CommandResult(exit_code=None, timed_out=True)
        failing = key in self._fail and key not in self._write_on_failure
        if failing:
            return CommandResult(exit_code=self._fail[key], timed_out=False)
        player = self._players().get(key)
        if player is None:
            raise UnscriptedCall(key)
        try:
            player(call)
        except Refused as refusal:
            self._log_text += f"error: {refusal}\n"
            return CommandResult(exit_code=2, timed_out=False)
        if key in self._after:
            self._after[key](call)
        if key in self._fail:
            return CommandResult(exit_code=self._fail[key], timed_out=False)
        return CommandResult(exit_code=0, timed_out=False)

    @staticmethod
    def _key(call: Invocation) -> str:
        key = " ".join(call.words)
        if key == "export convert":
            return f"{key} {call.options.get('--quantization', ['?'])[0]}"
        return key

    def _default_log(self, key: str) -> str:
        if key == "convert-script eval":
            return recorded_text("conv_eval.log")
        return NOISE if self._noisy else "done\n"

    def _players(self) -> dict[str, Callable[[Invocation], None]]:
        export = self._export
        return {
            "spm train": self._spm_train,
            "spm check": self._spm_check,
            "convert-script build-data": self._converter_build,
            "convert-script train": self._converter_train,
            "convert-script export": self._converter_export,
            "convert-script eval": self._converter_eval,
            "synth run": self._synth,
            "train ablate": self._ablate,
            "train launch": self._launch,
            "train average": self._average,
            "export convert float32": export,
            "export convert int8": export,
            "export parity": self._parity_run,
        }

    # ---- spm -----------------------------------------------------------------------------------

    def _spm_train(self, call: Invocation) -> None:
        config = yaml.safe_load(require_file(call.path("--config")).read_text("utf-8"))
        texts = sorted(require_dir(Path(config["corpus_dir"])).glob("*.txt"))
        if not texts or any(not path.read_text("utf-8") for path in texts):
            raise Refused("a language has no text")
        prefix = Path(config["output_prefix"])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{prefix}.model").write_bytes(b"sentencepiece model\n")
        size = (
            config["vocab_size"] if self._vocab_pieces is None else self._vocab_pieces
        )
        pieces = [f"piece{n}\t-0.0\n" for n in range(size)]
        Path(f"{prefix}.vocab").write_text("".join(pieces), encoding="utf-8")

    def _spm_check(self, call: Invocation) -> None:
        require_file(call.path("--model"))
        require_dir(call.path("--corpus-dir"))

    # ---- the converter -------------------------------------------------------------------------

    def _converter_build(self, call: Invocation) -> None:
        require_file(call.path("--pairs"))
        out = require_new(call.path("--out-dir"))
        out.mkdir(parents=True)
        (out / "manifest.json").write_text("{}\n", encoding="utf-8")

    def _converter_train(self, call: Invocation) -> None:
        if call.one("--profile") != "tiny":
            raise Refused("the profile is not the CPU one")
        require_file(require_dir(call.path("--data-dir")) / "manifest.json")
        try:
            load_char_model_config(require_file(call.path("--spm-config")))
        except (ConfigError, DataError) as error:
            raise Refused("the character config is not valid") from error
        call.number("--threads")
        call.number("--timeout")
        run = call.path("--run-dir")
        run.mkdir(parents=True, exist_ok=True)
        (run / "converter_char.model").write_bytes(b"character model\n")
        (run / "converter.yaml").write_text("seed: 1\n", encoding="utf-8")

    def _converter_export(self, call: Invocation) -> None:
        run = require_dir(call.path("--run-dir"))
        require_file(run / "converter_char.model")
        call.number("--timeout")
        out = require_new(call.path("--out-dir"))
        out.mkdir(parents=True)
        (out / "model.bin").write_bytes(b"converter model\n")

    def _converter_eval(self, call: Invocation) -> None:
        require_file(require_dir(call.path("--model-dir")) / "model.bin")
        require_file(call.path("--pairs"))

    # ---- synth ---------------------------------------------------------------------------------

    def _synth(self, call: Invocation) -> None:
        if not call.has("--no-qe"):
            raise Refused("the quality floor needs a scorer that is not here")
        mono = require_file(call.path("--input"))
        frame = pl.read_parquet(mono)
        if frame.schema != MONO_SCHEMA:
            raise Refused("the input is not a monolingual corpus")
        try:
            config = load_synth_config(require_file(call.path("--config")))
        except ConfigError as error:
            raise Refused("the synthetic config is not valid") from error
        job = config.jobs.get(call.positionals[0]) if call.positionals else None
        if job is None:
            raise Refused("the job is not in the config")
        call.number("--seed", 0)
        generator = config.generators[job.generator]
        assert isinstance(generator, VllmConfig)
        out = call.path("--out-dir")
        out.mkdir(parents=True, exist_ok=True)
        english = {row.khk: row.eng for row in shipped_fixtures().mono}
        texts = frame["text"].to_list()
        if self._contact_server:
            english = {
                text: self._ask(generator.base_url, text, generator.n) for text in texts
            }
        rows = [
            {
                "src_lang": job.other_lang,
                "tgt_lang": job.mono_lang,
                "src": english[text],
                "tgt": text,
                "origin": "bt",
                "corpus": "tiny-synth",
                "split": "train",
            }
            for text in texts
            if english.get(text)
        ]
        write_corpus(corpus_frame(rows, created=CREATED), out / "corpus.parquet")
        (out / "counts.json").write_text(recorded_text("counts.json"), encoding="utf-8")

    def _ask(self, base_url: str, text: str, n: int) -> str:
        """What the real child does: one chat request to the server of the config."""
        parts = urlsplit(base_url)
        connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
        body = json.dumps(
            {
                "model": "tiny-oracle",
                "messages": [{"role": "user", "content": f"Translate.\n\n{text}"}],
                "n": n,
            }
        )
        try:
            connection.request(
                "POST",
                f"{parts.path}/chat/completions",
                body,
                {"Content-Type": "application/json"},
            )
            reply = json.loads(connection.getresponse().read())
        finally:
            connection.close()
        self.requests_made += 1
        answer: str = reply["choices"][0]["message"]["content"]
        return answer

    # ---- the training ----------------------------------------------------------------------------

    def _ablate(self, call: Invocation) -> None:
        if not call.has("--prepare-only"):
            raise Refused("a real run would train the variants")
        try:
            matrix = load_ablation(require_file(call.path("--matrix")))
            frames = [
                scan_corpus(require_file(Path(path))).collect()
                for path in call.many("--corpus")
            ]
        except (ConfigError, DataError) as error:
            raise Refused("an input is not valid") from error
        if not frames:
            raise Refused("no corpus")
        require_file(call.path("--spm-model"))
        call.number("--threads")
        out = call.path("--out-dir")
        rows = pl.concat(frames).filter(pl.col("split").cast(pl.String) == "train")
        for variant in expand_variants(matrix):
            self._prepare_variant(out, matrix.base, matrix.overrides, variant, rows)

    @staticmethod
    def _prepare_variant(
        out: Path,
        base: Path,
        overrides: Mapping[str, Any],
        variant: Any,
        rows: pl.DataFrame,
    ) -> None:
        config = yaml.safe_load(base.read_text("utf-8"))
        for section, values in overrides.items():
            config[section] = {**config.get(section, {}), **values}
        directory = out / "variants" / variant.id
        config["seed"] = variant.seed
        config["training"]["model_path"] = str(directory / "run" / "ckpt")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        direction = pl.col("src_lang") + "-" + pl.col("tgt_lang")
        is_bt = pl.col("origin").cast(pl.String) == "bt"
        auth_counts = dict(
            rows.filter(~is_bt).group_by(direction.alias("d")).len().iter_rows()
        )
        bt_counts = dict(
            rows.filter(is_bt).group_by(direction.alias("d")).len().iter_rows()
        )
        bt_directory = (
            out / "data" / "bt" / f"r{id_number(variant.bt_ratio)}_s{variant.seed}"
        )
        for name, count in auth_counts.items():
            _sidecar(out / "data" / "auth", f"{name}.auth", count)
            used = min(bt_counts.get(name, 0), round(variant.bt_ratio * count))
            if used:
                _sidecar(bt_directory, f"{name}.bt", used)

    def _launch(self, call: Invocation) -> None:
        config = yaml.safe_load(require_file(call.path("--config")).read_text("utf-8"))
        run = call.path("--run-dir")
        training = config["training"]
        if Path(training["model_path"]).parent != run:
            raise Refused("the model path is not in the run directory")
        call.number("--threads")
        call.number("--timeout")
        steps = int(training["train_steps"])
        saves = range(steps, 0, -int(training["save_checkpoint_steps"]))
        for step in list(saves)[: int(training["keep_checkpoint"])]:
            weights = run / "ckpt" / f"step_{step}"
            weights.mkdir(parents=True, exist_ok=True)
            (weights / WEIGHTS).write_bytes(f"weights of step {step}\n".encode())
        manifest = Manifest(
            stage="train_launch",
            created="2026-11-11T00:00:00+00:00",
            code_version="fake",
            config_sha256="0" * 64,
            seed=int(config["seed"]),
            metrics={
                "steps": float(
                    steps if self._launched_steps is None else self._launched_steps
                )
            },
            status={"launch": "ok"},
        )
        store.write(run, manifest)

    def _average(self, call: Invocation) -> None:
        checkpoints = require_dir(call.path("--checkpoint-dir"))
        last = int(call.number("--last", 2))
        steps = sorted(
            (path for path in checkpoints.glob("step_*") if (path / WEIGHTS).is_file()),
            key=lambda path: int(path.name.removeprefix("step_")),
        )[-last:]
        if len(steps) < last:
            raise Refused("fewer checkpoints than --last")
        out = require_new(checkpoints.parent / "averaged")
        out.mkdir(parents=True)
        joined = ",".join(path.name for path in steps)
        (out / WEIGHTS).write_bytes(f"average of {joined}\n".encode())

    # ---- the export and the parity ---------------------------------------------------------------

    def _export(self, call: Invocation) -> None:
        if call.options.get("--tag-placement") != ["src"]:
            raise Refused("the tag is placed in the source")
        quantization = call.one("--quantization")
        if quantization not in ("float32", "int8"):
            raise Refused("a quantization that a CPU runs")
        weights = require_file(require_dir(call.path("--checkpoint")) / WEIGHTS)
        require_file(call.path("--spm-model"))
        out = require_new(call.path("--out-dir"))
        out.mkdir(parents=True)
        model = f"ct2 {quantization} {digest(weights)}\n"
        (out / "model.bin").write_text(model, encoding="utf-8")

    def _parity_run(self, call: Invocation) -> None:
        require_file(require_dir(call.path("--checkpoint")) / WEIGHTS)
        require_file(require_dir(call.path("--export")) / "model.bin")
        try:
            scan_corpus(require_file(call.path("--corpus"))).collect()
        except DataError as error:
            raise Refused("the corpus is not valid") from error
        call.number("--max-lines")
        call.number("--threads")
        call.number("--timeout")
        document = {**json.loads(recorded_text("parity.json")), **self._parity}
        out = call.path("--out-dir")
        out.mkdir(parents=True, exist_ok=True)
        (out / "parity.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )


def _python() -> str:
    return sys.executable


def _sidecar(directory: Path, name: str, rows: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    record = {"rows": rows, "src_sha256": "0" * 64, "tgt_sha256": "0" * 64}
    (directory / f"{name}.json").write_text(
        json.dumps(record, sort_keys=True), encoding="utf-8"
    )


# ---- the inputs and the stages -------------------------------------------------------------------


def write_library_stages(out_dir: Path) -> None:
    """Run the real corpus and filter stages into `out_dir`, as the pipeline would have."""
    context = make_context(out_dir)
    for run in (data_stages.run_corpus, data_stages.run_filter):
        result = run(context)
        assert result.record.state == "ok"


def copy_inputs(source: Path, out_dir: Path) -> None:
    """Make `out_dir` hold what `write_library_stages` wrote into `source`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage in ("corpus", "filter"):
        directory = stage_dir(source, stage)
        assert directory is not None
        shutil.copytree(directory, out_dir / directory.name)


def stage_function(
    stage_id: str, seed: int = SEED
) -> Callable[[StageContext], StageResult]:
    """The function of a model stage as the pipeline would call it: the seed bound where it is needed."""
    seeded: dict[str, Callable[[StageContext, int], StageResult]] = {
        "spm": model_stages.run_spm,
        "synth": model_stages.run_synth,
        "prepare": model_stages_rest.run_prepare,
    }
    plain: dict[str, Callable[[StageContext], StageResult]] = {
        "converter": model_stages.run_converter,
        "train": model_stages_rest.run_train,
        "average": model_stages_rest.run_average,
        "export": model_stages_rest.run_export,
        "parity": model_stages_rest.run_parity,
    }
    if stage_id in seeded:
        function = seeded[stage_id]
        return lambda ctx: function(ctx, seed)
    return plain[stage_id]


def run_stages(
    context: StageContext, upto: str, seed: int = SEED
) -> dict[str, StageResult]:
    """Run the model stages in order, up to and including `upto`, stopping at the first that is not ok."""
    results: dict[str, StageResult] = {}
    for stage_id in MODEL_STAGES:
        results[stage_id] = stage_function(stage_id, seed)(context)
        if stage_id == upto or results[stage_id].record.state != "ok":
            break
    return results


# ---- what the tests of the stages share ------------------------------------------------------------


@pytest.fixture(scope="module")
def library_outputs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The output of the real corpus and filter stages, made once for a module."""
    base = tmp_path_factory.mktemp("library")
    with offline_in(base):
        write_library_stages(base)
    yield base


@pytest.fixture
def out_dir(tmp_path: Path, library_outputs: Path) -> Path:
    """An output directory that holds the output of the library stages, for one test."""
    out = tmp_path / "out"
    copy_inputs(library_outputs, out)
    return out


def context(
    out: Path,
    *,
    runner: RecordingRunner | None = None,
    config: TinyConfig | None = None,
    remaining: Callable[[], float] | None = None,
) -> StageContext:
    """A context over `out` whose runner plays every command, unless another is given."""
    base = make_context(out, config=config)
    return dataclasses.replace(
        base,
        runner=runner or RecordingRunner(),
        remaining_seconds=remaining or base.remaining_seconds,
    )


def ready_for(
    out: Path,
    stage: str,
    *,
    config: TinyConfig | None = None,
    remaining: Callable[[], float] | None = None,
    **options: Any,
) -> tuple[StageContext, RecordingRunner]:
    """The context for `stage` over a runner with `options`, after the stages before it have run over
    a runner that plays every command (and the budget `remaining`, if any, only for `stage`)."""
    ctx = context(out, config=config)
    for before in MODEL_STAGES[: MODEL_STAGES.index(stage)]:
        result = stage_function(before)(ctx)
        assert result.record.state == "ok", before
    runner = RecordingRunner(**options)
    return dataclasses.replace(
        ctx, runner=runner, remaining_seconds=remaining or ctx.remaining_seconds
    ), runner


def run_one(
    out: Path, stage: str, **options: Any
) -> tuple[StageResult, RecordingRunner]:
    """Run `stage` after the stages before it, over a runner with `options`."""
    ctx, runner = ready_for(out, stage, **options)
    return stage_function(stage)(ctx), runner


def tree(root: Path) -> dict[str, bytes]:
    """Every file under `root` by its path relative to it."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def changed_config(**sections: Any) -> TinyConfig:
    """The shipped profile with some of its sections replaced (not validated again)."""
    return shipped_config().model_copy(update=sections)


def at(out: Path, stage: str, *parts: str) -> Path:
    """A path under the directory of `stage`."""
    directory = stage_dir(out, stage)
    assert directory is not None
    return directory.joinpath(*parts)


def variant_dir(out: Path) -> Path:
    """The directory of the one variant the preparation makes."""
    return at(out, "prepare", "ablate", "variants", VARIANT)
