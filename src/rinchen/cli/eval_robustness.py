"""`rinchen eval robustness`: measure how a system behaves on the inputs that break translation
quietly (D12): numbers, placeholders, names, glossary terms, the script of the output and a source
in the traditional script. Parsing, the order of the steps, the hashing and the exit codes only;
the work is `rinchen.eval`'s (the suite, the metrics, `robustness.json`, `report.json`), the
engine's (`rinchen.infer.engine.Engine`) and `cli/eval_robustness_output.py`'s (the lines, the
files and the manifest).

    rinchen eval robustness --suite FILE (--model DIR | --hypotheses FILE) --out-dir DIR
        [--system NAME] [--beam-size 4] [--max-decoding-length 256]
        [--script-mask retry|always|off] [--device cpu|cuda] [--threads 1]
        [--script-converter-ct2 DIR --script-converter-spm FILE]

`--suite` is a robustness suite (`rinchen.robustness.suite/1`). With `--model` (or
`RINCHEN_MODEL`), an export `rinchen export convert` wrote, the engine translates every request
of the suite in ONE call, on the CPU unless an operator says `cuda` (never run here: unmeasured;
`auto` is refused); with `--hypotheses` the texts are read from a file instead, which is how the
output of another system enters (`hypotheses.jsonl`, which a model run writes, is such a file).
Exactly one of the two is given. `--script-converter-ct2` and `--script-converter-spm` (both or
neither, only with `--model`) attach D5's script converter: the engine writes the traditional
script through it, and every `mong` row gains a third arm, its traditional source converted to
Cyrillic and translated, so that the converter is measured beside the model.

The command measures: it has no threshold, no pass and no fail, and exits 0 whenever it measured
(the floors belong to the gate). It prints `suite: <name>`, `system: <name>`, one line per section
the suite has (`numbers: 0.9292 (892/960), excluded 0`; `n/a (no scored row)` for a null figure),
`result: <O>/robustness.json`, `report: <O>/report.json` and `manifest: <O>/manifest.json`, and
writes those files into `--out-dir` with `hypotheses.jsonl` (with `--model`) and, LAST,
`manifest.json` (stage `eval_robustness`: every input with the SHA-256 taken before it was read,
the files written, the six values and the rows of each section).

The order: the options are checked; for `--model` the export is vouched for (`verify_export`);
the suite, the hypotheses file and the files of the converter are hashed BEFORE any of them is
read; the suite is read from the bytes that were hashed; an output directory that is an input's
or holds another stage's manifest is refused; the converter and the engine open and run; only
then is anything written, the old manifest first. A hash guards against a stale or swapped
file, not against one swapped between the hash and the load of the model; the engine verifies
the export again when it opens it, and this command refuses a model whose digests are not the
ones it hashed. No text of the suite or of a hypothesis is ever in a message of this command
(names, paths, counts and the section, id and arm of a request are).

Exit codes: 0; 2 for a bad option, both or neither of `--model` and `--hypotheses`, an export
that is not vouched for, a file that changed after it was hashed, a target or a script the model
cannot write, an output directory that is an input's; 3 for a suite or a hypotheses file that is
missing, unreadable or not one, and another stage's manifest in the output directory; 4 when
CTranslate2 fails; 1 when the output cannot be written; 130 when interrupted (a run that stops
before its first write leaves the results of the last run as they were).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Final, Literal, Protocol

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.eval_robustness_output import (
    REPORT_FILE,
    STAGE,
    Measured,
    commit,
    result_lines,
)
from rinchen.cli.eval_robustness_plan import (
    HypothesesPlan,
    ModelPlan,
    RunOptions,
    plan_of,
)
from rinchen.cli.train_common import hash_input, require_out_apart_from_inputs
from rinchen.errors import ConfigError, DataError
from rinchen.eval.robust_hypotheses import (
    Translates,
    hypotheses_from_bytes,
    run_requests,
)
from rinchen.eval.robust_report import ROBUSTNESS_FILE
from rinchen.eval.robust_suite import Suite, suite_requests
from rinchen.eval.robustness import RobustnessSystem, compute_robustness
from rinchen.export.record import CT2_FILES
from rinchen.export.verified import VerifiedExport, verified_content, verify_export
from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import (
    LoadOptions,
    ModelInfo,
    ScriptConverterLike,
    ScriptMode,
)
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import refuse_foreign_manifest
from rinchen.script_conv.runtime import ScriptConverter

_SUITE: Final = "suite file"
_HYPOTHESES: Final = "hypotheses file"
_CONVERTER_FILE: Final = "converter file"
# The arm of a `mong` row that the converter made, as `hypotheses.jsonl` writes it. A text
# cannot hold the quotes unescaped, so only the structure of the file matches.
_CONVERTED_ARM: Final = re.compile(rb'"arm"\s*:\s*"converted"')


class RobustnessEngine(Translates, Protocol):
    """What the command asks of an engine: `rinchen.infer.engine.Engine` has all of it."""

    @property
    def info(self) -> ModelInfo: ...

    def close(self) -> None: ...


def open_engine(
    model: Path, load: LoadOptions, converter: ScriptConverterLike | None
) -> RobustnessEngine:
    """The engine over the export `model`, with `converter` attached when there is one: the
    one place the command touches the model (the tests replace it).

    Raises:
        ConfigError, DataError: as `Engine.open`.
    """
    return Engine.open(model, load, script_converter=converter)


def load_converter(ct2_dir: Path, spm_model: Path, device: str) -> ScriptConverterLike:
    """D5's script converter over the CTranslate2 directory `ct2_dir` and the SentencePiece
    model `spm_model`, on `device`: the other place the command touches a model (the tests
    replace it).

    Raises:
        ConfigError, DataError: as `ScriptConverter`.
    """
    return ScriptConverter(ct2_dir, spm_model, device=device)


# The docstring below is the help text: typer keeps a line break inside a paragraph, so each
# paragraph is one source line and the help wraps it by the width of the terminal.
def robustness_command(
    suite: Annotated[
        Path, typer.Option("--suite", help="The robustness suite file (JSON).")
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: the result, the report, the hypotheses and the manifest.",
        ),
    ],
    model: Annotated[
        Path | None,
        typer.Option(
            "--model",
            envvar="RINCHEN_MODEL",
            help="The export directory `rinchen export convert` made; the engine translates.",
        ),
    ] = None,
    hypotheses: Annotated[
        Path | None,
        typer.Option(
            "--hypotheses",
            help=(
                "A hypotheses file (JSON Lines) instead of a model: what some system "
                "wrote for every request of the suite."
            ),
        ),
    ] = None,
    system: Annotated[
        str | None,
        typer.Option(
            "--system",
            help=(
                "Name of the system, 1 to 64 characters: letters, digits, '.', '_' and "
                "'-'. Default: the name of the export, or the stem of the hypotheses file."
            ),
        ),
    ] = None,
    beam_size: Annotated[
        int,
        typer.Option(
            "--beam-size",
            min=1,
            max=8,
            metavar="1..8",
            help="Beam width, 1 to 8 (--model).",
        ),
    ] = 4,
    max_decoding_length: Annotated[
        int,
        typer.Option(
            "--max-decoding-length",
            min=1,
            help="Longest hypothesis, in pieces (--model).",
        ),
    ] = 256,
    script_mask: Annotated[
        ScriptMode,
        typer.Option(
            "--script-mask",
            help=(
                "retry decodes again, with the other scripts suppressed, an output that is "
                "not in the target's script; always suppresses them throughout; off never."
            ),
        ),
    ] = "retry",
    device: Annotated[
        Literal["cpu", "cuda"],
        typer.Option(
            "--device",
            help="Where CTranslate2 runs; never auto, which would pick the GPU.",
        ),
    ] = "cpu",
    threads: Annotated[
        int,
        typer.Option(
            "--threads", min=1, help="Threads CTranslate2 uses per batch (--model)."
        ),
    ] = 1,
    script_converter_ct2: Annotated[
        Path | None,
        typer.Option(
            "--script-converter-ct2",
            help="D5's script converter, its CTranslate2 directory (with the next option).",
        ),
    ] = None,
    script_converter_spm: Annotated[
        Path | None,
        typer.Option(
            "--script-converter-spm",
            help="D5's script converter, its SentencePiece model (with the previous option).",
        ),
    ] = None,
) -> None:
    """Measure numbers, placeholders, names, terms, script purity and a traditional-script source.

    The suite is scored with what the engine writes for it (--model) or with the hypotheses of a file (--hypotheses). It measures and has no threshold: exit 0 whenever it measured."""
    with rinchen_errors_as_exit_codes():
        plan = plan_of(
            model, hypotheses, system, script_converter_ct2, script_converter_spm
        )
        options = RunOptions(
            beam_size, max_decoding_length, script_mask, device, threads
        )
        target = Path(os.path.abspath(out_dir))
        suite_path = Path(os.path.abspath(suite))
        if isinstance(plan, ModelPlan):
            measured = _run_model(suite_path, plan, options, target)
        else:
            measured = _run_hypotheses(suite_path, plan, target)
        manifest_path = commit(target, measured)
        for line in result_lines(measured.result):
            typer.echo(line)
        typer.echo(f"result: {target / ROBUSTNESS_FILE}")
        typer.echo(f"report: {target / REPORT_FILE}")
        typer.echo(f"manifest: {manifest_path}")


# ---- the inputs ------------------------------------------------------------------------------


def _read_suite(
    suite: ManifestFile, inputs: list[ManifestFile], out_dir: Path
) -> Suite:
    """The suite, from the bytes that were hashed; then `out_dir` is refused when it is the
    directory of an input or holds another stage's manifest."""
    parsed = Suite.from_bytes(verified_content(Path(suite.path), suite.sha256))
    sources = [Path(entry.path) for entry in inputs]
    require_out_apart_from_inputs(out_dir, sources, option="--out-dir")
    refuse_foreign_manifest(out_dir, STAGE)
    return parsed


def _export_entries(verified: VerifiedExport) -> list[ManifestFile]:
    """The five files of the export and its manifest with the digests `verify_export`
    took."""
    files = [
        (verified.directory / name, digest) for name, digest in verified.sha256.items()
    ]
    files.append((verified.directory / "manifest.json", verified.manifest_sha256))
    return [
        ManifestFile(path=str(path), sha256=digest, bytes=path.stat().st_size)
        for path, digest in files
    ]


def _converter_entries(ct2_dir: Path, spm_model: Path) -> list[ManifestFile]:
    return [
        hash_input(path, what=_CONVERTER_FILE, missing=ConfigError)
        for path in (*(ct2_dir / name for name in CT2_FILES), spm_model)
    ]


# ---- the runs --------------------------------------------------------------------------------


def _run_model(
    suite_path: Path, plan: ModelPlan, options: RunOptions, target: Path
) -> Measured:
    """A run in which the engine translates the suite. The export is vouched for first; the
    suite and the converter's files are hashed; the converter and the engine open only after
    every refusal that needs no model, and the engine is closed whatever happens."""
    verified = verify_export(plan.model)
    suite_entry = hash_input(suite_path, what=_SUITE, missing=DataError)
    inputs = [
        suite_entry,
        *_export_entries(verified),
        *(_converter_entries(*plan.converter) if plan.converter else ()),
    ]
    suite = _read_suite(suite_entry, inputs, target)
    converter = None
    if plan.converter is not None:
        converter = load_converter(*plan.converter, options.device)
    engine = open_engine(verified.directory, options.load(), converter)
    try:
        info = engine.info
        _require_the_files_that_were_hashed(info, verified)
        requests = suite_requests(suite, _converted_texts(suite, converter))
        hypotheses = run_requests(requests, engine, options.decode())
    finally:
        engine.close()
    record = options.record(converter=converter is not None)
    system = RobustnessSystem(
        name=plan.name or info.name,
        model_sha256=info.model_sha256,
        manifest_sha256=info.manifest_sha256,
        options=record,
    )
    return Measured(
        result=compute_robustness(
            suite, hypotheses, suite_sha256=suite_entry.sha256, system=system
        ),
        hypotheses=hypotheses,
        config={"mode": "model", **record},
        status={
            "robustness": "ok",
            "mode": "model",
            "script_mask": options.script_mask,
            "quantization": verified.record.quantization,
        },
        inputs=inputs,
    )


def _require_the_files_that_were_hashed(
    info: ModelInfo, verified: VerifiedExport
) -> None:
    """The engine verifies the export again when it opens it: what it loaded must be what
    the manifest of this run vouches for."""
    loaded = (
        ("model.bin", info.model_sha256, verified.sha256["model.bin"]),
        ("manifest.json", info.manifest_sha256, verified.manifest_sha256),
    )
    for name, digest, hashed in loaded:
        if digest != hashed:
            raise ConfigError(
                f"the engine loaded a {name} that is not the one that was hashed: the "
                "export changed while the run started"
            )


def _converted_texts(
    suite: Suite, converter: ScriptConverterLike | None
) -> dict[str, str] | None:
    """The traditional source of every `mong` row converted to Cyrillic, by row id; None
    without a converter."""
    if converter is None:
        return None
    return {
        request.key.id: converter.convert(request.text, "Cyrl").text
        for request in suite_requests(suite)
        if request.key.section == "mong" and request.key.arm == "mong"
    }


def _run_hypotheses(suite_path: Path, plan: HypothesesPlan, target: Path) -> Measured:
    """A run that scores the suite with the hypotheses of a file, read from the bytes that
    were hashed. A file that has the `converted` arm for the `mong` rows (a model run with a
    converter wrote it) is read with that arm."""
    suite_entry = hash_input(suite_path, what=_SUITE, missing=DataError)
    file_entry = hash_input(plan.path, what=_HYPOTHESES, missing=DataError)
    inputs = [suite_entry, file_entry]
    suite = _read_suite(suite_entry, inputs, target)
    data = verified_content(plan.path, file_entry.sha256)
    converted = None
    if _CONVERTED_ARM.search(data):
        converted = {row.id: "" for row in suite.sections.get("mong", ())}
    hypotheses = hypotheses_from_bytes(data, suite_requests(suite, converted))
    return Measured(
        result=compute_robustness(
            suite,
            hypotheses,
            suite_sha256=suite_entry.sha256,
            system=RobustnessSystem(
                name=plan.name, hypotheses_sha256=file_entry.sha256
            ),
        ),
        hypotheses=None,
        config={"mode": "hypotheses"},
        status={"robustness": "ok", "mode": "hypotheses"},
        inputs=inputs,
    )
