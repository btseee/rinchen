"""The first three CLI stages of the tiny pipeline: SentencePiece, the script converter and the
synthetic data (D15, spec P2, P3 and P17). The stages that train, average, export and compare the model
are in `model_stages_rest`.

Each stage writes what its commands read into its own directory, runs them as the real `rinchen`
commands through the runner of the context, and reads its facts from the files they wrote: the size of
the vocabulary from the `.vocab` file, the counts of the synthetic run from its `counts.json`. The one
printed text that is read is the five lines of `convert-script eval`, by shape. The synthetic stage is
the only function of the pipeline that opens a socket: the loopback listener of a `FakeChatServer`, which
the command, a child, connects to, and which is shut down on every way out.

The seed is the pipeline's, read from the profile file the command hashed; the context has no field for
it, so the stages that need it take it as a second argument.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import polars as pl

from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.schema import MONO_SCHEMA
from rinchen.errors import DataError
from rinchen.maintain.commands import read_complete_log
from rinchen.tiny import commands
from rinchen.tiny.context import StageContext, StageResult
from rinchen.tiny.fakes import FakeChatServer
from rinchen.tiny.fixtures import MonoRow
from rinchen.tiny.states import StageRecord, fail_check
from rinchen.tiny.step_runner import (
    StageCommands,
    StageStopped,
    path_in,
    read_text,
    run_stage,
)
from rinchen.utils.part_file import lines_as_bytes, replace_file

SYNTH_CHECK: Final = fail_check("synth_output")
KEPT_FILE: Final = "kept.parquet"
SPM_PREFIX: Final = "tiny"
SPM_MODEL_FILE: Final = f"{SPM_PREFIX}.model"
SPM_VOCAB_FILE: Final = f"{SPM_PREFIX}.vocab"
MONO_FILE: Final = "mono.parquet"
MONO_CORPUS: Final = "tiny-mono"
SYNTH_RUN_DIR: Final = "run"
SYNTH_CORPUS_FILE: Final = "corpus.parquet"
SYNTH_COUNTS_FILE: Final = "counts.json"
CONVERTER_MODEL_DIR: Final = "model"
CONVERTER_RUN_DIR: Final = "run"
CONVERTER_CHAR_MODEL: Final = "converter_char.model"


# ---- spm ----------------------------------------------------------------------------------------------


def _language_texts(kept: pl.DataFrame) -> dict[str, list[str]]:
    """The train sentences of each language, each once, in the order they were first seen: the source
    and the target of every kept row of the split `train`."""
    seen: dict[str, list[str]] = {}
    for row in kept.filter(pl.col("split").cast(pl.String) == "train").iter_rows(
        named=True
    ):
        seen.setdefault(row["src_lang"], []).append(row["src"])
        seen.setdefault(row["tgt_lang"], []).append(row["tgt"])
    return {language: list(dict.fromkeys(seen[language])) for language in sorted(seen)}


def _spm(stage: StageCommands, seed: int) -> Mapping[str, Any]:
    ctx = stage.ctx
    kept = scan_corpus(path_in(ctx, "filter", KEPT_FILE)).collect()
    texts = _language_texts(kept)
    text_dir = stage.directory / "text"
    text_dir.mkdir(exist_ok=True)
    for language, lines in texts.items():
        replace_file(text_dir / f"{language}.txt", lines_as_bytes(lines))
    config = stage.directory / "spm.yaml"
    commands.write_spm_config(
        config, ctx.config.spm, seed, text_dir, stage.directory / SPM_PREFIX
    )
    stage.run("train", commands.spm_train(config))
    stage.run("check", commands.spm_check(stage.directory / SPM_MODEL_FILE, text_dir))
    pieces = read_text(stage.directory / SPM_VOCAB_FILE).splitlines()
    if not pieces:
        raise DataError(f"{SPM_VOCAB_FILE}: the vocabulary is empty")
    return {"vocab_size": len(pieces), "languages": list(texts)}


def run_spm(ctx: StageContext, seed: int) -> StageResult:
    """Train the SentencePiece model on the kept train text of each language and check it.

    The text files and the config are written into `04_spm`, `rinchen spm train` and `rinchen spm
    check` run, and the facts are the number of pieces in the model's vocabulary file and the
    languages it was trained on. `spm check` writes nothing, so its verdict is its exit code and its
    coverage lines are not read.
    """
    return run_stage(ctx, "spm", lambda stage: _spm(stage, seed))


# ---- converter ----------------------------------------------------------------------------------------

_COUNT_FACTS: Final = ("n", "rule_fallbacks", "invalid_outputs")


def _converter(stage: StageCommands) -> Mapping[str, Any]:
    ctx = stage.ctx
    directory = stage.directory
    pairs = ctx.config.converter.pairs
    char_config = directory / "char.yaml"
    corpus = directory / "corpus"
    run = directory / CONVERTER_RUN_DIR
    model = directory / CONVERTER_MODEL_DIR
    commands.write_char_config(char_config, ctx.config.converter.char_vocab_size)
    stage.run("build", commands.converter_build_data(pairs, corpus))
    stage.run_bounded(
        "train",
        lambda timeout: commands.converter_train(
            corpus, run, char_config, ctx.threads, timeout
        ),
    )
    stage.run_bounded(
        "export", lambda timeout: commands.converter_export(run, model, timeout)
    )
    stage.run("eval", commands.converter_eval(model, pairs))
    printed = read_complete_log(stage.log_path("eval"))
    if printed is None:
        raise DataError("the converter evaluation log is too long to be five lines")
    numbers = commands.read_converter_eval(printed)
    return {
        key: int(value) if key in _COUNT_FACTS else value
        for key, value in numbers.items()
    }


def run_converter(ctx: StageContext) -> StageResult:
    """Build the corpus of the word pairs, train the character-level converter, export it and
    evaluate it, through `rinchen convert-script` in `05_converter`.

    The facts are the five numbers `convert-script eval` prints (`cer`, `word_accuracy`, `n`,
    `rule_fallbacks`, `invalid_outputs`): the one place a printed text is read, by shape.
    """
    return run_stage(ctx, "converter", _converter)


# ---- synth --------------------------------------------------------------------------------------------


def _write_mono(path: Path, rows: Sequence[MonoRow]) -> None:
    """The monolingual corpus of the job: the Khalkha line of each fixture row."""
    size = len(rows)
    frame = pl.DataFrame(
        {
            "id": [row.id for row in rows],
            "lang": [commands.MONO_LANGUAGE] * size,
            "text": [row.khk for row in rows],
            "corpus": [MONO_CORPUS] * size,
            "doc_id": [None] * size,
            "seg_idx": [None] * size,
            "url": [None] * size,
            "scores": [None] * size,
        },
        schema=MONO_SCHEMA,
    )
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    replace_file(path, buffer.getvalue())


def _synth(stage: StageCommands, seed: int) -> Mapping[str, Any]:
    ctx = stage.ctx
    section = ctx.config.synth
    mono = stage.directory / MONO_FILE
    run = stage.directory / SYNTH_RUN_DIR
    _write_mono(mono, ctx.fixtures.mono)
    oracle = {row.khk: row.eng for row in ctx.fixtures.mono}
    with FakeChatServer(oracle) as server:
        config = stage.directory / "synth.yaml"
        commands.write_synth_config(
            config, section.job, server.base_url, section.shard_size, section.n
        )
        stage.run("run", commands.synth_run(section.job, mono, config, run, seed))
        answered = server.requests_seen
    counts = commands.read_synth_counts(run / SYNTH_COUNTS_FILE)
    rows = scan_corpus(run / SYNTH_CORPUS_FILE).select(pl.len()).collect().item()
    # The fingerprint of the job names its shards and holds the port of the listener: not a fact.
    facts = {
        "job": counts["job"],
        "generator": counts["generator"],
        "qe_status": counts["qe_status"],
        "counts": counts["counts"],
        "shard_stats": counts["shard_stats"],
        "rows": rows,
        "chat_requests": answered,
    }
    if not _made_every_row(counts["counts"], rows, len(ctx.fixtures.mono)):
        raise StageStopped(StageRecord("synth", "fail", SYNTH_CHECK), facts)
    return facts


def _made_every_row(counts: Mapping[str, int], rows: int, fed: int) -> bool:
    """A server that answers nothing still lets `synth run` exit 0, with nothing selected and no row, and a
    run that read only part of its input would look whole: a run is complete when it read as many sources as
    the `fed` rows of the monolingual corpus the stage wrote, selected every one of them and made one row
    for each."""
    return (
        rows > 0
        and rows == counts["selected"]
        and counts["selected"] == counts["sources"]
        and counts["sources"] == fed
    )


def run_synth(ctx: StageContext, seed: int) -> StageResult:
    """Make the monolingual corpus of the job from the fixture, start the loopback chat server over
    the oracle of the fixture, write the config that names it and run `rinchen synth run` with
    `--no-qe` against it, in `06_synth`.

    The server is stopped, its socket closed and its thread joined however the stage ends, an
    exception of the runner included. The facts are the counts of the run (`counts.json`), the rows of
    its corpus and the requests the server answered. The stage fails the check `synth_output` (with
    the facts) when the run read other sources than the rows of the fixture it was fed, a source was
    not selected or the corpus does not hold one row for each selected source, or none: the command
    exits 0 for a server that answered nothing.
    """
    return run_stage(ctx, "synth", lambda stage: _synth(stage, seed))
