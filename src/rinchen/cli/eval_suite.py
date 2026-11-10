"""`rinchen eval suite`: fetch a benchmark, turn local files into an evaluation suite and look at one
(D13). Parsing, the order of the steps, the hashing and the exit codes only; the rows are read by
`rinchen.cli.eval_suite_files`, joined and checked by `rinchen.eval.suite_sources`, and the suite
file is `rinchen.eval.suites`'.

    rinchen eval suite fetch --benchmark B --split S --out-dir DIR [--revision HEX | --unpinned]
        [--config FILE]
    rinchen eval suite build --benchmark B --split S --from PATH --out-dir DIR [--name N]
        [--kind benchmark|fixture] [--config FILE]
    rinchen eval suite build --benchmark human --from FILE --out-dir DIR [--name N]
    rinchen eval suite inspect --suite FILE

`fetch` is the only step of D13 that reaches the network, and nothing here ever ran it: it calls
`rinchen.acquire.hf.fetch` for each language of the configuration (`configs/eval/suites.yaml`) and
writes `DIR/<split>/<language>/shard-*.parquet` and, last, `manifest.json` (stage
`eval_suite_fetch`). The revision must be pinned (`--revision`, or the configuration's);
`--unpinned` fetches the dataset as it is now, and the suites built from it say so.

`build` reads a directory of shards (what `fetch` wrote, or `<language>/shard-*.parquet`), of text
files (`<language>.txt`) or, for `human`, one export of D2's annotation app (`.jsonl`, `.tsv` or
`.parquet`), and writes `DIR/<name>.json` (`<benchmark>-<split>` unless named) and, last,
`manifest.json` (stage `eval_suite`). When `PATH` holds the manifest of a fetch, every shard it
lists must be there with its digest and no other may be, and the suite records the revision and
the dataset; loose files give a suite with no revision. `--config` is not read for `human`.

Every input is hashed BEFORE it is read and what is parsed is what was hashed. An output directory
that is an input's, or holds another stage's manifest, is refused; a refusal writes nothing and
leaves an earlier run's files as they were; the files an earlier manifest of the same stage lists
are removed before the new ones are written and no other file is touched. No text of a source is
in a message, and no token is read, printed or written (the dataset library finds its own).

Exit codes: 0; 2 for a bad option, a configuration that is not usable, a benchmark that cannot be
fetched, an output directory that is an input's, a file that changed after it was hashed; 3 for
an input that is missing, unreadable or not what the configuration says, a suite that is not one
and another stage's manifest in the output directory; 4 when the fetch fails upstream; 1 when
the output cannot be written; 130 when interrupted.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Final

import typer

from rinchen.acquire import hf
from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.eval_suite_files import (
    EXPORT_SUFFIXES,
    FETCH_STAGE,
    UNPINNED,
    FetchRecord,
    check_against_fetch,
    export_rows,
    language_inputs,
    parquet_height,
    parquet_rows,
    read_fetch_manifest,
)
from rinchen.cli.eval_suite_output import (
    BUILD_STAGE,
    SHARD_PATH_PARTS,
    Built,
    begin_outputs,
    commit_build,
    counted,
    description,
    options_digest,
    status_of,
    write_fetch_manifest,
)
from rinchen.cli.eval_suite_plan import (
    BuildPlan,
    SuiteKind,
    benchmark_of,
    pin_of,
    plan_of,
    revision_to_fetch,
)
from rinchen.cli.train_common import (
    create_out_dir,
    exit_cannot_write,
    hash_input,
    require_out_apart_from_inputs,
)
from rinchen.config.eval_suites import (
    DEFAULT_EVAL_SUITES_PATH,
    HUMAN_BENCHMARK,
    BenchmarkConfig,
    EvalSuitesConfig,
    load_eval_suites_config,
)
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.eval.suite_rules import HUB_LANGUAGE, MAX_SUITE_BYTES, SuiteSource
from rinchen.eval.suite_sources import (
    SourceRow,
    build_multiway,
    human_rows,
    lines_rows,
)
from rinchen.eval.suites import Suite
from rinchen.export.verified import verified_content
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import output_file, refuse_foreign_manifest

_CONFIG: Final = "eval suites config"

suite_app = typer.Typer(no_args_is_help=True)


# ---- fetch --------------------------------------------------------------------------------


@suite_app.command(name="fetch")
def fetch_command(
    benchmark: Annotated[
        str, typer.Option("--benchmark", help="A benchmark of the configuration.")
    ],
    split: Annotated[
        str, typer.Option("--split", help="A split of the benchmark: dev or devtest.")
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir", help="Output directory: <split>/<language>/shard-*.parquet."
        ),
    ],
    revision: Annotated[
        str | None,
        typer.Option(
            "--revision",
            help="The commit of the dataset to fetch: 40 lowercase hexadecimal characters.",
        ),
    ] = None,
    unpinned: Annotated[
        bool,
        typer.Option(
            "--unpinned",
            help="Fetch the dataset as it is now; the suites built from it say so.",
        ),
    ] = False,
    config: Annotated[
        Path,
        typer.Option("--config", help="The benchmarks (YAML)."),
    ] = DEFAULT_EVAL_SUITES_PATH,
) -> None:
    """Fetch a benchmark from the Hub as Parquet shards, one directory per language.

    The only step that reaches the network; the revision must be pinned (--revision or the configuration) unless --unpinned."""
    with rinchen_errors_as_exit_codes():
        pin = pin_of(revision, unpinned)
        target = Path(os.path.abspath(out_dir))
        config_path = Path(os.path.abspath(config))
        config_entry = hash_input(config_path, what=_CONFIG, missing=ConfigError)
        suites = _configuration(config_entry)
        bench = benchmark_of(suites, benchmark, split)
        if bench.reader != "parquet_shards":
            raise ConfigError(
                f"the benchmark {benchmark} is read from a directory of text files, not "
                "from the Hub: put the files in a directory and build from it"
            )
        if bench.dataset is None:
            raise ConfigError(
                f"the benchmark {benchmark} has no Hub dataset in the configuration, so "
                "nothing can be fetched"
            )
        pinned = revision_to_fetch(pin, bench)
        require_out_apart_from_inputs(target, [config_path], option="--out-dir")
        refuse_foreign_manifest(target, FETCH_STAGE)
        create_out_dir(target, what="fetched shards")
        try:
            begin_outputs(target, FETCH_STAGE, parts=SHARD_PATH_PARTS)
        except OSError as exc:
            exit_cannot_write(target, exc, what="fetched shards")
        shown = UNPINNED if pinned is None else pinned
        typer.echo(f"benchmark: {benchmark}")
        typer.echo(f"split: {split}")
        typer.echo(f"revision: {shown}")
        outputs: list[ManifestFile] = []
        counts: dict[str, int] = {}
        for language in bench.languages:
            shards = _fetch_language(
                bench.dataset, target / split, language, split, pinned
            )
            rows = [parquet_height(path) for path in shards]
            counts[language] = sum(rows)
            outputs += [
                output_file(path, n) for path, n in zip(shards, rows, strict=True)
            ]
            typer.echo(
                f"{language}: {counted(counts[language], 'row')}, "
                f"{counted(len(shards), 'shard')}"
            )
        manifest_path = write_fetch_manifest(
            target,
            config_entry,
            outputs,
            counts,
            benchmark=benchmark,
            split=split,
            dataset=bench.dataset,
            revision=shown,
        )
        typer.echo(f"manifest: {manifest_path}")


def _fetch_language(
    dataset: str, split_dir: Path, language: str, split: str, revision: str | None
) -> list[Path]:
    destination = split_dir / language
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        exit_cannot_write(destination, exc, what="fetched shards")
    try:
        shards = hf.fetch(
            dataset, destination, config=language, split=split, revision=revision
        )
    except Exception as error:
        raise UpstreamError(
            f"fetching {language} ({split}) of {dataset} failed: {type(error).__name__}"
        ) from error
    if not shards:
        raise UpstreamError(
            f"the Hub returned no row for {language} ({split}) of {dataset}"
        )
    return shards


# ---- build --------------------------------------------------------------------------------


@suite_app.command(name="build")
def build_command(
    benchmark: Annotated[
        str,
        typer.Option("--benchmark", help="A benchmark of the configuration, or human."),
    ],
    source: Annotated[
        Path,
        typer.Option(
            "--from",
            help="Shards or text files of the benchmark (a directory), or the export of a human set (a file).",
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Output directory: the suite and the manifest."),
    ],
    split: Annotated[
        str | None,
        typer.Option("--split", help="A split of the benchmark (human: test)."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name of the suite and its file; default <benchmark>-<split>.",
        ),
    ] = None,
    kind: Annotated[
        SuiteKind | None,
        typer.Option(
            "--kind", help="benchmark (default) or fixture; human for human only."
        ),
    ] = None,
    config: Annotated[
        Path,
        typer.Option("--config", help="The benchmarks (YAML); not read for human."),
    ] = DEFAULT_EVAL_SUITES_PATH,
) -> None:
    """Build an evaluation suite from shards, text files or the export of a human set.

    Every input is hashed before it is read; the manifest is written last."""
    with rinchen_errors_as_exit_codes():
        plan = plan_of(benchmark, split, kind, name)
        target = Path(os.path.abspath(out_dir))
        origin = Path(os.path.abspath(source))
        if plan.benchmark == HUMAN_BENCHMARK:
            built = _build_human(origin, plan, target)
        else:
            built = _build_multiway(origin, plan, Path(os.path.abspath(config)), target)
        suite_path, manifest_path, digest = commit_build(target, built)
        typer.echo(f"suite: {suite_path}")
        typer.echo(f"sha256: {digest}")
        typer.echo(f"manifest: {manifest_path}")


def _build_human(origin: Path, plan: BuildPlan, target: Path) -> Built:
    if origin.suffix not in EXPORT_SUFFIXES:
        raise ConfigError(
            f"--from must be a file ending in {', '.join(EXPORT_SUFFIXES)} for a human set"
        )
    entry = hash_input(origin, what="export file", missing=DataError)
    _require_apart_and_free([entry], target)
    rows = export_rows(origin, verified_content(origin, entry.sha256))
    source = SuiteSource(
        dataset=None,
        revision=None,
        pinned=False,
        files={origin.name: entry.sha256},
        missing_languages=(),
    )
    return Built(
        human_rows(rows, name=plan.name, source=source),
        [entry],
        status_of(plan, None),
        options_digest({"benchmark": plan.benchmark, "name": plan.name}),
    )


def _build_multiway(
    origin: Path, plan: BuildPlan, config_path: Path, target: Path
) -> Built:
    config_entry = hash_input(config_path, what=_CONFIG, missing=ConfigError)
    bench = benchmark_of(_configuration(config_entry), plan.benchmark, plan.split)
    if not origin.is_dir():
        raise DataError(f"the source {origin} is not a directory")
    fetch_entry, record = _vouching_fetch(origin, bench, plan)
    root = origin if record is None else origin / plan.split
    files = language_inputs(
        root, reader=bench.reader, configured=bench.languages, benchmark=plan.benchmark
    )
    if HUB_LANGUAGE not in files:
        raise DataError(
            f"the source {root} holds no input for the language {HUB_LANGUAGE}"
        )
    hashed = {
        language: [
            hash_input(path, what="input file", missing=DataError) for path in paths
        ]
        for language, paths in files.items()
    }
    if record is not None:
        check_against_fetch(hashed, record)
    entries = [
        config_entry,
        *([] if fetch_entry is None else [fetch_entry]),
        *(entry for language_entries in hashed.values() for entry in language_entries),
    ]
    _require_apart_and_free(entries, target)
    source = _source_of(bench, record, hashed)
    suite = build_multiway(
        name=plan.name,
        kind=plan.kind,
        benchmark=plan.benchmark,
        split=plan.split,
        languages={
            language: _rows_of(bench, language, language_entries)
            for language, language_entries in hashed.items()
        },
        configured=bench.languages,
        source=source,
    )
    options = {**plan._asdict(), "config": config_entry.sha256}
    return Built(suite, entries, status_of(plan, source), options_digest(options))


def _vouching_fetch(
    origin: Path, bench: BenchmarkConfig, plan: BuildPlan
) -> tuple[ManifestFile | None, FetchRecord | None]:
    """The manifest of a fetch that `origin` holds (hashed) and what it vouches for; neither for
    a source that holds none, which is a directory of loose shards or of text files."""
    manifest_file = origin / "manifest.json"
    if bench.reader != "parquet_shards" or not manifest_file.is_file():
        return None, None
    entry = hash_input(manifest_file, what="fetch manifest", missing=DataError)
    record = read_fetch_manifest(
        verified_content(manifest_file, entry.sha256),
        benchmark=plan.benchmark,
        split=plan.split,
    )
    return entry, record


def _source_of(
    bench: BenchmarkConfig,
    record: FetchRecord | None,
    hashed: Mapping[str, Sequence[ManifestFile]],
) -> SuiteSource:
    """Where the suite came from: the dataset and the revision a fetch vouches for (none for
    loose files) and every input file with its digest, named relative to the source."""
    return SuiteSource(
        dataset=None if record is None else record.dataset,
        revision=None if record is None else record.revision,
        pinned=record is not None and record.revision is not None,
        files={
            _relative_name(bench, language, entry): entry.sha256
            for language, language_entries in hashed.items()
            for entry in language_entries
        },
        missing_languages=(),
    )


def _rows_of(
    bench: BenchmarkConfig, language: str, entries: Sequence[ManifestFile]
) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for entry in entries:
        path = Path(entry.path)
        data = verified_content(path, entry.sha256)
        if bench.columns is None:
            rows += lines_rows(language, data)
        else:
            rows += parquet_rows(
                data, bench.columns, where=f"the shard {language}/{path.name}"
            )
    return rows


def _relative_name(bench: BenchmarkConfig, language: str, entry: ManifestFile) -> str:
    name = Path(entry.path).name
    return name if bench.reader == "lines_dir" else f"{language}/{name}"


def _require_apart_and_free(entries: Sequence[ManifestFile], target: Path) -> None:
    """`target` is refused when it is the directory of an input or holds another stage's
    manifest."""
    require_out_apart_from_inputs(
        target, [Path(entry.path) for entry in entries], option="--out-dir"
    )
    refuse_foreign_manifest(target, BUILD_STAGE)


@suite_app.command(name="inspect")
def inspect_command(
    suite: Annotated[Path, typer.Option("--suite", help="The suite file (JSON).")],
) -> None:
    """Print what a suite file is: name, kind, split, purpose, digest, rows and directions."""
    with rinchen_errors_as_exit_codes():
        path = Path(os.path.abspath(suite))
        entry = hash_input(path, what="suite file", missing=DataError)
        if entry.bytes is not None and entry.bytes > MAX_SUITE_BYTES:
            raise DataError(
                f"not an evaluation suite: the file {path} is larger than "
                f"{MAX_SUITE_BYTES // 2**20} MiB"
            )
        parsed = Suite.from_bytes(verified_content(path, entry.sha256))
        for line in description(parsed, entry.sha256):
            typer.echo(line)


def _configuration(entry: ManifestFile) -> EvalSuitesConfig:
    return load_eval_suites_config(verified_content(Path(entry.path), entry.sha256))
