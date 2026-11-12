"""The stages of the tiny pipeline that run in process: environment, fetch, corpus and filter (D15,
spec P2, P4, P6 and P15).

No command can do the job of these four from files (`rinchen data fetch` writes under the checkout's
`data/`, `rinchen filter run` takes no stand-in, the corpus build has no command), so each is a
function of a `StageContext`: the profile, the fixtures the command parsed from the bytes it hashed,
and the output directory. A stage reads nothing of the repository by itself, writes only into its own
directory and returns a `StageResult`: the record the report lists and the `facts` it observed (numbers
and closed identifiers; no text of a fixture ever appears in a record, a fact or a counts file).

`run_environment` also takes the bytes of `uv.lock` that the command hashed, because the context has no
field for them: the manifest names the lock's versions as the run's, which is true only if the
installed packages are the lock's.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import polars as pl

from rinchen.acquire.http import LOCAL_SCHEME, sha256_of
from rinchen.acquire.http import REPO_ROOT as CHECKOUT
from rinchen.acquire.registry import LocalSourceEntry, fetch_source, load_registry
from rinchen.config.filters import (
    Bounds,
    FilterConfig,
    LabseConfig,
    LengthConfig,
    MinHashConfig,
    QeConfig,
    StageName,
)
from rinchen.config.tiny import FilterSection
from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.writers import write_corpus
from rinchen.errors import DataError
from rinchen.filter.pipeline import kept_rows, run_pipeline
from rinchen.maintain.environment import check_environment, installed_versions
from rinchen.maintain.lock import read_lock
from rinchen.tiny.context import StageContext, StageResult, stage_dir
from rinchen.tiny.corpus import (
    NOISE_CORPUS,
    build_pair_corpus,
    clean_pair,
    created_stamp,
    expand_directions,
)
from rinchen.tiny.fakes import ConceptEncoder, ConceptQE
from rinchen.tiny.fixtures import FIXTURE_FILES, Fixtures, read_fixtures
from rinchen.tiny.states import (
    FAIL_INPUT,
    PART_IDS,
    SKIP_NO_LABSE,
    SKIP_NO_LID,
    SKIP_NO_QE,
    SKIP_TOKEN_RATIO,
    PartRecord,
    StageRecord,
    fail_check,
    fail_environment,
)
from rinchen.utils.part_file import replace_file

CORPUS_FILE: Final = "corpus.parquet"
COUNTS_FILE: Final = "counts.json"
PLANTED_CHECK: Final = fail_check("planted_noise")
SKIPPED_STAGE_CHECK: Final = fail_check("stage_skipped")
# The filter stages that run on the CPU without a model, in the order of precedence of the reasons.
CPU_STAGES: Final[tuple[StageName, ...]] = (
    "script_ratio",
    "numbers",
    "length",
    "minhash",
)
STAND_IN_STAGES: Final[tuple[StageName, ...]] = ("labse", "qe")
# The stage that gives each reason a planted row may expect.
STAGE_OF_REASON: Final = {
    "script_tgt": "script_ratio",
    "len_ratio_char": "length",
    "numbers": "numbers",
    "duplicate": "minhash",
    "labse": "labse",
}
# Why a part of the filter did not run when it is not among the stages that were asked.
UNASKED_PARTS: Final = {"lid": SKIP_NO_LID, "labse": SKIP_NO_LABSE, "qe": SKIP_NO_QE}


def _own_dir(ctx: StageContext, stage_id: str) -> Path:
    directory = stage_dir(ctx.out_dir, stage_id)
    if directory is None:
        raise ValueError("the stage writes no directory")
    return directory


def _write_counts(directory: Path, facts: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    document = json.dumps(facts, sort_keys=True, indent=2) + "\n"
    replace_file(directory / COUNTS_FILE, document.encode("utf-8"))


def _input_failure(stage_id: str, detail: Mapping[str, Any]) -> StageResult:
    record = StageRecord(stage_id, "fail", FAIL_INPUT, detail)
    return StageResult(record, detail)


def _counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    return dict(Counter(frame[column].to_list()))


# ---- environment -------------------------------------------------------------------------------


def run_environment(ctx: StageContext, lock_bytes: bytes) -> StageResult:
    """Compare the installed distributions with the lock in `lock_bytes`; nothing is written.

    The state is `fail` when a package is installed at a version the lock does not list or is not in
    the lock at all (the reason holds the two counts and no name); a locked package that is not
    installed is only counted, because a lock holds the packages of other platforms.

    Raises:
        DataError: `lock_bytes` is not a lock file.
    """
    lock = read_lock(lock_bytes)
    outcome = check_environment(lock, installed_versions())
    facts = {
        "installed": outcome.installed,
        "locked": len(lock.versions),
        "mismatched": len(outcome.mismatched),
        "extra": len(outcome.extra),
        "not_installed": outcome.not_installed,
    }
    if outcome.state == "ok":
        return StageResult(StageRecord("environment", "ok", detail=facts), facts)
    reason = fail_environment(len(outcome.mismatched), len(outcome.extra))
    return StageResult(StageRecord("environment", "fail", reason, facts), facts)


# ---- fetch -------------------------------------------------------------------------------------


def _copy_name(entry: LocalSourceEntry) -> str:
    """The name `fetch_source` gives the copy of a local entry: the last segment of its URL."""
    return entry.url.rstrip("/").split("/")[-1] or entry.id


def _registry_is_the_fixtures(entries: list[Any], directory: Path) -> bool:
    """Whether the registry lists each of the seven fixtures once, as a local entry for a file of
    the fixtures directory itself: no network kind, no file of another directory, no path that
    climbs out of it and no file that is a link under another name, so that the name the copy is
    given is the name of the fixture it is checked against."""
    home = directory.resolve()
    names = []
    for entry in entries:
        if not isinstance(entry, LocalSourceEntry) or not entry.url.startswith(
            LOCAL_SCHEME
        ):
            return False
        source = (CHECKOUT / entry.url[len(LOCAL_SCHEME) :]).resolve()
        if source.parent != home or source.name != _copy_name(entry):
            return False
        names.append(source.name)
    return sorted(names) == sorted(FIXTURE_FILES)


def run_fetch(ctx: StageContext) -> StageResult:
    """Copy the seven fixtures through the registry's `local` entries into `01_fetch/files` and
    check each copy against its pin.

    The pins are checked first (the files read are the profile's own), then the registry, and only
    then is a file copied, so a refusal leaves nothing behind. The corpus is built from the fixtures
    the command parsed, not from these copies: they are the proof that the acquisition path works.

    Raises:
        DataError: a fixture file is missing or differs from its pin.
        ConfigError: the registry is not readable or not valid.
    """
    pins = ctx.config.fixtures.sha256
    read_fixtures(ctx.config.fixtures.directory, pins)
    entries = load_registry(ctx.config.registry)
    if not _registry_is_the_fixtures(entries, ctx.config.fixtures.directory):
        return _input_failure("fetch", {"files": 0, "bytes": 0})
    copies = _own_dir(ctx, "fetch") / "files"
    files = size = 0
    for entry in entries:
        copy = fetch_source(entry, copies)
        if sha256_of(copy) != pins[copy.name]:
            return _input_failure("fetch", {"files": files, "bytes": size})
        files += 1
        size += copy.stat().st_size
    facts = {"files": files, "bytes": size}
    return StageResult(StageRecord("fetch", "ok", detail=facts), facts)


# ---- corpus ------------------------------------------------------------------------------------


def run_corpus(ctx: StageContext) -> StageResult:
    """Normalise and split the fixtures into the pair corpus, one row per pair, and write it with
    its counts to `02_corpus`. A pair with a side that is not one sentence is left out, counted and
    not an error."""
    built = build_pair_corpus(ctx.fixtures, created_stamp(ctx.config.created))
    directory = _own_dir(ctx, "corpus")
    write_corpus(built.frame, directory / CORPUS_FILE)
    dropped = sum(built.dropped.values())
    facts = {
        "rows": built.frame.height,
        "dropped": dropped,
        "dropped_by_corpus": dict(built.dropped),
        "corpora": _counts(built.frame, "corpus"),
        "splits": _counts(built.frame, "split"),
    }
    _write_counts(directory, facts)
    detail = {"rows": built.frame.height, "dropped": dropped}
    return StageResult(StageRecord("corpus", "ok", detail=detail), facts)


# ---- filter ------------------------------------------------------------------------------------


def _filter_config(section: FilterSection, *, qe_fake: bool) -> FilterConfig:
    stages = [*CPU_STAGES, *(STAND_IN_STAGES if qe_fake else ())]
    return FilterConfig(
        stages=stages,
        length=LengthConfig(
            char_ratio=Bounds(min=section.char_ratio_min, max=section.char_ratio_max)
        ),
        minhash=MinHashConfig(threshold=section.minhash_threshold),
        labse=LabseConfig(min_cosine=section.labse_min_cosine, device="cpu"),
        qe=QeConfig(max_metricx_qe=section.qe_max_metricx),
    )


def _parts(statuses: Mapping[str, str]) -> tuple[PartRecord, ...]:
    """The state of each of the seven filter stages. A stage that was not asked for is skipped
    with its reason; one that was asked for and that the pipeline skipped is a failure, because the
    stand-ins are always given and only an unexpected skip could do that."""
    parts = []
    for part in PART_IDS["filter"]:
        if part not in statuses:
            parts.append(PartRecord(part, "skipped", UNASKED_PARTS[part]))
        elif statuses[part].startswith("skipped"):
            parts.append(PartRecord(part, "fail", SKIPPED_STAGE_CHECK))
        else:
            parts.append(PartRecord(part, "ok"))
    return tuple(parts)


def _planted(
    scored: pl.DataFrame, fixtures: Fixtures, asked: tuple[str, ...]
) -> dict[str, Any]:
    """What the filter made of the planted rows: for each, the reason it expects against the one it
    got, counted only for rows whose reason an asked stage can give."""
    got: dict[tuple[str, str, str, str], str | None] = {
        (r["src_lang"], r["tgt_lang"], r["src"], r["tgt"]): r["filter_reason"]
        for r in scored.filter(pl.col("corpus") == NOISE_CORPUS).iter_rows(named=True)
    }
    expected: Counter[str] = Counter()
    found: Counter[str] = Counter()
    detectable = matched = 0
    for row in fixtures.noise:
        expected[row.expect] += 1
        pair = clean_pair(row.src_lang, row.src, row.tgt_lang, row.tgt)
        reason = None if pair is None else got.get((row.src_lang, row.tgt_lang, *pair))
        found[reason or "none"] += 1
        if STAGE_OF_REASON[row.expect] in asked:
            detectable += 1
            matched += reason == row.expect
    return {
        "total": len(fixtures.noise),
        "detectable": detectable,
        "matched": matched,
        "expected": dict(expected),
        "found": dict(found),
    }


def _filter_facts(
    scored: pl.DataFrame,
    kept: pl.DataFrame,
    train_rows: int,
    reason_counts: Mapping[str, int],
    planted: dict[str, Any],
) -> dict[str, Any]:
    authentic = scored.filter(pl.col("corpus") != NOISE_CORPUS)
    flagged = authentic.height - authentic["filter_reason"].null_count()
    return {
        "rows": scored.height,
        "kept": kept.height,
        "reasons": {reason: n for reason, n in reason_counts.items() if n},
        "authentic_rows": authentic.height,
        "authentic_flagged": flagged,
        "authentic_flagged_share": flagged / authentic.height
        if authentic.height
        else 0.0,
        "planted": planted,
        "train_rows": train_rows,
    }


def _filter_record(
    facts: Mapping[str, Any],
    parts: tuple[PartRecord, ...],
    section: FilterSection,
    *,
    token_check_skipped: bool,
) -> StageRecord:
    detail: dict[str, Any] = {
        "rows": facts["rows"],
        "kept": facts["kept"],
        "authentic_flagged": facts["authentic_flagged"],
        "planted_detectable": facts["planted"]["detectable"],
        "planted_matched": facts["planted"]["matched"],
        "train_rows": facts["train_rows"],
    }
    if token_check_skipped:
        detail["token_ratio_check"] = SKIP_TOKEN_RATIO
    planted = facts["planted"]
    if any(part.state == "fail" for part in parts):
        reason = SKIPPED_STAGE_CHECK
    elif (
        planted["matched"] < planted["detectable"]
        or facts["authentic_flagged_share"] > section.max_authentic_flagged_share
    ):
        reason = PLANTED_CHECK
    else:
        return StageRecord("filter", "ok", detail=detail, parts=parts)
    return StageRecord("filter", "fail", reason, detail, parts)


def _read_corpus(ctx: StageContext) -> pl.DataFrame | None:
    try:
        return scan_corpus(_own_dir(ctx, "corpus") / CORPUS_FILE).collect()
    except DataError, pl.exceptions.PolarsError:
        return None


def run_filter(ctx: StageContext) -> StageResult:
    """Filter the pair corpus of the `corpus` stage ONCE and write `scored`, `kept`, `train` (the kept
    train and dev rows expanded to both directions) and the counts to `03_filter`.

    With `ctx.qe_fake` the planted stand-ins serve the `labse` and `qe` stages; without it those two
    are not run (they would download a model) and the stage says so. The stage fails when a planted
    row that an asked stage can detect does not get the reason it expects, or when the stand-ins flag
    more authentic rows than the profile allows; the files are written all the same.
    """
    corpus = _read_corpus(ctx)
    if corpus is None:
        return _input_failure("filter", {"input": CORPUS_FILE})
    section = ctx.config.filter
    config = _filter_config(section, qe_fake=ctx.qe_fake)
    encoder = (
        ConceptEncoder(ctx.fixtures.nouns, ctx.fixtures.entities)
        if ctx.qe_fake
        else None
    )
    outcome = run_pipeline(
        corpus.lazy(),
        config,
        embedder=encoder,
        qe_scorer=None if encoder is None else ConceptQE(encoder),
    )
    kept = kept_rows(outcome.frame)
    train = expand_directions(kept, created_stamp(ctx.config.created))
    planted = _planted(outcome.frame, ctx.fixtures, tuple(config.stages))
    facts = _filter_facts(
        outcome.frame, kept, train.height, outcome.reason_counts, planted
    )
    directory = _own_dir(ctx, "filter")
    write_corpus(outcome.frame, directory / "scored.parquet")
    write_corpus(kept, directory / "kept.parquet")
    write_corpus(train, directory / "train.parquet")
    _write_counts(directory, facts)
    record = _filter_record(
        facts,
        _parts(outcome.stage_status),
        section,
        token_check_skipped=outcome.stage_status["length"].startswith("partial"),
    )
    return StageResult(record, facts)
