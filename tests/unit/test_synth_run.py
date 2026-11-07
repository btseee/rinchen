"""The job runner: it cuts the monolingual frame into shards, reuses the shards
a crashed or earlier run left (each valid file is decoded once), generates the
others (their candidates are used from memory, not read back) and selects shard
by shard, returning the corpus rows and the counts. The generator is a scripted
fake and the QE scorer a fake, so no test needs a server, a model or a GPU."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import polars as pl
import pytest

from rinchen.config.filters import QeConfig
from rinchen.config.synth import (
    GeneratorConfig,
    JobConfig,
    SelectionConfig,
    SynthConfig,
    VllmConfig,
)
from rinchen.corpus.schema import CORPUS_SCHEMA, make_row_id
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.filter.qe import QEScorer
from rinchen.synth.plan import Plan
from rinchen.synth.run import RunPlan, RunResult, ShardProgress, plan_run, run_job
from rinchen.synth.select import Kind, SourceRows, build_rows, select_candidates
from rinchen.synth.shards import (
    ShardStore,
    job_fingerprint,
    shard_directory,
)
from rinchen.synth.translator import Translator
from tests.support.planted_synth import (
    FIRST_RUN_STATS,
    PLANTED_COUNTS,
    FakeScorer,
    ScriptedTranslator,
    planted_monolingual,
    planted_scorer,
    planted_script,
)

GENERATOR = "vllm_default"
SEED = 7
SHARD_SIZE = 8
SHARD_FILES = [f"shard-{index:05d}.parquet" for index in range(5)]


def _job(kind: Kind = "bt", shard_size: int = SHARD_SIZE) -> JobConfig:
    return JobConfig(
        kind=kind,
        mono_lang="khk_Cyrl",
        other_lang="eng_Latn",
        generator=GENERATOR,
        shard_size=shard_size,
    )


def _run(
    out_dir: Path,
    kind: Kind = "bt",
    *,
    frame: pl.DataFrame | None = None,
    cfg: SynthConfig | None = None,
    job: JobConfig | None = None,
    generator_name: str = GENERATOR,
    generator_cfg: GeneratorConfig | None = None,
    translator: Translator | None = None,
    scorer: QEScorer | None | Literal["planted"] = "planted",
    seed: int = SEED,
    force: bool = False,
    progress: Callable[[ShardProgress], None] | None = None,
) -> RunResult:
    """`run_job` over the planted 40-row frame with the planted translator and
    scorer, 5 shards of 8 rows; the keyword arguments replace any of the inputs
    (`scorer=None` runs without a scorer)."""
    frame = planted_monolingual() if frame is None else frame
    cfg = SynthConfig() if cfg is None else cfg
    return run_job(
        _job(kind) if job is None else job,
        generator_name,
        cfg.generators[generator_name] if generator_cfg is None else generator_cfg,
        ScriptedTranslator(planted_script(frame)) if translator is None else translator,
        frame,
        out_dir=out_dir,
        cfg=cfg,
        scorer=planted_scorer(frame, kind) if isinstance(scorer, str) else scorer,
        seed=seed,
        force=force,
        progress=progress,
    )


def _shard_dir(
    out_dir: Path,
    kind: Kind = "bt",
    *,
    frame: pl.DataFrame | None = None,
    shard_size: int = SHARD_SIZE,
    cfg: SynthConfig | None = None,
    seed: int = SEED,
) -> Path:
    """Where the shards of a `_run` live, computed through the public shard
    functions rather than read from the result."""
    cfg = cfg or SynthConfig()
    frame = planted_monolingual() if frame is None else frame
    job = _job(kind, shard_size)
    fingerprint = job_fingerprint(
        frame, job, cfg.generators[GENERATOR], shard_size, seed
    )
    return shard_directory(out_dir, fingerprint)


def _files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def _shard_texts(*indexes: int) -> list[tuple[str, ...]]:
    texts = planted_monolingual()["text"].to_list()
    return [tuple(texts[index * 8 : index * 8 + 8]) for index in indexes]


class _RaisingScorer:
    requires_cuda = False

    def score(self, sources: Any, translations: Any) -> list[float]:
        raise RuntimeError("scorer out of memory")


def _without_created(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.drop("created")


def _number(row_index: int) -> str:
    return f"{row_index:03d}"


# What the selection keeps from the planted frame: rows of the classes 0, 6 and 7
# (modulo 8), the winner of each as the planted script and the chrF consensus
# make it.
def _expected_winner(row_index: int) -> str | None:
    number = _number(row_index)
    return {
        0: f"The state of Mongolia {number} stands",
        6: f"Mongolia is edge {number}",
        7: f"Mongolia is a country {number} indeed",
    }.get(row_index % 8)


SELECTED_ROWS = [index for index in range(40) if _expected_winner(index) is not None]


# --- the rows and the counts -------------------------------------------------


def test_a_back_translation_run_returns_the_selected_rows_with_the_synthetic_side_as_source(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, "bt")

    corpus = result.corpus
    assert corpus.schema == CORPUS_SCHEMA
    assert corpus.height == 15
    assert corpus["src"].to_list() == [_expected_winner(i) for i in SELECTED_ROWS]
    assert corpus["tgt"].to_list() == [
        f"Монгол улс {_number(i)}" for i in SELECTED_ROWS
    ]
    assert corpus["src_lang"].unique().to_list() == ["eng_Latn"]
    assert corpus["tgt_lang"].unique().to_list() == ["khk_Cyrl"]
    assert corpus["origin"].cast(pl.String).unique().to_list() == ["bt"]
    assert corpus["corpus"].unique().to_list() == ["bt:vllm_default"]
    assert corpus["split"].cast(pl.String).unique().to_list() == ["train"]


def test_a_back_translation_run_records_purity_and_qe_with_the_exact_first_row(
    tmp_path: Path,
) -> None:
    corpus = _run(tmp_path, "bt").corpus

    assert corpus.row(0, named=True) == {
        "id": make_row_id(
            "eng_Latn",
            "khk_Cyrl",
            "The state of Mongolia 000 stands",
            "Монгол улс 000",
        ),
        "src_lang": "eng_Latn",
        "tgt_lang": "khk_Cyrl",
        "src": "The state of Mongolia 000 stands",
        "tgt": "Монгол улс 000",
        "origin": "bt",
        "corpus": "bt:vllm_default",
        "doc_id": None,
        "seg_idx": None,
        "split": "train",
        "scores": {
            "lid_src": None,
            "lid_tgt": None,
            "script_src": 1.0,
            "script_tgt": None,
            "len_ratio": None,
            "num_ok": None,
            "labse": None,
            "metricx_qe": 1.0,
            "cometkiwi": None,
        },
        "filter_reason": None,
        "created": corpus["created"][0],
    }
    scores = corpus["scores"].struct
    assert scores.field("metricx_qe").to_list() == [1.0, 8.0, 1.0] * 5
    assert scores.field("script_src").to_list() == [1.0] * 15
    assert scores.field("script_tgt").null_count() == 15


def test_a_forward_translation_run_returns_the_synthetic_side_as_target(
    tmp_path: Path,
) -> None:
    corpus = _run(tmp_path, "ft").corpus

    assert corpus.schema == CORPUS_SCHEMA
    assert corpus["src"].to_list() == [
        f"Монгол улс {_number(i)}" for i in SELECTED_ROWS
    ]
    assert corpus["tgt"].to_list() == [_expected_winner(i) for i in SELECTED_ROWS]
    assert corpus["src_lang"].unique().to_list() == ["khk_Cyrl"]
    assert corpus["tgt_lang"].unique().to_list() == ["eng_Latn"]
    assert corpus["origin"].cast(pl.String).unique().to_list() == ["synth_llm"]
    assert corpus["corpus"].unique().to_list() == ["ft:vllm_default"]
    scores = corpus["scores"].struct
    assert scores.field("script_tgt").to_list() == [1.0] * 15
    assert scores.field("script_src").null_count() == 15
    assert scores.field("metricx_qe").to_list() == [1.0, 8.0, 1.0] * 5


def test_a_forward_translation_run_exact_first_row(tmp_path: Path) -> None:
    row = _run(tmp_path, "ft").corpus.row(0, named=True)

    assert row["id"] == make_row_id(
        "khk_Cyrl", "eng_Latn", "Монгол улс 000", "The state of Mongolia 000 stands"
    )
    assert (row["src"], row["tgt"]) == (
        "Монгол улс 000",
        "The state of Mongolia 000 stands",
    )


@pytest.mark.parametrize("kind", ["bt", "ft"])
def test_the_counts_of_the_planted_run_are_exact_for_both_kinds(
    tmp_path: Path, kind: Kind
) -> None:
    result = _run(tmp_path, kind)

    assert dict(result.counts) == PLANTED_COUNTS
    assert result.qe_status == "ok"


def test_the_generator_is_asked_to_translate_the_monolingual_language_into_the_other(
    tmp_path: Path,
) -> None:
    for kind in ("bt", "ft"):
        translator = ScriptedTranslator(planted_script(planted_monolingual()))

        _run(tmp_path / kind, kind, translator=translator)

        assert {(src, tgt) for _, src, tgt in translator.calls} == {
            ("khk_Cyrl", "eng_Latn")
        }


def test_a_ctranslate2_generator_makes_neural_origin_rows(tmp_path: Path) -> None:
    cfg = SynthConfig()
    job = JobConfig(
        kind="ft",
        mono_lang="khk_Cyrl",
        other_lang="eng_Latn",
        generator="ct2_nllb_default",
        shard_size=SHARD_SIZE,
    )

    corpus = _run(
        tmp_path,
        "ft",
        job=job,
        cfg=cfg,
        generator_name="ct2_nllb_default",
    ).corpus

    assert corpus["origin"].cast(pl.String).unique().to_list() == ["synth_nmt"]
    assert corpus["corpus"].unique().to_list() == ["ft:ct2_nllb_default"]


def test_a_run_without_a_scorer_skips_the_qe_floor_and_says_so(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, "bt", scorer=None)

    assert dict(result.counts) == {
        "sources": 40,
        "candidates": 60,
        "dropped_empty": 5,
        "dropped_purity": 10,
        "dropped_qe": 0,
        "dropped_copy": 0,
        "no_candidates": 5,
        "no_survivor": 10,
        "selected": 25,
    }
    assert result.qe_status == "skipped: no scorer"
    assert result.corpus["scores"].struct.field("metricx_qe").null_count() == 25
    assert result.corpus["scores"].struct.field("cometkiwi").null_count() == 25


def test_the_qe_score_goes_to_the_column_of_the_configured_scorer(
    tmp_path: Path,
) -> None:
    cfg = SynthConfig(selection=SelectionConfig(qe=QeConfig(scorer="cometkiwi")))
    frame = planted_monolingual(8)

    result = _run(
        tmp_path,
        "bt",
        frame=frame,
        cfg=cfg,
        scorer=FakeScorer({}, default=0.9),
    )

    scores = result.corpus["scores"].struct
    assert result.corpus.height == 5
    assert scores.field("cometkiwi").to_list() == pytest.approx([0.9] * 5, abs=1e-6)
    assert scores.field("metricx_qe").null_count() == 5


def test_a_real_text_in_the_synthetic_script_that_the_generator_copies_is_counted_as_a_copy(
    tmp_path: Path,
) -> None:
    frame = pl.DataFrame({"id": ["a"], "lang": ["khk_Cyrl"], "text": ["Hello world"]})
    translator = ScriptedTranslator({"Hello world": ["Hello world", "Hello there"]})

    result = _run(tmp_path, "bt", frame=frame, translator=translator, scorer=None)

    assert result.counts["dropped_copy"] == 1
    assert result.counts["selected"] == 1
    assert result.corpus["src"].to_list() == ["Hello there"]


@pytest.mark.parametrize("shard_size", [8, 1])
def test_refusals_are_counted_and_a_run_of_nothing_but_refusals_gives_no_rows(
    tmp_path: Path, shard_size: int
) -> None:
    frame = planted_monolingual(3)
    translator = ScriptedTranslator({text: [] for text in frame["text"]})

    result = _run(
        tmp_path,
        "bt",
        frame=frame,
        translator=translator,
        job=_job("bt", shard_size),
    )

    assert dict(result.counts) == {
        "sources": 3,
        "candidates": 0,
        "dropped_empty": 0,
        "dropped_purity": 0,
        "dropped_qe": 0,
        "dropped_copy": 0,
        "no_candidates": 3,
        "no_survivor": 0,
        "selected": 0,
    }
    assert result.corpus.schema == CORPUS_SCHEMA
    assert result.corpus.height == 0


def _pairs_asked(scorer: FakeScorer) -> list[tuple[str, str]]:
    return [
        pair
        for sources, translations in scorer.calls
        for pair in zip(sources, translations, strict=True)
    ]


def _one_selection_over_the_whole_frame(
    kind: Kind, *, with_scorer: bool
) -> tuple[pl.DataFrame, dict[str, int], str, list[tuple[str, str]]]:
    """What a single `select_candidates` and `build_rows` over all 40 planted
    rows give: the corpus, the counts, the QE status and the pairs the scorer
    was asked about."""
    frame = planted_monolingual()
    script = planted_script(frame)
    sources = SourceRows(ids=tuple(frame["id"]), texts=tuple(frame["text"]))
    candidates = [script[text] for text in sources.texts]
    selection_cfg = SynthConfig().selection
    scorer = planted_scorer(frame, kind) if with_scorer else None
    selection = select_candidates(
        sources,
        candidates,
        kind=kind,
        synthetic_lang="eng_Latn",
        cfg=selection_cfg,
        scorer=scorer,
    )
    corpus = build_rows(
        sources,
        candidates,
        selection,
        kind=kind,
        generator_kind="vllm",
        generator_name=GENERATOR,
        mono_lang="khk_Cyrl",
        other_lang="eng_Latn",
        qe_scorer=selection_cfg.qe.scorer if scorer is not None else None,
    )
    pairs = [] if scorer is None else _pairs_asked(scorer)
    return corpus, dict(selection.counts), selection.qe_status, pairs


@pytest.mark.parametrize("shard_size", [1, 3, 7, 8, 40, 100])
@pytest.mark.parametrize("with_scorer", [True, False])
@pytest.mark.parametrize("kind", ["bt", "ft"])
def test_selecting_shard_by_shard_equals_one_selection_over_the_whole_frame(
    tmp_path: Path, kind: Kind, with_scorer: bool, shard_size: int
) -> None:
    corpus, counts, qe_status, pairs = _one_selection_over_the_whole_frame(
        kind, with_scorer=with_scorer
    )
    scorer = planted_scorer(planted_monolingual(), kind) if with_scorer else None

    result = _run(tmp_path, kind, job=_job(kind, shard_size), scorer=scorer)

    assert _without_created(result.corpus).equals(_without_created(corpus))
    assert dict(result.counts) == counts
    assert result.qe_status == qe_status
    if scorer is not None:
        asked = _pairs_asked(scorer)
        assert len(asked) == len(pairs) == 45
        assert len(set(asked)) == len(asked)  # no pair is scored twice
        assert sorted(asked) == sorted(pairs)


def test_every_row_of_a_multi_shard_run_has_the_same_timestamp(tmp_path: Path) -> None:
    corpus = _run(tmp_path, job=_job("bt", 1)).corpus

    assert corpus.height == 15
    assert corpus["created"].n_unique() == 1


def test_the_frame_order_does_not_change_the_corpus_or_the_fingerprint(
    tmp_path: Path,
) -> None:
    frame = planted_monolingual()

    straight = _run(tmp_path / "straight", "bt", frame=frame)
    # The planted outcomes belong to the rows, not to their position in the frame.
    reversed_ = _run(
        tmp_path / "reversed",
        "bt",
        frame=frame.reverse(),
        translator=ScriptedTranslator(planted_script(frame)),
        scorer=planted_scorer(frame, "bt"),
    )

    assert straight.fingerprint == reversed_.fingerprint
    assert _without_created(straight.corpus).equals(_without_created(reversed_.corpus))


# --- shards, fingerprint, stats ----------------------------------------------


def test_a_first_run_generates_every_shard_into_the_directory_of_its_fingerprint(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "synth"

    result = _run(out_dir)

    fingerprint = job_fingerprint(
        planted_monolingual(),
        _job(),
        SynthConfig().generators[GENERATOR],
        SHARD_SIZE,
        SEED,
    )
    assert result.fingerprint == fingerprint
    assert result.shard_directory == out_dir / f"shards-{fingerprint[:12]}"
    assert _files(result.shard_directory) == SHARD_FILES
    assert dict(result.shard_stats) == FIRST_RUN_STATS


def test_the_translator_gets_each_shard_once_in_order(tmp_path: Path) -> None:
    translator = ScriptedTranslator(planted_script(planted_monolingual()))

    _run(tmp_path, translator=translator)

    assert translator.calls == [
        (texts, "khk_Cyrl", "eng_Latn") for texts in _shard_texts(0, 1, 2, 3, 4)
    ]


def test_a_crash_leaves_the_finished_shards_and_names_the_failed_one(
    tmp_path: Path,
) -> None:
    failure = UpstreamError("server gone")
    translator = ScriptedTranslator(
        planted_script(planted_monolingual()), fail_on_call={2: failure}
    )

    with pytest.raises(
        UpstreamError, match=r"^shard 2 of 5 failed: server gone$"
    ) as info:
        _run(tmp_path, translator=translator)

    assert info.value.__cause__ is failure
    assert _files(_shard_dir(tmp_path)) == SHARD_FILES[:2]
    assert len(translator.calls) == 3


def test_a_crash_on_the_first_shard_leaves_no_shard_files(tmp_path: Path) -> None:
    translator = ScriptedTranslator(
        planted_script(planted_monolingual()), fail_on_call={0: UpstreamError("down")}
    )

    with pytest.raises(UpstreamError, match=r"shard 0 of 5 failed: down"):
        _run(tmp_path, translator=translator)

    directory = _shard_dir(tmp_path)
    assert not directory.exists() or _files(directory) == []


def test_an_error_that_is_not_an_upstream_error_propagates_unchanged(
    tmp_path: Path,
) -> None:
    broken = RuntimeError("bug in the translator")
    translator = ScriptedTranslator(
        planted_script(planted_monolingual()), fail_on_call={1: broken}
    )

    with pytest.raises(RuntimeError, match="bug in the translator") as info:
        _run(tmp_path, translator=translator)

    assert info.value is broken
    assert _files(_shard_dir(tmp_path)) == SHARD_FILES[:1]


def test_a_rerun_after_a_crash_asks_only_for_the_unfinished_shards(
    tmp_path: Path,
) -> None:
    crashing = ScriptedTranslator(
        planted_script(planted_monolingual()), fail_on_call={2: UpstreamError("down")}
    )
    with pytest.raises(UpstreamError):
        _run(tmp_path / "resumed", translator=crashing)
    resuming = ScriptedTranslator(planted_script(planted_monolingual()))

    resumed = _run(tmp_path / "resumed", translator=resuming)

    assert resuming.calls == [
        (texts, "khk_Cyrl", "eng_Latn") for texts in _shard_texts(2, 3, 4)
    ]
    assert dict(resumed.shard_stats) == {
        "planned": 5,
        "reused": 2,
        "generated": 3,
        "regenerated_invalid": 0,
    }
    uninterrupted = _run(tmp_path / "uninterrupted")
    assert _without_created(resumed.corpus).equals(
        _without_created(uninterrupted.corpus)
    )
    assert dict(resumed.counts) == dict(uninterrupted.counts) == PLANTED_COUNTS


def test_a_rerun_of_a_finished_run_makes_no_request(tmp_path: Path) -> None:
    first = _run(tmp_path)
    translator = ScriptedTranslator(planted_script(planted_monolingual()))

    second = _run(tmp_path, translator=translator)

    assert translator.calls == []
    assert dict(second.shard_stats) == {
        "planned": 5,
        "reused": 5,
        "generated": 0,
        "regenerated_invalid": 0,
    }
    assert _without_created(second.corpus).equals(_without_created(first.corpus))


def _garbage(path: Path) -> None:
    path.write_bytes(b"this is not parquet")


def _truncated(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:40])


def _empty_file(path: Path) -> None:
    path.write_bytes(b"")


def _another_shards_ids(path: Path) -> None:
    ShardStore(path.parent).write(3, [f"x{index}" for index in range(8)], [["a"]] * 8)


@pytest.mark.parametrize(
    "damage", [_garbage, _truncated, _empty_file, _another_shards_ids]
)
def test_a_damaged_shard_file_is_regenerated_and_counted(
    tmp_path: Path, damage: Callable[[Path], None]
) -> None:
    first = _run(tmp_path)
    damage(first.shard_directory / SHARD_FILES[3])
    translator = ScriptedTranslator(planted_script(planted_monolingual()))

    second = _run(tmp_path, translator=translator)

    assert translator.calls == [
        (texts, "khk_Cyrl", "eng_Latn") for texts in _shard_texts(3)
    ]
    assert dict(second.shard_stats) == {
        "planned": 5,
        "reused": 4,
        "generated": 1,
        "regenerated_invalid": 1,
    }
    assert _without_created(second.corpus).equals(_without_created(first.corpus))
    third = _run(tmp_path, translator=ScriptedTranslator({}))
    assert dict(third.shard_stats)["reused"] == 5


def test_force_regenerates_every_shard_and_reuses_none(tmp_path: Path) -> None:
    first = _run(tmp_path)
    _garbage(first.shard_directory / SHARD_FILES[1])
    translator = ScriptedTranslator(planted_script(planted_monolingual()))

    forced = _run(tmp_path, translator=translator, force=True)

    assert translator.calls == [
        (texts, "khk_Cyrl", "eng_Latn") for texts in _shard_texts(0, 1, 2, 3, 4)
    ]
    assert dict(forced.shard_stats) == FIRST_RUN_STATS
    assert _without_created(forced.corpus).equals(_without_created(first.corpus))


def test_a_changed_shard_size_uses_a_new_directory_and_leaves_the_old_one_alone(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path)
    before = {path.name: path.read_bytes() for path in first.shard_directory.iterdir()}
    translator = ScriptedTranslator(planted_script(planted_monolingual()))

    second = _run(tmp_path, job=_job("bt", shard_size=10), translator=translator)

    assert second.shard_directory != first.shard_directory
    assert second.shard_directory == _shard_dir(tmp_path, shard_size=10)
    assert _files(second.shard_directory) == [
        f"shard-0000{i}.parquet" for i in range(4)
    ]
    assert dict(second.shard_stats) == {
        "planned": 4,
        "reused": 0,
        "generated": 4,
        "regenerated_invalid": 0,
    }
    assert len(translator.calls) == 4
    assert before == {
        path.name: path.read_bytes() for path in first.shard_directory.iterdir()
    }
    assert _without_created(second.corpus).equals(_without_created(first.corpus))


def test_a_changed_generator_config_uses_a_new_directory_and_leaves_the_old_one_alone(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path)
    before = {path.name: path.read_bytes() for path in first.shard_directory.iterdir()}
    changed = SynthConfig(generators={GENERATOR: VllmConfig(n=3)}, jobs={})
    translator = ScriptedTranslator(planted_script(planted_monolingual()))

    second = _run(tmp_path, cfg=changed, translator=translator)

    assert second.shard_directory != first.shard_directory
    assert second.shard_directory == _shard_dir(tmp_path, cfg=changed)
    assert len(translator.calls) == 5
    assert dict(second.shard_stats)["reused"] == 0
    assert before == {
        path.name: path.read_bytes() for path in first.shard_directory.iterdir()
    }


def test_the_seed_names_the_shard_directory_but_does_not_change_the_corpus(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path, seed=1)
    other = _run(tmp_path, seed=2)

    assert first.fingerprint != other.fingerprint
    assert first.shard_directory != other.shard_directory
    assert other.shard_directory == _shard_dir(tmp_path, seed=2)
    assert dict(other.shard_stats)["reused"] == 0
    assert _without_created(first.corpus).equals(_without_created(other.corpus))


def test_the_corpus_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", seed=3)
    second = _run(tmp_path / "second", seed=3)

    assert first.fingerprint == second.fingerprint
    assert _without_created(first.corpus).equals(_without_created(second.corpus))
    assert dict(first.counts) == dict(second.counts)


def test_the_translator_is_never_called_with_an_empty_batch(tmp_path: Path) -> None:
    for rows in (1, 16, 17, 39, 40):
        translator = ScriptedTranslator(planted_script(planted_monolingual(rows)))

        _run(
            tmp_path / str(rows),
            frame=planted_monolingual(rows),
            translator=translator,
        )

        sizes = [len(texts) for texts, _, _ in translator.calls]
        assert sizes == [8] * (rows // 8) + ([rows % 8] if rows % 8 else [])
        assert all(sizes)


def test_a_run_over_a_single_row_makes_one_shard_and_one_request(
    tmp_path: Path,
) -> None:
    frame = planted_monolingual(1)

    result = _run(tmp_path, frame=frame)

    assert dict(result.shard_stats) == {
        "planned": 1,
        "reused": 0,
        "generated": 1,
        "regenerated_invalid": 0,
    }
    assert result.corpus.height == 1


# --- progress ------------------------------------------------------------------


def test_progress_is_reported_once_per_shard_in_order(tmp_path: Path) -> None:
    events: list[ShardProgress] = []

    _run(tmp_path, progress=events.append)

    assert events == [ShardProgress(index, 5, False) for index in range(5)]


def test_progress_marks_the_reused_shards_after_a_crash(tmp_path: Path) -> None:
    crashed: list[ShardProgress] = []
    crashing = ScriptedTranslator(
        planted_script(planted_monolingual()), fail_on_call={2: UpstreamError("down")}
    )
    with pytest.raises(UpstreamError):
        _run(tmp_path, translator=crashing, progress=crashed.append)
    resumed: list[ShardProgress] = []

    _run(tmp_path, progress=resumed.append)

    assert crashed == [ShardProgress(0, 5, False), ShardProgress(1, 5, False)]
    assert resumed == [
        ShardProgress(0, 5, True),
        ShardProgress(1, 5, True),
        ShardProgress(2, 5, False),
        ShardProgress(3, 5, False),
        ShardProgress(4, 5, False),
    ]


def test_progress_is_reported_before_the_shard_is_selected(tmp_path: Path) -> None:
    events: list[ShardProgress] = []

    with pytest.raises(RuntimeError, match="scorer out of memory"):
        _run(tmp_path, scorer=_RaisingScorer(), progress=events.append)

    assert events == [ShardProgress(0, 5, False)]


def test_a_generated_shard_is_on_disk_when_its_progress_is_reported(
    tmp_path: Path,
) -> None:
    directory = _shard_dir(tmp_path)
    on_disk: list[bool] = []

    def record(event: ShardProgress) -> None:
        on_disk.append((directory / SHARD_FILES[event.index]).is_file())

    _run(tmp_path, progress=record)

    assert on_disk == [True] * 5


def _count_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch the Parquet reader to record the name of every file it opens."""
    opened: list[str] = []
    real_read = pl.read_parquet

    def counting_read(source: Any, *args: Any, **kwargs: Any) -> pl.DataFrame:
        opened.append(Path(source).name)
        return real_read(source, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", counting_read)
    return opened


def test_a_first_run_decodes_no_shard_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = _count_reads(monkeypatch)

    result = _run(tmp_path)

    assert opened == []
    assert result.corpus.height == 15


def test_a_rerun_decodes_each_existing_shard_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(tmp_path)
    opened = _count_reads(monkeypatch)

    _run(tmp_path)

    assert sorted(opened) == SHARD_FILES


def test_a_resumed_run_decodes_the_finished_shards_once_and_never_opens_a_missing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crashing = ScriptedTranslator(
        planted_script(planted_monolingual()), fail_on_call={2: UpstreamError("down")}
    )
    with pytest.raises(UpstreamError):
        _run(tmp_path, translator=crashing)
    opened = _count_reads(monkeypatch)

    _run(tmp_path)

    assert sorted(opened) == SHARD_FILES[:2]


def test_a_corrupt_shard_is_decoded_once_and_regenerated_without_a_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run(tmp_path)
    _garbage(first.shard_directory / SHARD_FILES[3])
    opened = _count_reads(monkeypatch)
    translator = ScriptedTranslator(planted_script(planted_monolingual()))

    second = _run(tmp_path, translator=translator)

    assert sorted(opened) == SHARD_FILES
    assert translator.calls == [
        (texts, "khk_Cyrl", "eng_Latn") for texts in _shard_texts(3)
    ]
    assert dict(second.shard_stats)["regenerated_invalid"] == 1


def test_force_does_not_open_the_shard_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(tmp_path)
    opened = _count_reads(monkeypatch)

    _run(tmp_path, force=True)

    assert opened == []


def test_progress_is_optional(tmp_path: Path) -> None:
    assert _run(tmp_path, progress=None).corpus.height == 15


def test_progress_and_run_result_are_frozen(tmp_path: Path) -> None:
    event = ShardProgress(0, 5, False)
    result = _run(tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.index = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.fingerprint = "x"  # type: ignore[misc]
    assert (event.index, event.total, event.reused) == (0, 5, False)


# --- what the translator may return --------------------------------------------


class _AlteredOnCall(ScriptedTranslator):
    """Answers like `ScriptedTranslator`, but passes the reply of the call
    number `call` through `alter` first."""

    def __init__(
        self,
        script: dict[str, list[str]],
        call: int,
        alter: Callable[[list[list[str]]], Any],
    ) -> None:
        super().__init__(script)
        self._call = call
        self._alter = alter

    def translate(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[list[str]]:
        reply = super().translate(texts, src_lang, tgt_lang)
        return self._alter(reply) if len(self.calls) - 1 == self._call else reply


@pytest.mark.parametrize(
    ("alter", "lists"),
    [
        (lambda reply: reply[:-1], 7),
        (lambda reply: [*reply, ["extra"]], 9),
    ],
)
def test_a_wrong_number_of_candidate_lists_is_a_data_error_and_writes_nothing(
    tmp_path: Path,
    alter: Callable[[list[list[str]]], list[list[str]]],
    lists: int,
) -> None:
    translator = _AlteredOnCall(planted_script(planted_monolingual()), 1, alter)

    with pytest.raises(
        DataError, match=rf"shard 1.*{lists} candidate lists for 8 texts"
    ):
        _run(tmp_path, translator=translator)

    assert _files(_shard_dir(tmp_path)) == SHARD_FILES[:1]
    assert len(translator.calls) == 2


@pytest.mark.parametrize(
    ("alter", "type_name"),
    [
        (lambda reply: None, "NoneType"),
        (lambda reply: (candidates for candidates in reply), "generator"),
        (lambda reply: tuple(reply), "tuple"),
        (lambda reply: "x" * len(reply), "str"),
    ],
    ids=["None", "generator", "tuple", "str"],
)
def test_a_reply_that_is_not_a_list_is_a_data_error_naming_the_shard_and_the_type(
    tmp_path: Path, alter: Callable[[list[list[str]]], Any], type_name: str
) -> None:
    translator = _AlteredOnCall(planted_script(planted_monolingual()), 1, alter)

    with pytest.raises(
        DataError,
        match=rf"^shard 1 of 5: the translator returned a {type_name}, not a list$",
    ):
        _run(tmp_path, translator=translator)

    assert _files(_shard_dir(tmp_path)) == SHARD_FILES[:1]
    assert len(translator.calls) == 2


@pytest.mark.parametrize("bad", [None, "a string", ("x",), [None], [1]])
def test_a_reply_that_is_not_a_list_of_candidate_strings_is_a_data_error(
    tmp_path: Path, bad: Any
) -> None:
    frame = planted_monolingual(2)

    class Bad:
        def translate(self, texts: Any, src_lang: str, tgt_lang: str) -> list[Any]:
            return [["fine"], bad]

    with pytest.raises(DataError, match=r"shard 0.*text 1"):
        _run(tmp_path, frame=frame, translator=Bad())

    directory = _shard_dir(tmp_path, frame=frame)
    assert not directory.exists() or _files(directory) == []


def test_a_candidate_list_that_is_empty_is_stored_as_a_refusal(tmp_path: Path) -> None:
    frame = planted_monolingual(2)
    translator = ScriptedTranslator({text: [] for text in frame["text"]})

    result = _run(tmp_path, frame=frame, translator=translator, scorer=None)

    store = ShardStore(result.shard_directory)
    assert store.read(0, frame["id"].to_list()) == [[], []]


# --- what the translator may raise ---------------------------------------------


@pytest.mark.parametrize("error_type", [DataError, ConfigError])
def test_a_data_or_config_error_of_the_translator_keeps_its_type_and_gains_the_shard(
    tmp_path: Path, error_type: type[Exception]
) -> None:
    original = error_type("the model file is gone")
    translator = ScriptedTranslator(
        planted_script(planted_monolingual()), fail_on_call={2: original}
    )

    with pytest.raises(
        error_type, match=r"^shard 2 of 5: the model file is gone$"
    ) as info:
        _run(tmp_path, translator=translator)

    assert type(info.value) is error_type
    assert info.value.__cause__ is original
    assert _files(_shard_dir(tmp_path)) == SHARD_FILES[:2]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (OSError("disk gone"), "disk gone"),
        (ConnectionResetError("reset by peer"), "reset by peer"),
        (TimeoutError("timed out"), "timed out"),
    ],
    ids=["OSError", "ConnectionResetError", "TimeoutError"],
)
def test_an_os_error_of_the_translator_becomes_an_upstream_error_naming_the_shard(
    tmp_path: Path, error: OSError, message: str
) -> None:
    translator = ScriptedTranslator(
        planted_script(planted_monolingual()), fail_on_call={2: error}
    )

    with pytest.raises(
        UpstreamError, match=rf"^shard 2 of 5 failed: {message}$"
    ) as info:
        _run(tmp_path, translator=translator)

    assert info.value.__cause__ is error
    assert _files(_shard_dir(tmp_path)) == SHARD_FILES[:2]


@pytest.mark.parametrize(
    "interrupt", [KeyboardInterrupt(), SystemExit(3)], ids=["Ctrl-C", "exit"]
)
def test_an_interrupt_in_the_translator_propagates_unchanged_with_the_finished_shards(
    tmp_path: Path, interrupt: BaseException
) -> None:
    translator = ScriptedTranslator(
        planted_script(planted_monolingual()), fail_on_call={2: interrupt}
    )

    with pytest.raises(type(interrupt)) as info:
        _run(tmp_path, translator=translator)

    assert info.value is interrupt
    # Two shard files and nothing else: no temporary file of the third.
    assert _files(_shard_dir(tmp_path)) == SHARD_FILES[:2]
    resuming = ScriptedTranslator(planted_script(planted_monolingual()))
    rerun = _run(tmp_path, translator=resuming)
    assert resuming.calls == [
        (texts, "khk_Cyrl", "eng_Latn") for texts in _shard_texts(2, 3, 4)
    ]
    assert dict(rerun.shard_stats)["reused"] == 2


# --- the inputs the runner refuses ----------------------------------------------


def test_a_generator_name_outside_the_config_is_a_config_error(tmp_path: Path) -> None:
    translator = ScriptedTranslator({})
    out_dir = tmp_path / "synth"

    job = JobConfig(
        kind="bt",
        mono_lang="khk_Cyrl",
        other_lang="eng_Latn",
        generator="nope",
        shard_size=SHARD_SIZE,
    )

    with pytest.raises(ConfigError, match=r"'nope' is not in the config"):
        _run(
            out_dir,
            job=job,
            generator_name="nope",
            generator_cfg=VllmConfig(),
            translator=translator,
        )

    assert translator.calls == []
    assert not out_dir.exists()


def test_a_job_that_names_another_generator_than_the_one_given_is_a_config_error(
    tmp_path: Path,
) -> None:
    translator = ScriptedTranslator({})
    out_dir = tmp_path / "synth"
    job = JobConfig(
        kind="bt",
        mono_lang="khk_Cyrl",
        other_lang="eng_Latn",
        generator="llamacpp_default",
        shard_size=SHARD_SIZE,
    )

    with pytest.raises(ConfigError, match=r"'llamacpp_default'.*'vllm_default'"):
        _run(out_dir, job=job, translator=translator)

    assert translator.calls == []
    assert not out_dir.exists()


def test_a_generator_config_that_differs_from_the_named_one_is_a_config_error(
    tmp_path: Path,
) -> None:
    translator = ScriptedTranslator({})
    out_dir = tmp_path / "synth"

    with pytest.raises(ConfigError, match="'vllm_default'"):
        _run(out_dir, generator_cfg=VllmConfig(n=3), translator=translator)

    assert translator.calls == []
    assert not out_dir.exists()


def test_a_frame_in_another_language_than_the_job_is_a_data_error(
    tmp_path: Path,
) -> None:
    translator = ScriptedTranslator({})
    out_dir = tmp_path / "synth"

    with pytest.raises(DataError, match="'eng_Latn'"):
        _run(
            out_dir,
            frame=planted_monolingual(3, lang="eng_Latn"),
            translator=translator,
        )

    assert translator.calls == []
    assert not out_dir.exists()


def test_a_frame_without_rows_is_a_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no monolingual rows"):
        _run(tmp_path, frame=planted_monolingual(1).clear())


@pytest.mark.parametrize("missing", ["id", "text"])
def test_a_frame_without_an_id_or_text_column_is_a_data_error_naming_it(
    tmp_path: Path, missing: str
) -> None:
    translator = ScriptedTranslator({})
    out_dir = tmp_path / "synth"

    with pytest.raises(DataError, match=rf"required columns: \['{missing}'\]"):
        _run(
            out_dir,
            frame=planted_monolingual(3).drop(missing),
            translator=translator,
            scorer=None,
        )

    assert translator.calls == []
    assert not out_dir.exists()


# --- planning -------------------------------------------------------------------


def _plan_run(
    out_dir: Path,
    *,
    job: JobConfig | None = None,
    generator_name: str = GENERATOR,
    generator_cfg: GeneratorConfig | None = None,
    frame: pl.DataFrame | None = None,
    seed: int = SEED,
) -> RunPlan:
    cfg = SynthConfig()
    return plan_run(
        _job() if job is None else job,
        generator_name,
        cfg.generators[generator_name] if generator_cfg is None else generator_cfg,
        planted_monolingual() if frame is None else frame,
        out_dir=out_dir,
        cfg=cfg,
        seed=seed,
    )


def test_plan_run_gives_the_plan_the_shards_the_fingerprint_and_the_directory(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "synth"

    run_plan = _plan_run(out_dir)

    fingerprint = job_fingerprint(
        planted_monolingual(),
        _job(),
        SynthConfig().generators[GENERATOR],
        SHARD_SIZE,
        SEED,
    )
    assert run_plan.plan == Plan(
        kind="bt",
        generator_src_lang="khk_Cyrl",
        generator_tgt_lang="eng_Latn",
        real_lang="khk_Cyrl",
        synthetic_lang="eng_Latn",
    )
    assert [shard.index for shard in run_plan.shards] == [0, 1, 2, 3, 4]
    assert run_plan.shards[0].ids == tuple(f"s{index:03d}" for index in range(8))
    assert [shard.texts for shard in run_plan.shards] == [
        texts for (texts,) in (_shard_texts(index) for index in range(5))
    ]
    assert run_plan.fingerprint == fingerprint
    assert run_plan.shard_directory == out_dir / f"shards-{fingerprint[:12]}"
    assert not out_dir.exists()


@pytest.mark.parametrize("kind", ["bt", "ft"])
def test_run_job_works_in_the_shard_directory_that_plan_run_names(
    tmp_path: Path, kind: Kind
) -> None:
    planned = plan_run(
        _job(kind),
        GENERATOR,
        SynthConfig().generators[GENERATOR],
        planted_monolingual(),
        out_dir=tmp_path,
        cfg=SynthConfig(),
        seed=SEED,
    )

    result = _run(tmp_path, kind)

    assert planned.plan.kind == kind
    assert (result.fingerprint, result.shard_directory) == (
        planned.fingerprint,
        planned.shard_directory,
    )
    assert _files(planned.shard_directory) == SHARD_FILES


def test_plan_run_follows_the_seed_and_the_shard_size(tmp_path: Path) -> None:
    base = _plan_run(tmp_path)

    other_seed = _plan_run(tmp_path, seed=SEED + 1)
    other_size = _plan_run(tmp_path, job=_job("bt", shard_size=10))

    assert other_seed.fingerprint != base.fingerprint
    assert other_seed.shard_directory != base.shard_directory
    assert len(base.shards) == 5
    assert len(other_size.shards) == 4
    assert other_size.fingerprint != base.fingerprint


def test_a_run_plan_is_frozen(tmp_path: Path) -> None:
    run_plan = _plan_run(tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        run_plan.fingerprint = "x"  # type: ignore[misc]


def test_plan_run_refuses_the_inputs_run_job_refuses_and_creates_nothing(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "synth"
    other_job = JobConfig(
        kind="bt",
        mono_lang="khk_Cyrl",
        other_lang="eng_Latn",
        generator="llamacpp_default",
        shard_size=SHARD_SIZE,
    )

    with pytest.raises(ConfigError, match=r"'llamacpp_default'.*'vllm_default'"):
        _plan_run(out_dir, job=other_job)
    with pytest.raises(ConfigError, match=r"'vllm_default'.*differs"):
        _plan_run(out_dir, generator_cfg=VllmConfig(n=3))
    with pytest.raises(DataError, match=r"'eng_Latn'"):
        _plan_run(out_dir, frame=planted_monolingual(3, lang="eng_Latn"))
    with pytest.raises(DataError, match=r"no monolingual rows"):
        _plan_run(out_dir, frame=planted_monolingual(1).clear())
    with pytest.raises(DataError, match=r"required columns: \['text'\]"):
        _plan_run(out_dir, frame=planted_monolingual(3).drop("text"))

    assert not out_dir.exists()


# --- failures after generation ---------------------------------------------------


def test_a_scorer_that_fails_leaves_the_shards_written_so_far_and_the_rerun_resumes(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="scorer out of memory"):
        _run(tmp_path, scorer=_RaisingScorer())
    # Shard 0 was written before its selection failed; the others were never asked for.
    assert _files(_shard_dir(tmp_path)) == SHARD_FILES[:1]
    translator = ScriptedTranslator(planted_script(planted_monolingual()))

    rerun = _run(tmp_path, translator=translator)

    assert translator.calls == [
        (texts, "khk_Cyrl", "eng_Latn") for texts in _shard_texts(1, 2, 3, 4)
    ]
    assert dict(rerun.shard_stats) == {
        "planned": 5,
        "reused": 1,
        "generated": 4,
        "regenerated_invalid": 0,
    }
    assert dict(rerun.counts) == PLANTED_COUNTS


class _ScorerFailingOnCall:
    """The planted scorer, which raises on its call number `call` (0-based)."""

    requires_cuda = False

    def __init__(self, inner: FakeScorer, call: int) -> None:
        self._inner = inner
        self._call = call

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        if len(self._inner.calls) == self._call:
            raise RuntimeError("scorer out of memory")
        return self._inner.score(sources, translations)


def test_a_scorer_that_fails_on_a_later_shard_keeps_the_shards_before_it_for_the_rerun(
    tmp_path: Path,
) -> None:
    scorer = _ScorerFailingOnCall(planted_scorer(planted_monolingual(), "bt"), call=2)
    events: list[ShardProgress] = []

    with pytest.raises(RuntimeError, match="scorer out of memory"):
        _run(tmp_path, scorer=scorer, progress=events.append)

    # Shard 2 was written before its selection failed; 3 and 4 were never asked for.
    assert _files(_shard_dir(tmp_path)) == SHARD_FILES[:3]
    assert events == [ShardProgress(index, 5, False) for index in range(3)]
    translator = ScriptedTranslator(planted_script(planted_monolingual()))

    rerun = _run(tmp_path, translator=translator)

    assert translator.calls == [
        (texts, "khk_Cyrl", "eng_Latn") for texts in _shard_texts(3, 4)
    ]
    assert dict(rerun.shard_stats) == {
        "planned": 5,
        "reused": 3,
        "generated": 2,
        "regenerated_invalid": 0,
    }
    assert dict(rerun.counts) == PLANTED_COUNTS


def _row_of(real_text: str) -> int:
    """The position of a planted real text (`Монгол улс 013`) in the frame."""
    return int(real_text.rsplit(" ", 1)[1])


@pytest.mark.parametrize("shard_size", [3, 8, 13, 40])
@pytest.mark.parametrize("kind", ["bt", "ft"])
def test_no_scorer_call_spans_two_shards_and_the_calls_follow_the_shard_order(
    tmp_path: Path, kind: Kind, shard_size: int
) -> None:
    scorer = planted_scorer(planted_monolingual(), kind)

    _run(tmp_path, kind, job=_job(kind, shard_size), scorer=scorer)

    # The pair is (candidate, real text) for bt and (real text, candidate) for ft.
    shards_per_call = [
        [
            _row_of(real) // shard_size
            for real in (translations if kind == "bt" else sources)
        ]
        for sources, translations in scorer.calls
    ]
    assert all(len(set(shards)) == 1 for shards in shards_per_call), shards_per_call
    first_shard_of_each_call = [shards[0] for shards in shards_per_call]
    assert first_shard_of_each_call == sorted(first_shard_of_each_call)
    assert sum(len(shards) for shards in shards_per_call) == 45


def test_a_scorer_that_returns_nan_for_every_pair_drops_every_candidate(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, scorer=FakeScorer({}, default=math.nan))

    assert result.corpus.height == 0
    assert result.counts["dropped_qe"] == 45
