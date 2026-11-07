"""Unit tests for `rinchen.synth.shards`: reading the monolingual input,
planning deterministic shards, fingerprinting a job and the crash-safe store of
per-shard generation results."""

from __future__ import annotations

import dataclasses
import re
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rinchen.config.synth import Ct2Config, JobConfig, LlamaCppConfig, VllmConfig
from rinchen.errors import DataError
from rinchen.synth.shards import (
    Invalid,
    Loaded,
    Missing,
    Shard,
    ShardStore,
    job_fingerprint,
    plan_shards,
    read_monolingual,
    shard_directory,
)

NNBSP = chr(0x202F)
NON_BMP = chr(0x1F600)
NUL = chr(0)
LONE_SURROGATE = chr(0xD800)
GOLDEN_DIGEST = "d47bc26907d18c0d81db55c6712b9f874d96e7c0a4639d9aa2a4c82acced45b4"

JOB = JobConfig(
    kind="bt", mono_lang="khk_Cyrl", other_lang="eng_Latn", generator="vllm_default"
)
GENERATOR = VllmConfig()

_STRING_COLUMNS = {"id": pl.String, "lang": pl.String, "text": pl.String}
_SHARD_SCHEMA = {"id": pl.String, "candidates": pl.List(pl.String)}


def _mono(rows: list[tuple[str, str, str]], **extra: list[Any]) -> pl.DataFrame:
    """A monolingual frame from (id, lang, text) rows, plus extra columns."""
    data: dict[str, list[Any]] = {
        "id": [row[0] for row in rows],
        "lang": [row[1] for row in rows],
        "text": [row[2] for row in rows],
        **extra,
    }
    return pl.DataFrame(data, schema_overrides=_STRING_COLUMNS)


def _letters(ids: str, lang: str = "khk_Cyrl") -> pl.DataFrame:
    """One row per character of `ids`, whose text is "text-<id>"."""
    return _mono([(char, lang, f"text-{char}") for char in ids])


def _write_parquet(path: Path, frame: pl.DataFrame) -> Path:
    frame.write_parquet(path)
    return path


def _mono_file(tmp_path: Path, frame: pl.DataFrame) -> Path:
    return _write_parquet(tmp_path / "mono.parquet", frame)


# --- read_monolingual ---------------------------------------------------------


def test_read_monolingual_returns_the_rows_sorted_by_id_in_string_order(
    tmp_path: Path,
) -> None:
    frame = _mono(
        [
            ("b", "khk_Cyrl", "second"),
            ("B", "khk_Cyrl", "capital"),
            ("10", "khk_Cyrl", "ten"),
            ("9", "khk_Cyrl", "nine"),
            ("a", "khk_Cyrl", "first"),
        ]
    )

    result = read_monolingual(_mono_file(tmp_path, frame))

    assert result["id"].to_list() == ["10", "9", "B", "a", "b"]
    assert result["text"].to_list() == ["ten", "nine", "capital", "first", "second"]


def test_read_monolingual_keeps_the_text_exactly_as_it_is_stored(
    tmp_path: Path,
) -> None:
    text = f"  padded{NNBSP}\n"
    frame = _mono([("a", "khk_Cyrl", text)])

    result = read_monolingual(_mono_file(tmp_path, frame))

    assert result["text"].to_list() == [text]


def test_read_monolingual_keeps_extra_columns(tmp_path: Path) -> None:
    frame = _mono(
        [("b", "khk_Cyrl", "two"), ("a", "khk_Cyrl", "one")],
        source=["web", "book"],
        rank=[2, 1],
    )

    result = read_monolingual(_mono_file(tmp_path, frame))

    assert set(result.columns) == {"id", "lang", "text", "source", "rank"}
    assert result["source"].to_list() == ["book", "web"]
    assert result["rank"].to_list() == [1, 2]


def test_read_monolingual_filters_to_the_requested_language(tmp_path: Path) -> None:
    frame = _mono(
        [
            ("1", "khk_Cyrl", "mongolian one"),
            ("2", "eng_Latn", "english"),
            ("3", "khk_Cyrl", "mongolian two"),
            ("4", "rus_Cyrl", "russian"),
        ]
    )

    result = read_monolingual(_mono_file(tmp_path, frame), lang="khk_Cyrl")

    assert result["id"].to_list() == ["1", "3"]
    assert set(result["lang"].to_list()) == {"khk_Cyrl"}


def test_read_monolingual_keeps_every_language_without_a_filter(
    tmp_path: Path,
) -> None:
    frame = _mono([("1", "khk_Cyrl", "a"), ("2", "eng_Latn", "b")])

    assert read_monolingual(_mono_file(tmp_path, frame))["id"].to_list() == ["1", "2"]


def test_read_monolingual_judges_only_the_rows_of_the_requested_language(
    tmp_path: Path,
) -> None:
    # The English rows never reach the generator, so their blank text and the id
    # they share with a Mongolian row are not this job's problem.
    frame = _mono(
        [
            ("1", "khk_Cyrl", "ok"),
            ("1", "eng_Latn", "same id, other language"),
            ("2", "eng_Latn", "   "),
        ]
    )

    result = read_monolingual(_mono_file(tmp_path, frame), lang="khk_Cyrl")

    assert result["id"].to_list() == ["1"]


def test_read_monolingual_rejects_a_filter_that_leaves_no_rows(tmp_path: Path) -> None:
    frame = _mono([("1", "eng_Latn", "a"), ("2", "rus_Cyrl", "b")])

    with pytest.raises(DataError, match=r"no rows.*khk_Cyrl") as caught:
        read_monolingual(_mono_file(tmp_path, frame), lang="khk_Cyrl")

    assert "eng_Latn" in str(caught.value)  # what the file does hold, for a typo


def test_read_monolingual_rejects_an_empty_file(tmp_path: Path) -> None:
    frame = _mono([])

    with pytest.raises(DataError, match=r"no rows"):
        read_monolingual(_mono_file(tmp_path, frame))


def test_read_monolingual_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match=r"not found.*absent\.parquet"):
        read_monolingual(tmp_path / "absent.parquet")


def test_read_monolingual_rejects_a_directory(tmp_path: Path) -> None:
    with pytest.raises(DataError, match=r"not found"):
        read_monolingual(tmp_path)


@pytest.mark.parametrize(
    "content",
    [b"", b"id,lang,text\n1,khk_Cyrl,hello\n", b"PAR1" + bytes(40) + b"PAR1"],
    ids=["zero bytes", "csv text", "parquet magic only"],
)
def test_read_monolingual_rejects_a_file_that_is_not_parquet(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "mono.parquet"
    path.write_bytes(content)

    with pytest.raises(DataError, match=r"cannot read.*mono\.parquet"):
        read_monolingual(path)


def _panicking_reader(*args: Any, **kwargs: Any) -> pl.DataFrame:
    raise pl.exceptions.PanicException("assertion failed in the parquet reader")


def test_read_monolingual_turns_a_reader_panic_into_a_data_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A damaged file can make the Rust reader panic, which is not an Exception.
    path = _mono_file(tmp_path, _letters("ab"))
    monkeypatch.setattr(pl, "read_parquet", _panicking_reader)

    with pytest.raises(DataError, match=r"cannot read.*mono\.parquet.*assertion"):
        read_monolingual(path)


@pytest.mark.parametrize("column", ["id", "lang", "text"])
def test_read_monolingual_rejects_a_missing_required_column(
    tmp_path: Path, column: str
) -> None:
    frame = _mono([("1", "khk_Cyrl", "a")]).drop(column)

    with pytest.raises(DataError, match=rf"missing required columns: \['{column}'\]"):
        read_monolingual(_mono_file(tmp_path, frame))


@pytest.mark.parametrize("column", ["id", "lang", "text"])
def test_read_monolingual_rejects_a_column_that_is_not_a_string(
    tmp_path: Path, column: str
) -> None:
    frame = _mono([("1", "khk_Cyrl", "a")]).with_columns(
        pl.Series(column, [7], dtype=pl.Int64)
    )

    with pytest.raises(
        DataError, match=rf"column '{column}' must be String, found Int64"
    ):
        read_monolingual(_mono_file(tmp_path, frame))


def test_read_monolingual_rejects_a_column_of_nothing_but_nulls(
    tmp_path: Path,
) -> None:
    # Parquet stores an all-null column untyped; it must not pass for strings.
    frame = pl.DataFrame({"id": ["1"], "lang": ["khk_Cyrl"], "text": [None]})

    with pytest.raises(DataError, match=r"column 'text' must be String, found Null"):
        read_monolingual(_mono_file(tmp_path, frame))


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(None, id="null"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="spaces"),
        pytest.param(f"\t\n {NNBSP}", id="tab newline NNBSP"),
        pytest.param(chr(0xA0), id="no-break space"),
        pytest.param(chr(0x1C), id="file separator"),
    ],
)
def test_read_monolingual_rejects_a_text_without_content_naming_the_row(
    tmp_path: Path, text: str | None
) -> None:
    frame = _mono(
        [("good", "khk_Cyrl", "fine"), ("bad", "khk_Cyrl", "x")]
    ).with_columns(pl.Series("text", ["fine", text], dtype=pl.String))

    with pytest.raises(
        DataError,
        match=r"null, empty or whitespace-only text: 1 \(first id 'bad'\)",
    ):
        read_monolingual(_mono_file(tmp_path, frame))


def test_read_monolingual_counts_every_row_without_text(tmp_path: Path) -> None:
    frame = _mono(
        [("1", "khk_Cyrl", " "), ("2", "khk_Cyrl", "ok"), ("3", "khk_Cyrl", "")]
    )

    with pytest.raises(DataError, match=r"whitespace-only text: 2 \(first id '1'\)"):
        read_monolingual(_mono_file(tmp_path, frame))


def test_read_monolingual_does_not_echo_the_text_in_the_error(tmp_path: Path) -> None:
    frame = _mono([("1", "khk_Cyrl", "ok"), ("1", "khk_Cyrl", "private words")])

    with pytest.raises(DataError) as caught:
        read_monolingual(_mono_file(tmp_path, frame))

    assert "private words" not in str(caught.value)


@pytest.mark.parametrize("bad_id", [None, ""], ids=["null", "empty"])
def test_read_monolingual_rejects_a_null_or_empty_id(
    tmp_path: Path, bad_id: str | None
) -> None:
    frame = _mono([("a", "khk_Cyrl", "x"), ("b", "khk_Cyrl", "y")]).with_columns(
        pl.Series("id", ["a", bad_id], dtype=pl.String)
    )

    with pytest.raises(DataError, match=r"null or empty id: 1"):
        read_monolingual(_mono_file(tmp_path, frame))


def test_read_monolingual_rejects_duplicate_ids_naming_one(tmp_path: Path) -> None:
    frame = _mono(
        [
            ("a", "khk_Cyrl", "1"),
            ("b", "khk_Cyrl", "2"),
            ("b", "khk_Cyrl", "3"),
            ("c", "khk_Cyrl", "4"),
            ("c", "khk_Cyrl", "5"),
            ("c", "khk_Cyrl", "6"),
        ]
    )

    with pytest.raises(DataError, match=r"duplicated ids: 2 \(first 'b'\)"):
        read_monolingual(_mono_file(tmp_path, frame))


def test_read_monolingual_accepts_an_id_that_is_a_prefix_of_another(
    tmp_path: Path,
) -> None:
    frame = _mono([("a", "khk_Cyrl", "x"), ("ab", "khk_Cyrl", "y")])

    assert read_monolingual(_mono_file(tmp_path, frame))["id"].to_list() == ["a", "ab"]


# --- plan_shards --------------------------------------------------------------


def _shard(index: int, ids: str) -> Shard:
    return Shard(
        index=index,
        ids=tuple(ids),
        texts=tuple(f"text-{char}" for char in ids),
    )


def test_plan_shards_cuts_contiguous_slices_with_a_shorter_last_shard() -> None:
    plan = plan_shards(_letters("abcdefg"), 3)

    assert plan == (_shard(0, "abc"), _shard(1, "def"), _shard(2, "g"))


def test_plan_shards_returns_a_tuple() -> None:
    assert isinstance(plan_shards(_letters("abc"), 2), tuple)


def test_plan_shards_gives_one_shard_when_the_size_exceeds_the_frame() -> None:
    assert plan_shards(_letters("abc"), 100) == (_shard(0, "abc"),)


def test_plan_shards_gives_one_shard_when_the_size_equals_the_frame() -> None:
    assert plan_shards(_letters("abc"), 3) == (_shard(0, "abc"),)


def test_plan_shards_of_size_one_gives_a_shard_per_row() -> None:
    assert plan_shards(_letters("abc"), 1) == (
        _shard(0, "a"),
        _shard(1, "b"),
        _shard(2, "c"),
    )


def test_plan_shards_of_an_empty_frame_is_empty() -> None:
    assert plan_shards(_mono([]), 3) == ()


def test_plan_shards_sorts_by_id_so_input_order_does_not_matter() -> None:
    shuffled = _letters("dfbgace")

    assert plan_shards(shuffled, 3) == plan_shards(_letters("abcdefg"), 3)
    assert plan_shards(shuffled, 3) == (
        _shard(0, "abc"),
        _shard(1, "def"),
        _shard(2, "g"),
    )


def test_plan_shards_sorts_ids_as_strings() -> None:
    frame = _mono([(id_, "khk_Cyrl", f"text-{id_}") for id_ in ("9", "10", "2")])

    plan = plan_shards(frame, 10)

    assert plan[0].ids == ("10", "2", "9")
    assert plan[0].texts == ("text-10", "text-2", "text-9")


def test_plan_shards_keeps_each_text_with_its_own_id_and_unmodified() -> None:
    text = f" a{NNBSP}b\n"
    frame = _mono([("2", "khk_Cyrl", text), ("1", "khk_Cyrl", "plain")])

    plan = plan_shards(frame, 5)

    assert plan == (Shard(0, ("1", "2"), ("plain", text)),)


@pytest.mark.parametrize("shard_size", [0, -1])
def test_plan_shards_rejects_a_size_below_one(shard_size: int) -> None:
    with pytest.raises(ValueError, match=r"shard_size"):
        plan_shards(_letters("abc"), shard_size)


def test_plan_shards_rejects_a_size_below_one_even_for_an_empty_frame() -> None:
    with pytest.raises(ValueError, match=r"shard_size"):
        plan_shards(_mono([]), 0)


def test_a_shard_is_frozen() -> None:
    shard = _shard(0, "ab")

    with pytest.raises(dataclasses.FrozenInstanceError):
        shard.index = 1  # type: ignore[misc]


# --- job_fingerprint ----------------------------------------------------------


def _base_frame() -> pl.DataFrame:
    return _mono(
        [
            ("a", "khk_Cyrl", "эхний"),
            ("b", "khk_Cyrl", f"хоёр{NNBSP}дахь"),
            ("c", "khk_Cyrl", f"гурав\n{NON_BMP}"),
        ]
    )


def _fingerprint(
    frame: pl.DataFrame | None = None,
    job: JobConfig = JOB,
    generator: Any = GENERATOR,
    shard_size: int = 3,
    seed: int = 7,
) -> str:
    return job_fingerprint(
        _base_frame() if frame is None else frame, job, generator, shard_size, seed
    )


def test_the_fingerprint_is_a_lowercase_sha256_hex_digest() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", _fingerprint())


def test_the_fingerprint_is_the_same_for_the_same_ingredients() -> None:
    assert _fingerprint() == _fingerprint()


def test_the_fingerprint_does_not_depend_on_the_row_order() -> None:
    frame = _base_frame()

    assert _fingerprint(frame.reverse()) == _fingerprint(frame)
    assert _fingerprint(frame[[1, 2, 0]]) == _fingerprint(frame)


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"source": ["web", "book", "news"]}, id="added column"),
        pytest.param({"lang": ["khk_Cyrl", "khk_Cyrl", "eng_Latn"]}, id="lang values"),
    ],
)
def test_the_fingerprint_counts_only_the_id_and_text_columns(
    extra: dict[str, list[str]],
) -> None:
    frame = _base_frame().with_columns(
        *(pl.Series(name, values) for name, values in extra.items())
    )

    assert _fingerprint(frame) == _fingerprint()


def test_the_fingerprint_does_not_depend_on_the_order_of_the_config_keys() -> None:
    class ReversedJob(JobConfig):
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            dumped = super().model_dump(**kwargs)
            return dict(reversed(dumped.items()))

    class ReversedGenerator(VllmConfig):
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            dumped = super().model_dump(**kwargs)
            return dict(reversed(dumped.items()))

    job = ReversedJob(**JOB.model_dump())
    generator = ReversedGenerator(**GENERATOR.model_dump())
    assert list(job.model_dump()) != list(JOB.model_dump())  # the order really differs

    assert _fingerprint(job=job, generator=generator) == _fingerprint()


def _with_text(text: str) -> pl.DataFrame:
    return _base_frame().with_columns(
        pl.when(pl.col("id") == "b")
        .then(pl.lit(text))
        .otherwise(pl.col("text"))
        .alias("text")
    )


_FRAME_CHANGES: list[Any] = [
    pytest.param(_with_text("өөр"), id="a text"),
    pytest.param(_with_text(f"хоёр{NNBSP}дахь "), id="a trailing space in a text"),
    pytest.param(
        _base_frame().with_columns(
            pl.when(pl.col("id") == "b")
            .then(pl.lit("bb"))
            .otherwise(pl.col("id"))
            .alias("id")
        ),
        id="an id",
    ),
    pytest.param(_base_frame().head(2), id="a dropped row"),
    pytest.param(
        pl.concat([_base_frame(), _mono([("d", "khk_Cyrl", "дөрөв")])]),
        id="an added row",
    ),
]


@pytest.mark.parametrize("frame", _FRAME_CHANGES)
def test_the_fingerprint_changes_when_the_input_changes(frame: pl.DataFrame) -> None:
    assert _fingerprint(frame) != _fingerprint()


def test_the_fingerprint_tells_an_id_from_a_text_of_the_same_characters() -> None:
    id_a = _mono([("a", "khk_Cyrl", "b")])
    id_b = _mono([("b", "khk_Cyrl", "a")])

    assert _fingerprint(id_a) != _fingerprint(id_b)


def test_the_fingerprint_tells_an_id_and_text_boundary_from_a_shifted_one() -> None:
    joined = _mono([("ab", "khk_Cyrl", "c")])
    shifted = _mono([("a", "khk_Cyrl", "bc")])

    assert _fingerprint(joined) != _fingerprint(shifted)


def test_the_fingerprint_tells_row_boundaries_from_shifted_ones() -> None:
    first = _mono([("a", "khk_Cyrl", "xy"), ("b", "khk_Cyrl", "z")])
    second = _mono([("a", "khk_Cyrl", "x"), ("b", "khk_Cyrl", "yz")])

    assert _fingerprint(first) != _fingerprint(second)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"kind": "ft"}, id="kind"),
        pytest.param({"other_lang": "rus_Cyrl"}, id="other_lang"),
        pytest.param(
            {"mono_lang": "eng_Latn", "other_lang": "khk_Cyrl"}, id="mono_lang"
        ),
        pytest.param({"generator": "llamacpp_default"}, id="generator name"),
        pytest.param({"shard_size": 64}, id="the job's own shard_size"),
    ],
)
def test_the_fingerprint_changes_when_the_job_changes(change: dict[str, Any]) -> None:
    job = JobConfig(**{**JOB.model_dump(), **change})

    assert _fingerprint(job=job) != _fingerprint()


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"n": 5}, id="n"),
        pytest.param({"temperature": 0.8}, id="temperature"),
        pytest.param({"top_p": 0.9}, id="top_p"),
        pytest.param({"max_tokens": 512}, id="max_tokens"),
        pytest.param({"base_url": "http://127.0.0.1:9000/v1"}, id="base_url"),
        pytest.param({"model": "another-model"}, id="model"),
        pytest.param({"concurrency": 4}, id="concurrency"),
        pytest.param({"timeout_s": 30.0}, id="timeout_s"),
        pytest.param({"retries": 1}, id="retries"),
        pytest.param({"backoff_s": 2.0}, id="backoff_s"),
    ],
)
def test_the_fingerprint_changes_when_one_generator_field_changes(
    change: dict[str, Any],
) -> None:
    generator = VllmConfig(**{**GENERATOR.model_dump(), **change})

    assert _fingerprint(generator=generator) != _fingerprint()


def test_the_fingerprint_changes_with_the_generator_kind_alone() -> None:
    same_fields = LlamaCppConfig(
        **{**GENERATOR.model_dump(exclude={"kind"}), "base_url": GENERATOR.base_url}
    )

    assert same_fields.model_dump(exclude={"kind"}) == GENERATOR.model_dump(
        exclude={"kind"}
    )
    assert _fingerprint(generator=same_fields) != _fingerprint()


def test_the_fingerprint_of_a_ctranslate2_generator_follows_its_model_paths() -> None:
    # The config holds Path values, which a plain JSON dump cannot encode.
    def ct2(model_dir: str) -> Ct2Config:
        return Ct2Config(
            model_dir=Path(model_dir),
            spm_model=Path(model_dir) / "sentencepiece.bpe.model",
            family="nllb",
        )

    first = _fingerprint(generator=ct2("/models/ct2/one"))

    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert first == _fingerprint(generator=ct2("/models/ct2/one"))
    assert first != _fingerprint(generator=ct2("/models/ct2/two"))


def test_the_fingerprint_of_a_job_with_a_non_json_value_still_encodes() -> None:
    class PathJob(JobConfig):
        def model_dump(self, *, mode: str = "python", **kwargs: Any) -> dict[str, Any]:
            dumped = super().model_dump(mode=mode, **kwargs)
            if mode == "python":
                dumped["note"] = Path("some/where")
            return dumped

    job = PathJob(**JOB.model_dump())

    assert re.fullmatch(r"[0-9a-f]{64}", _fingerprint(job=job))


def test_the_fingerprint_changes_with_the_shard_size() -> None:
    assert _fingerprint(shard_size=4) != _fingerprint(shard_size=3)


def test_the_fingerprint_changes_with_the_seed() -> None:
    assert _fingerprint(seed=8) != _fingerprint(seed=7)


def test_the_job_shard_size_and_the_argument_are_separate_ingredients() -> None:
    # Swapping the two values must not collide: each lands in its own slot.
    job = JobConfig(**{**JOB.model_dump(), "shard_size": 4})

    assert _fingerprint(job=job, shard_size=3) != _fingerprint(
        job=JobConfig(**{**JOB.model_dump(), "shard_size": 3}), shard_size=4
    )


def test_the_fingerprint_tells_the_seed_from_the_shard_size() -> None:
    assert _fingerprint(shard_size=3, seed=7) != _fingerprint(shard_size=7, seed=3)


# --- shard_directory ----------------------------------------------------------


def test_shard_directory_is_named_after_the_first_twelve_digits_of_the_fingerprint() -> (
    None
):
    fingerprint = "0123456789abcdef" * 4

    assert (
        shard_directory(Path("out"), fingerprint) == Path("out") / "shards-0123456789ab"
    )


# --- ShardStore: layout and round trip ----------------------------------------


@pytest.fixture
def shards_dir(tmp_path: Path) -> Path:
    return tmp_path / "shards"


@pytest.fixture
def store(shards_dir: Path) -> ShardStore:
    return ShardStore(shards_dir)


def test_path_for_names_the_file_after_the_zero_padded_index(tmp_path: Path) -> None:
    store = ShardStore(tmp_path / "d")

    assert store.path_for(0) == tmp_path / "d" / "shard-00000.parquet"
    assert store.path_for(7) == tmp_path / "d" / "shard-00007.parquet"
    assert store.path_for(123456) == tmp_path / "d" / "shard-123456.parquet"


def test_write_creates_the_directory_and_returns_the_file(tmp_path: Path) -> None:
    store = ShardStore(tmp_path / "deep" / "er" / "shards")

    path = store.write(3, ["a"], [["x"]])

    assert path == tmp_path / "deep" / "er" / "shards" / "shard-00003.parquet"
    assert path.is_file()


def test_write_then_read_round_trips_the_candidates(store: ShardStore) -> None:
    ids = ["a", "b", "c"]
    candidates = [["x", "y"], ["z"], ["p", "q", "r"]]

    store.write(0, ids, candidates)

    assert store.read(0, ids) == candidates


def test_an_empty_candidate_list_stays_an_empty_list(store: ShardStore) -> None:
    # A refusal yields no candidates for a text: that must not become a null.
    ids = ["a", "b", "c"]
    candidates: list[list[str]] = [[], ["z"], []]

    store.write(0, ids, candidates)

    assert store.read(0, ids) == [[], ["z"], []]


def test_the_file_holds_a_string_id_and_a_list_of_strings(store: ShardStore) -> None:
    path = store.write(0, ["a", "b"], [[], ["x"]])

    stored = pl.read_parquet(path)

    assert list(stored.schema.items()) == [
        ("id", pl.String),
        ("candidates", pl.List(pl.String)),
    ]
    assert stored["candidates"].null_count() == 0
    assert stored["candidates"].to_list() == [[], ["x"]]


def test_the_round_trip_keeps_hard_strings_exactly(store: ShardStore) -> None:
    candidates = [
        [f"left{NNBSP}right", "line one\nline two\r\n", NON_BMP * 3],
        ["  padded  ", "", "tab\there"],
        ["Өдрийн мэнд", "x" * 5000],
        [f"nul{NUL}inside", NUL],
    ]
    ids = ["a", "b", "c", "d"]

    store.write(0, ids, candidates)

    assert store.read(0, ids) == candidates


def test_the_round_trip_keeps_the_candidate_order(store: ShardStore) -> None:
    candidates = [["b", "a", "c"], ["z", "y"]]

    store.write(0, ["1", "2"], candidates)

    assert store.read(0, ["1", "2"]) == candidates


def test_the_round_trip_accepts_tuples(store: ShardStore) -> None:
    store.write(0, ("a", "b"), (("x",), ()))

    assert store.read(0, ("a", "b")) == [["x"], []]


def test_a_second_write_replaces_the_first(store: ShardStore) -> None:
    store.write(0, ["a"], [["old"]])
    store.write(0, ["a"], [["new", "newer"]])

    assert store.read(0, ["a"]) == [["new", "newer"]]


def test_the_shards_of_one_store_do_not_overwrite_each_other(
    store: ShardStore,
) -> None:
    store.write(0, ["a"], [["zero"]])
    store.write(1, ["b"], [["one"]])

    assert store.read(0, ["a"]) == [["zero"]]
    assert store.read(1, ["b"]) == [["one"]]


def test_write_leaves_only_the_final_file_in_the_directory(
    shards_dir: Path,
    store: ShardStore,
) -> None:
    store.write(0, ["a"], [["x"]])

    assert sorted(path.name for path in shards_dir.iterdir()) == ["shard-00000.parquet"]


def test_write_rejects_ids_and_candidates_of_different_lengths(
    store: ShardStore,
) -> None:
    with pytest.raises(ValueError, match=r"differ in length \(2 vs 1\)"):
        store.write(0, ["a", "b"], [["x"]])

    assert not store.path_for(0).exists()


def test_write_rejects_a_negative_index(store: ShardStore) -> None:
    with pytest.raises(ValueError, match=r"index.*-1"):
        store.write(-1, ["a"], [["x"]])


def test_a_negative_index_has_no_path_and_nothing_to_read(store: ShardStore) -> None:
    with pytest.raises(ValueError, match=r"index.*-2"):
        store.path_for(-2)
    with pytest.raises(ValueError, match=r"index.*-2"):
        store.read(-2, ["a"])


def test_write_rejects_a_candidate_that_is_not_a_string(
    shards_dir: Path, store: ShardStore
) -> None:
    with pytest.raises(ValueError, match=r"shard 4.*id 'b'"):
        store.write(4, ["a", "b"], [["x"], [None, "y"]])  # type: ignore[list-item]

    assert not shards_dir.exists()


def test_write_rejects_a_plain_string_as_a_candidate_list(
    shards_dir: Path, store: ShardStore
) -> None:
    # list("abc") would silently become three candidates.
    with pytest.raises(ValueError, match=r"shard 3.*id 'b'.*not a string"):
        store.write(3, ["a", "b"], [["x"], "abc"])

    assert not shards_dir.exists()


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param(LONE_SURROGATE, id="lone high surrogate"),
        pytest.param(f"secret{chr(0xDFFF)}", id="lone low surrogate"),
        pytest.param(f"secret{chr(0xD83D)}{chr(0xDE00)}", id="split surrogate pair"),
    ],
)
def test_write_rejects_a_candidate_utf8_cannot_encode_naming_the_shard_and_id(
    shards_dir: Path, store: ShardStore, candidate: str
) -> None:
    with pytest.raises(DataError, match=r"shard 7.*id 'b'") as caught:
        store.write(7, ["a", "b"], [["fine"], ["ok", candidate]])

    assert "secret" not in str(caught.value)
    assert candidate not in str(caught.value)
    assert not shards_dir.exists()  # rejected before the disk is touched


def test_write_keeps_the_earlier_file_when_a_candidate_is_unencodable(
    store: ShardStore,
) -> None:
    store.write(0, ["a"], [["old"]])

    with pytest.raises(DataError):
        store.write(0, ["a"], [[LONE_SURROGATE]])

    assert store.read(0, ["a"]) == [["old"]]


# --- ShardStore: atomic writes ------------------------------------------------


def test_a_failing_rename_leaves_no_final_file_and_no_part_file(
    shards_dir: Path, store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_replace(self: Path, target: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match=r"disk full"):
        store.write(0, ["a"], [["x"]])

    assert list(shards_dir.iterdir()) == []


def test_a_failure_while_writing_keeps_the_earlier_file_and_leaves_no_part(
    shards_dir: Path, store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write(0, ["a"], [["old"]])

    def half_written(self: pl.DataFrame, file: Any, *args: Any, **kwargs: Any) -> None:
        Path(file).write_bytes(b"PAR1 half a file")
        raise OSError("killed")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", half_written)

    with pytest.raises(OSError, match=r"killed"):
        store.write(0, ["a"], [["new"]])

    monkeypatch.undo()
    assert store.read(0, ["a"]) == [["old"]]
    assert [path.name for path in shards_dir.iterdir()] == ["shard-00000.parquet"]


def test_the_file_appears_only_through_a_rename_of_a_part_file(
    store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    renames: list[tuple[str, str]] = []
    real_replace = Path.replace

    def recording_replace(self: Path, target: Path) -> Path:
        renames.append((self.name, Path(target).name))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", recording_replace)

    store.write(4, ["a"], [["x"]])

    assert len(renames) == 1
    source, target = renames[0]
    assert target == "shard-00004.parquet"
    assert source.startswith("shard-00004.parquet.")
    assert source.endswith(".part")


def test_two_writers_interleaved_for_one_index_keep_their_own_part_files(
    shards_dir: Path, store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Writer A is cut off just before its rename; writer B runs to completion;
    # then A finishes. With a shared part file B would take A's file away.
    real_replace = Path.replace
    interleaved = {"done": False}

    def replace_after_the_other_writer(self: Path, target: Path) -> Path:
        if not interleaved["done"]:
            interleaved["done"] = True
            store.write(0, ["a"], [["from B"]])
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace_after_the_other_writer)

    store.write(0, ["a"], [["from A"]])

    monkeypatch.undo()
    assert store.read(0, ["a"]) == [["from A"]]  # the last rename wins
    assert [path.name for path in shards_dir.iterdir()] == ["shard-00000.parquet"]


def test_threads_writing_one_index_leave_a_valid_file_equal_to_one_of_them(
    shards_dir: Path,
    store: ShardStore,
) -> None:
    ids = [f"id-{n}" for n in range(40)]
    contents = [
        [[f"{writer}-{n}", "shared"] for n in range(40)] for writer in ("A", "B")
    ]
    start = threading.Barrier(2)
    errors: list[BaseException] = []

    def work(content: list[list[str]]) -> None:
        try:
            start.wait()
            for _ in range(15):
                store.write(0, ids, content)
        except BaseException as exc:  # noqa: BLE001 - reported to the test thread
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(content,)) for content in contents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.read(0, ids) in contents
    assert [path.name for path in shards_dir.iterdir()] == ["shard-00000.parquet"]


# --- ShardStore: read validation ----------------------------------------------

IDS = ["a", "b", "c"]
CANDIDATES = [["x"], [], ["y", "z"]]


def _good() -> pl.DataFrame:
    return pl.DataFrame({"id": IDS, "candidates": CANDIDATES}, schema=_SHARD_SCHEMA)


def _wrong_columns() -> pl.DataFrame:
    return _good().rename({"candidates": "hypotheses"})


def _extra_column() -> pl.DataFrame:
    return _good().with_columns(pl.lit(1).alias("note"))


def _only_ids() -> pl.DataFrame:
    return _good().select("id")


def _swapped_columns() -> pl.DataFrame:
    return _good().select("candidates", "id")


def _candidates_as_text() -> pl.DataFrame:
    return _good().with_columns(pl.col("candidates").list.join("|"))


def _ids_as_numbers() -> pl.DataFrame:
    return _good().with_columns(pl.Series("id", [1, 2, 3]))


def _candidates_of_numbers() -> pl.DataFrame:
    return pl.DataFrame(
        {"id": IDS, "candidates": [[1], [], [2, 3]]},
        schema={"id": pl.String, "candidates": pl.List(pl.Int64)},
    )


def _ids_in_another_order() -> pl.DataFrame:
    return _good().reverse()


def _ids_of_another_run() -> pl.DataFrame:
    return _good().with_columns(pl.Series("id", ["a", "b", "other"]))


def _one_row_too_many() -> pl.DataFrame:
    return pl.concat(
        [
            _good(),
            pl.DataFrame({"id": ["d"], "candidates": [["w"]]}, schema=_SHARD_SCHEMA),
        ]
    )


def _one_row_too_few() -> pl.DataFrame:
    return _good().head(2)


def _no_rows() -> pl.DataFrame:
    return _good().head(0)


def _a_null_list() -> pl.DataFrame:
    return pl.DataFrame(
        {"id": IDS, "candidates": [["x"], None, ["y", "z"]]}, schema=_SHARD_SCHEMA
    )


def _a_null_candidate() -> pl.DataFrame:
    return pl.DataFrame(
        {"id": IDS, "candidates": [["x"], [], [None, "z"]]}, schema=_SHARD_SCHEMA
    )


_COLUMNS_MESSAGE = r"expected columns \['id', 'candidates'\]"

_BAD_FRAMES: list[Any] = [
    pytest.param(_wrong_columns, _COLUMNS_MESSAGE, id="renamed column"),
    pytest.param(_extra_column, _COLUMNS_MESSAGE, id="extra column"),
    pytest.param(_only_ids, _COLUMNS_MESSAGE, id="missing column"),
    pytest.param(_swapped_columns, _COLUMNS_MESSAGE, id="columns in the other order"),
    pytest.param(
        _candidates_as_text,
        r"column 'candidates' must be List\(String\), found String",
        id="text candidates",
    ),
    pytest.param(
        _ids_as_numbers, r"column 'id' must be String, found Int64", id="numeric ids"
    ),
    pytest.param(
        _candidates_of_numbers,
        r"column 'candidates' must be List\(String\), found List\(Int64\)",
        id="int lists",
    ),
    pytest.param(
        _ids_in_another_order,
        r"id at index 0 is 'c', expected 'a'",
        id="wrong id order",
    ),
    pytest.param(
        _ids_of_another_run,
        r"id at index 2 is 'other', expected 'c'",
        id="foreign ids",
    ),
    pytest.param(_one_row_too_many, r"row count is 4, expected 3", id="extra row"),
    pytest.param(_one_row_too_few, r"row count is 2, expected 3", id="missing row"),
    pytest.param(_no_rows, r"row count is 0, expected 3", id="no rows"),
    pytest.param(_a_null_list, r"null candidate lists: 1", id="null list"),
    pytest.param(_a_null_candidate, r"null candidates: 1", id="null candidate"),
]


@pytest.mark.parametrize(("build", "message"), _BAD_FRAMES)
def test_read_rejects_an_invalid_file_naming_the_file_and_the_problem(
    shards_dir: Path,
    store: ShardStore,
    build: Callable[[], pl.DataFrame],
    message: str,
) -> None:
    shards_dir.mkdir()
    _write_parquet(store.path_for(0), build())

    with pytest.raises(DataError, match=message) as caught:
        store.read(0, IDS)

    assert "shard-00000.parquet" in str(caught.value)


@pytest.mark.parametrize(
    "content",
    [b"", b"definitely not parquet", b"PAR1" + bytes(40) + b"PAR1"],
    ids=["zero bytes", "text", "magic only"],
)
def test_read_rejects_bytes_that_are_not_parquet(
    shards_dir: Path, store: ShardStore, content: bytes
) -> None:
    shards_dir.mkdir()
    store.path_for(0).write_bytes(content)

    with pytest.raises(DataError, match=r"shard-00000\.parquet"):
        store.read(0, IDS)


def test_read_turns_a_reader_panic_into_a_data_error_and_completed_skips_it(
    store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(2)
    store.write(0, plan[0].ids, [["x"]] * 2)
    monkeypatch.setattr(pl, "read_parquet", _panicking_reader)

    with pytest.raises(DataError, match=r"shard-00000\.parquet.*assertion"):
        store.read(0, plan[0].ids)
    assert store.completed(plan) == frozenset()


def test_read_rejects_a_truncated_file(store: ShardStore) -> None:
    store.write(0, IDS, CANDIDATES)
    whole = store.path_for(0).read_bytes()
    store.path_for(0).write_bytes(whole[: len(whole) // 2])

    with pytest.raises(DataError, match=r"shard-00000\.parquet"):
        store.read(0, IDS)


def test_read_rejects_a_missing_file_naming_it(store: ShardStore) -> None:
    with pytest.raises(DataError, match=r"shard-00005\.parquet"):
        store.read(5, IDS)


def test_read_of_a_valid_file_against_the_wrong_ids_is_rejected(
    store: ShardStore,
) -> None:
    store.write(0, IDS, CANDIDATES)

    with pytest.raises(DataError, match=r"id at index 1 is 'b', expected 'c'"):
        store.read(0, ["a", "c", "b"])


def test_read_names_the_first_row_whose_id_differs(store: ShardStore) -> None:
    store.write(0, IDS, CANDIDATES)

    with pytest.raises(DataError, match=r"id at index 2 is 'c', expected 'zzz'"):
        store.read(0, ["a", "b", "zzz"])


def test_read_accepts_any_sequence_of_expected_ids(store: ShardStore) -> None:
    store.write(0, IDS, CANDIDATES)

    assert store.read(0, tuple(IDS)) == CANDIDATES


# --- ShardStore: completed ----------------------------------------------------


def _plan(count: int) -> tuple[Shard, ...]:
    return plan_shards(_letters("abcdefghij"[: count * 2]), 2)


def test_completed_is_empty_before_anything_is_written(store: ShardStore) -> None:
    assert store.completed(_plan(3)) == frozenset()


def test_completed_lists_the_indexes_of_the_valid_files(store: ShardStore) -> None:
    plan = _plan(4)
    store.write(0, plan[0].ids, [["x"]] * 2)
    store.write(2, plan[2].ids, [[], ["y"]])

    assert store.completed(plan) == frozenset({0, 2})


def test_completed_returns_a_frozenset(store: ShardStore) -> None:
    assert isinstance(store.completed(_plan(1)), frozenset)


def test_completed_treats_corrupt_short_and_foreign_files_as_not_completed(
    store: ShardStore,
) -> None:
    plan = _plan(5)
    store.write(0, plan[0].ids, [["ok"]] * 2)  # valid
    store.write(1, plan[1].ids, [["ok"]] * 2)  # valid, then truncated below
    whole = store.path_for(1).read_bytes()
    store.path_for(1).write_bytes(whole[:-9])
    store.write(2, plan[2].ids[:1], [["short"]])  # one row of two
    store.write(3, ["not", "these"], [["foreign"]] * 2)  # other ids
    store.path_for(4).write_bytes(b"garbage")  # not parquet

    assert store.completed(plan) == frozenset({0})


def test_completed_leaves_the_invalid_files_on_disk(
    shards_dir: Path, store: ShardStore
) -> None:
    plan = _plan(2)
    shards_dir.mkdir()
    store.path_for(0).write_bytes(b"garbage")

    store.completed(plan)

    assert store.path_for(0).read_bytes() == b"garbage"


def test_a_later_write_replaces_an_invalid_file_and_completes_the_shard(
    shards_dir: Path,
    store: ShardStore,
) -> None:
    plan = _plan(2)
    shards_dir.mkdir()
    store.path_for(1).write_bytes(b"garbage")
    assert store.completed(plan) == frozenset()

    store.write(1, plan[1].ids, [["fresh"]] * 2)

    assert store.completed(plan) == frozenset({1})
    assert store.read(1, plan[1].ids) == [["fresh"], ["fresh"]]


def test_completed_judges_each_file_against_its_own_planned_ids(
    store: ShardStore,
) -> None:
    plan = _plan(2)
    store.write(0, plan[1].ids, [["x"]] * 2)  # shard 1's rows under shard 0's name
    store.write(1, plan[0].ids, [["x"]] * 2)

    assert store.completed(plan) == frozenset()


def test_completed_ignores_files_outside_the_plan(store: ShardStore) -> None:
    plan = _plan(2)
    store.write(9, ["z"], [["x"]])

    assert store.completed(plan) == frozenset()


def test_completed_counts_a_shard_the_plan_never_had_as_missing(
    store: ShardStore,
) -> None:
    plan = _plan(3)
    store.write(0, plan[0].ids, [["x"]] * 2)

    assert store.completed(plan[:2]) == frozenset({0})


# --- ShardStore: invalid ------------------------------------------------------


def _count_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch the Parquet reader to record the name of every file it opens."""
    opened: list[str] = []
    real_read = pl.read_parquet

    def counting_read(source: Any, *args: Any, **kwargs: Any) -> pl.DataFrame:
        opened.append(Path(source).name)
        return real_read(source, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", counting_read)
    return opened


def _mixed_store(store: ShardStore) -> tuple[Shard, ...]:
    """Shard 0 valid, 1 corrupt, 2 missing, 3 short, 4 foreign ids, 5 valid."""
    plan = _plan(5)
    plan = plan + (Shard(5, ("y", "z"), ("text-y", "text-z")),)
    store.write(0, plan[0].ids, [["ok"]] * 2)
    store.path_for(1).write_bytes(b"garbage")
    store.write(3, plan[3].ids[:1], [["short"]])
    store.write(4, ["not", "these"], [["foreign"]] * 2)
    store.write(5, plan[5].ids, [[], ["fine"]])
    return plan


def test_invalid_lists_the_files_that_exist_but_fail_validation(
    store: ShardStore,
) -> None:
    plan = _mixed_store(store)

    assert store.invalid(plan) == frozenset({1, 3, 4})
    assert store.completed(plan) == frozenset({0, 5})


def test_invalid_returns_a_frozenset(store: ShardStore) -> None:
    assert isinstance(store.invalid(_plan(1)), frozenset)


def test_a_missing_shard_is_neither_completed_nor_invalid(store: ShardStore) -> None:
    plan = _plan(3)

    assert store.invalid(plan) == frozenset()
    assert store.completed(plan) == frozenset()


def test_a_stale_part_file_beside_a_missing_shard_is_ignored(
    store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(2)
    store.write(0, plan[0].ids, [["ok"]] * 2)
    stale = store.path_for(1).with_name("shard-00001.parquet.0123abcd.part")
    stale.write_bytes(b"half a file")
    opened = _count_reads(monkeypatch)

    assert store.completed(plan) == frozenset({0})
    assert store.invalid(plan) == frozenset()
    assert stale.read_bytes() == b"half a file"
    assert opened == ["shard-00000.parquet", "shard-00000.parquet"]


def test_a_valid_file_is_neither_invalid_nor_missing(store: ShardStore) -> None:
    plan = _plan(1)
    store.write(0, plan[0].ids, [["ok"]] * 2)

    assert store.invalid(plan) == frozenset()
    assert store.completed(plan) == frozenset({0})


def test_completed_decodes_each_existing_shard_once(
    store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _mixed_store(store)
    opened = _count_reads(monkeypatch)

    store.completed(plan)

    assert sorted(opened) == [
        "shard-00000.parquet",
        "shard-00001.parquet",
        "shard-00003.parquet",
        "shard-00004.parquet",
        "shard-00005.parquet",
    ]  # the missing shard 2 is not opened, none is opened twice


def test_invalid_decodes_each_existing_shard_once(
    store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _mixed_store(store)
    opened = _count_reads(monkeypatch)

    store.invalid(plan)

    assert sorted(opened) == [
        "shard-00000.parquet",
        "shard-00001.parquet",
        "shard-00003.parquet",
        "shard-00004.parquet",
        "shard-00005.parquet",
    ]


def test_a_later_write_moves_an_invalid_shard_to_completed(store: ShardStore) -> None:
    plan = _plan(2)
    store.write(0, plan[0].ids, [["ok"]] * 2)
    store.path_for(1).write_bytes(b"garbage")
    assert store.invalid(plan) == frozenset({1})

    store.write(1, plan[1].ids, [["fresh"]] * 2)

    assert store.invalid(plan) == frozenset()
    assert store.completed(plan) == frozenset({0, 1})


# --- ShardStore: load ---------------------------------------------------------


def test_load_of_a_valid_file_gives_its_candidates(store: ShardStore) -> None:
    store.write(0, IDS, CANDIDATES)

    loaded = store.load(0, IDS)

    assert loaded == Loaded(CANDIDATES)
    assert isinstance(loaded, Loaded)
    assert loaded.candidates == store.read(0, IDS)


def test_load_of_a_shard_without_a_file_is_missing(store: ShardStore) -> None:
    assert store.load(3, IDS) == Missing()


def test_load_of_a_shard_in_a_directory_that_does_not_exist_is_missing(
    tmp_path: Path,
) -> None:
    assert ShardStore(tmp_path / "never-made").load(0, IDS) == Missing()


@pytest.mark.parametrize(
    "damage",
    [
        lambda path: path.write_bytes(b"this is not parquet"),
        lambda path: path.write_bytes(b""),
        lambda path: path.write_bytes(path.read_bytes()[:30]),
    ],
    ids=["garbage", "empty", "truncated"],
)
def test_load_of_a_damaged_file_is_invalid(
    store: ShardStore, damage: Callable[[Path], None]
) -> None:
    store.write(0, IDS, CANDIDATES)
    damage(store.path_for(0))

    assert store.load(0, IDS) == Invalid()


def test_load_of_a_valid_file_for_other_ids_or_another_size_is_invalid(
    store: ShardStore,
) -> None:
    store.write(0, IDS, CANDIDATES)

    assert store.load(0, ["a", "c", "b"]) == Invalid()
    assert store.load(0, ["a", "b"]) == Invalid()


def test_load_turns_a_reader_panic_into_invalid(
    store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write(0, IDS, CANDIDATES)
    monkeypatch.setattr(pl, "read_parquet", _panicking_reader)

    assert store.load(0, IDS) == Invalid()


def test_load_tells_the_valid_the_damaged_and_the_missing_shards_apart(
    store: ShardStore,
) -> None:
    plan = _mixed_store(store)

    outcomes = [type(store.load(shard.index, shard.ids)) for shard in plan]

    assert outcomes == [Loaded, Invalid, Missing, Invalid, Invalid, Loaded]


def test_completed_and_invalid_are_the_loaded_and_the_invalid_shards(
    store: ShardStore,
) -> None:
    plan = _mixed_store(store)
    loads = {shard.index: store.load(shard.index, shard.ids) for shard in plan}

    assert store.completed(plan) == {
        index for index, load in loads.items() if isinstance(load, Loaded)
    }
    assert store.invalid(plan) == {
        index for index, load in loads.items() if isinstance(load, Invalid)
    }


def test_load_decodes_an_existing_file_once_and_never_opens_a_missing_one(
    store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _mixed_store(store)
    opened = _count_reads(monkeypatch)

    for shard in plan:
        store.load(shard.index, shard.ids)

    assert sorted(opened) == [
        "shard-00000.parquet",
        "shard-00001.parquet",
        "shard-00003.parquet",
        "shard-00004.parquet",
        "shard-00005.parquet",
    ]


def test_load_ignores_a_stale_part_file_beside_a_missing_shard(
    store: ShardStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(1)
    store.path_for(0).parent.mkdir(parents=True)
    stale = store.path_for(0).with_name("shard-00000.parquet.0123abcd.part")
    stale.write_bytes(b"half a file")
    opened = _count_reads(monkeypatch)

    assert store.load(0, plan[0].ids) == Missing()
    assert opened == []


def test_a_later_write_turns_an_invalid_load_into_a_loaded_one(
    store: ShardStore,
) -> None:
    plan = _plan(1)
    store.write(0, plan[0].ids, [["ok"]] * 2)
    store.path_for(0).write_bytes(b"garbage")
    assert store.load(0, plan[0].ids) == Invalid()

    store.write(0, plan[0].ids, [["fresh"]] * 2)

    assert store.load(0, plan[0].ids) == Loaded([["fresh"]] * 2)


def test_the_load_outcomes_are_frozen_values() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Loaded([]).candidates = [["x"]]  # type: ignore[misc]
    assert Missing() == Missing()
    assert Invalid() == Invalid()
    assert Missing() != Invalid()
    assert Loaded([["a"]]) != Loaded([["b"]])


# --- paths that look like glob patterns ---------------------------------------


def test_a_shard_directory_with_brackets_in_its_name_round_trips(
    tmp_path: Path,
) -> None:
    # `pl.read_parquet` would read `out[1]/...` as a glob matching `out1/...`.
    bracketed = ShardStore(tmp_path / "out[1]")
    neighbour = ShardStore(tmp_path / "out1")
    plan = _plan(2)
    neighbour.write(0, plan[0].ids, [["neighbour"]] * 2)
    bracketed.write(0, plan[0].ids, [["mine"]] * 2)
    bracketed.write(1, plan[1].ids, [["mine too"]] * 2)

    assert bracketed.read(0, plan[0].ids) == [["mine"], ["mine"]]
    assert bracketed.completed(plan) == frozenset({0, 1})
    assert bracketed.invalid(plan) == frozenset()


def test_a_shard_directory_of_wildcards_is_read_literally(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("Windows file names cannot hold * or ?")
    wild = ShardStore(tmp_path / "out*?")
    other = ShardStore(tmp_path / "outXY")
    plan = _plan(1)
    other.write(0, ["other-1", "other-2"], [["elsewhere"]] * 2)
    wild.write(0, plan[0].ids, [["mine"]] * 2)

    assert wild.read(0, plan[0].ids) == [["mine"], ["mine"]]
    assert wild.completed(plan) == frozenset({0})


def test_read_monolingual_reads_a_file_with_brackets_in_its_name(
    tmp_path: Path,
) -> None:
    literal = _write_parquet(tmp_path / "mono[1].parquet", _letters("ab"))
    _write_parquet(tmp_path / "mono1.parquet", _letters("xy"))  # what [1] would match

    assert read_monolingual(literal)["id"].to_list() == ["a", "b"]


def test_read_monolingual_reads_a_file_with_wildcards_in_its_name(
    tmp_path: Path,
) -> None:
    if sys.platform == "win32":
        pytest.skip("Windows file names cannot hold * or ?")
    literal = _write_parquet(tmp_path / "mono[1]*?.parquet", _letters("ab"))
    _write_parquet(tmp_path / "mono1xy.parquet", _letters("xy"))  # matches the pattern

    assert read_monolingual(literal)["id"].to_list() == ["a", "b"]


# --- the canonical form of the fingerprint ------------------------------------


def test_the_fingerprint_of_a_fixed_input_is_a_known_digest() -> None:
    # Pins the canonical form (key order, separators, ASCII escapes of non-ASCII
    # text, one JSON array per row). A new config field changes every digest, and
    # with it every shard directory name: update the literal on purpose.
    frame = _mono(
        [
            ("б2", "khk_Cyrl", f"Хоёр{NNBSP}дахь\n{NON_BMP}"),
            ("a1", "khk_Cyrl", "Сайн байна уу"),
        ]
    )
    job = JobConfig(
        kind="bt",
        mono_lang="khk_Cyrl",
        other_lang="eng_Latn",
        generator="vllm_default",
        shard_size=512,
    )
    generator = VllmConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="Hy-MT2-7B",
        n=4,
        temperature=0.7,
        top_p=0.95,
        max_tokens=256,
        concurrency=32,
        timeout_s=120.0,
        retries=4,
        backoff_s=1.0,
    )

    assert job_fingerprint(frame, job, generator, 16, 42) == GOLDEN_DIGEST
