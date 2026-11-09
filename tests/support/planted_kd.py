"""Planted inputs for the tests of the knowledge-distillation generation (D10): corpus files
that hold the rows a KD source must and must not be made of, `KdSources` built by hand, and
a teacher that answers from a script. No model, library or GPU is involved.

`planted_training_rows` writes four corpus files in two directions, English to Mongolian
(`EN_MN`) and Russian to Mongolian (`RU_MN`). The AUTHENTIC training rows, which are the
sources, are

    EN_MN  Hello, Good morning, Thank you (three rows, two targets), Water
    RU_MN  Привет, Спасибо, Книга

and every other row is one that must never be a source or a reference: a dev, a test and a
holdout row, a back-translated row, a forward-translated row, a synthetic-translation row,
and two filtered rows. Several of them repeat the text of a source, with another target, to
show that such a row adds neither a source nor a reference.

`ScriptedTeacher` is a `KdTranslator`: it answers each text with its scripted candidates (a
text the script lacks is a mistake of the test), records every call and can fail, or reply
something that is no list, on a chosen call.

`write_kd_files` writes a KD corpus file and the `distill_generate` manifest that vouches for
it, as the generation command leaves them, for the tests that read KD files back."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from rinchen.config.synth import SelectionConfig
from rinchen.corpus.schema import make_row_id
from rinchen.corpus.writers import write_corpus
from rinchen.filter.qe import QEScorer
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import output_file
from rinchen.train.distill_export import VerifiedExport
from rinchen.train.distill_generate import KdResult, KdTranslator, generate_kd
from rinchen.train.distill_sources import KdSources
from rinchen.train.distill_teacher import TeacherOptions
from rinchen.train.mixture import TrainingRows, load_training_rows
from tests.support.corpus_rows import EN_MN, RU_MN, corpus_row, write_rows
from tests.support.planted_synth import ScriptedTranslator

ENG, KHK, RUS = "eng_Latn", "khk_Cyrl", "rus_Cyrl"

# The sources and their authentic targets: what `kd_sources` must find in the planted files.
PLANTED_REFERENCES: dict[str, dict[str, list[str]]] = {
    EN_MN: {
        "Hello": ["Сайн байна уу"],
        "Good morning": ["Өглөөний мэнд"],
        "Thank you": ["Баярлалаа", "Баярлалаа танд"],
        "Water": ["Ус"],
    },
    RU_MN: {
        "Привет": ["Сайн уу"],
        "Спасибо": ["Баярлалаа"],
        "Книга": ["Ном"],
    },
}

# `make_row_id(src_lang, tgt_lang, text, "")` of each source, computed once by hand.
SOURCE_IDS: dict[str, str] = {
    "Hello": "29bc7ec8efab9ec2",
    "Good morning": "53855621fbc23c82",
    "Thank you": "ce3c60bd22eb00ce",
    "Water": "b7a3e2e9e81912de",
    "Привет": "8bc6a56a56313681",
    "Спасибо": "b294fa2f9d2ae629",
    "Книга": "e5bfff0fe74c87e4",
}

# The teacher's planted answers (see `PLANTED_COUNTS` for what the selection makes of them).
PLANTED_SCRIPT: dict[str, list[str]] = {
    "Hello": ["Сайн байна уу"],  # equals its reference
    "Good morning": ["Өглөөний мэнд"],  # equals its reference
    "Thank you": ["Баярлалаа танд"],  # equals the second reference
    "Water": ["Хүйтэн ус"],  # equals no reference
    "Привет": [],  # a refusal
    "Спасибо": ["Thank you very much"],  # the wrong script for a Mongolian target
    "Книга": ["Книга"],  # a copy of the source
}
# The answers of a teacher that selects every source (the refusal, the Latin and the copy of
# the planted script replaced by good ones).
ALL_SELECTED_SCRIPT: dict[str, list[str]] = {
    **PLANTED_SCRIPT,
    "Привет": ["Сайн байна"],
    "Спасибо": ["Баярлалаа"],
    "Книга": ["Ном"],
}
# What the planted answers give, per direction.
PLANTED_COUNTS: dict[str, dict[str, int]] = {
    EN_MN: {
        "sources": 4,
        "candidates": 4,
        "dropped_empty": 0,
        "dropped_purity": 0,
        "dropped_qe": 0,
        "dropped_copy": 0,
        "no_candidates": 0,
        "no_survivor": 0,
        "selected": 4,
    },
    RU_MN: {
        "sources": 3,
        "candidates": 2,
        "dropped_empty": 0,
        "dropped_purity": 1,
        "dropped_qe": 0,
        "dropped_copy": 1,
        "no_candidates": 1,
        "no_survivor": 2,
        "selected": 0,
    },
}


@dataclass(frozen=True, slots=True, eq=False)
class PlantedKd:
    """The planted corpus files, in the order they were written, and the training rows read
    from them in that order."""

    paths: tuple[Path, ...]
    training: TrainingRows


def _authentic_rows() -> list[dict[str, str]]:
    english = [
        corpus_row("Hello", "Сайн байна уу"),
        corpus_row("Good morning", "Өглөөний мэнд", origin="web"),
        corpus_row("Thank you", "Баярлалаа"),
        corpus_row("Thank you", "Баярлалаа танд", origin="web"),
        corpus_row("Thank you", "Баярлалаа", origin="mined"),
        corpus_row("Water", "Ус"),
    ]
    russian = [
        corpus_row("Привет", "Сайн уу", src_lang=RUS),
        corpus_row("Спасибо", "Баярлалаа", src_lang=RUS),
        corpus_row("Книга", "Ном", src_lang=RUS, origin="converted"),
    ]
    return [*english, *russian]


def _other_rows() -> list[dict[str, str]]:
    return [
        corpus_row("Dev only", "Хөгжүүлэлт", split="dev"),
        corpus_row("Test only", "Шалгалт", split="test"),
        corpus_row("Holdout only", "Тусгаарласан", split="holdout"),
        corpus_row("Back translated", "Эргэж орчуулсан", origin="bt"),
        corpus_row("Forward translated", "Урагш орчуулсан", origin="ft"),
        corpus_row("Synthetic", "Хиймэл", origin="synth_nmt"),
        # The text of a source with another target, in rows that are no references.
        corpus_row("Water", "Усан", split="dev"),
        corpus_row("Hello", "Мэнд", origin="bt"),
        corpus_row("Hello", "Мэндчилгээ", origin="ft"),
    ]


def _filtered_rows() -> list[dict[str, str]]:
    return [
        corpus_row("Filtered only", "Шүүсэн"),
        corpus_row("Water", "Шүүсэн ус"),
    ]


def planted_training_rows(
    directory: Path, *, shuffle_seed: int | None = None
) -> PlantedKd:
    """The four planted corpus files in `directory` (made when missing) and the training
    rows read from them. With `shuffle_seed` the rows of each file are shuffled: the sources
    must not depend on the order of the rows."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(shuffle_seed)

    def written(name: str, rows: list[dict[str, str]], **options: Any) -> Path:
        if shuffle_seed is not None:
            rows = rows[:]
            rng.shuffle(rows)
        return write_rows(directory, name, rows, **options)

    filtered = _filtered_rows()
    paths = (
        written("authentic_english", _authentic_rows()[:6]),
        written("authentic_russian", _authentic_rows()[6:]),
        written("others", _other_rows()),
        write_rows(
            directory,
            "filtered",
            filtered,
            filter_reasons=["length"] * len(filtered),
        ),
    )
    return PlantedKd(paths=paths, training=load_training_rows(list(paths)))


def sources_of(direction: str, references: Mapping[str, Sequence[str]]) -> KdSources:
    """The `KdSources` of `direction` (`<src_lang>-<tgt_lang>`) whose texts are the keys of
    `references` (the authentic targets of each), built by hand: ids as the module
    `distill_sources` documents them, the frame sorted by id."""
    src_lang, tgt_lang = direction.split("-")
    rows = sorted(
        (make_row_id(src_lang, tgt_lang, text, ""), text) for text in references
    )
    frame = pl.DataFrame(
        {
            "id": [row_id for row_id, _ in rows],
            "lang": [src_lang] * len(rows),
            "text": [text for _, text in rows],
        },
        schema={"id": pl.String, "lang": pl.String, "text": pl.String},
    )
    return KdSources(
        direction=direction,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        frame=frame,
        references={
            make_row_id(src_lang, tgt_lang, text, ""): frozenset(targets)
            for text, targets in references.items()
        },
    )


def planted_sources() -> list[KdSources]:
    """The sources `planted_training_rows` has, built by hand, sorted by direction."""
    return [
        sources_of(direction, PLANTED_REFERENCES[direction])
        for direction in (EN_MN, RU_MN)
    ]


class ScriptedTeacher(ScriptedTranslator):
    """A `KdTranslator`: `ScriptedTranslator` (scripted candidates by text, `calls`,
    `fail_on_call`) with the teacher's `counters()`, which are `counters` as given, and
    `replies`: the call numbers (0-based) whose answer is the given object instead of the
    script's, for a translator that breaks its contract."""

    def __init__(
        self,
        script: Mapping[str, Sequence[str]],
        *,
        fail_on_call: Mapping[int, BaseException] | None = None,
        replies: Mapping[int, object] | None = None,
        counters: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(script, fail_on_call=fail_on_call)
        self._replies = dict(replies or {})
        self._counters = dict(counters or {})

    def translate(self, texts: Sequence[str], src_lang: str, tgt_lang: str) -> Any:
        number = len(self.calls)
        if number not in self._replies:
            return super().translate(texts, src_lang, tgt_lang)
        self.calls.append((tuple(texts), src_lang, tgt_lang))
        return self._replies[number]

    def counters(self) -> dict[str, int]:
        return dict(self._counters)


def kd_run(
    tmp_path: Path,
    export: VerifiedExport,
    teacher: KdTranslator,
    *,
    sources: Sequence[KdSources] | None = None,
    options: TeacherOptions | None = None,
    selection: SelectionConfig | None = None,
    scorer: QEScorer | None = None,
    shard_size: int = 512,
    seed: int = 1234,
    name: str = "toy",
    **keywords: Any,
) -> KdResult:
    """`generate_kd` over the planted sources (or `sources`) into `tmp_path / "out"` with the
    default options and selection, no scorer, shards of `shard_size`; the keyword
    arguments replace any of them."""
    return generate_kd(
        planted_sources() if sources is None else sources,
        teacher,
        export,
        TeacherOptions() if options is None else options,
        SelectionConfig() if selection is None else selection,
        scorer,
        out_dir=tmp_path / "out",
        shard_size=shard_size,
        seed=seed,
        name=name,
        **keywords,
    )


KD_FILE = "kd.parquet"
KD_STAGE = "distill_generate"


def write_kd_files(
    directory: Path,
    frame: pl.DataFrame,
    *,
    name: str = KD_FILE,
    stage: str = KD_STAGE,
    status: Mapping[str, str] | None = None,
    listed: bool = True,
    rows: int | None = None,
) -> Path:
    """Write `frame` (a corpus frame) as `directory/name` and the manifest beside it that
    vouches for it the way `rinchen distill generate` does: stage `distill_generate`,
    status `generate: ok`, the file listed with its SHA-256 and its row count. The keywords
    make the manifests that must be refused: another `stage` or `status`, a file that is
    not `listed`, a recorded row count `rows` that is not the real one. Returns the path of
    the corpus file."""
    path = directory / name
    count = write_corpus(frame, path)
    entry = output_file(path, count if rows is None else rows)
    manifest = Manifest(
        stage=stage,
        created=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        code_version="test",
        config_sha256="0" * 64,
        seed=1234,
        outputs=[entry] if listed else [_other_file(directory)],
        status={"generate": "ok"} if status is None else dict(status),
    )
    store.write(directory, manifest)
    return path


def _other_file(directory: Path) -> ManifestFile:
    counts = directory / "counts.json"
    counts.write_text("{}\n", encoding="utf-8")
    return output_file(counts, None)
