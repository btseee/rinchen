"""Self-test of the D4 test support: the planted-failure corpus, the fake
embedders and the tiny LID model.

Every property a later filter stage will compute is recomputed here with plain
code (fastText `predict`, `script_of`, character sums, SentencePiece token
counts, `Counter` over ASCII digits, exact shingle Jaccard), so a mistake in
the corpus cannot hide behind a matching mistake in a stage.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import fasttext
import numpy as np
import pytest
import sentencepiece
from numpy.typing import NDArray

from rinchen.config.filters import FilterConfig
from rinchen.config.loader import REPO_ROOT
from rinchen.corpus.schema import CORPUS_SCHEMA, validate_corpus_schema
from rinchen.lang.scripts import Script, script_of
from tests.support import planted_corpus
from tests.support.embedders import ConceptEmbedder, HashEmbedder
from tests.support.planted_corpus import (
    PLANTED_REASON_COUNTS,
    PlantedCorpus,
    build_planted_corpus,
    tiny_spm_model,
)
from tests.support.tiny_lid import train_tiny_lid_model

CONFIG = FilterConfig()
MIN_SCRIPT_SHARE = CONFIG.script_ratio.min_ratio
CHAR_RATIO = CONFIG.length.char_ratio
TOKEN_RATIO = CONFIG.length.token_ratio
MIN_LID_PROB = CONFIG.lid.min_prob
LID_TOP_K = CONFIG.lid.top_k
MIN_COSINE = CONFIG.labse.min_cosine
MINHASH_THRESHOLD = CONFIG.minhash.threshold
SHINGLE_LENGTH = CONFIG.minhash.ngram

SRC_LANG = "khk_Cyrl"
TGT_LANG = "eng_Latn"
CLEAN_COUNT = 28
TOTAL_ROWS = 50
PURE_SCRIPT_SHARE = 0.97
CONFIDENT_LID = 0.9
LID_TRAINING_SECONDS = 10.0
MAX_DIFFERENT_CONCEPT_COSINE = 0.4
# Random 64-dimensional pairs sit near a cosine of 0.125; 0.6 is far beyond it.
MAX_UNMAPPED_TEXT_COSINE = 0.6
MIN_DUPLICATE_JACCARD = 0.9
MIN_ORIGINAL_LENGTH = 80
DIGITS = "0123456789"

# The stage that owns each planted reason, as named in `FilterConfig.stages`.
STAGE_OF_REASON = {
    "script_src": "script_ratio",
    "script_tgt": "script_ratio",
    "numbers": "numbers",
    "len_ratio_char": "length",
    "len_ratio_token": "length",
    "lid_src": "lid",
    "lid_tgt": "lid",
    "duplicate": "minhash",
    "labse": "labse",
}


@dataclass(frozen=True, slots=True)
class Measured:
    """One corpus row with every property recomputed by this test."""

    row_id: str
    src: str
    tgt: str
    reason: str | None
    src_script_share: float
    tgt_script_share: float
    char_ratio: float
    token_ratio: float
    src_lid: dict[str, float]
    tgt_lid: dict[str, float]


def _effective_length(text: str) -> float:
    # Latin and Cyrillic weigh 1.0 under the default weights; a test checks it.
    return float(sum(1 for char in text if not char.isspace()))


def _script_share(text: str, script: Script) -> float:
    return script_of(text).get(script, 0.0)


def _digits(text: str) -> Counter[str]:
    return Counter(char for char in text if char in DIGITS)


def _lid_probabilities(
    model: fasttext.FastText._FastText, text: str
) -> dict[str, float]:
    labels, probabilities = model.predict(text, k=-1)
    return {
        label.removeprefix("__label__"): float(probability)
        for label, probability in zip(labels, probabilities, strict=True)
    }


def _top_k_probability(
    model: fasttext.FastText._FastText, text: str, lang: str
) -> float:
    labels, probabilities = model.predict(text, k=LID_TOP_K)
    return sum(
        float(probability)
        for label, probability in zip(labels, probabilities, strict=True)
        if label == f"__label__{lang}"
    )


def _shingles(text: str) -> set[str]:
    return {text[i : i + SHINGLE_LENGTH] for i in range(len(text) - SHINGLE_LENGTH + 1)}


def _jaccard(a: str, b: str) -> float:
    left, right = _shingles(a), _shingles(b)
    return len(left & right) / len(left | right)


def _unit_cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.dot(a, b))


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("planted_support")


@pytest.fixture(scope="module")
def planted() -> PlantedCorpus:
    return build_planted_corpus()


@pytest.fixture(scope="module")
def spm(workdir: Path) -> sentencepiece.SentencePieceProcessor:
    model_path = tiny_spm_model(workdir / "spm")
    return sentencepiece.SentencePieceProcessor(model_file=str(model_path))


@pytest.fixture(scope="module")
def lid_model(workdir: Path) -> fasttext.FastText._FastText:
    return fasttext.load_model(str(train_tiny_lid_model(workdir / "lid" / "tiny.bin")))


@pytest.fixture(scope="module")
def measured(
    planted: PlantedCorpus,
    spm: sentencepiece.SentencePieceProcessor,
    lid_model: fasttext.FastText._FastText,
) -> list[Measured]:
    rows = []
    for row in planted.frame.iter_rows(named=True):
        src, tgt = row["src"], row["tgt"]
        rows.append(
            Measured(
                row_id=row["id"],
                src=src,
                tgt=tgt,
                reason=planted.expected_reason[row["id"]],
                src_script_share=_script_share(src, Script.CYRILLIC),
                tgt_script_share=_script_share(tgt, Script.LATIN),
                char_ratio=_effective_length(tgt) / _effective_length(src),
                token_ratio=len(spm.encode(tgt)) / len(spm.encode(src)),
                src_lid=_lid_probabilities(lid_model, src),
                tgt_lid=_lid_probabilities(lid_model, tgt),
            )
        )
    return rows


def _with_reason(measured: list[Measured], reason: str | None) -> list[Measured]:
    return [row for row in measured if row.reason == reason]


def _outside(value: float, low: float, high: float) -> bool:
    return value < low or value > high


# --- corpus shape -----------------------------------------------------------


def test_counts_match_the_plan(planted: PlantedCorpus) -> None:
    reasons = Counter(planted.expected_reason.values())

    assert planted.frame.height == TOTAL_ROWS
    assert reasons.pop(None) == CLEAN_COUNT
    assert dict(reasons) == PLANTED_REASON_COUNTS
    assert sum(PLANTED_REASON_COUNTS.values()) == TOTAL_ROWS - CLEAN_COUNT
    assert PLANTED_REASON_COUNTS == {
        "script_src": 2,
        "script_tgt": 2,
        "lid_src": 2,
        "lid_tgt": 2,
        "len_ratio_char": 3,
        "len_ratio_token": 2,
        "numbers": 3,
        "duplicate": 3,
        "labse": 3,
    }


def test_frame_is_a_valid_corpus_with_unique_ids(planted: PlantedCorpus) -> None:
    validate_corpus_schema(planted.frame.schema, source="planted corpus")

    assert planted.frame.schema == CORPUS_SCHEMA
    ids = planted.frame["id"].to_list()
    assert len(set(ids)) == TOTAL_ROWS
    assert set(planted.expected_reason) == set(ids)


def test_rows_share_direction_and_provenance(planted: PlantedCorpus) -> None:
    frame = planted.frame

    assert set(frame["src_lang"]) == {SRC_LANG}
    assert set(frame["tgt_lang"]) == {TGT_LANG}
    assert set(frame["origin"].cast(str)) == {"web"}
    assert set(frame["split"].cast(str)) == {"train"}
    assert set(frame["corpus"]) == {"planted"}
    assert set(frame["doc_id"]) == {"planted"}
    assert frame["seg_idx"].to_list() == list(range(TOTAL_ROWS))
    assert frame["scores"].null_count() == TOTAL_ROWS
    assert frame["filter_reason"].null_count() == TOTAL_ROWS


def test_planted_rows_are_interleaved_with_clean_rows(planted: PlantedCorpus) -> None:
    flags = [planted.expected_reason[row_id] is None for row_id in planted.frame["id"]]

    first_planted = flags.index(False)
    last_clean = len(flags) - 1 - flags[::-1].index(True)

    assert first_planted < last_clean
    assert flags[:CLEAN_COUNT] != [True] * CLEAN_COUNT


def test_two_builds_are_equal() -> None:
    first, second = build_planted_corpus(), build_planted_corpus()

    assert first.frame.equals(second.frame)
    assert first.concepts == second.concepts
    assert first.expected_reason == second.expected_reason
    assert first.isolated_failures == second.isolated_failures


# --- duplicates -------------------------------------------------------------


def test_duplicates_follow_a_long_original_and_overlap_it(
    measured: list[Measured],
) -> None:
    for position, row in enumerate(measured):
        if row.reason != "duplicate":
            continue
        originals = [
            earlier
            for earlier in measured[:position]
            if earlier.src == row.src and row.tgt.startswith(earlier.tgt)
        ]
        assert len(originals) == 1, row.row_id
        original = originals[0]
        assert original.reason is None
        assert len(row.tgt) == len(original.tgt) + 1
        assert len(original.src + "\t" + original.tgt) >= MIN_ORIGINAL_LENGTH
        assert (
            _jaccard(row.src + "\t" + row.tgt, original.src + "\t" + original.tgt)
            >= MIN_DUPLICATE_JACCARD
        )


def test_only_duplicates_overlap_an_earlier_row(measured: list[Measured]) -> None:
    for (i, first), (j, second) in combinations(enumerate(measured), 2):
        similar = (
            _jaccard(first.src + "\t" + first.tgt, second.src + "\t" + second.tgt)
            >= MINHASH_THRESHOLD
        )
        assert similar == (
            second.reason == "duplicate" and _is_copy_of(second, first)
        ), (
            i,
            j,
        )


def _is_copy_of(row: Measured, other: Measured) -> bool:
    return row.src == other.src and row.tgt.startswith(other.tgt)


# --- script shares ----------------------------------------------------------


def test_script_shares_match_the_ledger(measured: list[Measured]) -> None:
    for row in measured:
        if row.reason == "script_src":
            assert row.src_script_share < MIN_SCRIPT_SHARE - 0.2, row.src
        else:
            assert row.src_script_share > PURE_SCRIPT_SHARE, row.src
        if row.reason == "script_tgt":
            assert row.tgt_script_share < MIN_SCRIPT_SHARE - 0.2, row.tgt
        else:
            assert row.tgt_script_share > PURE_SCRIPT_SHARE, row.tgt


# --- lengths ----------------------------------------------------------------


def test_character_ratio_fails_exactly_the_three_length_rows(
    measured: list[Measured],
) -> None:
    outside = {
        row.row_id
        for row in measured
        if _outside(row.char_ratio, CHAR_RATIO.min, CHAR_RATIO.max)
    }

    assert outside == {row.row_id for row in _with_reason(measured, "len_ratio_char")}
    ratios = [row.char_ratio for row in _with_reason(measured, "len_ratio_char")]
    assert sum(ratio > 2.2 for ratio in ratios) == 2
    assert sum(ratio < 0.45 for ratio in ratios) == 1


def test_default_char_weights_leave_latin_and_cyrillic_at_one() -> None:
    assert Script.LATIN not in CONFIG.length.char_weights
    assert Script.CYRILLIC not in CONFIG.length.char_weights


def test_clean_rows_have_balanced_lengths(measured: list[Measured]) -> None:
    for row in _with_reason(measured, None):
        assert len(row.src) + len(row.tgt) >= MIN_ORIGINAL_LENGTH, row
        assert 0.7 <= row.char_ratio <= 1.4, row
        assert 0.6 <= row.token_ratio <= 1.6, row


def test_token_ratio_fails_exactly_the_two_fragmenting_rows(
    measured: list[Measured],
) -> None:
    outside = {
        row.row_id
        for row in measured
        if _outside(row.token_ratio, TOKEN_RATIO.min, TOKEN_RATIO.max)
    }
    fragmenting = _with_reason(measured, "len_ratio_token")

    assert outside == {row.row_id for row in fragmenting}
    for row in fragmenting:
        assert CHAR_RATIO.min < row.char_ratio < CHAR_RATIO.max
        assert row.token_ratio > 2.6


def test_rare_letters_fragment_into_byte_pieces(
    spm: sentencepiece.SentencePieceProcessor,
) -> None:
    pieces = spm.encode("qxy", out_type=str)

    assert pieces[-3:] == ["<0x71>", "<0x78>", "<0x79>"]


# --- numbers ----------------------------------------------------------------


def test_digit_multisets_differ_in_exactly_the_numbers_rows(
    measured: list[Measured],
) -> None:
    differing = {row.row_id for row in measured if _digits(row.src) != _digits(row.tgt)}

    assert differing == {row.row_id for row in _with_reason(measured, "numbers")}


def test_clean_rows_cover_decimal_separator_variants(measured: list[Measured]) -> None:
    with_digits = [row for row in _with_reason(measured, None) if _digits(row.src)]
    separator_rows = [
        row for row in with_digits if "1,5" in row.src and "1.5" in row.tgt
    ]

    assert len(with_digits) == 3
    assert len(separator_rows) == 3


# --- language identification -------------------------------------------------


def test_tiny_lid_model_trains_quickly(workdir: Path) -> None:
    started = time.perf_counter()
    path = train_tiny_lid_model(workdir / "timed" / "tiny.bin")

    assert time.perf_counter() - started < LID_TRAINING_SECONDS
    labels = fasttext.load_model(str(path)).get_labels()
    assert sorted(labels) == sorted(
        f"__label__{lang}" for lang in ("khk_Cyrl", "rus_Cyrl", "eng_Latn", "deu_Latn")
    )


def test_tiny_lid_model_is_deterministic(
    workdir: Path, measured: list[Measured]
) -> None:
    again = fasttext.load_model(
        str(train_tiny_lid_model(workdir / "again" / "tiny.bin"))
    )

    for row in measured:
        assert _lid_probabilities(again, row.src) == pytest.approx(row.src_lid)
        assert _lid_probabilities(again, row.tgt) == pytest.approx(row.tgt_lid)


def test_lid_labels_follow_the_ledger(measured: list[Measured]) -> None:
    for row in measured:
        if row.reason == "lid_src":
            assert row.src_lid["rus_Cyrl"] >= CONFIDENT_LID, row.src
            assert row.src_lid["khk_Cyrl"] < MIN_LID_PROB / 5, row.src
        else:
            assert row.src_lid["khk_Cyrl"] >= CONFIDENT_LID, row.src
        if row.reason == "lid_tgt":
            assert row.tgt_lid["deu_Latn"] >= CONFIDENT_LID, row.tgt
            assert row.tgt_lid["eng_Latn"] < MIN_LID_PROB / 5, row.tgt
        else:
            assert row.tgt_lid["eng_Latn"] >= CONFIDENT_LID, row.tgt


# --- concepts and the labse stage --------------------------------------------


def test_concepts_cover_every_text(planted: PlantedCorpus) -> None:
    texts = set(planted.frame["src"]) | set(planted.frame["tgt"])

    assert set(planted.concepts) == texts


def test_concept_ids_separate_the_labse_rows(planted: PlantedCorpus) -> None:
    for row in planted.frame.iter_rows(named=True):
        same = planted.concepts[row["src"]] == planted.concepts[row["tgt"]]
        assert same == (planted.expected_reason[row["id"]] != "labse")


def test_separated_concept_ids_stop_with_an_error_after_the_attempt_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(planted_corpus, "_MAX_ATTEMPTS", 3)

    with pytest.raises(
        RuntimeError, match=r"only \d+ pairwise separated .* first 3 seeds"
    ):
        list(planted_corpus._separated_concept_ids())


def test_concept_embedder_separates_concepts_in_the_planted_corpus(
    planted: PlantedCorpus,
) -> None:
    embedder = ConceptEmbedder(planted.concepts)
    by_concept = {concept: text for text, concept in planted.concepts.items()}
    vectors = embedder.embed(list(by_concept.values()))

    gram = vectors @ vectors.T

    assert (
        len(by_concept)
        == TOTAL_ROWS
        - PLANTED_REASON_COUNTS["duplicate"]
        + PLANTED_REASON_COUNTS["labse"]
    )
    assert np.allclose(np.diag(gram), 1.0, atol=1e-5)
    off_diagonal = np.abs(gram[~np.eye(len(gram), dtype=bool)])
    assert off_diagonal.max() < MAX_DIFFERENT_CONCEPT_COSINE


def test_concept_embedder_scores_each_row_like_the_labse_ledger(
    planted: PlantedCorpus,
) -> None:
    embedder = ConceptEmbedder(planted.concepts)

    for row in planted.frame.iter_rows(named=True):
        src, tgt = embedder.embed([row["src"], row["tgt"]])
        cosine = _unit_cosine(src, tgt)
        if planted.expected_reason[row["id"]] == "labse":
            assert abs(cosine) < MAX_DIFFERENT_CONCEPT_COSINE
        else:
            assert cosine == pytest.approx(1.0, abs=1e-5)


# --- the ledger of single-stage failures --------------------------------------


def _failing_stages(
    planted: PlantedCorpus,
    measured: list[Measured],
    lid_model: fasttext.FastText._FastText,
) -> dict[str, set[str]]:
    failing: dict[str, set[str]] = {
        stage: set() for stage in set(STAGE_OF_REASON.values())
    }
    embedder = ConceptEmbedder(planted.concepts)
    for position, row in enumerate(measured):
        if min(row.src_script_share, row.tgt_script_share) < MIN_SCRIPT_SHARE:
            failing["script_ratio"].add(row.row_id)
        if _digits(row.src) != _digits(row.tgt):
            failing["numbers"].add(row.row_id)
        if _outside(row.char_ratio, CHAR_RATIO.min, CHAR_RATIO.max) or _outside(
            row.token_ratio, TOKEN_RATIO.min, TOKEN_RATIO.max
        ):
            failing["length"].add(row.row_id)
        if (
            _top_k_probability(lid_model, row.src, SRC_LANG) < MIN_LID_PROB
            or _top_k_probability(lid_model, row.tgt, TGT_LANG) < MIN_LID_PROB
        ):
            failing["lid"].add(row.row_id)
        joined = row.src + "\t" + row.tgt
        if any(
            _jaccard(joined, earlier.src + "\t" + earlier.tgt) >= MINHASH_THRESHOLD
            for earlier in measured[:position]
        ):
            failing["minhash"].add(row.row_id)
        src_vector, tgt_vector = embedder.embed([row.src, row.tgt])
        if _unit_cosine(src_vector, tgt_vector) < MIN_COSINE:
            failing["labse"].add(row.row_id)
    return failing


def test_isolated_failures_list_the_rows_each_stage_flags_alone(
    planted: PlantedCorpus,
    measured: list[Measured],
    lid_model: fasttext.FastText._FastText,
) -> None:
    computed = _failing_stages(planted, measured, lid_model)

    assert set(planted.isolated_failures) == set(computed)
    for stage, flagged in computed.items():
        assert planted.isolated_failures[stage] == frozenset(flagged), stage


def test_every_planted_row_violates_exactly_its_own_stage(
    planted: PlantedCorpus,
) -> None:
    for row_id, reason in planted.expected_reason.items():
        stages = {
            stage
            for stage, flagged in planted.isolated_failures.items()
            if row_id in flagged
        }
        if reason is None:
            assert stages == set(), row_id
        else:
            assert stages == {STAGE_OF_REASON[reason]}, (row_id, reason)


# --- fake embedders -----------------------------------------------------------


def _row_texts(measured: list[Measured]) -> tuple[str, str]:
    clean = _with_reason(measured, None)
    return clean[0].tgt, clean[1].tgt


def test_hash_embedder_rows_are_unit_norm_and_stable(measured: list[Measured]) -> None:
    first, second = _row_texts(measured)
    embedder = HashEmbedder()

    vectors = embedder.embed([first, second])

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 128)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)
    assert _unit_cosine(vectors[0], embedder.embed([first])[0]) == pytest.approx(
        1.0, abs=1e-5
    )
    assert abs(_unit_cosine(vectors[0], vectors[1])) < 0.5


def test_hash_embedder_of_a_concatenation_is_close_to_the_sum(
    measured: list[Measured],
) -> None:
    first, second = _row_texts(measured)
    embedder = HashEmbedder()

    left, right = embedder.embed([first]), embedder.embed([second])
    joined = embedder.embed([first + " " + second])[0]
    total = (left + right)[0]
    total = total / np.linalg.norm(total)

    assert _unit_cosine(joined, total.astype(np.float32)) > 0.9


def _raw_token_vector(token: str, dim: int = 128) -> NDArray[np.float64]:
    """The documented token vector, recomputed here on purpose instead of
    imported: a standard-normal draw seeded by the first eight bytes of the
    token's SHA-256. Its norm is not 1, so an embedding of several tokens is
    the sum of these raw vectors, not of the unit vectors `embed` returns for
    single tokens."""
    seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal(dim)


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float32]:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


@pytest.mark.parametrize(
    ("text", "counts"),
    [
        ("alpha", {"alpha": 1}),
        ("alpha beta", {"alpha": 1, "beta": 1}),
        ("beta alpha", {"alpha": 1, "beta": 1}),
        ("alpha alpha beta", {"alpha": 2, "beta": 1}),
        ("alpha beta beta beta", {"alpha": 1, "beta": 3}),
        ("alpha beta gamma", {"alpha": 1, "beta": 1, "gamma": 1}),
    ],
)
def test_hash_embedder_is_the_normalised_sum_of_raw_token_vectors(
    text: str, counts: dict[str, int]
) -> None:
    expected = _unit(
        sum(count * _raw_token_vector(token) for token, count in counts.items())
    )

    (vector,) = HashEmbedder().embed([text])

    assert _unit_cosine(vector, expected) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("plain", "decorated"),
    [("hello world", "  Hello,   WORLD! "), ("alpha beta", '"alpha" (beta)')],
)
def test_hash_embedder_ignores_case_spacing_and_edge_punctuation(
    plain: str, decorated: str
) -> None:
    embedder = HashEmbedder()

    plain_vector, decorated_vector = embedder.embed([plain, decorated])

    assert _unit_cosine(plain_vector, decorated_vector) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("empty", ["", "   ", "...", "\t"])
def test_hash_embedder_gives_empty_text_one_fixed_non_zero_vector(empty: str) -> None:
    embedder = HashEmbedder()

    vector, other = embedder.embed([empty, "x"])

    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(vector, embedder.embed([""])[0])
    assert not np.array_equal(vector, other)


def test_concept_embedder_maps_texts_of_one_concept_to_one_vector() -> None:
    embedder = ConceptEmbedder({"a": 1, "b": 1, "c": 2}, dim=32)

    a, b, c = embedder.embed(["a", "b", "c"])

    assert a.shape == (32,)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_concept_embedder_vector_for_unknown_text_depends_on_the_text() -> None:
    embedder = ConceptEmbedder({"a": 1})

    first, again, other = embedder.embed(["unknown", "unknown", "different"])

    assert np.array_equal(first, again)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
    assert abs(_unit_cosine(first, other)) < MAX_UNMAPPED_TEXT_COSINE


def test_embedders_accept_an_empty_batch() -> None:
    assert HashEmbedder().embed([]).shape == (0, 128)
    assert ConceptEmbedder({}).embed([]).shape == (0, 64)


@pytest.mark.subprocess
def test_support_does_not_depend_on_the_python_hash_seed(tmp_path: Path) -> None:
    snippet = (
        "import hashlib, sys\n"
        "import fasttext\n"
        "from pathlib import Path\n"
        "from tests.support.embedders import ConceptEmbedder, HashEmbedder\n"
        "from tests.support.planted_corpus import build_planted_corpus\n"
        "from tests.support.tiny_lid import train_tiny_lid_model\n"
        "def digest(text):\n"
        "    return hashlib.sha256(text.encode('utf-8')).hexdigest()\n"
        "hashed = HashEmbedder().embed(['hello world', ''])\n"
        "concept = ConceptEmbedder({'a': 3}).embed(['a', 'unmapped text'])\n"
        "print('embedders', hashlib.sha256(\n"
        "    hashed.tobytes() + concept.tobytes()).hexdigest())\n"
        "planted = build_planted_corpus()\n"
        "print('planted', digest(\n"
        "    planted.frame.select('id', 'src', 'tgt').write_csv()\n"
        "    + repr(planted.concepts) + repr(planted.expected_reason)))\n"
        "model = fasttext.load_model(\n"
        "    str(train_tiny_lid_model(Path(sys.argv[1]) / 'tiny.bin')))\n"
        "lines = [*planted.frame['src'][:5], *planted.frame['tgt'][:5]]\n"
        "print('lid', digest(repr([model.predict(line, k=-1) for line in lines])))\n"
    )

    def digests(hash_seed: str) -> str:
        completed = subprocess.run(
            [sys.executable, "-c", snippet, str(tmp_path / hash_seed)],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    assert digests("1") == digests("2") == digests("random")
