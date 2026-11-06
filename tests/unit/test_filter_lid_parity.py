"""B7 parity: the `GlotLID` wrapper agrees with the raw fastText Python API on
the real GlotLID model for 200 fixture lines.

The golden file `glotlid_200.json` was captured once from `fasttext`'s
`predict(text, k=1)` on `model_v3.bin` of `cis-lmu/glotlid` (the fastText
command-line binary is not installed; spec ruling 9). Its label for a line of
traditional Mongolian script is `mn_Mong`, which the model does not know: the
wrapper answers those lines before the model runs. The model is downloaded to
the Hugging Face cache, outside the repository.

Caveats of the fixture:
- The 20 `mn_Mong` lines are 2-4 word phrases in traditional Mongolian script,
  not validated by a Mongolian reader. Their golden entries are synthetic, not
  model output (the model calls that script `und_Mong`): for them parity checks
  the pre-detector against the hard-coded label `mn_Mong` with probability 1.0.
- The Mongolian-Cyrillic and Kazakh sentences were not reviewed by a native
  speaker.
- The other 180 entries are the raw fastText top-1 label and probability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rinchen.config.filters import LidConfig
from rinchen.config.languages import known_language_codes
from rinchen.config.loader import REPO_ROOT
from rinchen.corpus.schema import corpus_frame
from rinchen.filter import lid
from rinchen.filter.lid import GlotLID

GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"
LINES_PATH = GOLDEN_DIR / "glotlid_200_lines.tsv"
GOLDEN_PATH = GOLDEN_DIR / "glotlid_200.json"
LINES_PER_LANGUAGE = 20
TRADITIONAL_FIRST = 0x1800
TRADITIONAL_LAST = 0x18AF
# The golden probabilities are stored rounded to six decimals.
PROBABILITY_TOLERANCE = 1e-5


def _fixture_lines() -> list[tuple[str, str]]:
    lines = []
    for row in LINES_PATH.read_text(encoding="utf-8").splitlines():
        label, text = row.split("\t", 1)
        lines.append((label, text))
    return lines


def _golden() -> dict[str, Any]:
    golden: dict[str, Any] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return golden


def _is_traditional(text: str) -> bool:
    letters = [char for char in text if not char.isspace()]
    in_block = sum(
        TRADITIONAL_FIRST <= ord(char) <= TRADITIONAL_LAST for char in letters
    )
    return bool(letters) and in_block * 2 >= len(letters)


# --- the fixture files --------------------------------------------------------------


def test_the_fixture_has_twenty_lines_for_every_registry_language() -> None:
    lines = _fixture_lines()

    assert len(lines) == len(known_language_codes()) * LINES_PER_LANGUAGE
    counts = {code: 0 for code in known_language_codes()}
    for label, text in lines:
        counts[label] += 1
        assert text.strip() == text
        assert text
    assert set(counts.values()) == {LINES_PER_LANGUAGE}
    assert len({text for _, text in lines}) == len(lines)


def test_only_the_mn_mong_lines_are_traditional_script() -> None:
    for label, text in _fixture_lines():
        assert _is_traditional(text) == (label == "mn_Mong"), (label, text)


def test_the_golden_file_has_one_entry_for_every_fixture_line() -> None:
    golden = _golden()

    assert golden["model"] == "cis-lmu/glotlid model_v3.bin"
    assert golden["reference"] == "fasttext python API predict, k=1"
    assert "synthetic" in golden["note"]
    assert "native speaker" in golden["note"]
    assert len(golden["lines"]) == len(_fixture_lines())
    for entry in golden["lines"]:
        assert set(entry) == {"label", "prob"}
        assert entry["label"]
        assert 0.0 <= entry["prob"] <= 1.0


def test_the_golden_label_of_a_traditional_script_line_is_mn_mong() -> None:
    for (_, text), entry in zip(_fixture_lines(), _golden()["lines"], strict=True):
        if _is_traditional(text):
            assert entry == {"label": "mn_Mong", "prob": 1.0}
        else:
            assert entry["label"] != "mn_Mong"


# --- the real model -----------------------------------------------------------------


@pytest.fixture(scope="module")
def glotlid_model_file() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id="cis-lmu/glotlid", filename="model_v3.bin"))


@pytest.mark.network
def test_the_wrapper_top_1_label_equals_the_golden_label_for_every_line(
    glotlid_model_file: Path,
) -> None:
    texts = [text for _, text in _fixture_lines()]
    golden = _golden()["lines"]

    predictions = GlotLID(glotlid_model_file).predict(texts, k=1)

    assert len(predictions) == len(golden)
    mismatches = [
        (text, entry["label"], top[0][0] if top else None)
        for text, entry, top in zip(texts, golden, predictions, strict=True)
        if not top or top[0][0] != entry["label"]
    ]
    assert mismatches == []
    for entry, top in zip(golden, predictions, strict=True):
        assert top[0][1] == pytest.approx(entry["prob"], abs=PROBABILITY_TOLERANCE)


@pytest.mark.network
def test_the_default_config_flags_none_of_the_fixture_lines(
    glotlid_model_file: Path,
) -> None:
    lines = _fixture_lines()
    frame = corpus_frame(
        [
            {
                "src_lang": label,
                "tgt_lang": label,
                "src": text,
                "tgt": text,
                "origin": "human",
                "corpus": "parity",
                "split": "train",
            }
            for label, text in lines
        ]
    )

    result = lid.apply(frame.lazy(), LidConfig(model_path=glotlid_model_file)).collect()

    chinese = result.filter(pl.col("src_lang") == "zho_Hans")
    assert chinese.height == LINES_PER_LANGUAGE
    assert chinese["filter_reason"].null_count() == LINES_PER_LANGUAGE
    assert chinese["scores"].struct.field("lid_src").min() >= LidConfig().min_prob
    flagged = result.filter(pl.col("filter_reason").is_not_null())
    assert flagged.select("src_lang", "filter_reason").rows() == []
